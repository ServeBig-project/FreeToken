"""Parallel DFlash proposals with the shared target verifier and token-page lifetime."""
from __future__ import annotations

import torch

from freetoken.speculative import DraftResult
from .block_cost import BlockController
from .dflash_cache import DFlashContext


class DFlashRuntime:
    def __init__(self, engine, model, layout):
        self.engine, self.model = engine, model
        self.context = DFlashContext(engine, model, layout)
        self.graphs = {}
        self.sizes = {}
        requests = engine.config.max_running_req
        maximum = requests * (engine.config.speculative_num_steps + 1)
        self.inputs, self.positions, self.locations = (
            torch.zeros(maximum, dtype=torch.int32, device=engine.device) for _ in range(3))
        # Per-round descriptors, packed: positions and flat page-table indices of every draft
        # position, each request's block offset and flat first-token index, the sampled rows.
        self.host = torch.empty(3 * maximum + 2 * requests, dtype=torch.int64, pin_memory=True)
        self.staged = torch.empty_like(self.host, device=engine.device)
        self.first_tokens = torch.empty(requests, dtype=torch.int32, device=engine.device)
        self.uploaded = torch.cuda.Event()
        self.logits = None
        limit = engine.config.speculative_num_steps
        self.widths = sorted({1, limit, *(n for n in (2, 4, 8) if n <= limit)})

    @property
    def feature_bytes_per_token(self):
        return len(self.model.target_layer_ids) * self.model.hidden_size * self.engine.dtype.itemsize

    def feature_buffers(self, hidden):
        return {layer: torch.zeros_like(hidden) for layer in self.model.target_layer_ids}

    def record(self, layer, hidden, residual, features=None):
        self.context.record(layer, hidden, residual, features)

    def flush(self, batch, features=None):
        self.context.flush(batch, features)

    def capture_stores(self):
        return self.context.full_stores()

    def protect_capture(self, tokens):
        """Target graph capture writes context at locations [0, tokens): point their window
        bindings at the sentinel meanwhile; the returned function restores them."""
        pool = self.context.pool
        if pool is None:
            return lambda: None
        mapping = pool.full_to_swa_index_mapping[:tokens]
        saved = mapping.clone()
        mapping.zero_()
        return lambda: mapping.copy_(saved)

    def geometry(self):
        out = self.context.geometry()
        out["weight_bytes"] = self.model.weight_bytes
        out["context_bytes"] = out["full_context_bytes"] + out["window_context_bytes"]
        out["reserved_bytes"] = (out["weight_bytes"] + out["context_bytes"]
                                 + out["metadata_bytes"] + out["workspace_bytes"])
        return dict(active=True, **out)

    def _forward(self, batch_size, tokens):
        embeddings = self.engine.model.model.embed_tokens.forward(self.inputs[:tokens])
        embeddings = embeddings * self.model.input_embedding_scale
        locations = self.context.slots(self.locations[:tokens])
        hidden = self.model(
            embeddings, self.positions[:tokens],
            lambda layer, q, k, v: self.context.attend(
                layer, q, k, v, batch_size=batch_size, locations=locations))
        logits = self.engine.model.lm_head.forward_selected(hidden).float()
        logits = logits * self.model.output_multiplier
        softcap = self.model.final_logit_softcapping
        if softcap is not None and softcap > 0:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def capture_graphs(self, runner):
        if runner.speculative is None:
            return
        before = torch.cuda.memory_reserved(self.engine.device)
        self._capture_graphs(runner)
        runner.speculative_reserved_bytes += max(0, torch.cuda.memory_reserved(self.engine.device) - before)

    def _capture_graphs(self, runner):
        self.logits = runner.speculative.buffer.logits
        dummy = self.engine.config.max_running_req
        self.locations.fill_(self.engine.num_pages)
        self.inputs.fill_(self.model.mask_token_id)
        for batch in reversed(self.context.batch_sizes):
            if batch not in runner.speculative.batch_sizes or batch > self.logits.shape[0]:
                continue
            sizes = sorted({min(batch * (n + 1), self.logits.shape[0]) for n in self.widths})
            self.sizes[batch] = sizes
            for tokens in reversed(sizes):
                remaining, widths = tokens - batch, []
                for _ in range(batch):
                    extra = min(remaining, self.engine.config.speculative_num_steps)
                    widths.append(1 + extra)
                    remaining -= extra
                positions = [j for width in widths for j in range(width)]
                self.positions[:tokens].copy_(torch.tensor(positions, device=self.engine.device))
                self.context.plan([dummy] * batch, [0] * batch, widths)
                self.logits[:tokens] = self._forward(batch, tokens)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=runner.pool, stream=self.engine.stream):
                    self.logits[:tokens] = self._forward(batch, tokens)
                self.graphs[batch, tokens] = graph

    def destroy_graphs(self):
        self.graphs.clear()
        self.sizes.clear()
        self.logits = None

    def rebuild(self):
        self.context.rebuild()

    def physical_tokens(self, count, actual):
        """Positions a draft of ``actual`` tokens over ``count`` requests runs at."""
        return next((n for n in self.sizes.get(count, ()) if n >= actual), actual)

    def propose_logits(self, batch, lengths, token_table):
        count = batch.size
        rows = [r.table_idx for r in batch.reqs]
        firsts = [r.cached_len for r in batch.reqs]
        widths = [n + 1 for n in lengths]
        offsets, offset = [], 0
        for width in widths:
            offsets.append(offset)
            offset += width
        actual = offset
        physical = self.physical_tokens(count, actual)
        graph = self.graphs.get((count, physical))
        stride = self.engine.page_table.shape[1]
        positions, flat = [], []
        for row, first, width in zip(rows, firsts, widths, strict=True):
            for position in range(first, first + width):
                positions.append(position)
                flat.append(row * stride + position)
        sampled = [off + 1 + j for off, n in zip(offsets, lengths, strict=True) for j in range(n)]
        self.uploaded.synchronize()  # the previous round's upload has read the host buffer
        host, heads = self.host.numpy(), 2 * actual + 2 * count
        used = heads + len(sampled)
        host[:actual] = positions
        host[actual:2 * actual] = flat
        host[2 * actual:2 * actual + count] = offsets
        host[2 * actual + count:heads] = [row * stride + p for row, p in zip(rows, firsts)]
        host[heads:used] = sampled
        staged = self.staged[:used]
        staged.copy_(self.host[:used], non_blocking=True)
        self.uploaded.record()
        self.positions[:physical].zero_()
        self.positions[:actual].copy_(staged[:actual])
        self.locations[:physical].fill_(self.engine.num_pages)
        torch.index_select(self.engine.page_table.view(-1), 0, staged[actual:2 * actual],
                           out=self.locations[:actual])
        self.inputs[:physical].fill_(self.model.mask_token_id)
        torch.index_select(token_table.view(-1), 0, staged[2 * actual + count:heads],
                           out=self.first_tokens[:count])
        self.inputs.index_copy_(0, staged[2 * actual:2 * actual + count], self.first_tokens[:count])
        self.context.plan(rows, firsts, widths)
        if graph is not None:
            graph.replay()
            result = self.logits[:actual]
            key = ("draft", count, actual, physical)
            counters = self.engine.graph_runner.replay_counts
            counters[key] = counters.get(key, 0) + 1
        else:
            if self.graphs:
                self.engine.graph_runner.eager_counts["draft"] += 1
            result = self._forward(count, actual)[:actual]
        return result.index_select(0, staged[heads:used])


class DFlashDrafter:
    uses_target_state = False

    def __init__(self, engine, table, generator):
        self.engine, self.table, self.generator = engine, table, generator
        self.runtime = engine.dflash
        self.control = (BlockController(engine, self.runtime)
                        if engine.config.speculative_adaptive_cost else None)
        self.positions = [0, 0]  # real and physical draft positions

    def plan(self, batch, lengths):
        return self.control.plan(batch, lengths) if self.control is not None else lengths

    def propose(self, batch, views, starts, lengths):
        engine, sampler = self.engine, self.engine.sampler
        selected = self.runtime.propose_logits(batch, lengths, self.table.token_pool)
        real = batch.size + sum(lengths)
        self.positions[0] += real
        self.positions[1] += self.runtime.physical_tokens(batch.size, real)
        probabilities = sampler.probabilities(selected, sampler.prepare(batch, repeats=lengths))
        samples = torch.multinomial(probabilities, 1, generator=self.generator).flatten().to(torch.int32)
        width = max(lengths) + 1
        tokens = torch.zeros(batch.size, width, dtype=torch.int32, device=engine.device)
        q = torch.zeros(batch.size, width, sampler.vocab_size, dtype=torch.float32, device=engine.device)
        offset = 0
        for i, length in enumerate(lengths):
            tokens[i, :length] = samples[offset:offset + length]
            q[i, :length] = probabilities[offset:offset + length]
            self.table.token_pool[views[i].table_idx, starts[i]:starts[i] + length] = tokens[i, :length]
            offset += length
        return DraftResult(tokens, q, lengths)

    def snapshot(self):
        out = dict(drafter="dflash", residency_stops=0, draft_expert_loads=0,
                   dflash_control="fixed", dflash_draft_positions=self.positions[0],
                   dflash_draft_physical_positions=self.positions[1])
        if self.control is not None:
            out.update(self.control.snapshot())
        return out
