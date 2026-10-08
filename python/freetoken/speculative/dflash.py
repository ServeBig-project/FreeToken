"""Parallel DFlash proposals with the shared target verifier and token-page lifetime."""
from __future__ import annotations

import torch

from freetoken.speculative import DraftResult
from .block_cost import BlockDraftCost
from .dflash_cache import DFlashContext


class DFlashRuntime:
    def __init__(self, engine, model):
        self.engine, self.model = engine, model
        self.context = DFlashContext(engine, model)
        self.graphs = {}
        self.sizes = {}
        maximum = engine.config.max_running_req * (engine.config.speculative_num_steps + 1)
        self.inputs, self.positions, self.locations = (
            torch.zeros(maximum, dtype=torch.int32, device=engine.device) for _ in range(3))
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
        return [(self.context.kv[0, layer], self.context.kv[1, layer])
                for layer in range(self.model.config.num_hidden_layers)]

    @property
    def storage_bytes_actual(self):
        return self.storage_bytes_for_pages(self.engine.num_pages)

    def storage_bytes_for_pages(self, pages):
        buffers = (self.inputs, self.positions, self.locations, self.model.inv_freq)
        index_width = min(self.engine.config.max_seq_len, pages)
        index_delta = ((index_width - self.context.index_width) * 4
                       * sum(self.context.batch_sizes) * len(set(self.model.attention_modes)))
        return ((pages + 1) * self.context.unit_bytes + self.context.metadata_bytes
                + index_delta + sum(t.numel() * t.element_size() for t in buffers))

    def geometry(self):
        context = (self.engine.num_pages + 1) * self.context.unit_bytes
        return dict(active=True, weight_bytes=self.model.weight_bytes, context_bytes=context,
                    metadata_bytes=self.storage_bytes_actual - context,
                    reserved_bytes=self.model.weight_bytes + self.storage_bytes_actual)

    def _forward(self, batch_size, tokens):
        embeddings = self.engine.model.model.embed_tokens.forward(self.inputs[:tokens])
        embeddings = embeddings * self.model.input_embedding_scale
        hidden = self.model(
            embeddings, self.positions[:tokens],
            lambda layer, q, k, v: self.context.attend(
                layer, q, k, v, batch_size=batch_size, locations=self.locations[:tokens]))
        logits = self.engine.model.lm_head.forward_selected(hidden).float()
        logits = logits * self.model.output_multiplier
        softcap = self.model.final_logit_softcapping
        if softcap is not None and softcap > 0:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def capture_graphs(self, runner):
        if runner.speculative is None:
            return
        self.logits = runner.speculative.buffer.logits
        dummy = self.engine.config.max_running_req
        self.locations.fill_(self.engine.num_pages)
        self.inputs.fill_(self.model.mask_token_id)
        for batch in reversed(self.context.batch_sizes):
            if batch not in runner.graph_bs_list or batch > self.logits.shape[0]:
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
        physical = next((n for n in self.sizes.get(count, ()) if n >= actual), actual)
        graph = self.graphs.get((count, physical))
        device = self.engine.device
        to = lambda x: torch.tensor(x, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        positions = to([p + j for p, w in zip(firsts, widths, strict=True) for j in range(w)])
        table_rows = to([row for row, w in zip(rows, widths, strict=True) for _ in range(w)])
        total = actual
        self.inputs[:physical].fill_(self.model.mask_token_id)
        self.positions[:physical].zero_()
        self.positions[:total].copy_(positions)
        self.locations[:physical].fill_(self.engine.num_pages)
        self.locations[:total].copy_(self.engine.page_table[table_rows, positions])
        self.inputs[to(offsets)] = token_table[to(rows[:count]), to(firsts[:count])]
        self.context.plan(rows, firsts, widths)
        if graph is not None:
            graph.replay()
            result = self.logits[:actual]
            key = ("draft", count, actual, physical)
            counters = self.engine.graph_runner.replay_counts
            counters[key] = counters.get(key, 0) + 1
        else:
            result = self._forward(count, total)[:actual]
        return result, offsets


class DFlashDrafter:
    uses_target_state = False

    def __init__(self, engine, table, generator):
        self.engine, self.table, self.generator = engine, table, generator
        self.runtime = engine.dflash
        self.cost = BlockDraftCost(engine)

    def plan(self, batch, lengths):
        return self.cost.plan(lengths)

    def propose(self, batch, views, starts, lengths):
        engine, sampler = self.engine, self.engine.sampler
        self.cost.begin()
        logits, offsets = self.runtime.propose_logits(batch, lengths, self.table.token_pool)
        self.cost.end(batch.size, max(lengths))
        query_rows = [off + 1 + j for off, n in zip(offsets, lengths, strict=True) for j in range(n)]
        selected = logits[query_rows]
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
        return dict(drafter="dflash", residency_stops=0, draft_expert_loads=0, **self.cost.snapshot())

    def observe_acceptance(self, lengths, accepted):
        self.cost.observe_acceptance(lengths, accepted)
