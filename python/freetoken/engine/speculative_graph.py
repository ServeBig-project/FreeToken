"""Shared capture and replay of draft and ragged verification forwards."""
from copy import copy

import torch

from freetoken.core import Batch, get_global_ctx
from .graph import GraphCaptureBuffer
from .model_forward import forward_model


class SpeculativeGraphs:
    def __init__(self, runner, model, config, max_seq_len: int, vocab_size: int):
        self.runner = runner
        self.top_k = config.speculative_draft_experts
        self.router = (config.speculative_draft_residency == "router"
                       and not config.speculative_draft_load_missing)
        self.graphs = {}
        self.verify_sizes: dict[int, list[int]] = {}  # captured verify token counts per batch size
        self.attention = runner.attn_backend.create_speculative_graphs(max_seq_len)
        ctx = get_global_ctx()
        kv = ctx.kv_cache
        stores = []
        for layer in range(kv.num_layers):
            try:
                stores.append((kv.k_cache(layer), kv.v_cache(layer)))
            except KeyError:  # linear-attention layer without paged KV
                continue
        usable_tokens = stores[0][0].flatten(0, 1).shape[0] - 1
        if ctx.draft_context is not None:
            stores.extend(ctx.draft_context.capture_stores())
        max_tokens = min(runner.max_graph_bs * (config.speculative_num_steps + 1), usable_tokens)
        self.max_tokens = max_tokens
        self.query_width = config.speculative_num_steps + 1
        # Admission is LRU up to this many query tokens and layer-distance above it, so
        # padding a smaller verification batch past it could change the eviction policy.
        cache = runner.moe_offload_cache
        self.exact_tokens = max_tokens if cache is None else max(4, cache.decode_cache_size // (
            config.model_config.num_experts_per_tok * config.model_config.num_moe_layers))
        self.real_tokens = torch.empty((), dtype=torch.int32, device=runner.device)
        self.dummy_slot = ctx.page_table[runner.dummy_req.table_idx, 0]
        self.buffer = GraphCaptureBuffer.init(max_tokens, vocab_size, runner.device)
        state_pool = ctx.linear_state_pool
        self.state = (state_pool.create_speculative_graphs(runner.max_graph_bs, self.query_width, runner.device)
                      if state_pool is not None else None)
        self.available = (
            torch.ones(config.model_config.num_moe_layers, config.model_config.num_experts,
                       dtype=torch.bool, device=runner.device) if self.router else None
        )
        # Rebuild can preserve existing prefix KV. Capture scratch must not overwrite it.
        scratch = [storage.flatten(0, 1)[:max_tokens] for pair in stores for storage in pair]
        saved = [tensor.clone() for tensor in scratch]
        restore_windows = (ctx.draft_context.protect_capture(max_tokens)
                           if ctx.draft_context is not None else lambda: None)
        try:
            for bs in reversed(runner.graph_bs_list):
                if bs > max_tokens:
                    continue
                if config.speculative_draft_model_path is None:
                    self._capture(model, "draft", [1] * bs)
                limit = min(bs * self.query_width, max_tokens)
                # Exact shapes where the admission policy could flip, and the full window. The
                # power-of-two batch sizes most rounds run at also get 2-5 inputs per request
                # (up to four drafts), replayed at the smallest shape that holds the round;
                # every captured graph stays resident (~15 MB each on Qwen3.6), so rare batch
                # sizes keep only the full window.
                sizes = {*range(bs, min(limit, self.exact_tokens) + 1), limit}
                if bs & (bs - 1) == 0:
                    sizes |= {min(k * bs, limit) for k in range(2, 6)}
                sizes = sorted(sizes)
                self.verify_sizes[bs] = sizes
                # The first (largest) plan sets FlashInfer's maximum total query-row bound.
                for tokens in reversed(sizes):
                    remaining = tokens - bs
                    lengths = []
                    for _ in range(bs):
                        extra = min(remaining, config.speculative_num_steps)
                        lengths.append(1 + extra)
                        remaining -= extra
                    self._capture(model, "verify", lengths)
        finally:
            for tensor, original in zip(scratch, saved, strict=True):
                tensor.copy_(original)
            restore_windows()
            runner._reset_moe_offload_cache()
            torch.cuda.synchronize(runner.device)

    def _capture(self, model, phase, lengths):
        runner = self.runner
        tokens, bs = sum(lengths), len(lengths)
        table = torch.zeros(bs, max(lengths), dtype=torch.int32, device=runner.device)
        reqs = []
        offset = 0
        self.buffer.input_ids[:tokens].zero_()
        for index, length in enumerate(lengths):
            req = copy(runner.dummy_req)
            req.table_idx, req.cached_len, req.device_len = index, 0, length
            reqs.append(req)
            table[index, :length] = torch.arange(offset, offset + length, device=runner.device)
            self.buffer.out_loc[offset : offset + length] = table[index, :length]
            self.buffer.positions[offset : offset + length] = torch.arange(length, device=runner.device)
            offset += length
        batch = Batch(reqs, decode_size=bs if phase == "draft" else 0,
                      draft_experts=self.top_k if phase == "draft" else None,
                      is_speculative_verify=phase == "verify",
                      draft_available_experts=self.available if phase == "draft" else None)
        if phase == "verify" and tokens > self.exact_tokens:
            self.real_tokens.fill_(tokens)
            batch.num_token_non_padded = self.real_tokens
        batch.padded_reqs = reqs
        batch.input_ids = self.buffer.input_ids[:tokens]
        batch.positions = self.buffer.positions[:tokens]
        batch.out_loc = self.buffer.out_loc[:tokens]
        if self.state is not None:
            self.state.prepare_capture(batch, lengths, tokens)
        self.attention.prepare_capture(batch, table)
        graph = torch.cuda.CUDAGraph()
        with get_global_ctx().forward_batch(batch):
            # Admission reads GPU state on replay, so warmups can retain expert residency.
            self.buffer.logits[:tokens] = forward_model(model)
            with torch.cuda.graph(graph, pool=runner.pool, stream=runner.stream):
                self.buffer.logits[:tokens] = forward_model(model)
        self.graphs[(phase, bs, tokens)] = graph

    def verify_tokens(self, batch_size, tokens):
        """Physical size a verify batch of ``tokens`` queries replays at, shared by replay and
        the SD cost model: the smallest captured shape that holds it, else eager ``tokens``."""
        return next((size for size in self.verify_sizes.get(batch_size, ()) if size >= tokens),
                    tokens)

    def _key(self, batch):
        tokens = batch.positions.numel()
        if batch.is_speculative_verify:
            tokens = self.verify_tokens(batch.size, tokens)
        return ("verify" if batch.is_speculative_verify else "draft", batch.size, tokens)

    def can_replay(self, batch) -> bool:
        if batch.draft_experts is not None and batch.draft_experts != self.top_k:
            return False
        return self._key(batch) in self.graphs

    def replay(self, batch):
        key = self._key(batch)
        real = batch.positions.numel()
        if batch.is_speculative_verify and real > self.exact_tokens:
            self.real_tokens.fill_(real)
            if real < key[2]:
                self.buffer.input_ids[real:key[2]].zero_()
                self.buffer.positions[real:key[2]].zero_()
                self.buffer.out_loc[real:key[2]] = self.dummy_slot
        self.buffer.copy_from(batch)
        if batch.draft_experts is not None and self.available is not None:
            self.available.copy_(batch.draft_available_experts)
        if self.state is not None:
            self.state.prepare_replay(batch, key[2])
        self.attention.prepare_replay(batch)
        self.graphs[key].replay()
        return self.buffer.logits[:real]
