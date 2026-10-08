"""Shared capture and replay of draft and ragged verification forwards."""
from copy import copy

import torch

from freetoken.core import Batch, get_global_ctx
from .graph import GraphCaptureBuffer, _LayerRangeCapture
from .model_forward import forward_model


class SpeculativeGraphs:
    def __init__(self, runner, model, config, max_seq_len: int, vocab_size: int):
        self.runner = runner
        self.top_k = config.speculative_draft_experts
        self.router = (config.speculative_draft_residency == "router"
                       and not config.speculative_draft_load_missing)
        self.graphs = {}
        # Full-window verify rows over the decode layer ranges, for rounds beside a wave.
        self.ranges: dict[tuple[int, int, int], _LayerRangeCapture] = {}
        self.range_inputs = None
        self._prepared_range_batch = None
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
        self.exact_tokens = max(4, runner.moe_offload_cache.decode_cache_size // (
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
            if config.speculative_phase != "outwave" and runner.layer_range_group_end_candidates:
                self._capture_ranges(model, runner.graph_bs_list)
        finally:
            for tensor, original in zip(scratch, saved, strict=True):
                tensor.copy_(original)
            runner._reset_moe_offload_cache()
            torch.cuda.synchronize(runner.device)

    def _capture(self, model, phase, lengths):
        batch = self._prepare_capture(phase, lengths)
        tokens = sum(lengths)
        graph = torch.cuda.CUDAGraph()
        with get_global_ctx().forward_batch(batch):
            # Admission reads GPU state on replay, so warmups can retain expert residency.
            self.buffer.logits[:tokens] = forward_model(model)
            with torch.cuda.graph(graph, pool=self.runner.pool, stream=self.runner.stream):
                self.buffer.logits[:tokens] = forward_model(model)
        self.graphs[(phase, len(lengths), tokens)] = graph

    def _capture_ranges(self, model, batch_sizes):
        runner = self.runner
        adapter = runner.layered_execution_adapter
        groups = [(start, end) for start, ends in sorted(runner.layer_range_group_end_candidates.items())
                  for end in ends]
        for bs in sorted(batch_sizes, reverse=True):
            tokens = bs * self.query_width
            if tokens > self.max_tokens:
                continue
            batch = self._prepare_capture("verify", [self.query_width] * bs)
            pool = None  # one pool per size, as for decode ranges
            with get_global_ctx().forward_batch(batch):
                if self.range_inputs is None:
                    seed = model.begin_layer_group_prefill(batch.input_ids)
                    seed = model.advance_layer_group_prefill(seed, groups[0][1])
                    self.range_inputs = adapter.create_range_graph_inputs(seed)
                    del seed
                for start, end in groups:
                    def run():
                        state = (model.begin_layer_group_prefill(batch.input_ids) if start == 0 else
                                 adapter.make_range_graph_state(self.range_inputs, start, tokens))
                        return model.advance_layer_group_prefill(state, end)

                    run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=runner.stream):
                        captured = run()
                    runner._reset_moe_offload_cache()
                    pool = pool or graph.pool()
                    self.ranges[start, end, bs] = _LayerRangeCapture(graph, captured)

    def has_ranges(self, batch) -> bool:
        tokens = batch.positions.numel()
        return tokens == batch.size * self.query_width and any(
            key[2] == batch.size for key in self.ranges)

    def prepare_ranges(self, batch) -> None:
        if batch is self._prepared_range_batch:
            return
        # An eager stage of this batch may have planned its verify path already.
        batch.attn_metadata.prefill.initialized = False
        self._stage(batch, batch.positions.numel())
        self._prepared_range_batch = batch

    def replay_range(self, batch, state, start, end):
        adapter = self.runner.layered_execution_adapter
        tokens = batch.positions.numel()
        capture = self.ranges[start, end, batch.size]
        if start:
            adapter.stage_range_graph_inputs(self.range_inputs, state, tokens, start)
        capture.graph.replay()
        return adapter.finish_range_graph_replay(capture.output, tokens, end)

    def _prepare_capture(self, phase, lengths):
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
        return batch

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
        if batch.draft_experts is not None and (
                batch.draft_experts != self.top_k
                or not self.runner.moe_offload_cache.captured_drafts_safe):
            return False
        return self._key(batch) in self.graphs

    def replay(self, batch):
        key = self._key(batch)
        self._stage(batch, key[2])
        self.graphs[key].replay()
        return self.buffer.logits[:batch.positions.numel()]

    def _stage(self, batch, physical):
        """Copy a real batch into the captured inputs, padding to ``physical`` rows."""
        real = batch.positions.numel()
        if batch.is_speculative_verify and real > self.exact_tokens:
            self.real_tokens.fill_(real)
            if real < physical:
                self.buffer.input_ids[real:physical].zero_()
                self.buffer.positions[real:physical].zero_()
                self.buffer.out_loc[real:physical] = self.dummy_slot
        self.buffer.copy_from(batch)
        if batch.draft_experts is not None:
            self.runner.moe_offload_cache.wait_resident_copies()
            if self.available is not None:
                self.available.copy_(batch.draft_available_experts)
        if self.state is not None:
            self.state.prepare_replay(batch, physical)
        self.attention.prepare_replay(batch)
