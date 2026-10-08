"""DFlash context KV follows the target token pages, including public prefixes."""
from __future__ import annotations

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from freetoken.kernel import store_cache


class DFlashContext:
    def __init__(self, engine, model):
        self.engine, self.model = engine, model
        c = model.config
        self.unit_bytes = 2 * c.num_hidden_layers * c.num_key_value_heads * c.head_dim * engine.dtype.itemsize
        self.feature_indices = {layer: i for i, layer in enumerate(model.target_layer_ids)}
        self.features = [None] * len(self.feature_indices)
        maximum = engine.config.max_running_req
        self.batch_sizes = list(range(1, maximum + 1))
        self.wrappers, self.integer_workspaces = {}, {}
        self.index_width = engine.max_seq_len
        self.allocate()
        self._prime_plans()

    def _prime_plans(self):
        # Graph-mode plans also serve eager fallback: the first plan must cover
        # later, longer blocks, even when the first real request has a short tail.
        width = min(self.engine.config.speculative_num_steps + 1, self.engine.max_seq_len)
        dummy = self.engine.config.max_running_req
        for batch in self.batch_sizes:
            self.plan([dummy] * batch, [0] * batch, [width] * batch)

    def allocate(self):
        c = self.model.config
        self.kv = torch.empty(
            (2, c.num_hidden_layers, self.engine.num_pages + 1, 1, c.num_key_value_heads, c.head_dim),
            dtype=self.engine.dtype, device=self.engine.device)

    def paged_views(self):
        """Per-layer ``[pages, 2, 1, heads, head_dim]`` views of the draft context KV, which
        follows the target's token pages."""
        return [self.kv[:, layer].movedim(1, 0) for layer in range(self.kv.shape[1])]

    def _wrapper(self, batch, mode):
        key = batch, mode
        if key not in self.wrappers:
            zeros = lambda n: torch.zeros(n, dtype=torch.int32, device=self.engine.device)
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.engine.attn_backend.float_workspace_buffer, kv_layout="NHD", backend="fa2",
                use_cuda_graph=True, qo_indptr_buf=zeros(batch + 1),
                paged_kv_indptr_buf=zeros(batch + 1),
                paged_kv_indices_buf=zeros(batch * self.index_width),
                paged_kv_last_page_len_buf=zeros(batch))
            if mode not in self.integer_workspaces:
                self.integer_workspaces[mode] = wrapper._int_workspace_buffer
            else:
                wrapper._int_workspace_buffer = self.integer_workspaces[mode]
            self.wrappers[key] = wrapper
        return self.wrappers[key]

    def record(self, layer, hidden, residual, features=None):
        """Keep a target layer's output; layered execution passes its state's dict."""
        if layer in self.feature_indices:
            output = hidden if residual is None else hidden + residual
            if features is None:
                self.features[self.feature_indices[layer]] = output
            else:
                features[layer] = output

    def flush(self, batch, features=None):
        features = torch.cat(self.features if features is None else
                             [features[layer] for layer in self.feature_indices], dim=-1)
        self.model.project_context(
            features, batch.positions,
            lambda layer, k, v: self.store(layer, k, v, batch.out_loc))
        self.features = [None] * len(self.feature_indices)

    def store(self, layer, key, value, locations):
        c = self.model.config
        shape = (-1, c.num_key_value_heads, c.head_dim)
        store_cache(k_cache=self.kv[0, layer].view(shape), v_cache=self.kv[1, layer].view(shape),
                    indices=locations, k=key.flatten(1), v=value.flatten(1))

    def plan(self, rows, firsts, widths):
        pin = dict(dtype=torch.int32, pin_memory=True)
        queries = torch.tensor([0, *widths], **pin).cumsum(0, dtype=torch.int32)
        lengths = [p + w for p, w in zip(firsts, widths, strict=True)]
        indptr = torch.tensor([0, *lengths], **pin).cumsum(0, dtype=torch.int32)
        indices = torch.cat([self.engine.page_table[row, :n] for row, n in zip(rows, lengths, strict=True)])
        last = torch.ones(len(rows), **pin)
        c = self.model.config
        for mode in set(self.model.attention_modes):
            self._wrapper(len(rows), mode).plan(
                queries, indptr, indices, last, c.num_attention_heads, c.num_key_value_heads,
                c.head_dim, 1, causal=mode[0], window_left=mode[1],
                q_data_type=self.engine.dtype, kv_data_type=self.engine.dtype,
                non_blocking=True)

    def attend(self, layer, query, key, value, *, batch_size, locations):
        self.store(layer, key, value, locations)
        wrapper = self.wrappers[batch_size, self.model.attention_modes[layer]]
        return wrapper.run(query, (self.kv[0, layer], self.kv[1, layer]))

    @property
    def metadata_bytes(self):
        tensors = {}
        shared = self.engine.attn_backend.float_workspace_buffer
        for wrapper in self.wrappers.values():
            for value in vars(wrapper).values():
                if torch.is_tensor(value) and value.is_cuda and value is not shared:
                    tensors[value.data_ptr()] = value.numel() * value.element_size()
        return sum(tensors.values())

    def rebuild(self):
        self.kv = None
        self.wrappers.clear()
        self.integer_workspaces.clear()
        self.index_width = self.engine.max_seq_len
        self.allocate()
        self._prime_plans()
