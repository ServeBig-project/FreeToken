"""DFlash context KV. Full-history layers follow the target token pages; windowed layers keep a
bounded window behind the full->window slot mapping that the prefix cache shares."""
from __future__ import annotations

import numpy as np
import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from freetoken.kernel import store_cache
from freetoken.utils import align_ceil

# FlashInfer allocates both per wrapper; the integer workspace is then shared per attention mode.
_INT_WORKSPACE_BYTES = 8 << 20
_KV_LENS_BYTES = 32768 * 4


class DFlashLayout:
    """Which drafter layers keep full history and which a bounded window, and the GPU bytes
    that costs at a given target page count. Pure arithmetic: startup prices it before any
    tensor exists, and rebuild and status use the same numbers."""

    def __init__(self, config, draft):
        cap = config.dflash_attention_window
        # Committed history each layer reads: a native window, the cap on full layers, or all.
        self.limits = [w if w >= 0 else cap or None for _, w in draft.attention_modes]
        self.modes = draft.attention_modes
        limited = [i for i, limit in enumerate(self.limits) if limit is not None]
        self.window_layers = tuple(limited) if config.dflash_compact_kv else ()
        self.full_layers = tuple(i for i in range(len(self.limits)) if i not in self.window_layers)
        self.window = max((self.limits[i] for i in self.window_layers), default=0)
        self.row_bytes = 2 * draft.num_key_value_heads * draft.head_dim * config.dtype.itemsize
        self.config = config

    @property
    def request_slots(self) -> int:
        """Window slots one running request may hold between two releases."""
        from freetoken.scheduler.cache import _SWA_EVICTION_INTERVAL

        p = self.config.page_size
        steps = self.config.speculative_num_steps
        return align_ceil(self.window + _SWA_EVICTION_INTERVAL + steps + 1, p) + 2 * p

    def window_capacity(self, pages: int) -> int:
        """Usable window slots, never more than the target positions they map: per running
        request its read window, the committed tokens one release interval lets pass and one
        draft block (plus two pages of slack); one prefill batch; one window of prefix-cache
        copies when the host tier is on; and a retained tool-call window per request when
        anchors are on."""
        if not self.window_layers:
            return 0
        from freetoken.scheduler.cache import _SWA_RETAIN_GAP

        c, p = self.config, self.config.page_size
        up = lambda n: align_ceil(n, p)
        prefill = up(getattr(c, "max_extend_tokens", c.max_seq_len))
        copies = up(self.window) if c.prefix_cache_host_gib > 0 else 0
        anchors = (c.max_running_req * up(self.window + _SWA_RETAIN_GAP)
                   if getattr(c, "special_token_ckpt", False) else 0)
        return min(pages, c.max_running_req * self.request_slots + prefill + copies + anchors)

    def history_width(self, limit: int | None, pages: int) -> int:
        """Index entries one request's history plus its draft block can need for a mode."""
        width = min(self.config.max_seq_len, pages)
        return width if limit is None else min(width, limit + self.config.speculative_num_steps + 1)

    def bytes(self, pages: int) -> dict:
        c = self.config
        capacity = self.window_capacity(pages)
        batches = c.max_running_req * (c.max_running_req + 1) // 2  # wrappers for sizes 1..C
        modes = {mode: limit for mode, limit in zip(self.modes, self.limits)}
        indices = sum(self.history_width(limit, pages) for limit in modes.values())
        indptr = len(modes) * sum(3 * b + 2 for b in range(1, c.max_running_req + 1))
        # FlashInfer's per-wrapper buffers, then one staged history index list per mode.
        metadata = 4 * (batches * indices + indptr) + 4 * c.max_running_req * indices
        if self.window_layers:
            metadata += 8 * c.max_running_req * max(
                self.history_width(limit, pages) for mode, limit in modes.items()
                if limit is not None)
        if self.window_layers:
            metadata += 8 * (pages + 3) + 8 * capacity  # full->window mapping, free ring
        tokens = c.max_running_req * (c.speculative_num_steps + 1)
        workspace = (len(modes) * (_INT_WORKSPACE_BYTES + c.max_running_req * _KV_LENS_BYTES)
                     + 3 * 4 * tokens + 8 * (3 * tokens + 2 * c.max_running_req)
                     + 4 * c.max_running_req)
        return dict(full_context_bytes=len(self.full_layers) * (pages + 1) * self.row_bytes,
                    window_context_bytes=(len(self.window_layers) * (capacity + 1) * self.row_bytes
                                          if self.window_layers else 0),
                    metadata_bytes=metadata, workspace_bytes=workspace)

    def total_bytes(self, pages: int) -> int:
        return sum(self.bytes(pages).values())


class DFlashContext:
    def __init__(self, engine, model, layout: DFlashLayout):
        self.engine, self.model, self.layout = engine, model, layout
        self.window = layout.window
        self.feature_indices = {layer: i for i, layer in enumerate(model.target_layer_ids)}
        self.features = [None] * len(self.feature_indices)
        maximum = engine.config.max_running_req
        self.batch_sizes = list(range(1, maximum + 1))
        self.wrappers, self.integer_workspaces = {}, {}
        # Plan inputs per attention mode: its history limit and whether the window pool holds it.
        self.modes = {}
        for layer, (mode, limit) in enumerate(zip(model.attention_modes, layout.limits)):
            self.modes[mode] = (limit, layer in layout.window_layers)
        # Host plan descriptors, written in place: queries, last-page lengths, one indptr per mode.
        self.host_plan = torch.empty(2 * maximum + 1 + len(self.modes) * (maximum + 1),
                                     dtype=torch.int32, pin_memory=True)
        self.planned = torch.cuda.Event()  # the last plan's uploads have read host_plan
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
        from freetoken.attention.base import AttnType
        from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
        from freetoken.models.config import KVCacheGroupSpec

        c, layout, pages = self.model.config, self.layout, self.engine.num_pages
        requests = self.engine.config.max_running_req
        device = self.engine.device
        self.history = {mode: torch.empty(requests * layout.history_width(limit, pages),
                                          dtype=torch.int32, device=device)
                        for mode, (limit, _) in self.modes.items()}
        widths = [layout.history_width(limit, pages) for limit, windowed in self.modes.values()
                  if windowed]
        self.window_history = (torch.empty(requests * max(widths), dtype=torch.int64, device=device)
                               if widths else None)
        self.kv = self.pool = None
        if not layout.window_layers:
            self.kv = torch.empty(
                (2, c.num_hidden_layers, pages + 1, 1, c.num_key_value_heads, c.head_dim),
                dtype=self.engine.dtype, device=self.engine.device)
            return
        groups = [
            KVCacheGroupSpec("full", layout.full_layers, c.num_key_value_heads, c.head_dim, None),
            KVCacheGroupSpec("swa", layout.window_layers, c.num_key_value_heads, c.head_dim,
                             self.window, attn_type=AttnType.SWA)]
        self.pool = HybridSWAKVCache(
            groups, c.num_hidden_layers, pages + 1, 1, self.engine.dtype, self.engine.device,
            layout.window_capacity(pages) + 1)

    # ---- the window pool the prefix cache drives (only when windowed layers exist) ----
    @property
    def swa_paged(self) -> bool:
        return self.pool is not None

    @property
    def swa_num_tokens(self) -> int:
        return self.pool.swa_num_tokens

    def alloc_swa(self, indices):
        self.pool.alloc_swa(indices)

    def free_swa(self, indices):
        self.pool.free_swa(indices)

    def swa_available_size(self) -> int:
        return self.pool.swa_available_size()

    def window_units(self, indices):
        return self.pool.window_units(indices)

    def window_views(self):
        return self.pool.window_views()

    def paged_views(self):
        """Per-layer ``[pages, 2, 1, heads, head_dim]`` views of the full-history context KV."""
        if self.pool is not None:
            return self.pool.paged_views()
        return [self.kv[:, layer].movedim(1, 0) for layer in range(self.kv.shape[1])]

    def layer_cache(self, layer):
        if self.pool is not None:
            return self.pool.k_cache(layer), self.pool.v_cache(layer)
        return self.kv[0, layer], self.kv[1, layer]

    def full_stores(self):
        """K/V tensors indexed by target locations (graph capture scratch)."""
        if self.pool is None:
            return [self.layer_cache(layer) for layer in range(self.kv.shape[1])]
        return [self.layer_cache(layer) for layer in self.layout.full_layers]

    def reset_window(self):
        if self.pool is not None:
            self.pool._init_swa_paged_state()

    def geometry(self) -> dict:
        layout, pages = self.layout, self.engine.num_pages
        out = layout.bytes(pages)
        out.update(compact_kv=bool(self.engine.config.dflash_compact_kv),
                   attention_window=self.engine.config.dflash_attention_window,
                   window_tokens=self.window,
                   full_token_bytes=len(layout.full_layers) * layout.row_bytes,
                   window_token_bytes=len(layout.window_layers) * layout.row_bytes,
                   window_capacity_limit=layout.window_capacity(1 << 62))
        if self.pool is not None:
            out.update(window_slots=self.pool.swa_num_tokens - 1,
                       window_free_slots=self.pool.swa_available_size())
        return out

    def _wrapper(self, batch, mode, limit):
        key = batch, mode
        if key not in self.wrappers:
            zeros = lambda n: torch.zeros(n, dtype=torch.int32, device=self.engine.device)
            width = self.layout.history_width(limit, self.engine.num_pages)
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.engine.attn_backend.float_workspace_buffer, kv_layout="NHD", backend="fa2",
                use_cuda_graph=True, qo_indptr_buf=zeros(batch + 1),
                paged_kv_indptr_buf=zeros(batch + 1),
                paged_kv_indices_buf=zeros(batch * width),
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
        locations = self.slots(batch.out_loc)
        self.model.project_context(
            features, batch.positions, lambda layer, k, v: self.store(layer, k, v, locations))
        self.features = [None] * len(self.feature_indices)

    def slots(self, full):
        """Target locations and, once per batch, their window slots for every windowed layer."""
        return full, (self.window_units(full) if self.pool is not None else None)

    def store(self, layer, key, value, locations):
        c = self.model.config
        shape = (-1, c.num_key_value_heads, c.head_dim)
        keys, values = self.layer_cache(layer)
        full, window = locations
        indices = window if layer in self.layout.window_layers else full
        store_cache(k_cache=keys.view(shape), v_cache=values.view(shape),
                    indices=indices, k=key.flatten(1), v=value.flatten(1))

    def plan(self, rows, firsts, widths):
        self.planned.synchronize()  # already done in steady serving; matters back to back
        count, host = len(rows), self.host_plan.numpy()
        maximum = self.engine.config.max_running_req
        queries, last = self.host_plan[:count + 1], self.host_plan[maximum + 1:maximum + 1 + count]
        np.cumsum([0, *widths], out=host[:count + 1])
        host[maximum + 1:maximum + 1 + count] = 1
        ends = [p + w for p, w in zip(firsts, widths, strict=True)]
        table = self.engine.page_table
        c = self.model.config
        for slot, (mode, (limit, windowed)) in enumerate(self.modes.items()):
            # The history ends where the block starts; the whole block stays visible.
            starts = [0 if limit is None else max(0, first - limit) for first in firsts]
            base = 2 * maximum + 1 + slot * (maximum + 1)
            np.cumsum([0, *(e - s for s, e in zip(starts, ends))], out=host[base:base + count + 1])
            indices = self.history[mode][:int(host[base + count])]
            torch.cat([table[row, s:e] for row, s, e in zip(rows, starts, ends, strict=True)],
                      out=indices)
            if windowed:
                units = self.window_history[:indices.numel()]
                torch.index_select(self.pool.full_to_swa_index_mapping, 0, indices, out=units)
                indices.copy_(units)
            self._wrapper(count, mode, limit).plan(
                queries, self.host_plan[base:base + count + 1], indices, last,
                c.num_attention_heads, c.num_key_value_heads, c.head_dim, 1, causal=mode[0],
                window_left=mode[1], q_data_type=self.engine.dtype,
                kv_data_type=self.engine.dtype, non_blocking=True)
        self.planned.record()

    def attend(self, layer, query, key, value, *, batch_size, locations):
        self.store(layer, key, value, locations)
        wrapper = self.wrappers[batch_size, self.model.attention_modes[layer]]
        return wrapper.run(query, self.layer_cache(layer))

    def rebuild(self):
        self.kv = self.pool = None
        self.wrappers.clear()
        self.integer_workspaces.clear()
        self.allocate()
        self._prime_plans()
