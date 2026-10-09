from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import KVCacheGroupSpec
from freetoken.utils import align_ceil, div_even

from .base import BaseKVCachePool
from .runtime_pool import upload


@dataclass(frozen=True)
class _LayerRef:
    group: str
    index: int


@dataclass(frozen=True)
class _KVGroupStorage:
    buffer: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    storage_shape: tuple[int, int, int]
    banks: list | None = None  # shared runtime banks of the buffer


class HybridSWAKVCache(BaseKVCachePool):
    """SGLang-style wrapper for hybrid full/SWA attention KV storage."""

    def __init__(
        self,
        groups: Sequence[KVCacheGroupSpec],
        num_layers: int,
        num_full_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_swa_tokens: int | None = None,
        runtime=None,
    ) -> None:
        # A drafter whose layers are all windowed still brings an empty full group.
        specs = {group.name: group for group in groups if group.num_layers > 0 or group.name == "full"}
        if set(specs) != {"full", "swa"}:
            raise ValueError(f"HybridSWAKVCache requires full and swa groups, got {sorted(specs)}")

        self._num_layers = num_layers
        self._device = device
        self._dtype = dtype
        self._full_num_tokens = num_full_pages * page_size
        self._swa_num_tokens = num_swa_tokens if num_swa_tokens is not None else self._full_num_tokens
        self._page_size = page_size
        # Global-paged SWA (== sglang SWAKVPool): a swa pool reached through a dense full->swa
        # slot mapping + an independent swa free-list. Used with and without prefix reuse, which
        # differ only in whether the prefix tree keeps windows. Always paged; the mapping/free-list are built below
        # (translate/store_kv unconditionally index full_to_swa_index_mapping).
        self._swa_paged = True

        tp_size = get_tp_info().size
        # Shared runtime: full-group pages map with the target's pages (``page_banks``); window
        # slots are chosen on the host and mapped as they are bound (``slot_units``).
        self.page_banks, self.slot_units = [], None
        self.full_kv_pool = self._allocate_group(
            specs["full"],
            tp_size=tp_size,
            outer_size=num_full_pages,
            inner_size=page_size,
            dtype=dtype,
            device=device,
            runtime=runtime,
        )
        self.swa_kv_pool = self._allocate_group(
            specs["swa"],
            tp_size=tp_size,
            outer_size=self._swa_num_tokens,
            inner_size=1,
            dtype=dtype,
            device=device,
            runtime=runtime,
        )
        if runtime is not None:
            from .runtime_pool import Units

            self.page_banks = self.full_kv_pool.banks
            self.slot_units = Units(self.swa_kv_pool.banks)
            self.slot_units.pin([0])  # the sentinel slot, and every layer view's base
        self._storages = {
            "full": self.full_kv_pool,
            "swa": self.swa_kv_pool,
        }
        self.layers_mapping = self._build_layers_mapping(num_layers, specs)
        if self._swa_paged:
            self._init_swa_paged_state()

    @staticmethod
    def _allocate_group(
        spec: KVCacheGroupSpec,
        tp_size: int,
        outer_size: int,
        inner_size: int,
        dtype: torch.dtype,
        device: torch.device,
        runtime=None,
    ) -> _KVGroupStorage:
        local_kv_heads = div_even(spec.num_kv_heads, tp_size, allow_replicate=True)
        shape = (2, spec.num_layers, outer_size, inner_size, local_kv_heads, spec.head_dim)
        banks = None
        if runtime is None or spec.num_layers == 0:
            buffer = torch.empty(shape, device=device, dtype=dtype)
        else:
            from .runtime_pool import banked

            # On a shared runtime this pool only holds the DFlash drafter's history.
            buffer, banks = banked(runtime, f"draft_{spec.name}", shape, dtype)
        return _KVGroupStorage(
            buffer=buffer,
            k_buffer=buffer[0],
            v_buffer=buffer[1],
            storage_shape=(outer_size * inner_size, local_kv_heads, spec.head_dim),
            banks=banks or [],
        )

    @staticmethod
    def _build_layers_mapping(
        num_layers: int, specs: dict[str, KVCacheGroupSpec]
    ) -> tuple[_LayerRef, ...]:
        mapping: list[_LayerRef | None] = [None] * num_layers
        for group_name in ("full", "swa"):
            for local_index, layer_id in enumerate(specs[group_name].layer_ids):
                if layer_id < 0 or layer_id >= num_layers:
                    raise ValueError(f"KV layer id {layer_id} is outside [0, {num_layers})")
                if mapping[layer_id] is not None:
                    raise ValueError(f"KV layer id {layer_id} appears in more than one group")
                mapping[layer_id] = _LayerRef(group=group_name, index=local_index)

        missing = [layer_id for layer_id, ref in enumerate(mapping) if ref is None]
        if missing:
            raise ValueError(f"KV layer ids missing from full/swa groups: {missing}")
        return tuple(ref for ref in mapping if ref is not None)

    def is_full_layer(self, layer_id: int) -> bool:
        return self.layers_mapping[layer_id].group == "full"

    def is_swa_layer(self, layer_id: int) -> bool:
        return self.layers_mapping[layer_id].group == "swa"

    def group_of(self, layer_id: int) -> str:
        return self.layers_mapping[layer_id].group

    def translate_loc_from_full_to_swa(self, out_loc: torch.Tensor) -> torch.Tensor:
        # Global-paged SWA: a full slot's swa slot is read from the dense mapping (0 = no live
        # swa slot). Computed/reused full slots' swa entries are written by alloc_swa before any
        # store/gather; out-of-window slots map back to 0.
        return self.full_to_swa_index_mapping[out_loc.to(torch.int64)].to(torch.int32)

    # ---- Option A: global-paged SWA allocator (== sglang SWATokenToKVPoolAllocator) ----

    def _init_swa_paged_state(self) -> None:
        """(Re)build the global-paged SWA allocator state: the dense full->swa slot mapping
        (slot 0 = 'no live SWA slot' sentinel) and the ring of free swa slots. Called from
        __init__ and from rebuild() so both always match the freshly (re)allocated swa buffer
        (idle-only on rebuild)."""
        n = self._full_num_tokens
        ps = self._page_size
        dev = self._device
        # Every full slot maps to 0 (= no live swa slot) until alloc_swa writes it. The
        # trailing -1 lets a -1 "last_loc" map to -1 (matches sglang allocator/swa.py).
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(n + ps, dtype=torch.int64, device=dev),
                torch.tensor([-1], dtype=torch.int64, device=dev),
            ]
        )
        # swa slots 1.._swa_num_tokens-1 are allocatable; slot 0 is the reserved sentinel. The
        # free slots are the ``_swa_count`` ring entries from ``_swa_head``: the host knows how
        # many it takes and returns, so no call reads the device.
        self._swa_free = torch.arange(1, self._swa_num_tokens, dtype=torch.int64, device=dev)
        self._swa_head = 0
        self._swa_count = self._swa_free.numel()
        if self.slot_units is not None:
            # The host's copy of the mapping and its free slots (tail first: low, then reused).
            self._swa_host = torch.zeros(n + ps + 1, dtype=torch.int64)
            self._swa_host[-1] = -1
            self._swa_free_host = list(range(self._swa_num_tokens - 1, 0, -1))

    def _ring(self, start: int, n: int) -> list[torch.Tensor]:
        """The ring entries [start, start + n), as at most two contiguous slices."""
        start %= self._swa_free.numel()
        first = self._swa_free[start : start + n]
        return [first] if first.numel() == n else [first, self._swa_free[: n - first.numel()]]

    def alloc_swa(self, full_indices: torch.Tensor) -> None:
        """Bind one free swa slot to each full slot. The caller must check
        ``swa_available_size()`` first. Granularity-agnostic: at page_size>1 the caller hands
        whole pages (allocate_paged's _page_to_token expansion)."""
        n = int(full_indices.numel())
        if n == 0:
            return
        if self.slot_units is not None:
            raise RuntimeError("a shared runtime binds window slots through a claim")
        if n > self._swa_count:
            raise RuntimeError(f"SWA pool exhausted: need {n}, have {self._swa_count}")
        slots = self._ring(self._swa_head, n)
        self.full_to_swa_index_mapping.index_copy_(
            0, full_indices.to(torch.int64), slots[0] if len(slots) == 1 else torch.cat(slots))
        self._swa_head = (self._swa_head + n) % self._swa_free.numel()
        self._swa_count -= n

    def free_swa(self, full_indices: torch.Tensor) -> None:
        """Return the swa slots bound to ``full_indices`` and reset their mapping entries to the
        0 sentinel. Every index must hold a live binding the caller owns. Never touches the
        full pool."""
        n = int(full_indices.numel())
        if n == 0:
            return
        fi = full_indices.to(torch.int64)
        if self.slot_units is not None:
            return self._unbind_host(fi)
        offset = 0
        for part in self._ring(self._swa_head + self._swa_count, n):
            torch.index_select(self.full_to_swa_index_mapping, 0, fi[offset : offset + part.numel()],
                               out=part)
            offset += part.numel()
        # index_fill_ takes the 0 as a kernel argument; ``mapping[fi] = 0`` would first copy a
        # host scalar to the device and wait for it.
        self.full_to_swa_index_mapping.index_fill_(0, fi, 0)
        self._swa_count += n

    def swa_available_size(self) -> int:
        if self.slot_units is not None:  # free ids; their memory is taken when bound
            return len(self._swa_free_host)
        return self._swa_count

    def next_slots(self, n: int) -> torch.Tensor | None:
        """The window slots the next binding of ``n`` locations takes (not yet held)."""
        free = self._swa_free_host
        return torch.tensor(free[len(free) - n:], dtype=torch.int64) if n <= len(free) else None

    def bind_slots(self, full: torch.Tensor, slots: torch.Tensor) -> None:
        """Record ``next_slots`` as bound to ``full`` once their memory is held."""
        del self._swa_free_host[len(self._swa_free_host) - len(slots):]
        self._swa_host[full] = slots
        self.full_to_swa_index_mapping.index_copy_(
            0, upload(full, self._device), upload(slots, self._device))

    def _unbind_host(self, full: torch.Tensor) -> None:
        slots = self._swa_host[full]
        slots = slots[slots > 0]  # positions never bound, or released already
        self._swa_host[full] = 0
        self.full_to_swa_index_mapping.index_fill_(0, upload(full, self._device), 0)
        self._swa_free_host.extend(slots.tolist())
        self._swa_free_host.sort(reverse=True)  # lowest slot next: windows pack into few blocks
        self.slot_units.release(slots.numpy())

    def paged_views(self) -> list[torch.Tensor]:
        """Per-layer views ``[pages, 2, page_size, heads, head_dim]`` of the paged KV, for
        copying whole pages between tiers."""
        buf = self.full_kv_pool.buffer
        return [buf[:, layer].movedim(1, 0) for layer in range(buf.shape[1])]

    def window_views(self) -> list[torch.Tensor]:
        """Per-layer views ``[window slots, 2, 1, heads, head_dim]`` of the window KV."""
        buf = self.swa_kv_pool.buffer
        return [buf[:, layer].movedim(1, 0) for layer in range(buf.shape[1])]

    def window_units(self, full_locs: torch.Tensor) -> torch.Tensor:
        """Window slot of each full location (one unit per token), on the locations' device."""
        mapping = self.full_to_swa_index_mapping
        if not full_locs.is_cuda and self.slot_units is not None:
            mapping = self._swa_host
        return mapping[full_locs.to(torch.int64)]

    @property
    def swa_paged(self) -> bool:
        return self._swa_paged

    def k_cache(self, index: int) -> torch.Tensor:
        ref = self.layers_mapping[index]
        return self._storages[ref.group].k_buffer[ref.index]

    def v_cache(self, index: int) -> torch.Tensor:
        ref = self.layers_mapping[index]
        return self._storages[ref.group].v_buffer[ref.index]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        from freetoken.kernel import store_cache

        ref = self.layers_mapping[layer_id]
        storage = self._storages[ref.group]
        indices = out_loc
        if ref.group == "swa":
            indices = self.translate_loc_from_full_to_swa(out_loc)
        store_cache(
            k_cache=storage.k_buffer[ref.index].view(storage.storage_shape),
            v_cache=storage.v_buffer[ref.index].view(storage.storage_shape),
            indices=indices,
            k=k,
            v=v,
        )

    @property
    def full_num_tokens(self) -> int:
        return self._full_num_tokens

    @property
    def swa_num_tokens(self) -> int:
        return self._swa_num_tokens

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @staticmethod
    def _group_geometry(group: _KVGroupStorage) -> tuple:
        # Everything the realloc needs that does NOT pin the old buffer alive: layer count,
        # kv heads, head_dim, device, dtype. (Plain ints + device/dtype handles, no tensor.)
        _, num_layers, _old_outer, _old_inner, local_kv_heads, head_dim = group.buffer.shape
        return (num_layers, local_kv_heads, head_dim, group.buffer.device, group.buffer.dtype)

    @staticmethod
    def _alloc_group(geom: tuple, outer_size: int, inner_size: int) -> _KVGroupStorage:
        # Only the outer (page/token) dimension changes; the rest comes from ``geom``.
        num_layers, local_kv_heads, head_dim, device, dtype = geom
        buffer = torch.empty(
            (2, num_layers, outer_size, inner_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        return _KVGroupStorage(
            buffer=buffer,
            k_buffer=buffer[0],
            v_buffer=buffer[1],
            storage_shape=(outer_size * inner_size, local_kv_heads, head_dim),
        )

    def rebuild(self, num_full_pages: int, num_swa_tokens: int | None = None) -> None:
        """Reallocate both group buffers IN PLACE for new sizes.

        ``page_size`` (the full group's inner dim) and group geometry are read from the
        existing buffers; ``layers_mapping`` is unchanged. Object identity is preserved.
        """
        page_size = self.full_kv_pool.buffer.shape[3]
        self._full_num_tokens = num_full_pages * page_size
        self._swa_num_tokens = num_swa_tokens if num_swa_tokens is not None else self._full_num_tokens
        # Capture geometry, then DROP all references to the old buffers before allocating
        # the replacements so empty_cache() can actually reclaim them. Otherwise the old
        # and new KV buffers are live simultaneously and the rebuild can OOM even when the
        # target geometry alone would fit.
        full_geom = self._group_geometry(self.full_kv_pool)
        swa_geom = self._group_geometry(self.swa_kv_pool)
        self.full_kv_pool = None
        self.swa_kv_pool = None
        self._storages = {}
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        self.full_kv_pool = self._alloc_group(full_geom, outer_size=num_full_pages, inner_size=page_size)
        self.swa_kv_pool = self._alloc_group(swa_geom, outer_size=self._swa_num_tokens, inner_size=1)
        self._storages = {"full": self.full_kv_pool, "swa": self.swa_kv_pool}
        if self._swa_paged:
            # Reset the full->swa mapping (all 0) + swa free-list (all free) to the new
            # geometry, atomic with the buffer realloc. The fresh empty SWA radix tree built
            # by CacheManager.rebuild is then consistent by construction (idle-only).
            self._init_swa_paged_state()

    @classmethod
    def kv_cost(cls, config, *, num_swa_pages: int | None = None) -> tuple[int, int, int, int]:
        """Full groups ride cache_per_page; the window pool is either a pinned absolute
        window (fixed) or ratio x full (per-page) + the concurrency floor reservation, and
        the dense full_to_swa mapping (8B int64/slot) always scales with the full pages.
        ``num_swa_pages`` is THIS family's own keyword: a rebuild's target window, priced
        before the config override is written; None reads config's truth (its override,
        else the ratio)."""
        from .base import spec_kv_bytes_per_token

        swa_pin = (
            num_swa_pages if num_swa_pages is not None
            else config.swa_num_pages_override
        )
        cache_per_page = 0
        fixed_cache_size = 0
        for spec in config.model_config.kv_cache_group_specs():
            per_token = spec_kv_bytes_per_token(spec, config)
            if not spec.is_swa:
                cache_per_page += per_token * config.page_size
                continue
            cache_per_page += 8 * config.page_size  # full_to_swa_index_mapping bytes/page
            if config.cache_type != "swa_radix":
                # naive swa pool (see _naive_swa_num_tokens): fixed concurrency x window.
                fixed_cache_size += per_token * _naive_swa_num_tokens(config)
            elif swa_pin is not None:
                # pinned window == _swa_paged_num_tokens(override): max(floor, pin) + 1.
                fixed_cache_size += per_token * (
                    max(_swa_pool_floor(config), int(swa_pin)) + 1
                )
            else:
                cache_per_page += int(per_token * config.page_size * config.swa_full_tokens_ratio)
                fixed_cache_size += per_token * _swa_pool_floor(config)  # concurrency floor
        return cache_per_page, fixed_cache_size, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        # radix sizes the window by ratio (cross-request reuse), naive by concurrency x window.
        num_swa_tokens = (
            _swa_paged_num_tokens(config, num_pages + 1, num_swa_pages=num_swa_pages)
            if config.cache_type == "swa_radix"
            else _naive_swa_num_tokens(config)
        )
        # +1 for the dummy page (matches create_kvcache_pool)
        self.rebuild(num_full_pages=num_pages + 1, num_swa_tokens=num_swa_tokens)

    def unit_bytes(self) -> tuple[int, int]:
        full = self.full_kv_pool.buffer
        swa = self.swa_kv_pool.buffer
        full_tokens = int(full.shape[2]) * int(full.shape[3])
        return (
            int(full.numel() * full.element_size()) // full_tokens,
            int(swa.numel() * swa.element_size()) // self._swa_num_tokens,
        )


# ---- SWA pool sizing (pure arithmetic; the pool family's geometry formulas) ----


def _naive_swa_num_tokens(config) -> int:
    """Naive SWA pool = one window+headroom buffer per concurrent request. Sized so a request's
    whole swa footprint (up to max_forward_len) fits without prefill-time out-of-window freeing;
    the decode driver still bounds it during generation. == sglang's concurrency x window cap on
    the shared paged pool (memory-efficient naive with prefill-time freeing is a follow-up)."""
    swa_group = config.model_config.swa_attention_group()
    assert swa_group is not None
    width = align_ceil(swa_group.sliding_window + config.max_forward_len + 1, 32)
    return (config.max_running_req + 1) * width


def _swa_per_req_swa_floor(config) -> int:
    """One request's NON-EVICTABLE swa while it decodes, in tokens:

      - the trailing window the prefill-boundary commit locks -- window + retain gap, page-rounded
        (_cache_req_swa splits the committed node there so inc_lock pins that and no more);
      - its own decode tail up to the first out-of-window free: the driver runs only every
        _SWA_EVICTION_INTERVAL forwards and floors its frees at the committed length, so the
        request grows window + 2 pages + one interval before it can reclaim anything.

    Neither is reachable by evict_swa, so the pool must hold both for every running request."""
    from freetoken.scheduler.cache import _SWA_EVICTION_INTERVAL, _SWA_RETAIN_GAP

    window = next(g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa)
    ps = config.page_size
    locked = ((window + _SWA_RETAIN_GAP + ps - 1) // ps) * ps
    floor = locked + window + _SWA_EVICTION_INTERVAL + 2 * ps
    if getattr(config, "special_token_ckpt", False):
        floor += window + _SWA_RETAIN_GAP + _SWA_EVICTION_INTERVAL
    return floor


def _swa_pool_floor(config) -> int:
    """The swa pool's hard floor: max_running_req x the per-request non-evictable footprint.
    Admission gates only the incoming chunk (PrefillAdder's need_swa is one window) and never
    reserves the decode growth of the requests already running, so a full batch can drive the
    pool to zero -- and the alloc_swa that then raises has no handler above it."""
    return config.max_running_req * _swa_per_req_swa_floor(config)


def _swa_paged_num_tokens(config, num_full_pages: int, num_swa_pages: int | None = None) -> int:
    """SWA pool size = max(concurrency floor, target tokens) + 1 (slot-0 sentinel). The target is
    a pinned absolute window (``num_swa_pages`` if given, else swa_num_pages_override, usable
    tokens), else ratio x full-pool tokens. The floor keeps a full batch always fitting;
    < 1.0 ratio trades reuse for memory."""
    floor = _swa_pool_floor(config)
    override = num_swa_pages if num_swa_pages is not None else config.swa_num_pages_override
    if override is not None:
        return max(floor, int(override)) + 1
    full_tokens = num_full_pages * config.page_size
    return max(floor, int(config.swa_full_tokens_ratio * full_tokens)) + 1
