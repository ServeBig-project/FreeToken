from __future__ import annotations

import math

import torch
from freetoken.distributed import get_tp_info
from freetoken.env import ENV
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.utils import div_even

_SSM_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def ssm_state_dtype() -> torch.dtype:
    """Recurrent (SSM) state dtype, from FREETOKEN_MAMBA_SSM_DTYPE (default fp32)."""
    return _SSM_DTYPES.get(str(ENV.MAMBA_SSM_DTYPE).lower(), torch.float32)


def slot_major(shape, dtype, device) -> torch.Tensor:
    """Zeroed ``[layers, slots, ...]`` view whose storage keeps each slot's layers adjacent, so
    one slot (or record row) is a single address range that can be mapped on its own."""
    layers, slots, *rest = shape
    return torch.zeros((slots, layers, *rest), dtype=dtype, device=device).transpose(0, 1)


def _linear_local_dims(
    group: LinearGatedDeltaGroupConfig, tp_size: int
) -> tuple[int, int, int, int]:
    """TP-local ``(n_layers, conv_dim, v_heads, k_heads)`` for the GDN state tensors -- the
    single source of the sharding math shared by the pool allocation and the byte estimate."""
    local_k_heads = div_even(group.num_key_heads, tp_size, allow_replicate=True)
    local_v_heads = div_even(group.num_value_heads, tp_size, allow_replicate=True)
    local_conv_dim = 2 * local_k_heads * group.key_head_dim + local_v_heads * group.value_head_dim
    return len(group.layer_ids), local_conv_dim, local_v_heads, local_k_heads


def _replay_shapes(group, tp_size, dtype, records):
    """ReplaySSM buffers for ``records = (rows, ring, draft_steps, graph_rows)``."""
    from .gdn_replay import replay_shapes

    n_layers, conv_dim, v_heads, k_heads = _linear_local_dims(group, tp_size)
    return replay_shapes(n_layers, conv_dim, v_heads, k_heads, group.key_head_dim,
                         group.value_head_dim, group.conv_kernel_dim, dtype, *records)


class LinearStatePool:
    """Per-request recurrent state (conv + SSM) for GatedDeltaNet layers.

    Hybrid caching allocates live states and snapshots from one free list. Naive
    caching reserves ``fixed_slots`` entries for table-indexed live states; only
    entries beyond those and the padding sink may be allocated as scratch. With a shared
    ``runtime`` every slot is allocated, the padding sink is slot 0, and a slot's memory is
    mapped when it is allocated.

    A model declares extra per-request tensors on the same slots through
    ``ModelConfig.slot_states`` (``SlotStateSpec``); they clear, copy, snapshot and rebuild
    with the GDN state and are read back through ``slot_state(name, layer_id)``.
    """

    def __init__(
        self,
        group: LinearGatedDeltaGroupConfig,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
        tp_size: int | None = None,
        fixed_slots: int = 0,
        records: tuple[int, int, int, int] | None = None,
        slot_states: tuple[SlotStateSpec, ...] = (),
        runtime=None,
    ) -> None:
        if tp_size is None:
            tp_size = get_tp_info().size

        self._group = group
        self._device = device
        self._conv_dtype = dtype

        n_layers, local_conv_dim, local_v_heads, _ = _linear_local_dims(group, tp_size)

        # conv left-context: the last (kernel-1) timesteps of the conv input stream.
        conv = ((n_layers, num_slots, local_conv_dim, group.conv_kernel_dim - 1), dtype)
        # SSM recurrent state. fp32 by default (matches HF mamba_ssm_dtype); the dtype is
        # overridable via FREETOKEN_MAMBA_SSM_DTYPE (see ssm_state_dtype).
        rec = ((n_layers, num_slots, local_v_heads, group.key_head_dim, group.value_head_dim),
               ssm_state_dtype())
        self._local_index = {layer_id: i for i, layer_id in enumerate(group.layer_ids)}
        self._slot_specs = tuple(slot_states)
        self._state_layer_index = {
            spec.name: {lid: i for i, lid in enumerate(spec.layer_ids)} for spec in slot_states
        }

        # Naive live slots belong to TableManager, so they must never enter this allocator.
        self.padding_slot = fixed_slots
        self.replay = None
        # ReplaySSM: each live slot is a checkpoint completed by its request's update records.
        self._replay_shapes = (_replay_shapes(group, tp_size, dtype, records)
                               if records is not None else None)
        self._allocate(conv, rec, runtime)

    def _allocate(self, conv, rec, runtime) -> None:
        """States (and records) for ``conv``/``rec`` = (shape, dtype), all slots free."""
        num_slots = conv[0][1]
        self._num_slots = num_slots
        self._free_slots: list[int] = list(range(self.padding_slot + 1, num_slots))
        self.slot_states = {}
        self.units = None
        if runtime is None:
            self.slot_states = self._alloc_slot_states(num_slots)
            self.conv_states = slot_major(*conv, self._device)
            self.recurrent_states = slot_major(*rec, self._device)
        else:
            from .runtime_pool import Units, slot_rows

            self.conv_states, conv_bank = slot_rows(runtime, "gdn_conv", *conv)
            self.recurrent_states, rec_bank = slot_rows(runtime, "gdn_state", *rec)
            banks = [rec_bank, conv_bank]
            for spec in self._slot_specs:
                tensor, bank = slot_rows(runtime, spec.name,
                    (max(1, len(spec.layer_ids)), num_slots, *spec.shape),
                    spec.dtype if spec.dtype is not None else self._conv_dtype)
                self.slot_states[spec.name] = tensor
                banks.append(bank)
            self.units = Units(banks)
            # The padding sink stays mapped: every layer view starts inside it.
            self.units.pin([self.padding_slot])
            # Live rows initialize from their component's fresh-input path; only the
            # permanent padding row must be initialized before graph capture.
            for spec in self._slot_specs:
                self.slot_states[spec.name][:, self.padding_slot].fill_(spec.fill_value)
            self._free_slots.reverse()  # low slots first, so released memory is reused first
        if self._replay_shapes is not None and (self.replay is None or runtime is not None):
            from .gdn_replay import GdnReplay

            self.replay = GdnReplay(self, self._replay_shapes, self._device, runtime)

    def _alloc_slot_states(self, num_slots: int) -> dict[str, torch.Tensor]:
        return {
            spec.name: torch.full(
                (max(1, len(spec.layer_ids)), num_slots, *spec.shape),
                spec.fill_value,
                dtype=spec.dtype if spec.dtype is not None else self._conv_dtype,
                device=self._device,
            )
            for spec in self._slot_specs
        }

    def slot_state(self, name: str, layer_id: int | None = None) -> torch.Tensor:
        """One declared sibling state, ``[num_slots, *shape]``; ``layer_id`` picks its layer row."""
        layers = self._state_layer_index[name]
        return self.slot_states[name][0 if layer_id is None else layers[layer_id]]

    def capture_position(self, start: int, target: int) -> int:
        """Deepest position at or before ``target`` whose state a prefill extend starting at
        ``start`` produces: the chunked recurrence has one every CHUNK_SIZE tokens."""
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        return start + (target - start) // CHUNK_SIZE * CHUNK_SIZE

    def check_page_size(self, page_size: int) -> None:
        """Captured states must sit on page boundaries for the prefix tree to hold them."""
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        if CHUNK_SIZE % page_size:
            raise ValueError(f"state caching needs page_size dividing {CHUNK_SIZE}, got {page_size}")

    def can_export(self, req, position: int) -> bool:
        """Whether the complete state after ``position`` inputs can still be produced. Replay
        reaches past positions of the GDN state only; declared slot states keep just the
        current one, so with them a past position is not a complete state."""
        if position == req.cached_len:
            return True
        return self.replay is not None and not self._slot_specs and self.replay.can_export(req, position)

    def export(self, req, position: int, dst: int) -> None:
        if self.replay is None:
            self.copy_from(req.linear_slot_idx, dst)
            return
        self.replay.export(req, position, dst)
        src = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        for t in self.slot_states.values():
            t[:, dst].copy_(t[:, src])

    def materialize(self, req) -> None:
        """Make the request's own slot its complete current state."""
        if self.replay is not None:
            self.replay.materialize(req)

    def speculative_size(self, lengths):
        if self.replay is not None:
            return 0  # records replace draft and verify slots
        # Verify needs at least as many slots as the one-per-active-request draft phase.
        return sum(length + 1 for length in lengths)

    def limit_speculation(self, lengths, available):
        if self.replay is not None:
            return lengths
        for limit in range(max(lengths), 0, -1):
            capped = [min(length, limit) for length in lengths]
            if self.speculative_size(capped) <= available:
                return capped
        return [0] * len(lengths)

    def begin_speculation(self, reqs, views, lengths, *, draft=True, slots=None):
        if self.replay is not None:
            return self.replay.begin_round(reqs, lengths)
        return LinearSpeculativeState(self, reqs, views, lengths, draft=draft, slots=slots)

    def create_speculative_graphs(self, max_batch, query_width, device):
        from freetoken.attention.linear import FLASpeculativeGraphs, ReplaySpeculativeGraphs

        if self.replay is not None:
            return ReplaySpeculativeGraphs(self)
        return FLASpeculativeGraphs(self, max_batch, query_width, device)

    @property
    def unused_slots(self) -> int:
        """Slot ids nobody holds (with a shared runtime, not all of them can be mapped)."""
        return len(self._free_slots)

    @property
    def num_free_slots(self) -> int:
        """Free slot ids; with a shared runtime a slot may still lack memory (``try_alloc``)."""
        return len(self._free_slots)

    def alloc(self, n: int = 1) -> list[int]:
        """Pop ``n`` free slot ids (LIFO). Raises if the pool is exhausted."""
        slots = self.try_alloc(n)
        if slots is None:
            raise RuntimeError(f"LinearStatePool exhausted: need {n}, have "
                               f"{len(self._free_slots)} ids and their memory")
        return slots

    def try_alloc(self, n: int = 1) -> list[int] | None:
        """``n`` slots with their memory mapped, or None (nothing taken)."""
        slots = self.peek(n)
        if slots is None or (self.units is not None and not self.units.acquire(slots)):
            return None
        self.take(slots)
        return slots

    def peek(self, n: int) -> list[int] | None:
        """The slots the next allocation of ``n`` takes (not taken yet)."""
        return self._free_slots[len(self._free_slots) - n:][::-1] if n <= len(self._free_slots) else None

    def take(self, slots) -> None:
        """Record ``peek``'s slots as allocated once their memory is held."""
        del self._free_slots[len(self._free_slots) - len(slots):]

    def reclaim_all_slots(self) -> None:
        """Restore slots after the fixed live slots and padding. Idle-only: the caller
        (e.g. a CacheManager rebuild that discards donated snapshots) must guarantee no
        running request holds a slot, otherwise live state would be handed out twice."""
        self._free_slots = list(range(self.padding_slot + 1, self._num_slots))

    def rebuild(self, num_slots: int, runtime=None) -> None:
        """Reallocate the conv + recurrent state tensors for ``num_slots`` slots IN PLACE, on
        ``runtime`` when shared (its record buffers move there too).

        Geometry (layers, conv dim, head dims) and dtypes are taken from the existing
        tensors; only the slot count changes. Object identity is preserved so cached
        references (ctx.linear_state_pool) stay valid. Idle-only and destructive: every
        live/snapshot state is dropped, so the caller must guarantee no running request
        holds a slot and the radix tree owning donated snapshots is discarded too.
        """
        n_layers, _, local_conv_dim, km1 = self.conv_states.shape
        _, _, local_v_heads, key_head_dim, value_head_dim = self.recurrent_states.shape
        conv_dtype, rec_dtype = self.conv_states.dtype, self.recurrent_states.dtype
        device = self._device
        self.conv_states = None
        self.recurrent_states = None
        self.slot_states = {}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._allocate(((n_layers, num_slots, local_conv_dim, km1), conv_dtype),
                       ((n_layers, num_slots, local_v_heads, key_head_dim, value_head_dim),
                        rec_dtype), runtime)

    def free(self, slots) -> None:
        """Return slot ids to the free-list. Accepts an int, list, or 1-D tensor."""
        if isinstance(slots, torch.Tensor):
            slots = slots.flatten().tolist()
        elif isinstance(slots, int):
            slots = [slots]
        self._free_slots.extend(int(s) for s in slots)
        if self.units is not None:
            self._free_slots.sort(reverse=True)  # lowest slot next: states pack into few blocks
            self.units.release(slots)

    def clear_slots(self, slots) -> None:
        """Zero conv + recurrent state at ``slots`` across all linear layers (fresh sequence)."""
        if isinstance(slots, (list, tuple)):
            slots = torch.as_tensor(slots, dtype=torch.long, device=self._device)
        self.conv_states[:, slots] = 0
        self.recurrent_states[:, slots] = 0
        for spec in self._slot_specs:
            self.slot_states[spec.name][:, slots] = spec.fill_value

    def copy_from(self, src: int, dst: int) -> None:
        """Copy a whole-sequence snapshot (conv + recurrent, all layers) from slot ``src`` to
        ``dst``. Used for COW-on-restore (donated snapshot -> fresh live slot)."""
        self.conv_states[:, dst].copy_(self.conv_states[:, src])
        self.recurrent_states[:, dst].copy_(self.recurrent_states[:, src])
        for t in self.slot_states.values():
            t[:, dst].copy_(t[:, src])

    def state_views(self) -> list[torch.Tensor]:
        """Per-layer ``[slots, ...]`` views of the conv, recurrent and declared slot states."""
        views = [*self.conv_states.unbind(0), *self.recurrent_states.unbind(0)]
        for t in self.slot_states.values():
            views.extend(t.unbind(0))
        return views

    def is_linear_layer(self, layer_id: int) -> bool:
        return layer_id in self._local_index

    def local_index(self, layer_id: int) -> int:
        return self._local_index[layer_id]

    def conv_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.conv_states[self._local_index[layer_id], table_idx]

    def recurrent_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.recurrent_states[self._local_index[layer_id], table_idx]

    def reset(self, table_idx: int) -> None:
        """Zero a slot across all linear layers (new request takes this table_idx)."""
        self.conv_states[:, table_idx].zero_()
        self.recurrent_states[:, table_idx].zero_()
        for spec in self._slot_specs:
            self.slot_states[spec.name][:, table_idx] = spec.fill_value

    @property
    def num_linear_layers(self) -> int:
        return len(self._local_index)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def device(self) -> torch.device:
        return self._device

    def bytes_per_slot(self) -> int:
        """Total state bytes for one request (all linear layers)."""
        per = (
            self.conv_states[:, 0].numel() * self.conv_states.element_size()
            + self.recurrent_states[:, 0].numel() * self.recurrent_states.element_size()
        )
        for t in self.slot_states.values():
            per += t[:, 0].numel() * t.element_size()
        return int(per)


def linear_state_bytes_per_req(
    group: LinearGatedDeltaGroupConfig,
    tp_size: int,
    dtype: torch.dtype,
    slot_states: tuple[SlotStateSpec, ...] = (),
) -> int:
    """Linear-state bytes for one request across all linear layers (TP-local), plus the
    declared slot states."""
    n_layers, local_conv_dim, local_v_heads, _ = _linear_local_dims(group, tp_size)

    conv_elems = local_conv_dim * (group.conv_kernel_dim - 1)
    rec_elems = local_v_heads * group.key_head_dim * group.value_head_dim
    conv_bytes = conv_elems * dtype.itemsize  # conv state in model dtype
    rec_bytes = rec_elems * ssm_state_dtype().itemsize  # recurrent state (default fp32)
    total = n_layers * (conv_bytes + rec_bytes)
    for spec in slot_states:
        item = (spec.dtype if spec.dtype is not None else dtype).itemsize
        total += max(1, len(spec.layer_ids)) * math.prod(spec.shape) * item
    return int(total)


def state_banks(config) -> list[tuple[int, int]]:
    """(banks, bytes of one slot in each) of the GDN states on a shared runtime: one bank of
    recurrent and one of conv states, each slot's layers adjacent."""
    group = config.model_config.linear_attention_group()
    if group is None:
        return []
    layers, conv_dim, v_heads, _ = _linear_local_dims(group, config.tp_info.size)
    banks = [(1, layers * v_heads * group.key_head_dim * group.value_head_dim
             * ssm_state_dtype().itemsize),
            (1, layers * conv_dim * (group.conv_kernel_dim - 1) * config.dtype.itemsize)]
    banks += [(1, max(1, len(spec.layer_ids)) * math.prod(spec.shape)
               * (spec.dtype if spec.dtype is not None else config.dtype).itemsize)
              for spec in getattr(config.model_config, "slot_states", ())]
    return banks


def record_banks(config) -> list[tuple[int, int]]:
    """(banks, bytes of one record row in each) of the ReplaySSM records on a shared runtime."""
    records = replay_records(config)
    if records is None:
        return []
    shapes = _replay_shapes(config.model_config.linear_attention_group(), config.tp_info.size,
                            config.dtype, (1, records[1], records[2], 0))
    return [(1, math.prod(shape) * dtype.itemsize) for name, (shape, dtype) in shapes.items()
            if name in ("u", "k", "g", "window")]


__all__ = ["LinearStatePool", "linear_state_bytes_per_req"]


def state_pool_bytes(config, num_slots: int | None = None) -> int:
    """Total GDN state bytes at ``num_slots`` PHYSICAL slots (default: the startup slot
    count), including the fixed ReplaySSM buffers. The engine adds this to the KV family's
    fixed cost when budgeting -- the state pool is a sibling pool, not a KV tier."""
    linear_group = config.model_config.linear_attention_group()
    if linear_group is None:
        return 0
    slots = num_slots if num_slots is not None else _linear_pool_num_slots(config)
    return _bytes_per_req(config) * slots + replay_buffer_bytes(config)


def _bytes_per_req(config) -> int:
    return linear_state_bytes_per_req(
        config.model_config.linear_attention_group(), config.tp_info.size, config.dtype,
        getattr(config.model_config, "slot_states", ()))  # duck-typed test configs omit it


def replay_records(config) -> tuple[int, int, int, int] | None:
    """``(rows, ring, draft_steps, graph_rows)`` of the ReplaySSM buffers, None when not
    active. ``graph_rows`` bounds every batch size GraphRunner may capture."""
    if not config.enable_gdn_replayssm or config.model_config.linear_attention_group() is None:
        return None
    if config.cuda_graph_bs is not None:
        graph_rows = max(config.cuda_graph_bs, default=0)
    else:
        graph_rows = 256 if config.cuda_graph_max_bs is None else max(config.cuda_graph_max_bs, 0)
    return (config.max_running_req, config.gdn_replay_buffer_len, config.speculative_num_steps,
            graph_rows)


def replay_buffer_bytes(config) -> int:
    records = replay_records(config)
    if records is None:
        return 0
    shapes = _replay_shapes(config.model_config.linear_attention_group(), config.tp_info.size,
                            config.dtype, records)
    return sum(math.prod(shape) * dtype.itemsize for shape, dtype in shapes.values())


def _default_pool_slots(config) -> int:
    """Replay-off slots: on-demand snapshots share the free slots with live states,
    reusable public prefixes and speculative scratch."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1  # live + dummy/padding
    ratio = 2.0 if config.linear_state_cache_ratio is None else config.linear_state_cache_ratio
    return 4 * mr + max(4, int(ratio * mr)) + 1


def gdn_state_budget(config) -> int:
    """Startup GDN state bytes: the explicit budget, else the replay-off pool's bytes."""
    return config.gdn_state_budget_bytes or _default_pool_slots(config) * _bytes_per_req(config)


def _linear_pool_num_slots(config) -> int:
    """Full-state slots that fit the GDN state budget next to the fixed ReplaySSM buffers."""
    if config.gdn_state_budget_bytes is None and replay_records(config) is None:
        return _default_pool_slots(config)
    per_slot = _bytes_per_req(config)
    budget = gdn_state_budget(config)
    fixed = replay_buffer_bytes(config)
    slots = (budget - fixed) // per_slot
    if slots < _linear_pool_min_slots(config):
        raise ValueError(
            f"GDN state budget {budget} bytes holds {max(slots, 0)} full states after "
            f"{fixed} bytes of replay storage; at least {_linear_pool_min_slots(config)} "
            f"states of {per_slot} bytes are needed, raise --gdn-state-budget-bytes"
        )
    return slots


def speculative_state_slots(config) -> int:
    """Scratch states one request's full SD window needs beyond the live states."""
    if not config.speculative_num_steps or replay_records(config) is not None:
        return 0  # ReplaySSM records replace draft and verify slots
    return config.speculative_num_steps + 1


def _linear_pool_min_slots(config) -> int:
    """Keep the existing conservative rebuild floor while changing slot ownership only."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1
    return 4 * mr + 1


class LinearSpeculativeState:
    """Own temporary draft/verify states until the host decides what output is retained."""

    def __init__(self, pool, reqs, views, lengths, *, draft=True, slots=None):
        """``slots``: scratch states a shared runtime claimed for the whole round."""
        self.pool, self.claimed = pool, slots
        count = sum(length > 0 for length in lengths) if draft else 0
        self.slots = slots[:count] if slots is not None else pool.alloc(count)
        self.live_slots = [req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
                           for req in reqs]
        draft_slots = iter(self.slots)
        for live, view, length in zip(self.live_slots, views, lengths, strict=True):
            if draft and length:
                slot = next(draft_slots)
                pool.copy_from(live, slot)
                view.linear_slot_idx = slot

    def prepare_verify(self, batch, lengths):
        # Draft and verify are ordered on the engine stream; returning indices does
        # not free tensor storage, and verify overwrites them only after draft finishes.
        size = self.pool.speculative_size(lengths)
        if self.claimed is not None:
            self.slots = self.claimed[:size]
        else:
            self.pool.free(self.slots)
            self.slots = self.pool.alloc(size)
        self.states, offset = [], 0
        for live, length in zip(self.live_slots, lengths, strict=True):
            self.states.append((live, self.slots[offset:offset + length + 1]))
            offset += length + 1
        batch.speculative_states = self.states

    def commit(self, retained):
        # Output token j was sampled after computing position j. A stop at output j
        # therefore commits scratch[j], not the last algorithmically accepted position.
        selected = [(live, scratch[n - 1]) for (live, scratch), n in
                    zip(self.states, retained, strict=True) if n]
        if selected:
            live, src = (torch.tensor(xs, dtype=torch.int64, device=self.pool.device)
                         for xs in zip(*selected))
            for layer in range(self.pool.num_linear_layers):
                for tensor in (self.pool.recurrent_states[layer], self.pool.conv_states[layer]):
                    tensor.index_copy_(0, live, tensor.index_select(0, src))
        self.pool.free(self.claimed if self.claimed is not None else self.slots)
