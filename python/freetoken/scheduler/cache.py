from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Tuple

import torch
from freetoken.core import Batch, Req
from freetoken.kvcache.prefix_policy import ANCHOR, INPUT, OUTPUT, POLICIES
from freetoken.kvcache.radix_cache import CacheHandle, RadixCache

from .host_tier import HostTier, build_components, wait_ms
from freetoken.utils import align_down, div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq

# A decoding request releases its out-of-window windows every `interval` committed tokens.
def _swa_eviction_interval() -> int:
    raw = os.environ.get("FREETOKEN_SWA_EVICTION_INTERVAL", "128")
    try:
        return max(1, int(raw))
    except ValueError:
        raise ValueError(f"FREETOKEN_SWA_EVICTION_INTERVAL must be an integer, got {raw!r}")


_SWA_EVICTION_INTERVAL = _swa_eviction_interval()

# Finish-time retention keeps [P - window - gap, P) swa-live for the next turn's cut near the
# prompt end. The gap covers templates whose generation prompt injects tokens that vanish when
# the client drops reasoning (Qwen's "<think>\n": the re-render diverges 2 tokens BEFORE P).
_SWA_RETAIN_GAP = 16

# A few contiguous row writes are cheaper than allocating and uploading two index
# vectors. Above this point, one batched advanced-index write avoids excessive launches.
_MAX_DIRECT_PAGE_TABLE_COPY_SEGMENTS = 8


@dataclass
class _DecodePageReservation:
    table_idx: int
    cached_len: int
    device_len: int
    first_page: int
    last_page: int
    allocated: torch.Tensor


@dataclass(eq=False)
class StateCapture:
    """A recurrent state the request should freeze on its way through ``target``: the forward
    takes the deepest position it can produce at or before it (``pos``) into ``slot``."""

    target: int
    purpose: str
    slot: int | None = None
    pos: int | None = None


@dataclass(frozen=True)
class _DeferredPrefillAllocation:
    first_pos: int
    last_pos: int
    allocated: torch.Tensor


class _PrefillExecutionSession:
    """Cache-owned SWA bindings replayed for each resident expert stage."""

    def __init__(self, manager: CacheManager, reqs: List[Req]) -> None:
        self._manager = manager
        self._owners = {req.table_idx: req for req in reqs}
        if len(self._owners) != len(reqs):
            raise RuntimeError("layered prefill requests must have distinct page-table rows")
        self._deferred: dict[int, _DeferredPrefillAllocation] = {}
        self._initial_pos = {req.table_idx: req.cached_len for req in reqs}
        self._next_pos = dict(self._initial_pos)
        self._evicted_pos = {
            req.table_idx: max(
                req.swa_evicted_seqlen,
                req.cache_handle.cached_len,
            )
            for req in reqs
        }
        self._initial_evicted_pos = dict(self._evicted_pos)
        self._active_signature: tuple[tuple[int, int, int], ...] | None = None
        self._closed = False

    def owns(self, req: Req) -> bool:
        return req.table_idx in self._owners

    def record_allocation(
        self,
        req: Req,
        first_page: int,
        last_page: int,
        allocated: torch.Tensor,
    ) -> None:
        if not self.owns(req):
            raise RuntimeError("prefill execution received an unowned allocation")
        if req.table_idx in self._deferred:
            raise RuntimeError("prefill execution allocated one request more than once")
        first_pos = first_page * self._manager.page_size
        last_pos = last_page * self._manager.page_size
        if int(allocated.numel()) != last_pos - first_pos:
            raise RuntimeError("prefill execution allocation has the wrong page span")
        self._deferred[req.table_idx] = _DeferredPrefillAllocation(
            first_pos=first_pos,
            last_pos=last_pos,
            allocated=allocated,
        )

    def activate(self, reqs: Iterable[Req]) -> None:
        """Release the prior window and bind the next tile's allocated pages."""
        if self._closed:
            raise RuntimeError("prefill execution session is closed")
        tile_reqs = list(reqs)
        signature = tuple(
            (req.table_idx, req.cached_len, req.device_len) for req in tile_reqs
        )
        if signature == self._active_signature:
            return

        for req in tile_reqs:
            owner = self._owners.get(req.table_idx)
            if owner is None:
                raise RuntimeError("prefill tile contains a request outside its logical wave")
            expected = self._next_pos[req.table_idx]
            if req.cached_len != expected or req.device_len <= req.cached_len:
                raise RuntimeError(
                    "prefill tiles must advance each request contiguously"
                )

        manager = self._manager
        page_size = manager.page_size
        for table_idx, _, device_len in self._active_signature or ():
            new_evicted = align_down(
                device_len - manager.sliding_window_size - page_size,
                page_size,
            )
            start = self._evicted_pos[table_idx]
            if new_evicted > start:
                manager._free_swa(manager.page_table[table_idx, start:new_evicted])
                self._evicted_pos[table_idx] = new_evicted

        bindings: list[torch.Tensor] = []
        for req in tile_reqs:
            deferred = self._deferred.get(req.table_idx)
            if deferred is None:
                continue
            first_pos = max(
                div_ceil(req.cached_len, page_size) * page_size,
                deferred.first_pos,
            )
            last_pos = min(
                div_ceil(req.device_len, page_size) * page_size,
                deferred.last_pos,
            )
            if last_pos > first_pos:
                offset = first_pos - deferred.first_pos
                bindings.append(deferred.allocated[offset : offset + last_pos - first_pos])

        for allocated in bindings:
            manager._bind_swa(allocated)
        for req in tile_reqs:
            self._next_pos[req.table_idx] = req.device_len
        self._active_signature = signature

    def rewind(self) -> None:
        """Return session-owned bindings and replay the same tiles at the next stage."""
        if self._closed:
            raise RuntimeError("prefill execution session is closed")
        for deferred in self._deferred.values():
            self._manager._free_swa(deferred.allocated)
        self._next_pos = dict(self._initial_pos)
        self._evicted_pos = dict(self._initial_evicted_pos)
        self._active_signature = None

    def close(self) -> None:
        if not self._closed:
            for table_idx, owner in self._owners.items():
                owner.swa_evicted_seqlen = max(
                    owner.swa_evicted_seqlen,
                    self._evicted_pos[table_idx],
                )
            self._closed = True
            self._manager._close_prefill_execution(self)

    def cancel(self) -> None:
        if self._closed:
            return
        for deferred in self._deferred.values():
            self._manager._free_swa(deferred.allocated)
        self._closed = True
        self._manager._close_prefill_execution(self)


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str,
                 linear_state_pool=None, swa_pool=None, sliding_window_size=None,
                 policy: str = "baseline", host_bytes: int = 0, draft_kv=None,
                 tp_group=None):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.linear_state_pool = linear_state_pool
        kv_pool = swa_pool
        if draft_kv is not None and draft_kv.swa_paged:
            # SD targets have no window of their own: the drafter's window rides the tree.
            swa_pool, sliding_window_size = draft_kv, draft_kv.window
        self.swa_pool = swa_pool
        self.sliding_window_size = sliding_window_size
        # swa_paged: the pool keeps window KV behind a full->window mapping with its own slots,
        # with or without prefix reuse; it gates the window slot lifecycle.
        self.swa_paged = swa_pool is not None and getattr(swa_pool, "swa_paged", False)
        # Owned-pool capability pickup: a plugged-in swa pool may cap the prefill chunk (DSV4:
        # ~half the window working set). Instance attrs shadow the class defaults; absent
        # attributes leave the defaults untouched (Gemma4).
        if swa_pool is not None:
            self.prefill_chunk_budget = getattr(swa_pool, "prefill_chunk_budget", None)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self.reuse = type != "naive"
        # The prefix tree carries each component the pools hold: recurrent states when a state
        # pool exists, windows when the pool pages its window KV.
        self.state_cache = self.reuse and linear_state_pool is not None
        self.window_cache = self.reuse and self.swa_paged
        if self.state_cache:
            linear_state_pool.check_page_size(page_size)
        self.policy_name = policy
        self.policy = POLICIES[policy]()
        self.stats = dict.fromkeys((
            "checkpoint_created", "checkpoint_deduplicated", "checkpoint_pruned",
            "checkpoint_evicted", "gpu_checkpoint_peak", "host_checkpoint_peak",
            "gpu_reused_tokens", "host_reused_tokens", "recomputed_tokens"), 0)
        self.tree = self._make_tree()
        if host_bytes and not self.reuse:
            raise ValueError("--prefix-cache-host-gib needs prefix reuse (--cache-type radix)")
        self.components = build_components(
            kv_pool, linear_state_pool, draft_kv, window_pool=swa_pool, window=self.window_cache,
            state=self.state_cache, required=bool(host_bytes))
        self.tp_group = tp_group  # CPU group when TP > 1: ranks agree on finished copies
        self.host = HostTier(self, host_bytes, self.components) if host_bytes else None
        self._host_reuse: dict[int, int] = {}  # uid -> tokens its admission got from the host
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self._decode_page_reservations: dict[Req, _DecodePageReservation] = {}
        self._prefill_execution: _PrefillExecutionSession | None = None

    # ----- capability hooks (defaults; plugged-in pools may narrow them) -----
    supports_runtime_rebuild = True
    prefill_chunk_budget = None  # generic shared page pool: no per-model prefill chunk cap

    def _make_tree(self) -> RadixCache | None:
        if not self.reuse:
            return None
        return RadixCache(
            self.device, self.page_size, self.policy, self.stats,
            window=self.sliding_window_size if self.window_cache else None,
            has_state=self.state_cache,
        )

    def page_usage(self) -> tuple[int, int]:
        """(used_pages, total_pages): allocated, non-evictable pages over the pool total
        (active requests + protected prefix; evictable prefix-cache pages are excluded)."""
        total = self.num_pages
        return total - len(self.free_slots) - self._evictable("kv") // self.page_size, total

    def _evictable(self, kind: str) -> int:
        return self.tree.evictable[kind] if self.tree is not None else 0

    def match_req(self, req: PendingReq) -> CacheHandle:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        # Multimodal requests must not reuse a shared prefix: image-placeholder tokens
        # have identical ids across images but carry different content (and KV).
        if self.tree is None or req.mm_embeds is not None:
            return CacheHandle(0, None, self.empty)
        return self.tree.match(req.input_ids[: input_len - 1], req.cache_group, reuse=True)

    @property
    def available_size(self) -> int:
        return self._evictable("kv") + len(self.free_slots) * self.page_size

    @property
    def mamba_available_size(self) -> int:
        """Free state slots + unlocked tree states."""
        return self.linear_state_pool.num_free_slots + self._evictable("state")

    @property
    def swa_available_size(self) -> int:
        """Free window slots + unlocked live tree window tokens."""
        return self.swa_pool.swa_available_size() + self._evictable("window")

    def decode_swa_reservation(self, reqs: Iterable[Req]) -> int:
        """SWA token slots needed by these requests' pending query rows this forward."""
        if not self.swa_paged:
            return 0
        ps = self.page_size
        total = 0
        reservations = getattr(self, "_decode_page_reservations", {})
        for req in reqs:
            first_page = div_ceil(req.cached_len, ps)
            last_page = div_ceil(req.device_len, ps)
            reservation = reservations.get(req) if reservations else None
            if self._reservation_matches(
                req, reservation, first_page=first_page, last_page=last_page
            ):
                continue
            total += (last_page - first_page) * ps
        return total

    def decode_reserved_tokens_for(self, reqs: Iterable[Req]) -> int:
        """Full-token capacity already held for these runnable decode requests."""
        return sum(
            reservation.allocated.numel()
            for req in reqs
            if (reservation := self._decode_page_reservations.get(req)) is not None
        )

    def ensure_swa_slots(self, n: int) -> None:
        """Free window slots until >= ``n`` are available by evicting LRU tree windows."""
        while self.swa_pool.swa_available_size() < n:
            ev = self.tree.evict_window(n - self.swa_pool.swa_available_size())
            if ev.window.numel() == 0:
                break
            self._release(ev)

    def ensure_mamba_slots(self, n: int) -> None:
        """Free state slots until >= ``n`` are available by evicting LRU tree states."""
        while self.linear_state_pool.num_free_slots < n:
            ev = self.tree.evict_states(n - self.linear_state_pool.num_free_slots)
            if not ev.states:
                break
            self._release(ev)

    def _release(self, ev) -> None:
        """Return what the tree gave up to its pools."""
        self._free_swa(ev.window)
        self._free(ev.kv)
        if ev.states:
            self.linear_state_pool.free(ev.states)

    def plan_captures(self, req: PendingReq, handle: CacheHandle) -> List[StateCapture]:
        """The states a freshly admitted request should freeze during its prefill."""
        if not self.state_cache or req.mm_embeds is not None:
            return []
        targets = self.policy.checkpoint_positions(req.input_len, handle.cached_len,
                                                   handle.matched_len)
        return [StateCapture(target, purpose) for target, purpose in targets]

    def status(self) -> dict:
        """The public ``prefix_cache`` status object (host-side counters only, no sync)."""
        states = self.tree.state_count if self.tree is not None else 0
        pool = self.linear_state_pool
        active = pool.num_slots - 1 - pool.num_free_slots - states if self.state_cache else 0
        out = {"enabled": False, "policy": self.policy_name, "gpu_checkpoint_count": states,
               "active_state_count": active, "host_budget_bytes": 0,
               "host_allocated_bytes": 0, "host_used_bytes": 0, "host_inflight_bytes": 0,
               "transfer_device_bytes": 0, "host_checkpoint_count": 0, "restore_wait_ms": 0.0,
               "h2d_bytes": 0, "d2h_bytes": 0, "h2d_batches": 0, "d2h_batches": 0,
               "h2d_time_ms": 0.0, "d2h_time_ms": 0.0, **self.stats}
        if self.host is not None:
            out.update(self.host.status())
        if self.window_cache:
            # Physical slots: free, held by the tree (locked by request handles or window
            # copies, else evictable), and the rest owned by running requests. Copies in
            # flight are a share of the locked and request slots, not extra ones.
            slots, free = self.swa_pool.swa_num_tokens - 1, self.swa_pool.swa_available_size()
            locked, unlocked = self.tree.protected["window"], self.tree.evictable["window"]
            out["window_slots"] = dict(
                total=slots, free=free, tree_locked=locked, tree_evictable=unlocked,
                request_owned=slots - free - locked - unlocked,
                copy_inflight=self.host.window_inflight if self.host is not None else 0)
        comps = self.components
        out["components"] = [
            {"name": c.name, "storage_kind": c.storage_kind,
             "device_allocated_bytes": sum(_storage_bytes(v) for v in c.views()),
             **(self.host.component_status(c) if self.host is not None else
                {"host_used_bytes": 0, "h2d_bytes": 0, "d2h_bytes": 0})}
            for c in (*comps["paged"], comps["window"], comps["state"]) if c is not None]
        return out

    def start_restore(self, handle: CacheHandle) -> bool:
        """Bring a deeper host-resident prefix back for an admitting request; True means the
        request waits for it (other requests keep being admitted)."""
        return (self.host is not None and handle.restore is not None
                and self.host.restore(handle))

    def admitted(self, req: PendingReq, handle: CacheHandle) -> None:
        """Account an admitted request that waited for a restore: its wait, and the part of its
        reused prefix that came back from the host."""
        if req.restore_wait is None:
            return
        since, ready, restored = req.restore_wait
        self.host.restore_wait_ms += wait_ms(since)
        self._host_reuse[req.uid] = max(0, min(handle.cached_len, restored) - ready)

    def count_reuse(self, uid: int, cached: int) -> None:
        host = self._host_reuse.pop(uid, 0)
        self.stats["host_reused_tokens"] += host
        self.stats["gpu_reused_tokens"] += cached - host

    def _alloc_state(self) -> int | None:
        self.ensure_mamba_slots(1)
        pool = self.linear_state_pool
        return pool.alloc(1)[0] if pool.num_free_slots else None

    def prepare_prefill_snapshots(self, reqs: List[Req]) -> None:
        """Give every capture this chunk passes a slot; the state pool says where inside the
        extend a capture lands (at or before its target)."""
        self.stats["recomputed_tokens"] += sum(req.extend_len for req in reqs)
        if not self.state_cache:
            return
        pool = self.linear_state_pool
        for req in reqs:
            for c in req.state_captures:
                if c.slot is not None or not req.cached_len <= c.target < req.device_len:
                    continue
                pos = pool.capture_position(req.cached_len, c.target)
                if pos > req.cache_handle.cached_len and pos % self.page_size == 0:
                    c.slot = self._alloc_state()

    def snapshot_toolcall_anchor(self, reqs: List[Req]) -> None:
        """Freeze a reusable tool-call prefix before the next decode advances live state."""
        if not self.state_cache:
            return
        pool = self.linear_state_pool
        for req in reqs:
            anchor = req.toolcall_anchor_len
            if (anchor is None or any(c.purpose == ANCHOR for c in req.state_captures)
                    or anchor % self.page_size != 0 or not pool.can_export(req, anchor)):
                continue
            slot = self._alloc_state()
            if slot is not None:
                pool.export(req, anchor, slot)
                req.state_captures.append(StateCapture(anchor, ANCHOR, slot, anchor))

    def begin_linear_records(self, batch: Batch) -> None:
        """ReplaySSM: a prefill leaves a complete state, so its request restarts its records."""
        pool = self.linear_state_pool
        if pool is not None and pool.replay is not None:
            pool.replay.begin_prefill(batch.prefill_reqs)

    def _release_captures(self, req: Req, *, keep_future: bool = False) -> None:
        """Free the request's frozen states that were not published; ``keep_future`` keeps
        the captures its prefill has not reached yet."""
        future = [c for c in req.state_captures
                  if keep_future and c.slot is None and c.target >= req.cached_len]
        for c in req.state_captures:
            if c.slot is not None:
                self.linear_state_pool.free(c.slot)
        req.state_captures[:] = future

    def maybe_free_swa_out_of_window(self, reqs: List[Req], *, force: bool = False) -> None:
        """Proactively release each decoding request's now-out-of-window SWA slots, bounding its
        swa footprint to ~one window so a smaller-than-full swa pool (swa_full_tokens_ratio<1)
        stays viable. Mirrors sglang ``ScheduleBatch.maybe_evict_swa``: run every ``interval``
        committed tokens of a request (``force``: now, under pool pressure) -- counting tokens,
        not forwards, keeps a speculative round's several tokens inside the same bound; skip a
        request's first decode step (its extend forward may still be in-flight under overlap);
        free its OWN slots above the reused prefix and drop its lock on the reused prefix's
        window below the same frontier (other holders keep theirs); and keep a
        ``window + page_size`` margin so the freed slots are out-of-window for every in-flight
        forward."""
        if not self.swa_paged:
            return
        window = self.sliding_window_size
        for req in reqs:
            if req.decode_batch_idx < 1:
                continue                       # overlap guard: extend forward may still be running
            if not force and req.cached_len < req.swa_next_reclaim:
                continue
            req.swa_next_reclaim = req.cached_len + _SWA_EVICTION_INTERVAL
            floor = req.cache_handle.cached_len   # reused prefix -> its swa is tree-owned, not ours
            threshold = (req.device_len - 1) - window - self.page_size
            if req.toolcall_anchor_len is not None:
                # Keep the window ending at the anchor resumable: a client-side rewrite of the
                # echoed tool call forks after the anchor, and a resume there needs
                # [anchor - window, anchor) live. The finish-insert then adopts (rather than
                # tombstones) these never-evicted slots; they stay unlocked, so real pool
                # pressure can still reclaim them (same soft retention as the prompt-end pin).
                cap = req.toolcall_anchor_len - window - _SWA_RETAIN_GAP
                if threshold - cap > window + _SWA_RETAIN_GAP:
                    # The decode ran on far past the anchor (a tool call is normally within
                    # tens of tokens of the end). Holding the cap would grow this request's
                    # live swa without bound ("SWA pool exhausted" is unhandled) -- drop the
                    # anchor and let normal eviction resume. This bound is what the
                    # anchor-retention term in _swa_per_req_swa_floor sizes the pool for.
                    req.toolcall_anchor_len = None
                else:
                    threshold = min(threshold, cap)
            new_evicted = align_down(threshold, self.page_size)
            if self.window_cache:
                self.tree.advance_window(req.cache_handle, new_evicted)
            start = max(req.swa_evicted_seqlen, floor)
            if new_evicted > start:
                self._free_swa(self.page_table[req.table_idx, start:new_evicted])
                req.swa_evicted_seqlen = new_evicted

    def free_swa_out_of_window_extend(self, reqs: List[Req]) -> None:
        """Prefill sibling of ``maybe_free_swa_out_of_window``: before allocating a chunk, return
        each request's now-out-of-window SWA slots so a chunked prompt's live swa stays ~one window
        regardless of prompt length (else a prompt longer than the swa pool exhausts alloc_swa).
        Runs on EVERY prefill batch -- no eviction-interval cadence, since a long prompt would
        overflow the pool before a cadence fires. The frontier is based on ``cached_len`` (the
        pre-chunk, already-forwarded length; the chunk ``[cached_len, device_len)`` is allocated by
        the following ``allocate_paged``, not here), so only positions prior chunks consumed are
        freed, floored at the tree-owned reused prefix. Overlap-safe by the same scheduler stream
        gate + ``window + page_size`` margin the decode driver relies on; ``free_swa`` is idempotent
        over the sentinel, so re-freeing an earlier chunk's range is a no-op. The pool is always
        sized > one window (see the swa-pool floor), so a chunk can always make forward progress."""
        if not self.swa_paged:
            return
        window = self.sliding_window_size
        for req in reqs:
            floor = req.cache_handle.cached_len   # reused prefix -> its swa is tree-owned, not ours
            new_evicted = align_down(req.cached_len - window - self.page_size, self.page_size)
            start = max(req.swa_evicted_seqlen, floor)
            if new_evicted > start:
                self._free_swa(self.page_table[req.table_idx, start:new_evicted])
                req.swa_evicted_seqlen = new_evicted

    def open_prefill_execution(self, batch: Batch) -> _PrefillExecutionSession | None:
        """Open incremental SWA bindings when a logical prefill exceeds the pool budget."""
        if not self.swa_paged or self.prefill_chunk_budget is None:
            return None
        binding_limit = int(self.prefill_chunk_budget)
        if binding_limit < 1:
            raise RuntimeError("prefill execution budget must be positive")
        if batch.log_new_tokens <= binding_limit:
            return None
        reqs = batch.prefill_reqs
        if self._prefill_execution is not None:
            raise RuntimeError("a prefill execution session is already active")
        session = _PrefillExecutionSession(self, reqs)
        self._prefill_execution = session
        return session

    def incremental_prefill_window_reservation(self, extend_len: int) -> int | None:
        """SWA slots reserved per request when a cache session binds the range incrementally."""
        if (
            not self.swa_paged
            or self.prefill_chunk_budget is None
            or self.sliding_window_size is None
        ):
            return None
        return div_ceil(
            min(max(extend_len, 1), self.sliding_window_size) + 1,
            self.page_size,
        ) * self.page_size

    def _close_prefill_execution(self, session: _PrefillExecutionSession) -> None:
        if self._prefill_execution is not session:
            raise RuntimeError("closing an inactive prefill execution session")
        self._prefill_execution = None

    def lock(self, handle: CacheHandle) -> None:
        if handle.node is not None:
            self.tree.lock(handle)

    def unlock(self, handle: CacheHandle) -> None:
        if handle.node is not None:
            self.tree.unlock(handle)
            self._release(self.tree.take_released())

    def _free_swa(self, indices: torch.Tensor) -> None:
        """Free the swa-pool slots bound to ``indices`` (full-pool slots); each must still hold
        a binding the caller owns."""
        if self.swa_paged and len(indices) > 0:
            self.swa_pool.free_swa(indices)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        allocation_reqs: List[Req] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            reservation = self._decode_page_reservations.pop(req, None)
            if self._reservation_matches(
                req, reservation, first_page=first_page, last_page=last_page
            ):
                continue
            if reservation is not None:
                self._free_decode_reservation(reservation)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
                allocation_reqs.append(req)
        if needed_pages > 0:
            execution = self._prefill_execution
            if execution is None or not any(execution.owns(req) for req in allocation_reqs):
                self._allocate_paged_rows(needed_pages, allocation_info)
                return

            allocated = self._allocate_paged_rows(
                needed_pages,
                allocation_info,
                bind_swa=False,
            )
            offset = 0
            for req, (_, first_page, last_page) in zip(
                allocation_reqs, allocation_info, strict=True
            ):
                length = (last_page - first_page) * self.page_size
                req_allocated = allocated[offset : offset + length]
                if execution.owns(req):
                    execution.record_allocation(
                        req,
                        first_page,
                        last_page,
                        req_allocated,
                    )
                else:
                    self._bind_swa(req_allocated)
                offset += length

    def reserve_next_decode(self, reqs: List[Req]) -> None:
        """Reserve the next query pages while the current decode graph is running.

        The current graph reads only through ``cached_len``. The rows reserved here start at
        that exclusive boundary, so their page-table writes can run on the scheduler stream
        while the graph consumes the earlier row prefix on the engine stream.
        """
        if self.device.type != "cuda":
            return

        reservation_info: List[Tuple[Req, int, int]] = []
        needed_pages = 0
        for req in reqs:
            previous = self._decode_page_reservations.pop(req, None)
            if previous is not None:
                self._free_decode_reservation(previous)
            if not req.can_decode:
                continue
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                reservation_info.append((req, first_page, last_page))
        if needed_pages == 0:
            return

        allocation_info = [
            (req.table_idx, first_page, last_page)
            for req, first_page, last_page in reservation_info
        ]
        allocated = self._allocate_paged_rows(needed_pages, allocation_info)
        offset = 0
        for req, first_page, last_page in reservation_info:
            length = (last_page - first_page) * self.page_size
            self._decode_page_reservations[req] = _DecodePageReservation(
                table_idx=req.table_idx,
                cached_len=req.cached_len,
                device_len=req.device_len,
                first_page=first_page,
                last_page=last_page,
                allocated=allocated[offset : offset + length],
            )
            offset += length

    def limit_speculation(self, lengths):
        pool = self.linear_state_pool
        if pool is None:
            return lengths
        available = self.mamba_available_size if self.state_cache else pool.num_free_slots
        return pool.limit_speculation(lengths, available)

    def begin_speculation(self, reqs, views, lengths, *, draft=True):
        pool = self.linear_state_pool
        if pool is None:
            return None
        if self.state_cache:
            self.ensure_mamba_slots(pool.speculative_size(lengths))
        return pool.begin_speculation(reqs, views, lengths, draft=draft)

    def release_speculative(self, req: Req, allocated_len: int) -> None:
        """Return whole provisional pages beyond the committed target KV."""
        start = div_ceil(req.cached_len, self.page_size) * self.page_size
        end = div_ceil(allocated_len, self.page_size) * self.page_size
        pages = self.page_table[req.table_idx, start:end]
        self._free_swa(pages)
        self._free(pages)

    def _allocate_paged_rows(
        self,
        needed_pages: int,
        allocation_info: List[Tuple[int, int, int]],
        *,
        bind_swa: bool = True,
    ) -> torch.Tensor:
        allocated = self._page_to_token(self._allocate(needed_pages))
        if bind_swa:
            self._bind_swa(allocated)
        _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
        return allocated

    def _bind_swa(self, allocated: torch.Tensor) -> None:
        if not self.swa_paged or allocated.numel() == 0:
            return
        # Each newly-allocated full token needs a swa-pool slot (where its SWA-layer KV is
        # written; read back via the full->swa mapping). radix reuses the existing prefix's
        # live slots and evicts tree swa if the pool is short; naive has no tree.
        if self.window_cache:
            self.ensure_swa_slots(len(allocated))
        self.swa_pool.alloc_swa(allocated)

    @staticmethod
    def _reservation_matches(
        req: Req,
        reservation: _DecodePageReservation | None,
        *,
        first_page: int,
        last_page: int,
    ) -> bool:
        return (
            reservation is not None
            and reservation.table_idx == req.table_idx
            and reservation.cached_len == req.cached_len
            and reservation.device_len == req.device_len
            and reservation.first_page == first_page
            and reservation.last_page == last_page
        )

    def _free_decode_reservation(
        self, reservation: _DecodePageReservation
    ) -> None:
        if self.swa_paged:
            self._free_swa(reservation.allocated)
        self._free(reservation.allocated)

    def _cancel_decode_reservation(self, req: Req) -> None:
        reservation = self._decode_page_reservations.pop(req, None)
        if reservation is not None:
            self._free_decode_reservation(reservation)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        """Publish the request's committed prefix; on finish also release what it owns.

        The tree may already hold part of the prefix: the request's own pages for it are
        duplicates and go back to the pools, and an unfinished request's row is re-pointed at
        the tree's pages because the next allocation hands the duplicates to someone else."""
        self._cancel_decode_reservation(req)
        old = req.cache_handle
        if self.tree is None or req.mm_embeds is not None:
            # No reuse (or multimodal content the token ids do not identify): nothing is shared.
            self.unlock(old)
            if finished:
                self._free_tail(req, old.cached_len)
                self._free_req_slots(req)
            return
        boundaries = self._commit_boundaries(req, finished=finished)
        if not boundaries and not finished:
            self._release_captures(req, keep_future=True)
            return  # nothing new is resumable yet; the request keeps its handle
        pages = self.page_table[req.table_idx, : req.cached_len]
        free_upto, published, ends = old.cached_len, [], []
        for pos, capture, purpose in boundaries:
            slot = capture.slot if capture is not None else (
                req.linear_slot_idx if purpose == OUTPUT else None)
            matched, freed, taken, node = self.tree.insert(
                req.input_ids[:pos], pages[:pos], group=req.cache_group, state=slot,
                purpose=purpose, update_after=free_upto,
                window_freed_before=req.swa_evicted_seqlen)
            self._free_swa(freed.window)
            self._free(freed.kv)
            if taken and capture is not None:
                capture.slot = None
            elif taken:
                req.linear_slot_idx = None
            if node is None:
                # A copy is still filling the tree there: what was published before it is
                # re-pointed below, the request keeps its own pages from there on.
                free_upto = max(free_upto, matched)
                break
            if slot is not None:
                published.append(node)
            ends.append(node)
            free_upto = max(free_upto, pos)
        req.round_states.extend(published)
        self.unlock(old)
        if finished:
            if published:
                self._prune_round(req, published[-1])
            self._free_tail(req, free_upto)
            self._free_req_slots(req)
            if self.window_cache:
                self._retain_prompt_window(req)
            self._backup(ends)
            return
        if self.window_cache:
            # Locks are node-granular: cut a node boundary a window back so the request's lock
            # pins only the trailing window it still reads, not the whole chunk.
            keep_from = align_down(
                max(free_upto - self.sliding_window_size - _SWA_RETAIN_GAP, 0), self.page_size)
            if keep_from > 0:
                self.tree.match(req.input_ids[:keep_from], req.cache_group)
        # Re-point and lock all the tree now holds of the committed prefix, not just its
        # resumable part: the duplicates freed above were the request's pages for all of it.
        handle = self.tree.published(req.input_ids[:free_upto], req.cache_group)
        if handle.cached_len > old.cached_len:
            self.page_table[req.table_idx, old.cached_len : handle.cached_len].copy_(
                handle.kv_indices[old.cached_len :])
        req.cache_handle = handle
        self.lock(handle)
        self._release_captures(req, keep_future=True)
        self._backup(ends)

    def _backup(self, ends) -> None:
        if self.host is not None:
            self.host.backup([n for n in ends if not n.is_root()])

    def poll(self) -> None:
        """Publish finished host copies; called every scheduling pass, never waits."""
        if self.host is not None:
            self.host.poll()
            self._release(self.tree.take_released())

    def _prune_round(self, req: Req, deepest) -> None:
        """Let the policy drop the states this finished round replaced above ``deepest``. The
        state the round restored from is its prompt-end point unless it froze a deeper one."""
        restored, *published = req.round_states
        kept = {n for n in published if n.state is not None}
        if restored is not None and not any(n.purpose == INPUT for n in kept):
            kept.add(restored)
        self.tree.drop_states(self.policy.prune_after_commit(
            kept, self.tree.state_chain(deepest)))
        self._release(self.tree.take_released())

    def _commit_boundaries(self, req: Req, *, finished: bool):
        """(position, capture, purpose) of each resumable boundary this commit publishes, in
        order; a finished request's own live state comes last as the round's committed end."""
        if not self.state_cache:
            pos = align_down(req.cached_len, self.page_size)
            return [(pos, None, None)] if pos > 0 else []
        out = sorted(
            ((c.pos, c, c.purpose) for c in req.state_captures
             if c.slot is not None and c.pos is not None and 0 < c.pos <= req.cached_len
             and c.pos % self.page_size == 0),
            key=lambda b: b[0])
        if finished and req.cached_len > 0 and req.cached_len % self.page_size == 0:
            self.linear_state_pool.materialize(req)
            out.append((req.cached_len, None, OUTPUT))
        return out

    def _retain_prompt_window(self, req: Req) -> None:
        """Soft-pin the prompt-end window after finish: decode never re-stamps the prompt path,
        so it would be the first window victim, yet a follow-up turn that drops reasoning
        diverges right at the prompt end and needs only that trailing window. Free the head's
        window eagerly (full KV stays) and re-stamp the tail; it stays unlocked."""
        prompt_len = align_down(req.max_device_len - req.output_len, self.page_size)
        if prompt_len <= 0:
            return
        # With recurrent states the resume point is the prompt-end state, which the chunked
        # recurrence can only freeze up to one chunk before the prompt end.
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        resume = prompt_len - (CHUNK_SIZE if self.state_cache else 0)
        keep_from = align_down(
            max(resume - self.sliding_window_size - _SWA_RETAIN_GAP, 0), self.page_size)
        if keep_from > 0:
            self._free_swa(self.tree.trim_head_window(
                req.input_ids[:prompt_len], keep_from, req.cache_group))
        self.tree.match(req.input_ids[:prompt_len], req.cache_group)

    def discard_incomplete_layered_wave(
        self,
        handle: CacheHandle,
        table_idx: int,
        allocated_device_len: int,
    ) -> None:
        """Discard an unfinished full-KV layered prompt without caching it.

        Layer-major prefill keeps the original prefix handle locked while several
        prompt chunks share one page-table row.  None of that new KV is a complete
        model prefix until the wave reaches the final layer, so abort must return
        only the request-owned pages and must never publish them.
        """
        if self.state_cache or self.swa_paged:
            raise RuntimeError(
                "incomplete layered-wave discard supports full-KV models only"
            )
        start = div_ceil(handle.cached_len, self.page_size) * self.page_size
        end = div_ceil(allocated_device_len, self.page_size) * self.page_size
        self.unlock(handle)
        self._free(self.page_table[table_idx, start:end])

    def _padded_tail(self, req: Req, start: int) -> torch.Tensor:
        """The request's OWN slice [start, page_ceil(cached_len)) of the page table. A finish
        frees through the page-CEIL bound, not cached_len: allocate_paged allocates (and, when
        swa_paged, charges swa for) whole pages, so the padding [cached_len, page_ceil) belongs
        to the finishing request. ``start`` is page-aligned (a match/insert boundary), so the
        full-pool page bases derived via ``[::page_size]`` are identical to the unpadded slice."""
        end = div_ceil(req.cached_len, self.page_size) * self.page_size
        return self.page_table[req.table_idx, start:end]

    def _free_tail(self, req: Req, start: int) -> None:
        """Free the request's own pages from ``start``; its window only where still bound."""
        tail = self._padded_tail(req, start)
        self._free_swa(tail[max(0, req.swa_evicted_seqlen - start):])
        self._free(tail)

    def _free_req_slots(self, req: Req) -> None:
        """Return remaining private state; donated public states belong to the cache."""
        self._release_captures(req)
        if req.linear_slot_idx is not None:
            self.linear_state_pool.free(req.linear_slot_idx)
        req.linear_slot_idx = None

    def check_integrity(self) -> None:
        if self.host is not None:
            # Idle: finish this cache's own copies so restored pages are in the tree.
            self.host.drain()
            self._release(self.tree.take_released())
        cache_pages = 0
        if self.tree is not None:
            tree = self.tree
            tree.check_integrity()
            cache_pages = tree.kv_tokens // self.page_size
            if self.state_cache:
                # free + tree-held states can never exceed the usable pool; running requests
                # hold the remainder.
                pool = self.linear_state_pool
                held = tree.evictable["state"] + tree.protected["state"]
                assert pool.num_free_slots + held <= pool.num_slots - 1, (
                    f"state-slot leak: free({pool.num_free_slots}) + tree({held}) > "
                    f"capacity({pool.num_slots - 1})"
                )
            if self.window_cache:
                # Idle-only: no request holds a window slot, so free + tree must equal the
                # capacity exactly (slot 0 is the reserved sentinel).
                held = tree.evictable["window"] + tree.protected["window"]
                cap = self.swa_pool.swa_num_tokens - 1
                assert self.swa_pool.swa_available_size() + held == cap, (
                    f"window-slot leak/double-free: free({self.swa_pool.swa_available_size()}) + "
                    f"tree({held}) != capacity({cap})"
                )
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    def rebuild(self, num_pages: int, page_table: torch.Tensor) -> None:
        """Re-point the page table and reset page accounting + prefix tree IN PLACE.

        Idle-only: assumes no request holds a live handle.
        """
        device = page_table.device
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * self.page_size
        self._decode_page_reservations.clear()
        if self.host is not None:
            # Same layout and components: host data stays reusable; GPU copies come back by
            # restore into the new pools.
            self.host.drain()
            self.tree.drop_gpu()
        else:
            self.tree = self._make_tree()
        # The discarded tree owned donated states; rebuild is idle-only, so reclaim the whole
        # state free-list (else those slots leak -> admission hangs).
        if self.state_cache:
            self.linear_state_pool.reclaim_all_slots()

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            # clone: callers pass page-table VIEWS, and the deferred concat below only reads them
            # when the region exits. A commit that re-points the row in between (the dedup
            # re-point, the SWA one) would otherwise rewrite the pending free list underneath us
            # and return the tree's canonical pages instead of the request's duplicates.
            if len(indices) > 0:
                lazy_free_list.append(indices[:: self.page_size].clone())

        def lazy_free_swa(indices: torch.Tensor) -> None:
            # One window release per region instead of one per request.
            if self.swa_paged and len(indices) > 0:
                lazy_swa_list.append(indices.clone())

        lazy_free_list: List[torch.Tensor] = []
        lazy_swa_list: List[torch.Tensor] = []
        try:
            self._free, self._free_swa = lazy_free, lazy_free_swa
            yield
        finally:
            del self._free, self._free_swa
            if lazy_swa_list:
                self.swa_pool.free_swa(torch.cat(lazy_swa_list))
            if lazy_free_list:
                self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)) and self.tree is not None:
            ev = self.tree.evict_kv((needed_pages - free_pages) * self.page_size)
            self._free_swa(ev.window)
            if ev.states:
                self.linear_state_pool.free(ev.states)
            self.free_slots = torch.cat([self.free_slots, ev.kv[:: self.page_size]])
        assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    if len(allocation_info) <= _MAX_DIRECT_PAGE_TABLE_COPY_SEGMENTS:
        offset = 0
        for table_idx, first_page, last_page in allocation_info:
            first_pos, last_pos = first_page * page_size, last_page * page_size
            length = last_pos - first_pos
            page_table[table_idx, first_pos:last_pos].copy_(
                allocated[offset : offset + length]
            )
            offset += length
        assert offset == needed_tokens, "Mismatch in allocated tokens and copied tokens."
        return

    # Pinned only when there is a device to copy to asynchronously; CPU-only runs (unit tests,
    # a CPU CI runner) would otherwise raise instead of just doing a plain host allocation.
    pin = torch.cuda.is_available()
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=pin)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=pin)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    assert allocated.dtype == page_table.dtype, (
        f"allocated dtype {allocated.dtype} != page_table dtype {page_table.dtype}"
    )
    page_table[table_idxs, offsets] = allocated


def _storage_bytes(view: torch.Tensor) -> int:
    return view.numel() * view.element_size()
