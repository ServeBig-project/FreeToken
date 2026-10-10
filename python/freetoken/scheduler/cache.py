from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Tuple

import numpy as np
import torch
from freetoken.core import Batch, Req
from freetoken.kvcache.prefix_policy import ANCHOR, INPUT, OUTPUT, POLICIES
from freetoken.kvcache.radix_cache import CacheHandle, RadixCache
from freetoken.kvcache.runtime_pool import Claim, upload

from .host_tier import HostTier, build_components, wait_ms
from freetoken.kvcache.prefix_store import units_of
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

# Window retention reaches this far before a resume point (the commit's node cut, tool-call
# anchors). The gap covers templates whose generation prompt injects tokens that vanish when
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


DEFERRED = object()  # a paused request's host copy must wait for copy budget
_ALL = lambda tokens: tokens  # bind a window slot to every claimed location


@dataclass(eq=False)
class PausedState:
    """A paused request's model state on the host: one copy per component, and the positions
    whose window it holds. ``saved``/``loaded`` turn true when the copies finished."""

    window: torch.Tensor
    copies: list | None = None
    saved: bool = False
    loaded: bool = False


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
                manager._free_swa(manager.rows[table_idx, start:new_evicted])
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
                 tp_group=None, page_units=None):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        # Shared runtime: the host decides which pages exist before a kernel touches them, so
        # page ids, the prefix tree and a mirror of the page table (``rows``) stay on the CPU.
        # Page 0 is the engine's dummy page, and the tail of the free list (recently freed, still
        # mapped) is allocated first.
        self.page_units = page_units
        self.index_device = torch.device("cpu") if page_units is not None else device
        self.page_size = page_size
        self._reset_pages(num_pages, page_table)
        self.linear_state_pool = linear_state_pool
        kv_pool = swa_pool
        self.residency = getattr(kv_pool, "residency", None)
        if draft_kv is not None and draft_kv.swa_paged:
            # SD targets have no window of their own: the drafter's window rides the tree.
            swa_pool, sliding_window_size = draft_kv, draft_kv.window
            self.draft_block = draft_kv.layout.config.speculative_num_steps + 1
            self.anchor_windows = getattr(draft_kv.layout.config, "special_token_ckpt", False)
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
        self.reuse = type != "naive"
        # The prefix tree carries each component the pools hold: recurrent states when a state
        # pool exists, windows when the pool pages its window KV.
        self.state_cache = self.reuse and linear_state_pool is not None
        # Live GDN states come from the pool (not the table row) with a tree or a shared runtime.
        self.private_states = linear_state_pool is not None and (
            self.state_cache or page_units is not None)
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
        if host_bytes and not self.reuse and page_units is None:
            raise ValueError("--prefix-cache-host-gib needs prefix reuse (--cache-type radix)")
        # A shared runtime also keeps paused requests on the host, with or without a tree.
        self.components = build_components(
            kv_pool, linear_state_pool, draft_kv, window_pool=swa_pool,
            window=self.window_cache or (page_units is not None and self.swa_paged),
            state=self.state_cache or (page_units is not None and linear_state_pool is not None),
            required=bool(host_bytes))
        self.tp_group = tp_group  # CPU group when TP > 1: ranks agree on finished copies
        self.host = HostTier(self, host_bytes, self.components) if host_bytes else None
        self._host_reuse: dict[int, int] = {}  # uid -> tokens its admission got from the host
        self.empty = torch.empty(0, dtype=torch.int32, device=self.index_device)
        self._decode_page_reservations: dict[Req, _DecodePageReservation] = {}
        self._prefill_execution: _PrefillExecutionSession | None = None
        self.speculative_slots: list[int] | None = None  # shared runtime: claimed for one round
        # Shared runtime: requests paused under memory pressure, kept on the host or recomputed.
        self.paused_stats = dict(paused=0, recompute=0, restored=0, recomputed_tokens=0,
                                 paused_ms=0.0, short_decode=0, short_prefill=0, compactions=0,
                                 claim_ms=0.0)

    # ----- capability hooks (defaults; plugged-in pools may narrow them) -----
    supports_runtime_rebuild = True
    prefill_chunk_budget = None  # generic shared page pool: no per-model prefill chunk cap
    draft_block = 0  # tokens one draft round adds when the window pool is the drafter's
    anchor_windows = False  # a retained tool-call anchor may hold one more window

    def _make_tree(self) -> RadixCache | None:
        if not self.reuse:
            return None
        return RadixCache(
            self.index_device, self.page_size, self.policy, self.stats,
            window=self.sliding_window_size if self.window_cache else None,
            has_state=self.state_cache,
        )

    def page_usage(self) -> tuple[int, int]:
        """(used_pages, total_pages): allocated, non-evictable pages over the pool total
        (active requests + protected prefix; evictable prefix-cache pages are excluded)."""
        if self.page_units is not None:  # page ids are address space; memory is in the runtime
            return 0, 0
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
    def copies_inflight(self) -> bool:
        """Host copies still running (their nodes cannot be evicted meanwhile)."""
        return self.host is not None and self.host.transfer.inflight_bytes > 0

    @property
    def available_size(self) -> int:
        # A shared runtime counts free page ids; whether memory backs them is settled when
        # pages are taken (``take``), which evicts or fails as a whole.
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

    def wave_window_growth(self, reqs: Iterable[Req]) -> int:
        """Window slots decoding requests may take before their next release while a layered
        wave runs without new admissions: the rest of a window not filled yet, then one release
        interval and one draft round (plus the retained window before an anchor drops), never
        more than each can still generate. Drafter pool only (it is sized per running request);
        a reused prefix holds at most one window."""
        if not self.draft_block:
            return 0
        window = self.sliding_window_size
        steady = _SWA_EVICTION_INTERVAL + self.draft_block + 2 * self.page_size
        if self.anchor_windows:
            steady += window + _SWA_RETAIN_GAP
        return sum(
            min(req.max_device_len - req.device_len + self.draft_block,
                window - min(req.device_len - req.swa_evicted_seqlen, window) + steady)
            for req in reqs)

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
        active = pool.num_slots - 1 - pool.unused_slots - states if self.state_cache else 0
        out = {"enabled": False, "policy": self.policy_name, "gpu_checkpoint_count": states,
               "active_state_count": active, "host_budget_bytes": 0,
               "host_allocated_bytes": 0, "host_used_bytes": 0, "host_inflight_bytes": 0,
               "transfer_device_bytes": 0, "host_checkpoint_count": 0, "restore_wait_ms": 0.0,
               "h2d_bytes": 0, "d2h_bytes": 0, "h2d_batches": 0, "d2h_batches": 0,
               "h2d_time_ms": 0.0, "d2h_time_ms": 0.0, **self.stats}
        if self.host is not None:
            out.update(self.host.status())
        if self.page_units is not None:
            out["runtime"] = dict(self.page_units.blocks.status(), **self.paused_stats,
                                  evictable_bytes=self._evictable_bytes())
            if self.residency is not None:
                out["runtime"].update(self.residency.status())
        if self.swa_paged:
            # Physical slots: free, held by the tree (locked by request handles or window
            # copies, else evictable), and the rest owned by running requests. Copies in
            # flight are a share of the locked and request slots, not extra ones.
            slots, free = self.swa_pool.swa_num_tokens - 1, self.swa_pool.swa_available_size()
            locked, unlocked = ((self.tree.protected["window"], self.tree.evictable["window"])
                                if self.window_cache else (0, 0))
            out["window_slots"] = dict(
                total=slots, free=free, tree_locked=locked, tree_evictable=unlocked,
                request_owned=slots - free - locked - unlocked,
                copy_inflight=self.host.window_inflight if self.host is not None else 0)
        comps = self.components
        held = (self.page_units.blocks.held_bytes if self.page_units is not None
                else lambda views: sum(_storage_bytes(v) for v in views))
        out["components"] = [
            {"name": c.name, "storage_kind": c.storage_kind,
             "device_allocated_bytes": held(c.views()),
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
        if self.page_units is not None:
            got = self.claim(states=1)
            return got[1][0] if got is not None else None
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
                self._free_swa(self.rows[req.table_idx, start:new_evicted])
                req.swa_evicted_seqlen = new_evicted

    def free_swa_out_of_window_extend(self, reqs: List[Req], *, before: int | None = None) -> None:
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
        sized > one window (see the swa-pool floor), so a chunk can always make forward progress.
        ``before`` frees less: only below it."""
        if not self.swa_paged:
            return
        window = self.sliding_window_size
        for req in reqs:
            floor = req.cache_handle.cached_len   # reused prefix -> its swa is tree-owned, not ours
            frontier = req.cached_len - window - self.page_size
            new_evicted = align_down(frontier if before is None else min(before, frontier),
                                     self.page_size)
            start = max(req.swa_evicted_seqlen, floor)
            if new_evicted > start:
                self._free_swa(self.rows[req.table_idx, start:new_evicted])
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
                if self._allocate_paged_rows(needed_pages, allocation_info) is None:
                    raise RuntimeError("shared runtime: a batch ran rows nobody reserved")
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
        for req in reqs:
            self._cancel_decode_reservation(req)
        self.reserve([req for req in reqs if req.can_decode])

    def reserve(self, reqs: List[Req]) -> bool:
        """Take the pages (and their window slots) of each request's pending rows
        ``[cached_len, device_len)`` ahead of the batch that runs them, which then reuses
        them; all or nothing. With a shared runtime False means memory is short now."""
        info = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if self._reservation_matches(req, self._decode_page_reservations.get(req),
                                         first_page=first_page, last_page=last_page):
                continue
            self._cancel_decode_reservation(req)
            if last_page > first_page:
                info.append((req, first_page, last_page))
        if not info:
            return True
        allocated = self._allocate_paged_rows(
            sum(last - first for _, first, last in info),
            [(req.table_idx, first, last) for req, first, last in info])
        if allocated is None:
            return False
        self._record_reservations(info, allocated)
        return True

    def hold_rows(self, req: Req, tokens: torch.Tensor) -> None:
        """Point a request's pending rows at pages a claim already took, as a reservation the
        batch that runs them reuses."""
        info = [(req, div_ceil(req.cached_len, self.page_size),
                 div_ceil(req.device_len, self.page_size))]
        if len(tokens):
            self._write_rows(tokens, [(req.table_idx, info[0][1], info[0][2])])
            self._record_reservations(info, tokens)

    def _record_reservations(self, info, allocated: torch.Tensor) -> None:
        offset = 0
        for req, first_page, last_page in info:
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
        if self.page_units is not None:
            return lengths  # the round's scratch states come with its pages (reserve_round)
        available = self.mamba_available_size if self.state_cache else pool.num_free_slots
        return pool.limit_speculation(lengths, available)

    def begin_speculation(self, reqs, views, lengths, *, draft=True):
        pool = self.linear_state_pool
        if pool is None:
            return None
        if self.page_units is not None and pool.replay is None:
            slots, self.speculative_slots = self.speculative_slots, None
            return pool.begin_speculation(reqs, views, lengths, draft=draft, slots=slots)
        if self.state_cache:
            self.ensure_mamba_slots(pool.speculative_size(lengths))
        return pool.begin_speculation(reqs, views, lengths, draft=draft)

    def reserve_round(self, views: List[Req], states: int) -> bool:
        """Shared runtime: an SD round's draft pages (the views' pending rows, with their
        window slots) and its ``states`` scratch states in one claim; False, with nothing
        taken, when they do not fit. ``start`` runs the views and uses the states."""
        self.drop_speculative_slots()
        info = []
        for req in views:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                info.append((req, first_page, last_page))
        got = self.claim(sum(last - first for _, first, last in info),
                         window=_ALL if self.swa_paged else None, states=states)
        if got is None:
            return False
        allocated, self.speculative_slots, _ = got
        if info:
            self._write_rows(allocated, [(r.table_idx, first, last) for r, first, last in info])
            self._record_reservations(info, allocated)
        return True

    def drop_speculative_slots(self) -> None:
        """Give back scratch states claimed for a round that does not run."""
        if self.speculative_slots:
            self.linear_state_pool.free(self.speculative_slots)
        self.speculative_slots = None

    def release_speculative(self, req: Req, allocated_len: int) -> None:
        """Return whole provisional pages beyond the committed target KV."""
        start = div_ceil(req.cached_len, self.page_size) * self.page_size
        end = div_ceil(allocated_len, self.page_size) * self.page_size
        pages = self.rows[req.table_idx, start:end]
        self._free_swa(pages)
        self._free(pages)

    def _allocate_paged_rows(
        self,
        needed_pages: int,
        allocation_info: List[Tuple[int, int, int]],
        *,
        bind_swa: bool = True,
    ) -> torch.Tensor | None:
        if self.page_units is None:
            allocated = self._page_to_token(self._allocate(needed_pages))
            if bind_swa:
                self._bind_swa(allocated)
        else:  # None: the runtime cannot back them now, nothing taken
            got = self.claim(needed_pages, window=_ALL if bind_swa and self.swa_paged else None)
            if got is None:
                return None
            allocated = got[0]
        self._write_rows(allocated, allocation_info)
        return allocated

    def _write_rows(self, allocated: torch.Tensor, allocation_info) -> None:
        on_device = allocated
        if self.page_units is not None:
            _write_page_table(self.rows, allocated, allocation_info, self.page_size)
            on_device = upload(allocated, self.device)
        _write_page_table(self.page_table, on_device, allocation_info, self.page_size)

    def write_row(self, table_idx: int, start: int, values: torch.Tensor) -> None:
        """Point ``page_table[table_idx, start:]`` at ``values`` (page ids on the index device)."""
        end = start + len(values)
        if self.page_units is not None:
            self.rows[table_idx, start:end] = values
            values = upload(values, self.device)
        self.page_table[table_idx, start:end].copy_(values)

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
        if not finished and self.draft_block:
            # The drafter's pool is sized per running request, but a layered wave can prefill
            # far more than one window and decode's own release skips a request's first step:
            # return what lies a window before the first resume point now. Holding it would
            # also stop the commit at a shared node whose window another request released.
            bounds = self._commit_boundaries(req, finished=False)
            if bounds:
                self.free_swa_out_of_window_extend(
                    [req], before=bounds[0][0] - self.sliding_window_size - _SWA_RETAIN_GAP)
        self._commit_req(req, finished=finished)

    def _commit_req(self, req: Req, *, finished: bool) -> None:
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
        pages = self.rows[req.table_idx, : req.cached_len]
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
            self.write_row(req.table_idx, old.cached_len, handle.kv_indices[old.cached_len :])
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
            if self.tree is not None:
                self._release(self.tree.take_released())
        if self.residency is not None:
            self.residency.poll()

    def backup_completed(self, req: Req, length: int) -> None:
        if self.residency is not None and self.host is not None:
            end = align_down(length, self.page_size)
            self.host.backup_active(units_of(self.rows[req.table_idx, :end], self.page_size))

    # ----- paused requests (shared runtime) -----
    def pause(self, req: Req) -> PausedState | None:
        """Start keeping a drained request's committed model state (KV, window and recurrent
        state through ``cached_len``) on the host; None when the host budget cannot hold it,
        so it will be recomputed. Its GPU data stays until ``release_paused``."""
        self._cancel_decode_reservation(req)
        if self.linear_state_pool is not None:
            self.linear_state_pool.materialize(req)
        return self.save_paused(req) if self.host is not None else None

    def save_paused(self, req: Req) -> PausedState | None:
        """Copy a paused request's state to the host; None when the budget cannot hold it,
        DEFERRED while the copy queue or window copy budget is busy."""
        units, window = self._paused_units(req)
        if not self.host.credit(len(window), save=True):
            return DEFERRED
        state = PausedState(window=window)
        state.copies = self.host.save(units, len(window), lambda: setattr(state, "saved", True))
        if self._agree(state.copies is not None):
            return state
        if state.copies is not None:  # another TP rank keeps no copy: all recompute
            self.discard_paused(state)
        return None

    def _evictable_bytes(self) -> int:
        """Bytes of cached prefix data no request holds (KV, GDN checkpoints, window slots):
        claims reclaim it for any component. Logical bytes, not whole physical blocks."""
        unit = lambda units: sum(length for *_, length in units.banks) if units else 0
        pages = self._evictable("kv") // self.page_size * unit(self.page_units)
        states = self._evictable("state") * unit(getattr(self.linear_state_pool, "units", None))
        windows = (self._evictable("window") * unit(self.swa_pool.slot_units)
                   if self.swa_paged else 0)
        return pages + states + windows

    def drop_cached(self) -> None:
        """Release every unlocked cached prefix from the GPU, so the next claims, taking the
        lowest free ids, lay a request out packed."""
        while self._evict_any(1 << 62):  # any amount: everything unlocked goes
            pass

    def release_paused(self, req: Req) -> None:
        """Give back a paused request's GPU data, as a finish would, without publishing it."""
        self._cancel_decode_reservation(req)
        self.unlock(req.cache_handle)
        self._free_tail(req, req.cache_handle.cached_len)
        self._free_req_slots(req)

    def restore_paused(self, req: Req, state: PausedState, table) -> bool:
        """Take a new table row and fresh storage for a paused request's saved state in one
        claim and copy it back (``state.loaded`` once done); False when the runtime cannot
        hold it now."""
        c = req.cached_len
        if not self.host.credit(len(state.window), save=False):
            return False
        got = self.claim(div_ceil(c, self.page_size),
                         window=(lambda tokens: tokens[state.window]) if len(state.window)
                         else None,
                         states=int(self.private_states), table=table,
                         payload_pages=int(c % self.page_size != 0) if self.residency else None)
        if got is None:
            return False
        tokens, slots, req.table_idx = got
        self.write_row(req.table_idx, 0, tokens)
        req.linear_slot_idx = slots[0] if slots else None
        if slots and self.linear_state_pool.replay is not None:
            self.linear_state_pool.replay.restart([req], [c])
        req.cache_handle = CacheHandle(0, None, self.empty)
        req.swa_evicted_seqlen = int(state.window[0]) if len(state.window) else c
        units, _ = self._paused_units(req)

        def loaded():
            for copy in state.copies:
                copy.release()
            state.loaded = True
        self.host.load(units, state.copies, len(state.window), loaded)
        return True

    def discard_paused(self, state: PausedState) -> None:
        for copy in state.copies:
            copy.release()

    def _paused_units(self, req: Req):
        """(component, unit indices) of a request's state through ``cached_len``, and the
        positions whose window is bound (the drafter's trailing window)."""
        ps, c, comps = self.page_size, req.cached_len, self.components
        row = self.rows[req.table_idx]
        units = [(comp, units_of(row[: div_ceil(c, ps) * ps], ps)) for comp in comps["paged"]]
        window = self.empty
        if comps["window"] is not None:
            start = max(align_down(c - self.sliding_window_size - ps, ps), 0)
            bound = self.swa_pool.window_units(row[start:c])
            window = torch.nonzero(bound > 0).flatten() + start
            units.append((comps["window"], bound[bound > 0]))
        if comps["state"] is not None:
            units.append((comps["state"], torch.tensor([req.linear_slot_idx])))
        return units, window

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
        diverges right at the prompt end. Re-stamp the path, its head oldest; it stays unlocked.
        The head's windows are not dropped here: other resume points on a shared path still
        need them, and window pressure evicts the head first anyway."""
        prompt_len = align_down(req.prompt_len, self.page_size)
        if prompt_len > 0:
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
        self._free(self.rows[table_idx, start:end])

    def _padded_tail(self, req: Req, start: int) -> torch.Tensor:
        """The request's OWN slice [start, page_ceil(cached_len)) of the page table. A finish
        frees through the page-CEIL bound, not cached_len: allocate_paged allocates (and, when
        swa_paged, charges swa for) whole pages, so the padding [cached_len, page_ceil) belongs
        to the finishing request. ``start`` is page-aligned (a match/insert boundary), so the
        full-pool page bases derived via ``[::page_size]`` are identical to the unpadded slice."""
        end = div_ceil(req.cached_len, self.page_size) * self.page_size
        return self.rows[req.table_idx, start:end]

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
            if self.tree is not None:
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

    def rebuild(self, num_pages: int, page_table: torch.Tensor, page_units=None) -> None:
        """Re-point the page table and reset page accounting + prefix tree IN PLACE (onto a
        shared runtime's new ``page_units``, whose pools come back empty).

        Idle-only: assumes no request holds a live handle.
        """
        device = page_table.device
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_units = page_units
        self._reset_pages(num_pages, page_table)
        self._decode_page_reservations.clear()
        self.speculative_slots = None
        if self.host is not None:
            # Same layout and components: host data stays reusable; GPU copies come back by
            # restore into the new pools.
            self.host.drain()
            if self.tree is not None:
                self.tree.drop_gpu()
        else:
            self.tree = self._make_tree()
        # The discarded tree owned donated states; rebuild is idle-only, so reclaim the whole
        # state free-list (else those slots leak -> admission hangs).
        if self.state_cache and page_units is None:
            self.linear_state_pool.reclaim_all_slots()

    def _reset_pages(self, num_pages: int, page_table: torch.Tensor) -> None:
        """Every page free (with a shared runtime, page 0 is the dummy page)."""
        if self.page_units is None:
            self.free_slots = torch.arange(
                num_pages, dtype=torch.int32, device=page_table.device) * self.page_size
            self.rows = page_table
        else:
            self.free_slots = torch.arange(num_pages, 0, -1, dtype=torch.int32) * self.page_size
            self.rows = torch.zeros(page_table.shape, dtype=page_table.dtype)

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
                self._return_pages(torch.cat(lazy_free_list))

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

    def claim(self, pages: int = 0, *, window=None, states: int = 0, table=None,
              payload_pages: int | None = None):
        """Shared runtime: one operation's units, held together or not at all -- ``pages``
        pages (as token locations), window slots for the locations ``window(tokens)`` names,
        ``states`` GDN state slots, and ``table``'s next row with its records.
        Cached prefix data is evicted while they do not fit, and every TP rank must hold them
        before any is recorded as taken. Returns (tokens, slots, row), or None."""
        begin = time.perf_counter()
        try:
            return self._claim(pages, window, states, table, payload_pages)
        finally:  # time on the scheduler thread spent taking memory, evictions included
            self.paused_stats["claim_ms"] += (time.perf_counter() - begin) * 1e3

    def _claim(self, pages, window, states, table, payload_pages):
        pool, ps, out = self.linear_state_pool, self.page_size, {}

        def build():
            claim = Claim()
            ids = self.free_slots[len(self.free_slots) - pages:] if pages else self.empty
            if len(ids) < pages:
                return None
            tokens = self._page_to_token(ids)
            claim.add(self.page_units, ids.numpy() // ps, lambda _: self._pop_pages(pages))
            if self.residency is not None:
                page_ids = ids.numpy() // ps
                count = pages if payload_pages is None else payload_pages
                writable = page_ids[-count:] if count else page_ids[:0]
                claim.add(self.residency.payload, writable,
                          lambda _: self.residency.allocated(page_ids, writable))
            if window is not None:
                locs = window(tokens).to(torch.int64)
                slots = self.swa_pool.next_slots(len(locs))
                if slots is None:
                    return None
                claim.add(self.swa_pool.slot_units, slots.numpy(),
                          lambda _: self.swa_pool.bind_slots(locs, slots))
            state = pool.peek(states) if states else []
            if state is None:
                return None
            if states:
                claim.add(pool.units, state, pool.take)
            row = table.peek() if table is not None else None
            if table is not None:
                if row is None:
                    return None
                claim.add(table.row_units, [row], lambda _: table.take(row))
            out.update(tokens=tokens, slots=state, row=row)
            return claim

        # Cached data goes one entry at a time, and only while the claim really lacks ids or
        # blocks: never more than the operation needs.
        while (claim := build()) is None or not self.page_units.blocks.acquire(claim.plan):
            if not (self._evict_any(ps) or
                    (self.residency is not None and self.residency.evict_one())):
                claim = None
                break
        if not self._agree(claim is not None):
            if claim is not None:  # another TP rank could not: release, record nothing
                claim.release()
            return None
        claim.commit()
        return out["tokens"], out["slots"], out["row"]

    def _pop_pages(self, pages: int) -> None:
        self.free_slots = self.free_slots[: len(self.free_slots) - pages]

    def _agree(self, ok: bool) -> bool:
        """Whether every TP rank's shared-runtime step succeeded (each rank asks at the same
        point of the same scheduling pass)."""
        if self.tp_group is None:
            return ok
        flag = torch.tensor([int(ok)], dtype=torch.int64)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=self.tp_group)
        return bool(flag.item())

    def _evict_any(self, tokens: int) -> bool:
        """Release one batch of unlocked cached prefix data; False when nothing is left."""
        if self.tree is None:
            return False
        kinds = [(self.tree.evict_kv, tokens), (self.tree.evict_states, 1)]
        if self.swa_paged:
            kinds.append((self.tree.evict_window, tokens))
        for evict, amount in kinds:
            ev = evict(amount)
            if ev.kv.numel() or ev.window.numel() or ev.states:
                self._release(ev)
                return True
        return False

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._return_pages(indices[:: self.page_size])

    def _return_pages(self, pages: torch.Tensor) -> None:
        if self.page_units is None:
            self.free_slots = torch.cat([self.free_slots, pages])
            return
        if self.residency is not None:
            self.residency.release(pages.numpy() // self.page_size)
        # Kept descending, the lowest page next (claims take the tail): live pages pack into
        # the fewest blocks. Inserted in place of a full sort: returns happen every step.
        free, back = self.free_slots.numpy()[::-1], np.sort(pages.numpy())
        self.free_slots = torch.from_numpy(
            np.insert(free, np.searchsorted(free, back), back)[::-1].copy())
        self.page_units.release(pages.numpy() // self.page_size)

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=pages.device, dtype=torch.int32)
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
