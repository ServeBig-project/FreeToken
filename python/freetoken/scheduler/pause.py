"""Shared runtime: which requests give way under memory pressure, and their pause lifecycle.

The scheduler calls ``relieve`` before choosing each batch and ``advance`` at the end of each
round. Under pressure the newest active requests give way: a running one is held out of new
batches and paused once its last batch drained, a half-prefilled prompt drops its pages. The
oldest is protected; until it finishes nothing new is admitted or restored, so a request just
paused is not swapped straight back in. A paused request keeps its model state on the host
when there is room and is copied back later, else it is recomputed through ordinary prefill.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .cache import DEFERRED
from .utils import PendingReq

if TYPE_CHECKING:
    from freetoken.core import Req

    from .cache import CacheManager, PausedState
    from .decode import DecodeManager
    from .prefill import PrefillManager
    from .table import TableManager


@dataclass(eq=False)
class _Paused:
    req: Req
    state: PausedState | object | None  # host copy, DEFERRED, or None (recompute)
    since: float = field(default_factory=time.monotonic)


class PauseManager:
    def __init__(self, cache: CacheManager, table: TableManager, decode: DecodeManager,
                 prefill: PrefillManager, inflight: Callable[[], set],
                 fail: Callable[[int, str, Req | None], None]):
        self.cache, self.table, self.decode, self.prefill = cache, table, decode, prefill
        self.inflight = inflight  # requests in a launched batch whose results are unprocessed
        self.fail = fail  # end a request (uid, error, its Req if it has one) with a public error
        self.protected: int | None = None  # uid of the request kept running
        self.victims: list[Req] = []  # held out of batches, to pause once drained; by arrival
        # uid -> position at which a request last gave way while alone. Short again there
        # once back (packed), it cannot be served at all.
        self.walls: dict[int, int] = {}
        self.packing: set[int] = set()  # gave way to be packed: cached data goes on release
        self.unsaved: list[_Paused] = []    # host copy waits for copy budget
        self.saving: list[_Paused] = []     # host copy in flight; GPU data still held
        self.waiting: list[_Paused] = []    # on the host, by arrival
        self.restoring: list[_Paused] = []  # copy back in flight
        self.stats = cache.paused_stats

    @property
    def runnable(self) -> bool:
        return bool(self.decode.held or self.unsaved or self.saving or self.waiting
                    or self.restoring)

    # ---- before choosing a batch ----
    def relieve(self) -> None:
        """Take the running requests' next-step pages; while that fails, or a prompt's next
        chunk got no page in the last pass, the newest active request gives way. Then tell
        prefill which arrivals it may admit."""
        for req in sorted(self.decode.held - set(self.victims), key=lambda r: r.arrival):
            self.decode.unhold(req)  # parked last round: try again
        running = sorted(self.decode.running_reqs, key=lambda r: r.arrival)
        inflight = {r.uid for r in self.inflight()}
        # A prompt with a chunk inside an open wave or a launched batch is read until it drains.
        chunked = sorted((p for p in self.prefill.pending_list if p.chunked_req is not None
                          and p.uid not in inflight), key=lambda p: p.arrival)
        stalled, self.prefill.stalled = self.prefill.stalled, False
        while len(running) + len(chunked) > 1:
            if self.cache.reserve(running) and not stalled:
                break
            self.stats["short_prefill" if stalled else "short_decode"] += 1
            if chunked and (not running or chunked[-1].arrival > running[-1].arrival):
                self._drop_chunk(chunked.pop())
            else:
                self._hold(running.pop())
            stalled = False  # one prompt gives way per pass; the next pass retries
            self.protected = min(running + [p.chunked_req for p in chunked],
                                 key=lambda r: r.arrival).uid
        # Alone: nothing else holds runtime memory, not even a batch still draining (a chunk
        # mid-wave, a request that just finished): that gives way or goes next pass instead.
        alone = (not self.decode.held and not (self.unsaved or self.saving or self.restoring)
                 and not (inflight - {r.uid for r in running}))
        if running and not self.cache.reserve(running):
            if len(running) == 1 and not chunked and alone:
                self._compact(running[0], lambda: self._hold(running[0]))
            else:
                for req in running:  # cannot step now: parked for this round
                    self.decode.hold(req)
        elif stalled and not running and len(chunked) == 1 and alone:
            pending = chunked[0]
            self._compact(pending.chunked_req, lambda: self._drop_chunk(pending), pending)
        elif alone and not running and not chunked and self.prefill.blocked_head is not None:
            # The queue's head could not even start in an otherwise empty runtime (its cached
            # prefix plus one page is already too much): it cannot be served.
            pending, self.prefill.blocked_head = self.prefill.blocked_head, None
            self.prefill.pending_list.remove(pending)
            self.fail(pending.uid, "request does not fit the shared runtime", pending.paused)
        if self.protected is not None and not self._active(self.protected):
            self.protected = None
        # New and paused requests wait for the protected one; restores keep arrival order.
        oldest = self.waiting[0].req.arrival if self.waiting else None
        self.prefill.admit_before = 0 if self.protected is not None else oldest
        self.prefill.empty = (alone and not running and not chunked
                              and not self.cache.copies_inflight)

    def _compact(self, req: Req, give_way, pending: PendingReq | None = None) -> None:
        """A request alone in the runtime cannot get its next page: fragmentation, so it gives
        way (``give_way``) and comes back packed. Short again no further along than last time,
        it cannot be served at all and ends with an error."""
        if req.cached_len <= self.walls.get(req.uid, -1):
            if pending is not None:
                self.prefill.pending_list.remove(pending)
            self.walls.pop(req.uid)
            self.fail(req.uid, "request no longer fits the shared runtime even alone", req)
            return
        self.walls[req.uid] = req.cached_len
        self.packing.add(req.uid)
        if len(self.walls) > 1024:  # finished requests are not reported here; bound it
            self.walls.pop(next(iter(self.walls)))
        self.stats["compactions"] += 1
        give_way()

    def _hold(self, req: Req) -> None:
        self.victims.append(req)
        self.decode.hold(req)

    def _active(self, uid: int) -> bool:
        return any(r.uid == uid for r in self.decode.running_reqs | self.decode.held) or any(
            p.uid == uid and p.chunked_req is not None for p in self.prefill.pending_list)

    def _drop_chunk(self, pending: PendingReq) -> None:
        """A queued prompt drops its partial prefill; it restarts from the reusable prefix."""
        req, pending.chunked_req, pending.layered_cached_len = pending.chunked_req, None, None
        pending.paused_since = time.monotonic()
        self.cache.release_paused(req)
        self.table.free(req.table_idx)
        req.table_idx = -1  # released: the scheduler's free paths are no-ops now
        self._pack(req)
        self.stats["paused"] += 1
        self.stats["recompute"] += 1

    # ---- at the end of a round ----
    def advance(self, inflight: set) -> None:
        """Pause held requests whose batches drained, move host copies along, and restore
        paused requests in arrival order while nothing protected is running."""
        # Every TP rank walks the same requests in the same (arrival) order.
        for req in sorted((r for r in self.victims if r not in inflight), key=lambda r: r.arrival):
            self.victims.remove(req)
            self.decode.held.discard(req)
            if req.table_idx != -1:  # finished or aborted meanwhile
                self._pause(req)
        for paused in list(self.unsaved):
            self.unsaved.remove(paused)
            paused.state = self.cache.save_paused(paused.req)
            self._route(paused)
        for paused in [p for p in self.saving if p.state.saved]:
            self.saving.remove(paused)
            self._release(paused)
        for paused in [p for p in self.restoring if p.state.loaded]:
            self.restoring.remove(paused)
            if paused.req.aborted:
                self.cache.release_paused(paused.req)
                self.table.free(paused.req.table_idx)
                paused.req.table_idx = -1
                continue
            self.decode.running_reqs.add(paused.req)
            self.stats["restored"] += 1
            self.stats["paused_ms"] += (time.monotonic() - paused.since) * 1e3
        while self.waiting and self.protected is None and self._restore(self.waiting[0]):
            self.restoring.append(self.waiting.pop(0))


    def _pause(self, req: Req) -> None:
        self.stats["paused"] += 1
        self._route(_Paused(req, self.cache.pause(req)))

    def _route(self, paused: _Paused) -> None:
        """By how its state is kept: copying (released when done), waiting for copy budget
        (retried next round), or none (released now, to be recomputed). Without room, the
        host gives up newer paused requests' copies first."""
        if paused.state is None and self.cache.host is not None:
            for victim in sorted((p for p in self.waiting if p.req.arrival > paused.req.arrival),
                                 key=lambda p: -p.req.arrival):
                self.waiting.remove(victim)
                self.cache.discard_paused(victim.state)
                victim.state = None
                self._requeue(victim)
                paused.state = self.cache.save_paused(paused.req)
                if paused.state is not None:
                    break
        if paused.state is DEFERRED:
            self.unsaved.append(paused)
        elif paused.state is None:
            self._release(paused)
        else:
            self.saving.append(paused)

    def _release(self, paused: _Paused) -> None:
        """Its GPU data goes; it waits on the host or is recomputed through prefill."""
        req = paused.req
        self.cache.release_paused(req)
        self.table.free(req.table_idx)
        req.table_idx = -1  # released: the scheduler's free paths are no-ops now
        self._pack(req)
        if req.aborted:
            if paused.state is not None:
                self.cache.discard_paused(paused.state)
        elif paused.state is None:
            self._requeue(paused)
        else:
            self.waiting.append(paused)
            self.waiting.sort(key=lambda p: p.req.arrival)

    def _pack(self, req: Req) -> None:
        if req.uid in self.packing:
            self.packing.discard(req.uid)
            self.cache.drop_cached()

    def _requeue(self, paused: _Paused) -> None:
        self.stats["recompute"] += 1
        req = paused.req
        self.prefill.requeue(PendingReq(req.uid, req.input_ids.clone(), req.sampling_params,
                                        cache_group=req.cache_group, arrival=req.arrival,
                                        paused=req, paused_since=paused.since))

    def _restore(self, paused: _Paused) -> bool:
        """Start copying a paused request's state back; False when it does not fit now."""
        req = paused.req
        if not self.cache.restore_paused(req, paused.state, self.table):
            return False
        self.table.token_pool[req.table_idx, : req.device_len].copy_(
            req.input_ids.pin_memory(), non_blocking=True)
        return True

    def abort(self, uid: int) -> None:
        """Cancel a paused request: its host copy goes, its GPU data once no copy uses it."""
        for paused in self.unsaved + self.waiting:
            if paused.req.uid == uid:
                paused.req.aborted = True
                if paused in self.unsaved:  # nothing copied yet: GPU data goes now
                    self.unsaved.remove(paused)
                    paused.state = None
                    self._release(paused)
                else:
                    self.waiting.remove(paused)
                    self.cache.discard_paused(paused.state)
        for paused in self.saving + self.restoring:
            if paused.req.uid == uid:
                paused.req.aborted = True  # released when its copy finishes
