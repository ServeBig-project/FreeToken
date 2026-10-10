"""Moves prefix-cache data between the GPU tree and host memory for the CacheManager.

The tree decides what a resume point needs (``plan_backup`` / ``plan_restore``) and holds
locks while a copy runs; this module reserves the destinations, builds the copy tasks, and
attaches the result when the transfer reports completion. Nothing is published before its
copy finished.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, List

import torch
from freetoken.kvcache.prefix_store import (
    Component, CopyTask, HostCopy, HostSpan, HostStore, PrefixTransfer, units_of,
)
from freetoken.utils import align_ceil

if TYPE_CHECKING:
    from .cache import CacheManager


def build_components(kv_pool, state_pool, draft_kv, *, window_pool, window: bool, state: bool,
                     required: bool) -> dict:
    """The cache's storage components by role. ``required`` (host tier on) turns a pool without
    the needed layout views into a startup error naming the missing storage capability.
    ``window_pool`` holds the window KV: the target pool, or the drafter's context."""
    def comp(pool, method, name, kind):
        if hasattr(pool, method):
            return Component(name, kind, getattr(pool, method))
        if required:
            raise ValueError(f"--prefix-cache-host-gib needs {kind} storage views from "
                             f"{type(pool).__name__} ({method}); this cache layout has none")
        return None

    paged = [comp(kv_pool, "paged_views", "paged_kv", "paged")]
    if draft_kv is not None and draft_kv.paged_views():
        paged.append(comp(draft_kv, "paged_views", "draft_kv", "paged"))
    out = {"paged": [c for c in paged if c is not None], "window": None, "state": None}
    if window:
        name = "draft_window" if window_pool is draft_kv else "window_kv"
        out["window"] = comp(window_pool, "window_views", name, "window")
    if state:
        out["state"] = comp(state_pool, "state_views", "recurrent_state", "boundary_state")
    return out


class HostTier:
    def __init__(self, manager: CacheManager, host_bytes: int, components: dict) -> None:
        self.m = manager
        self.store = HostStore(host_bytes)
        self.transfer = PrefixTransfer(manager.device)
        self.comps = components
        self.host_bytes = {c.name: 0 for c in self._all_comps()}
        self.restore_wait_ms = 0.0
        # Window slots copies may hold beyond what running requests need: one window in all,
        # which the window pool reserves.
        self.window_limit = (align_ceil(manager.sliding_window_size, manager.page_size)
                             if components["window"] else 0)
        self.window_inflight = 0

    def _take_window(self, plan) -> int | None:
        """Window slots this plan holds, if the copy budget has room for them."""
        tokens = plan.window_tokens
        if self.window_inflight + tokens > self.window_limit:
            return None
        self.window_inflight += tokens
        return tokens

    def _done(self, tokens: int, finish) -> None:
        self.window_inflight -= tokens
        finish()

    def _all_comps(self) -> List[Component]:
        return [c for c in (*self.comps["paged"], self.comps["window"], self.comps["state"]) if c]

    # ---------------------------------------------------------------- backup
    def backup(self, nodes) -> None:
        """Copy each published resume point's missing host data, shallowest first. A full
        queue or host budget skips the backup; requests never wait for it."""
        tree = self.m.tree
        for node in nodes:
            if self.transfer.backup_full:
                return
            plan = tree.plan_backup(node)
            if plan is None:
                continue
            need = (sum(units for _, units in plan.kv) * sum(c.unit_bytes() for c in self.comps["paged"])
                    + (sum(units for _, units in plan.window) * self.comps["window"].unit_bytes()
                       if plan.window else 0)
                    + (self.comps["state"].unit_bytes() if plan.state else 0))
            free = self.store.budget - self.store.used
            if need > free and need > free + tree.host_freeable_bytes():
                tree.abandon(plan)  # cannot fit even after evicting: keep older host data
                continue
            window_tokens = self._take_window(plan)
            if window_tokens is None:
                tree.abandon(plan)  # restores come first: skip this optional copy
                continue
            kv = [self._copies(self.comps["paged"], units) for _, units in plan.kv]
            win = [self._copies([self.comps["window"]], units) for _, units in plan.window]
            st = self._copies([self.comps["state"]], 1) if plan.state else []
            if None in (*kv, *win, st):
                for copies in (*kv, *win, st):
                    for c in copies or ():
                        c.release()
                self.window_inflight -= window_tokens
                tree.abandon(plan)
                continue
            tasks, keep = [], []
            for (n, units), copies in zip(plan.kv, kv):
                idx = units_of(n.value, self.m.page_size)
                keep.append(idx)
                tasks += [CopyTask(c.comp, idx, HostSpan(c, 0, units)) for c in copies]
            for (n, units), copies in zip(plan.window, win):
                idx = self.m.swa_pool.window_units(n.value)
                keep.append(idx)
                tasks.append(CopyTask(copies[0].comp, idx, HostSpan(copies[0], 0, units)))
            if plan.state:
                idx = torch.tensor([node.state], dtype=torch.int64, device=self.m.device)
                keep.append(idx)
                tasks.append(CopyTask(st[0].comp, idx, HostSpan(st[0], 0, 1)))
            self.transfer.submit(
                "d2h", tasks,
                lambda plan=plan, kv=kv, win=win, st=st, w=window_tokens: self._done(
                    w, lambda: tree.finish_backup(plan, kv, win, st if plan.state else None)),
                keep)

    def _copies(self, comps, units) -> list | None:
        """Host copies of ``units`` units for each component, freeing host data if needed."""
        out = []
        for comp in comps:
            copy = HostCopy(self.store, comp, units)
            if copy.where is None and self.m.tree is not None:
                self.m.tree.evict_host(lambda n=copy.nbytes: self.store.fits(n))
                self.m._release(self.m.tree.take_released())
                copy = HostCopy(self.store, comp, units)
            if copy.where is None:
                for c in out:
                    c.release()
                return None
            self.host_bytes[comp.name] += copy.nbytes
            copy.on_free = lambda c=copy: self._freed(c)
            out.append(copy)
        return out

    def _freed(self, copy: HostCopy) -> None:
        self.host_bytes[copy.comp.name] -= copy.nbytes

    # ---------------------------------------------------------------- restore
    def restore(self, handle) -> bool:
        """Start (or join) bringing ``handle.restore`` back to the GPU. True: the request should
        wait for it; False: restoring is not possible now, use the GPU-ready prefix."""
        m, tree, node = self.m, self.m.tree, handle.restore
        if node.busy:
            return True  # another request's restore (or a backup) works on it
        plan = tree.plan_restore(node)
        if plan is None:
            return True
        ps = m.page_size
        tokens = sum(n.length for n, _ in plan.kv)
        win_tokens = sum(n.length for n, _ in plan.window)
        if tokens > m.available_size or (
                plan.window and m.swa_available_size < win_tokens) or (
                plan.state and m.mamba_available_size < 1):
            tree.abandon(plan)
            return False
        window_tokens = self._take_window(plan)
        if window_tokens is None:
            tree.abandon(plan)
            return True  # wait for the copies in flight to return their window budget
        def split(locs):
            """Each node's new KV locations, and every window node's locations."""
            values, off = [], 0
            for n, _ in plan.kv:
                values.append(locs[off : off + n.length])
                off += n.length
            new = {id(n): v for (n, _), v in zip(plan.kv, values)}
            return values, [new.get(id(n), n.value) for n, _ in plan.window]

        if m.page_units is not None:  # one claim: pages, window slots and the state
            got = m.claim(tokens // ps, states=int(plan.state),
                          window=(lambda t: torch.cat(split(t)[1])) if plan.window else None)
            if got is None:
                self.window_inflight -= window_tokens
                tree.abandon(plan)
                return False
            locs, slot = got[0], (got[1][0] if plan.state else None)
            values, win_locs = split(locs)
        else:
            locs = m._page_to_token(m._allocate(tokens // ps)) if tokens else m.empty
            values, win_locs = split(locs)
            if plan.window:
                m.ensure_swa_slots(win_tokens)
                m.swa_pool.alloc_swa(torch.cat(win_locs))
            slot = m._alloc_state() if plan.state else None
        tasks, keep = [], [locs]
        for (n, units), value in zip(plan.kv, values):
            idx = units_of(value, ps)
            keep.append(idx)
            tasks += [CopyTask(c, idx, span) for c, span in zip(self.comps["paged"], n.host)]
        for (n, units), value in zip(plan.window, win_locs):
            idx = m.swa_pool.window_units(value)
            keep.append(idx)
            tasks.append(CopyTask(self.comps["window"], idx, n.host_window[0]))
        if plan.state:
            idx = torch.tensor([slot], dtype=torch.int64, device=m.device)
            keep.append(idx)
            tasks.append(CopyTask(self.comps["state"], idx, node.host_state[0]))
        self.transfer.submit("h2d", tasks, lambda: self._done(
            window_tokens, lambda: tree.finish_restore(plan, values, slot)), keep)
        return True

    # ---------------------------------------------------------------- paused requests
    def credit(self, window_tokens: int, save: bool) -> bool:
        """Whether a private copy holding ``window_tokens`` window slots may start now, under
        the copy queue and window budget the prefix copies use (one copy larger than the
        whole window budget may run alone, so every request can still be offloaded)."""
        if save and self.transfer.backup_full:
            return False
        return (not self.window_inflight
                or self.window_inflight + window_tokens <= self.window_limit)

    def save(self, units, window_tokens: int, done) -> list | None:
        """Private host copies of a paused request's ``units`` (component, unit indices) pairs,
        ``done`` once copied; None, with nothing copied, when the host budget cannot hold them.
        Cold prefix data may be evicted for them; they are never evicted for prefixes."""
        copies = []
        for comp, idx in units:
            got = self._copies([comp], len(idx))
            if got is None:
                for c in copies:
                    c.release()
                return None
            copies.append(got[0])
        self.window_inflight += window_tokens
        self.transfer.submit("d2h", self._tasks(units, copies),
                             lambda: self._done(window_tokens, done))
        return copies

    def load(self, units, copies, window_tokens: int, done) -> None:
        """Copy saved units back into ``units`` (component, new unit indices) pairs."""
        self.window_inflight += window_tokens
        self.transfer.submit("h2d", self._tasks(units, copies),
                             lambda: self._done(window_tokens, done))

    @staticmethod
    def _tasks(units, copies) -> list:
        return [CopyTask(comp, idx, HostSpan(copy, 0, len(idx)))
                for (comp, idx), copy in zip(units, copies, strict=True)]

    # ---------------------------------------------------------------- lifecycle
    def poll(self) -> None:
        self.transfer.poll(self.m.tp_group)

    def drain(self) -> None:
        self.transfer.drain(self.m.tp_group)

    def status(self) -> dict:
        t = self.transfer
        return {
            "enabled": True,
            # Values are this worker's (TP rank 0's); an instance total is per-rank x tp_size.
            "scope": "worker",
            "tp_size": 1 if self.m.tp_group is None else self.m.tp_group.size(),
            "host_budget_bytes": self.store.budget,
            "host_allocated_bytes": self.store.budget,
            "host_used_bytes": self.store.used,
            "host_inflight_bytes": t.inflight_bytes,
            "transfer_device_bytes": t.device_bytes,
            "host_checkpoint_count": self.m.tree.host_states if self.m.tree else 0,
            "restore_wait_ms": self.restore_wait_ms,
            **t.stats,
        }

    def component_status(self, comp: Component) -> dict:
        per = self.transfer.comp_bytes.get(comp.name, {})
        return {"host_used_bytes": self.host_bytes.get(comp.name, 0),
                "h2d_bytes": per.get("h2d", 0), "d2h_bytes": per.get("d2h", 0)}


def wait_ms(since: float) -> float:
    return (time.monotonic() - since) * 1000.0
