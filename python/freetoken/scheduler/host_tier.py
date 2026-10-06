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

if TYPE_CHECKING:
    from .cache import CacheManager


def build_components(kv_pool, state_pool, draft_kv, *, window: bool, state: bool,
                     required: bool) -> dict:
    """The cache's storage components by role. ``required`` (host tier on) turns a pool without
    the needed layout views into a startup error naming the missing storage capability."""
    def comp(pool, method, name, kind):
        if hasattr(pool, method):
            return Component(name, kind, getattr(pool, method))
        if required:
            raise ValueError(f"--prefix-cache-host-gib needs {kind} storage views from "
                             f"{type(pool).__name__} ({method}); this cache layout has none")
        return None

    paged = [comp(kv_pool, "paged_views", "paged_kv", "paged")]
    if draft_kv is not None:
        paged.append(comp(draft_kv, "paged_views", "draft_kv", "paged"))
    out = {"paged": [c for c in paged if c is not None], "window": None, "state": None}
    if window:
        out["window"] = comp(kv_pool, "window_views", "window_kv", "window")
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
            kv = [self._copies(self.comps["paged"], units) for _, units in plan.kv]
            win = [self._copies([self.comps["window"]], units) for _, units in plan.window]
            st = self._copies([self.comps["state"]], 1) if plan.state else []
            if None in (*kv, *win, st):
                for copies in (*kv, *win, st):
                    for c in copies or ():
                        c.release()
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
                lambda plan=plan, kv=kv, win=win, st=st: tree.finish_backup(
                    plan, kv, win, st if plan.state else None),
                keep)

    def _copies(self, comps, units) -> list | None:
        """Host copies of ``units`` units for each component, freeing host data if needed."""
        out = []
        for comp in comps:
            copy = HostCopy(self.store, comp, units)
            if copy.where is None:
                self.m.tree.evict_host(copy.nbytes, lambda n=copy.nbytes: self.store.fits(n))
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
        locs = m._page_to_token(m._allocate(tokens // ps)) if tokens else m.empty
        values, off = [], 0
        for n, _ in plan.kv:
            values.append(locs[off : off + n.length])
            off += n.length
        new = {id(n): v for (n, _), v in zip(plan.kv, values)}
        win_locs = [new.get(id(n), n.value) for n, _ in plan.window]
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
        self.transfer.submit("h2d", tasks,
                             lambda: tree.finish_restore(plan, values, slot), keep)
        return True

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
            "host_allocated_bytes": self.store.allocated,
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
