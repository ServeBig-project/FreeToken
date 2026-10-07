"""Host copies of prefix-cache data and the copies that move them.

A *component* is one kind of cache storage described as tensor views whose first dimension is
a storage unit: a KV page, a window unit, or a state slot. Every view of a component is
indexed by the same unit, so a node's data for that component is just a set of units. On the
host a copy of ``n`` units is laid out view after view (``n`` rows of view 0, then of view 1,
...), and node splits share one copy through spans.

``PrefixTransfer`` runs resolved copies on its own streams and reports completion by event;
it owns only its staging buffers, never the cache data, and never decides what to copy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, List

import torch

_ALIGN = 512
STAGING_BYTES = 8 << 20  # per copy direction; a starting point, not a tuned value


def transfer_device_bytes(config) -> int:
    """GPU bytes the copy module takes when the host tier is on (charged to the KV budget)."""
    return 2 * STAGING_BYTES if config.prefix_cache_host_gib > 0 else 0


class HostStore:
    """Pinned host memory under one byte budget, grown in slabs and carved first-fit."""

    def __init__(self, budget: int, slab_bytes: int = 256 << 20) -> None:
        self.budget = budget
        self.slab_bytes = slab_bytes
        self.slabs: List[torch.Tensor] = []
        self.free_extents: List[List[List[int]]] = []  # per slab: sorted [offset, size]
        self.used = 0

    @property
    def allocated(self) -> int:
        return sum(s.numel() for s in self.slabs)

    def fits(self, nbytes: int) -> bool:
        nbytes = -(-nbytes // _ALIGN) * _ALIGN
        empty = sum(s.numel() for i, s in enumerate(self.slabs)
                    if s.numel() and self.free_extents[i] == [[0, s.numel()]])
        return (self.allocated - empty + nbytes <= self.budget
                or any(ext[1] >= nbytes for extents in self.free_extents for ext in extents))

    def alloc(self, nbytes: int) -> tuple[int, int] | None:
        nbytes = -(-nbytes // _ALIGN) * _ALIGN
        for i, extents in enumerate(self.free_extents):
            for ext in extents:
                if ext[1] >= nbytes:
                    off = ext[0]
                    ext[0] += nbytes
                    ext[1] -= nbytes
                    if ext[1] == 0:
                        extents.remove(ext)
                    self.used += nbytes
                    return i, off
        if self.allocated + nbytes > self.budget:
            self._release_empty_slabs()  # their budget can back a differently sized slab
        # A new slab takes what the budget has left, up to the slab size, but at least this.
        size = max(min(self.slab_bytes, self.budget - self.allocated), nbytes)
        if self.allocated + size > self.budget:
            return None
        self.slabs.append(torch.empty(size, dtype=torch.uint8, pin_memory=True))
        self.free_extents.append([[nbytes, size - nbytes]] if size > nbytes else [])
        self.used += nbytes
        return len(self.slabs) - 1, 0

    def _release_empty_slabs(self) -> None:
        for i, slab in enumerate(self.slabs):
            if slab.numel() and self.free_extents[i] == [[0, slab.numel()]]:
                self.slabs[i] = slab.new_empty(0)  # keeps slab indices of live copies stable
                self.free_extents[i] = []

    def free(self, slab: int, off: int, nbytes: int) -> None:
        nbytes = -(-nbytes // _ALIGN) * _ALIGN
        self.used -= nbytes
        extents = self.free_extents[slab]
        extents.append([off, nbytes])
        extents.sort()
        merged = [extents[0]]
        for ext in extents[1:]:
            if merged[-1][0] + merged[-1][1] == ext[0]:
                merged[-1][1] += ext[1]
            else:
                merged.append(ext)
        self.free_extents[slab] = merged


@dataclass(eq=False)
class Component:
    """``views()`` returns the current tensor views (pools re-allocate on rebuild)."""

    name: str
    storage_kind: str  # paged | window | boundary_state | composite
    views: Callable[[], List[torch.Tensor]]

    def row_bytes(self) -> List[int]:
        return [v[0].numel() * v.element_size() for v in self.views()]

    def unit_bytes(self) -> int:
        return sum(self.row_bytes())


class HostCopy:
    """``units`` units of one component on the host; spans of split nodes share it."""

    def __init__(self, store: HostStore, comp: Component, units: int) -> None:
        self.store, self.comp, self.units = store, comp, units
        self.rows = comp.row_bytes()
        self.nbytes = units * sum(self.rows)
        self.refs = 1
        self.where = store.alloc(self.nbytes)
        self.on_free: Callable[[], None] | None = None

    def view(self, family: int, start: int, count: int) -> torch.Tensor:
        slab, off = self.where
        off += self.units * sum(self.rows[:family]) + start * self.rows[family]
        return self.store.slabs[slab][off : off + count * self.rows[family]]

    def release(self) -> None:
        self.refs -= 1
        if self.refs == 0 and self.where is not None:
            self.store.free(*self.where, self.nbytes)
            if self.on_free is not None:
                self.on_free()


@dataclass(frozen=True)
class HostSpan:
    copy: HostCopy
    start: int
    count: int

    def split(self, k: int) -> tuple[HostSpan, HostSpan]:
        self.copy.refs += 1
        return HostSpan(self.copy, self.start, k), HostSpan(self.copy, self.start + k, self.count - k)

    @property
    def nbytes(self) -> int:
        return self.count * sum(self.copy.rows)


@dataclass(frozen=True)
class CopyTask:
    """Copy units ``index`` (GPU int64) of ``comp`` to/from host span ``span``."""

    comp: Component
    index: torch.Tensor
    span: HostSpan


class _Job:
    def __init__(self, tasks, done, keep) -> None:
        self.start, self.end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        self.done, self.keep = done, keep  # keep: tensors the copy reads until it finishes
        self.nbytes = sum(t.span.nbytes for t in tasks)
        self.bytes_by_comp: dict[str, int] = {}
        for t in tasks:
            self.bytes_by_comp[t.comp.name] = self.bytes_by_comp.get(t.comp.name, 0) + t.span.nbytes


class PrefixTransfer:
    """Host<->GPU copies on two side streams (restores never queue behind backups), each with
    a fixed staging buffer reused in stream order."""

    def __init__(self, device: torch.device, staging_bytes: int = STAGING_BYTES,
                 max_backup_jobs: int = 64) -> None:
        self.device = device
        self.max_backup_jobs = max_backup_jobs
        self.streams = {d: torch.cuda.Stream(device) for d in ("d2h", "h2d")}
        self.staging = {d: torch.empty(staging_bytes, dtype=torch.uint8, device=device)
                        for d in ("d2h", "h2d")}
        self.jobs = {d: deque() for d in ("d2h", "h2d")}
        self.stats = {"h2d_bytes": 0, "d2h_bytes": 0, "h2d_batches": 0, "d2h_batches": 0,
                      "h2d_time_ms": 0.0, "d2h_time_ms": 0.0}
        self.comp_bytes: dict[str, dict[str, int]] = {}

    @property
    def device_bytes(self) -> int:
        return sum(t.numel() for t in self.staging.values())

    @property
    def backup_full(self) -> bool:
        return len(self.jobs["d2h"]) >= self.max_backup_jobs

    @property
    def inflight_bytes(self) -> int:
        return sum(j.nbytes for q in self.jobs.values() for j in q)

    def submit(self, direction: str, tasks: List[CopyTask], done: Callable[[], None],
               keep=()) -> None:
        """Run ``tasks`` after all work already queued on the current stream; ``done`` runs
        from ``poll`` once they completed."""
        stream = self.streams[direction]
        stream.wait_stream(torch.cuda.current_stream(self.device))
        job = _Job(tasks, done, keep)
        with torch.cuda.stream(stream):
            job.start.record()
            for task in tasks:
                for family, view in enumerate(task.comp.views()):
                    self._copy(direction, view, family, task)
            job.end.record()
        self.jobs[direction].append(job)

    def _copy(self, direction: str, view: torch.Tensor, family: int, task: CopyTask) -> None:
        stage = self.staging[direction]
        row = view[0].numel() * view.element_size()
        per = max(1, stage.numel() // row)
        assert row <= stage.numel(), "a storage row exceeds the staging buffer"
        n = task.index.numel()
        for lo in range(0, n, per):
            k = min(per, n - lo)
            idx = task.index[lo : lo + k]
            host = task.span.copy.view(family, task.span.start + lo, k)
            buf = stage[: k * row].view(view.dtype).view(k, *view.shape[1:])
            if direction == "d2h":
                torch.index_select(view, 0, idx, out=buf)
                host.copy_(stage[: k * row], non_blocking=True)
            else:
                stage[: k * row].copy_(host, non_blocking=True)
                view.index_copy_(0, idx, buf)

    def poll(self, group=None) -> None:
        """Finish completed copies in submission order; never waits. With a TP ``group``,
        only copies every rank has completed finish, so all ranks publish the same data."""
        ready = []
        for queue in self.jobs.values():
            n = 0
            while n < len(queue) and queue[n].end.query():
                n += 1
            ready.append(n)
        if group is not None:
            counts = torch.tensor(ready, dtype=torch.int64)
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.MIN, group=group)
            ready = counts.tolist()
        for (direction, queue), n in zip(self.jobs.items(), ready):
            for _ in range(n):
                job = queue.popleft()
                self.stats[f"{direction}_bytes"] += job.nbytes
                self.stats[f"{direction}_batches"] += 1
                self.stats[f"{direction}_time_ms"] += job.start.elapsed_time(job.end)
                for name, n in job.bytes_by_comp.items():
                    per = self.comp_bytes.setdefault(name, {"h2d": 0, "d2h": 0})
                    per[direction] += n
                job.done()

    def drain(self, group=None) -> None:
        """Maintenance only: wait for this module's own copies, then finish them."""
        for stream in self.streams.values():
            stream.synchronize()
        self.poll(group)


def units_of(values: torch.Tensor, per_unit: int) -> torch.Tensor:
    """Unit index of each ``per_unit``-token unit of page-aligned ``values`` (GPU, int64)."""
    return torch.div(values[::per_unit].to(torch.int64), per_unit, rounding_mode="floor")


__all__ = ["Component", "CopyTask", "HostCopy", "HostSpan", "HostStore", "PrefixTransfer",
           "units_of"]
