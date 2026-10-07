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
from freetoken.kernel.pinned import alloc_pinned_tensor

_ALIGN = 512
STAGING_BYTES = 8 << 20  # per copy direction; a starting point, not a tuned value


def transfer_device_bytes(config) -> int:
    """GPU bytes the copy module takes when the host tier is on (charged to the KV budget)."""
    return 2 * STAGING_BYTES if config.prefix_cache_host_gib > 0 else 0


class HostStore:
    """Pinned host memory of one fixed budget, carved first-fit. It is pinned whole at startup:
    pinning on demand stalls the scheduler for hundreds of milliseconds per block."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.buf = alloc_pinned_tensor(budget, dtype=torch.uint8)
        self.free_extents: List[List[int]] = [[0, budget]]  # sorted [offset, size]
        self.used = 0

    def fits(self, nbytes: int) -> bool:
        nbytes = _aligned(nbytes)
        return any(ext[1] >= nbytes for ext in self.free_extents)

    def alloc(self, nbytes: int) -> int | None:
        nbytes = _aligned(nbytes)
        for ext in self.free_extents:
            if ext[1] >= nbytes:
                off = ext[0]
                ext[0] += nbytes
                ext[1] -= nbytes
                if ext[1] == 0:
                    self.free_extents.remove(ext)
                self.used += nbytes
                return off
        return None

    def free(self, off: int, nbytes: int) -> None:
        nbytes = _aligned(nbytes)
        self.used -= nbytes
        extents = sorted(self.free_extents + [[off, nbytes]])
        merged = [extents[0]]
        for ext in extents[1:]:
            if merged[-1][0] + merged[-1][1] == ext[0]:
                merged[-1][1] += ext[1]
            else:
                merged.append(ext)
        self.free_extents = merged


def _aligned(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


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
        off = self.where + self.units * sum(self.rows[:family]) + start * self.rows[family]
        return self.store.buf[off : off + count * self.rows[family]]

    def release(self) -> None:
        self.refs -= 1
        if self.refs == 0 and self.where is not None:
            self.store.free(self.where, self.nbytes)
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
        tasks = _merge_adjacent(tasks)  # before the wait: the side stream reads merged indices
        stream = self.streams[direction]
        stream.wait_stream(torch.cuda.current_stream(self.device))
        job = _Job(tasks, done, [*keep, *(t.index for t in tasks)])
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


def _merge_adjacent(tasks: List[CopyTask]) -> List[CopyTask]:
    """One task per run of consecutive units of one host copy (the pieces of a split node, in
    whatever order they were listed), so a fragmented path costs one copy per layer instead of
    one per piece and layer. Destinations are disjoint, so task order does not matter."""
    group: dict = {}
    for t in tasks:
        group.setdefault((t.comp, t.span.copy), len(group))
    out, runs = [], []
    for t in sorted(tasks, key=lambda t: (group[t.comp, t.span.copy], t.span.start)):
        p = out[-1] if out else None
        if (p is not None and p.comp is t.comp and p.span.copy is t.span.copy
                and p.span.start + p.span.count == t.span.start):
            out[-1] = CopyTask(t.comp, p.index, HostSpan(p.span.copy, p.span.start,
                                                         p.span.count + t.span.count))
            runs[-1].append(t.index)
        else:
            out.append(t)
            runs.append([t.index])
    return [CopyTask(t.comp, torch.cat(r) if len(r) > 1 else t.index, t.span)
            for t, r in zip(out, runs)]


def units_of(values: torch.Tensor, per_unit: int) -> torch.Tensor:
    """Unit index of each ``per_unit``-token unit of page-aligned ``values`` (GPU, int64)."""
    return torch.div(values[::per_unit].to(torch.int64), per_unit, rounding_mode="floor")


__all__ = ["Component", "CopyTask", "HostCopy", "HostSpan", "HostStore", "PrefixTransfer",
           "units_of"]
