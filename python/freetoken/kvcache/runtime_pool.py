"""Shared runtime pool: one fixed set of physical GPU blocks behind stable reserved addresses.

Components keep their tensor layouts over reserved address ranges. A unit -- a KV page, a
state slot, a record row -- is usable once every block its bytes touch is mapped. Blocks
whose units are all released stay mapped for the same component and are unmapped only when
another component needs physical space, after the GPU work that last used them finished."""
from __future__ import annotations

import importlib
import math
import time
from functools import lru_cache

import numpy as np
import torch
from freetoken.utils import align_ceil


@lru_cache(maxsize=1)
def _vmm():
    return importlib.import_module("freetoken.kernel._vmm")


def granularity(device: torch.device) -> int:
    """The device's physical block size (its minimum VMM allocation granularity)."""
    return _vmm().granularity(device.index)


class PhysicalBlocks:
    """``budget // granularity`` physical blocks, created once and lent to regions."""

    def __init__(self, budget_bytes: int, device: torch.device, stream: torch.cuda.Stream):
        vmm, index = _vmm(), device.index
        if not vmm.supported(index):
            raise ValueError("--runtime-cache-gib needs CUDA virtual memory management, "
                             f"which GPU {index} does not support")
        self.device, self.stream = device, stream
        self.granularity = granularity(device)
        count = budget_bytes // self.granularity
        if count < 1:
            raise ValueError(f"--runtime-cache-gib is below one {self.granularity}-byte block")
        self.handles = [vmm.create(index, self.granularity) for _ in range(count)]
        self.free = list(range(count))
        self.regions: list[Region] = []
        self.units: list[Units] = []
        self.releases = 0  # orders idle chunks by when they became idle
        self.pending_releases: dict[int, _Release] = {}  # releases not yet complete
        self.map_count = self.unmap_count = 0
        self.map_seconds = 0.0  # host time in map/unmap, including waits for the last user
        self.limits: dict = {}  # context and concurrency limits the engine derived from it

    @property
    def total_bytes(self) -> int:
        return len(self.handles) * self.granularity

    def region(self, name: str, nbytes: int) -> Region:
        region = Region(self, name, nbytes)
        self.regions.append(region)
        return region

    def acquire(self, plan) -> bool:
        """Hold ``plan``'s units (pairs of Units and unit ids) together: map what they touch,
        taking idle blocks of any component if needed; all or nothing."""
        holds = []  # (units, ids, [(region, chunk of every hold)])
        touched: dict[Region, np.ndarray] = {}
        for units, ids in plan:
            ids = np.asarray(ids, dtype=np.int64)
            pieces = [(region, chunks) for region, chunks, _ in units._chunks(ids)]
            holds.append((units, ids, pieces))
            for region, chunks in pieces:
                touched[region] = np.union1d(touched.get(region, chunks), chunks)
        missing = [(r, c[r.block[c] < 0]) for r, c in touched.items()]
        short = sum(len(c) for _, c in missing) - len(self.free)
        if short > 0 and not self.reclaim(short, touched):
            return False
        for region, chunks in missing:
            chunks = np.sort(chunks)
            for run in np.split(chunks, np.flatnonzero(np.diff(chunks) != 1) + 1):
                if len(run):
                    region.map_run(int(run[0]), len(run))
        for units, ids, pieces in holds:
            for region, chunks in pieces:
                np.add.at(region.refs, chunks, 1)
                region.idle[chunks] = -1
                region.idle_event[chunks] = None
            units.held += len(ids)
        return True

    def reclaim(self, count: int, keep: dict) -> bool:
        """Unmap ``count`` idle chunks, oldest release first, sparing ``keep`` (region ->
        chunk ids); False, with nothing unmapped, when there are not that many."""
        candidates = []
        for index, region in enumerate(self.regions):  # deterministic order on every TP rank
            idle = np.flatnonzero(region.idle >= 0)
            if region in keep:
                idle = np.setdiff1d(idle, keep[region], assume_unique=True)
            candidates.extend((int(region.idle[c]), index, int(c), region) for c in idle)
        if len(candidates) < count:
            return False
        for _, _, chunk, region in sorted(candidates, key=lambda c: c[:3])[:count]:
            region.unmap(chunk)
        return True

    def status(self) -> dict:
        """Physical bytes by state: free; held by units (``used`` of it is the units' own
        bytes, the rest block-rounding waste); idle, which another component may take, except
        what work queued before its release still protects."""
        g = self.granularity
        used = dict.fromkeys(self.regions, 0)
        for units in self.units:
            for region, _, _, length in units.banks:
                used[region] += units.held * length
        held = lambda r: int(((r.block >= 0) & (r.refs > 0)).sum()) * g
        idle = lambda r: int((r.idle >= 0).sum()) * g
        # Releases are recorded in order on one stream and complete in order: chunks idle since
        # the oldest unfinished one are still protected by the work queued before them.
        self.prune_releases()
        oldest = next(iter(self.pending_releases), None)
        protected = lambda r: 0 if oldest is None else int((r.idle >= oldest).sum()) * g
        components = {r.name: dict(held_bytes=held(r), used_bytes=used[r],
                                   waste_bytes=held(r) - used[r], idle_bytes=idle(r),
                                   protected_bytes=protected(r), address_bytes=r.size,
                                   maps=r.maps, unmaps=r.unmaps)
                      for r in self.regions}
        return dict(budget_bytes=self.total_bytes, granularity_bytes=g,
                    free_bytes=len(self.free) * g,
                    **{key: sum(c[key] for c in components.values()) for key in
                       ("held_bytes", "used_bytes", "waste_bytes", "idle_bytes",
                        "protected_bytes")},
                    components=components,
                    map_count=self.map_count, unmap_count=self.unmap_count,
                    map_ms=self.map_seconds * 1e3, **self.limits)

    def prune_releases(self) -> None:
        for release, event in list(self.pending_releases.items()):
            if not event.query():
                break
            del self.pending_releases[release]

    def held_bytes(self, tensors) -> int:
        """Physical bytes held by the regions these views live in (each region once)."""
        regions = {r for t in tensors for r in self.regions
                   if r.base <= t.data_ptr() < r.base + r.size}
        g = self.granularity
        return sum(int(((r.block >= 0) & (r.refs > 0)).sum()) * g for r in regions)

    def close(self) -> None:
        """Unmap and release everything; every tensor view must already be dropped."""
        torch.cuda.synchronize(self.device)
        vmm = _vmm()
        for region in self.regions:
            for chunk in np.flatnonzero(region.block >= 0):
                vmm.unmap(region.base + int(chunk) * self.granularity, self.granularity)
            vmm.free_range(region.base, region.size)
        for handle in self.handles:
            vmm.release(handle)
        self.regions, self.handles, self.free = [], [], []


class _Release:
    """The GPU work queued when units were released: on the engine stream and on the stream
    that released them (the scheduler's commits and copies)."""

    def __init__(self, streams):
        self.events = []
        for stream in {s.cuda_stream: s for s in streams}.values():
            event = torch.cuda.Event()
            event.record(stream)
            self.events.append(event)

    def query(self) -> bool:
        return all(event.query() for event in self.events)

    def synchronize(self) -> None:
        for event in self.events:
            event.synchronize()


class Region:
    """A reserved address range whose granularity-sized chunks are mapped on demand."""

    def __init__(self, blocks: PhysicalBlocks, name: str, nbytes: int):
        self.blocks, self.name = blocks, name
        g = blocks.granularity
        self.size = align_ceil(max(nbytes, 1), g)
        self.base = _vmm().reserve(blocks.device.index, self.size, g)
        count = self.size // g
        self.block = np.full(count, -1, dtype=np.int64)  # physical block backing each chunk
        self.refs = np.zeros(count, dtype=np.int64)      # unit holds on each chunk
        self.idle = np.full(count, -1, dtype=np.int64)   # release order of a mapped, unheld chunk
        self.idle_event = np.empty(count, dtype=object)  # the _Release that made a chunk idle
        self.init_event = np.empty(count, dtype=object)  # the zeroing of a freshly mapped chunk
        self.maps = self.unmaps = 0  # chunks mapped and unmapped here, for the status

    def tensor(self, shape, strides, dtype: torch.dtype, offset: int = 0) -> torch.Tensor:
        return _vmm().view(self.base + offset, list(shape), list(strides), dtype,
                           self.blocks.device.index)

    @property
    def mapped(self) -> int:
        return int((self.block >= 0).sum())

    def map_run(self, first: int, count: int) -> None:
        """Map ``count`` consecutive chunks from ``first``: one access grant and one zeroing
        for the run (fresh memory, not a former owner's bytes: padding rows read whatever a
        unit holds)."""
        blocks, g, vmm = self.blocks, self.blocks.granularity, _vmm()
        begin = time.perf_counter()
        for chunk in range(first, first + count):
            block = blocks.free.pop()
            vmm.map(blocks.device.index, self.base + chunk * g, g, blocks.handles[block])
            self.block[chunk] = block
        vmm.set_access(blocks.device.index, self.base + first * g, count * g)
        blocks.map_count += count
        self.maps += count
        self.tensor((count * g,), (1,), torch.uint8, first * g).zero_()
        event = torch.cuda.Event()
        event.record()
        self.init_event[first:first + count] = event
        blocks.map_seconds += time.perf_counter() - begin

    def unmap(self, chunk: int) -> None:
        blocks = self.blocks
        begin = time.perf_counter()
        # Its zeroing and the last kernel or copy using it have finished.
        self.init_event[chunk].synchronize()
        self.idle_event[chunk].synchronize()
        _vmm().unmap(self.base + chunk * blocks.granularity, blocks.granularity)
        blocks.free.append(int(self.block[chunk]))
        self.block[chunk], self.idle[chunk] = -1, -1
        self.idle_event[chunk] = self.init_event[chunk] = None
        blocks.unmap_count += 1
        self.unmaps += 1
        blocks.map_seconds += time.perf_counter() - begin


class Claim:
    """The units one operation needs, across components, before any is taken: parts of
    (Units or None, ids, commit), where ``commit(ids)`` records the ids as taken once the
    whole claim's memory is held."""

    def __init__(self):
        self.parts: list = []

    def add(self, units, ids, commit) -> None:
        self.parts.append((units, ids, commit))

    @property
    def plan(self) -> list:
        return [(units, ids) for units, ids, _ in self.parts if units is not None]

    def release(self) -> None:
        for units, ids in self.plan:
            units.release(ids)

    def commit(self) -> None:
        for _, ids, commit in self.parts:
            commit(ids)


class Units:
    """Equal units laid out in banks ``(region, offset, stride, length)``: unit ``i`` covers
    ``[offset + i*stride, offset + i*stride + length)`` of every bank."""

    def __init__(self, banks: list[tuple[Region, int, int, int]]):
        self.banks = banks
        self.blocks = banks[0][0].blocks
        self.blocks.units.append(self)
        self.held = 0  # unit holds, for the status's used bytes
        g = self.blocks.granularity
        # The most chunks one unit can touch (a range may straddle one more chunk boundary).
        self.per_unit = sum((length + g - 1) // g + 1 for *_, length in banks)

    def _chunks(self, units: np.ndarray):
        """Per bank: (region, the chunk of every (unit, chunk) hold, the unit's position)."""
        g = self.blocks.granularity
        for region, offset, stride, length in self.banks:
            start = offset + units * stride
            first, last = start // g, (start + length - 1) // g
            chunks = first[:, None] + np.arange((length + g - 1) // g + 1)
            keep = chunks <= last[:, None]
            yield region, chunks[keep], np.nonzero(keep)[0]

    def pin(self, units) -> None:
        """Acquire what must exist for the pools to work at all (sentinels, capture scratch)."""
        if not self.acquire(units):
            raise RuntimeError("shared runtime cannot map its fixed sentinel/scratch blocks")

    def acquire(self, units) -> bool:
        """Map every chunk the units touch and hold it; all or nothing."""
        return self.blocks.acquire([(self, units)])

    def release(self, units) -> None:
        """Drop the units' holds. A chunk nobody holds stays mapped; another component may
        take it once the work already queued on the engine stream is done."""
        units = np.asarray(units, dtype=np.int64)
        self.held -= len(units)
        release = None
        for region, chunks, _ in self._chunks(units):
            np.subtract.at(region.refs, chunks, 1)
            free = np.unique(chunks[region.refs[chunks] == 0])
            if len(free):
                if release is None:
                    release = _Release([self.blocks.stream, torch.cuda.current_stream()])
                    self.blocks.releases += 1
                    self.blocks.prune_releases()
                    self.blocks.pending_releases[self.blocks.releases] = release
                region.idle[free] = self.blocks.releases
                region.idle_event[free] = release


class RuntimeLayout:
    """What the pools build on the runtime, as (banks, bytes of one unit in each) per unit kind
    -- pages, windows, states, rows -- and how many blocks a count of each touches, units packed
    from the start of every bank. Nothing is allocated."""

    def __init__(self, granularity: int, **kinds: list[tuple[int, int]]):
        self.granularity, self.kinds = granularity, kinds

    def unit_bytes(self, kind: str) -> int:
        return sum(banks * unit for banks, unit in self.kinds[kind])

    def blocks(self, **counts: int) -> int:
        g = self.granularity
        return sum(banks * -(-counts[kind] * unit // g)
                   for kind, parts in self.kinds.items() for banks, unit in parts)


def upload(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host indices on the device without making the host wait for the copy."""
    return values.pin_memory().to(device, non_blocking=True)


def _contiguous(shape) -> list[int]:
    return [math.prod(shape[i + 1:]) for i in range(len(shape))]


def banked(blocks: PhysicalBlocks, name: str, shape, dtype: torch.dtype):
    """``[kv, layers, units, *unit]`` over one region with every (kv, layer) bank starting on
    a block, so a unit maps the same chunk of every bank. Returns the view and its banks."""
    kv, layers, units, *unit = shape
    unit_bytes = math.prod(unit) * dtype.itemsize
    bank = align_ceil(units * unit_bytes, blocks.granularity)
    region = blocks.region(name, kv * layers * bank)
    elems = bank // dtype.itemsize
    view = region.tensor(shape, [layers * elems, elems, *_contiguous([units, *unit])], dtype)
    return view, [(region, i * bank, unit_bytes, unit_bytes) for i in range(kv * layers)]


def slot_rows(blocks: PhysicalBlocks, name: str, shape, dtype: torch.dtype):
    """``[layers, slots, *slot]`` view whose storage keeps each slot's layers adjacent (one
    address range per slot). Returns the view and its bank."""
    layers, slots, *rest = shape
    per_layer = math.prod(rest)
    row = layers * per_layer
    region = blocks.region(name, slots * row * dtype.itemsize)
    view = region.tensor(shape, [per_layer, row, *_contiguous(rest)], dtype)
    return view, (region, 0, row * dtype.itemsize, row * dtype.itemsize)
