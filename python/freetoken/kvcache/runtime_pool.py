"""Shared runtime pool: one fixed set of physical GPU blocks behind stable reserved addresses.

Components keep their tensor layouts over reserved address ranges. A unit -- a KV page, a
state slot, a record row -- is usable once every block its bytes touch is mapped. Blocks
whose units are all released stay mapped for the same component and are unmapped only when
another component needs physical space, after the GPU work that last used them finished."""
from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np
import torch
from freetoken.utils import align_ceil


@lru_cache(maxsize=1)
def _vmm():
    return importlib.import_module("freetoken.kernel._vmm")


class PhysicalBlocks:
    """``budget // granularity`` physical blocks, created once and lent to regions."""

    def __init__(self, budget_bytes: int, device: torch.device, stream: torch.cuda.Stream):
        vmm, index = _vmm(), device.index
        if not vmm.supported(index):
            raise ValueError("--runtime-cache-gib needs CUDA virtual memory management, "
                             f"which GPU {index} does not support")
        self.device, self.stream = device, stream
        self.granularity = vmm.granularity(index)
        count = budget_bytes // self.granularity
        if count < 1:
            raise ValueError(f"--runtime-cache-gib is below one {self.granularity}-byte block")
        self.handles = [vmm.create(index, self.granularity) for _ in range(count)]
        self.free = list(range(count))
        self.regions: list[Region] = []
        self.releases = 0  # orders idle chunks by when they became idle
        self.map_count = self.unmap_count = 0

    @property
    def total_bytes(self) -> int:
        return len(self.handles) * self.granularity

    def region(self, name: str, nbytes: int) -> Region:
        region = Region(self, name, nbytes)
        self.regions.append(region)
        return region

    def reclaim(self, count: int, keep: dict) -> bool:
        """Unmap ``count`` idle chunks, oldest release first, sparing ``keep`` (region ->
        chunk ids); False, with nothing unmapped, when there are not that many."""
        candidates = []
        for region in self.regions:
            idle = np.flatnonzero(region.idle >= 0)
            if region in keep:
                idle = np.setdiff1d(idle, keep[region], assume_unique=True)
            candidates.extend((int(region.idle[c]), id(region), region, int(c)) for c in idle)
        if len(candidates) < count:
            return False
        for *_, region, chunk in sorted(candidates)[:count]:
            region.unmap(chunk)
        return True

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
        self.idle_event = np.empty(count, dtype=object)  # engine-stream work queued at release

    def tensor(self, shape, strides, dtype: torch.dtype, offset: int = 0) -> torch.Tensor:
        return _vmm().view(self.base + offset, list(shape), list(strides), dtype,
                           self.blocks.device.index)

    @property
    def mapped(self) -> int:
        return int((self.block >= 0).sum())

    def map(self, chunk: int) -> None:
        blocks = self.blocks
        block = blocks.free.pop()
        _vmm().map(blocks.device.index, self.base + chunk * blocks.granularity,
                   blocks.granularity, blocks.handles[block])
        self.block[chunk] = block
        blocks.map_count += 1

    def unmap(self, chunk: int) -> None:
        self.idle_event[chunk].synchronize()  # the last kernel or copy using it has finished
        blocks = self.blocks
        _vmm().unmap(self.base + chunk * blocks.granularity, blocks.granularity)
        blocks.free.append(int(self.block[chunk]))
        self.block[chunk], self.idle[chunk], self.idle_event[chunk] = -1, -1, None
        blocks.unmap_count += 1


class Units:
    """Equal units laid out in banks ``(region, offset, stride, length)``: unit ``i`` covers
    ``[offset + i*stride, offset + i*stride + length)`` of every bank."""

    def __init__(self, banks: list[tuple[Region, int, int, int]]):
        self.banks = banks
        self.blocks = banks[0][0].blocks

    def _chunks(self, units: np.ndarray):
        """Per bank: (region, the chunk of every (unit, chunk) hold, one entry per hold)."""
        g = self.blocks.granularity
        for region, offset, stride, length in self.banks:
            first = (offset + units * stride) // g
            last = (offset + units * stride + length - 1) // g
            span = (length + g - 1) // g + 1  # a range of ``length`` touches at most this many
            chunks = first[:, None] + np.arange(span)
            yield region, chunks[chunks <= last[:, None]]

    def acquire(self, units) -> bool:
        """Map every chunk the units touch and hold it; all or nothing."""
        units = np.asarray(units, dtype=np.int64)
        held = list(self._chunks(units))
        touched: dict[Region, np.ndarray] = {}
        for region, chunks in held:
            touched[region] = np.union1d(touched.get(region, chunks), chunks)
        missing = [(r, c[r.block[c] < 0]) for r, c in touched.items()]
        short = sum(len(c) for _, c in missing) - len(self.blocks.free)
        if short > 0 and not self.blocks.reclaim(short, touched):
            return False
        for region, chunks in missing:
            for chunk in chunks:
                region.map(int(chunk))
        for region, chunks in held:
            np.add.at(region.refs, chunks, 1)
            region.idle[chunks] = -1
            region.idle_event[chunks] = None
        return True

    def release(self, units) -> None:
        """Drop the units' holds. A chunk nobody holds stays mapped; another component may
        take it once the work already queued on the engine stream is done."""
        units = np.asarray(units, dtype=np.int64)
        event = None
        for region, chunks in self._chunks(units):
            np.subtract.at(region.refs, chunks, 1)
            free = np.unique(chunks[region.refs[chunks] == 0])
            if len(free):
                if event is None:
                    event = torch.cuda.Event()
                    event.record(self.blocks.stream)
                    self.blocks.releases += 1
                region.idle[free] = self.blocks.releases
                region.idle_event[free] = event
