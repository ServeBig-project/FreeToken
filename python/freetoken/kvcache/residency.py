"""Host-backed page payloads; logical pages and their index remain in the shared runtime."""
from __future__ import annotations

import numpy as np
import torch

from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
from .prefix_store import CopyTask, HostSeries, HostSpan
from .runtime_pool import Units, _Release, slot_rows, upload

KV_PLACEMENTS = ("gpu", "tiered")


class HostKV:
    """Own the host references of live logical pages and only release payload under pressure."""

    def __init__(self, pool, runtime, banks):
        self.pool = pool
        self.backups: dict[int, HostSpan] = {}
        self.pending = []
        self.read_bytes = torch.zeros(1, dtype=torch.int64, device=runtime.device)
        self._read_host = alloc_pinned_tensor(1, dtype=torch.int64)
        self._read_stream = torch.cuda.Stream(device=runtime.device)
        self._read_event = torch.cuda.Event()
        self._read_pending = False
        self._last_read_bytes = 0
        self.rebuild(runtime, banks)

    def rebuild(self, runtime, banks) -> None:
        for span in self.backups.values():
            span.copy.release()
        for event, copies in self.pending:
            event.synchronize()
            for copy in copies:
                copy.release()
        self.backups, self.pending = {}, []
        self.runtime = runtime
        self.payload = Units(banks)
        self.payload.pin([0])
        self.gpu = {0}
        self.host_pages = 0
        self.immutable = set()
        self.families = len(self.pool.payload_views())
        self.addresses, bank = slot_rows(runtime, "kv_host_addresses",
            (self.families + 1, self.pool.payload_views()[0].shape[0]), torch.int64)
        self.banks = [bank]

    @property
    def page_bytes(self) -> int:
        """K/V payload and scale bytes of one page, on either side."""
        return sum(length for *_, length in self.payload.banks)

    @property
    def gpu_flags(self):
        return self.addresses[-1]

    def initialize_dummy(self) -> None:
        self.gpu_flags[0] = 1

    def allocated(self, ids, writable) -> None:
        self.gpu.update(int(p) for p in writable)
        idx = upload(torch.as_tensor(np.asarray(ids), dtype=torch.int64), self.runtime.device)
        self.gpu_flags.index_fill_(0, idx, 0)
        if len(writable):
            idx = upload(torch.as_tensor(np.asarray(writable), dtype=torch.int64), self.runtime.device)
            self.gpu_flags.index_fill_(0, idx, 1)

    def missing_bytes(self, comp, indices) -> int:
        return sum(int(p) not in self.backups for p in indices) * comp.unit_bytes()

    def backup(self, comp, indices, allocate):
        """A private/prefix snapshot references the same host pages as active attention."""
        pages = [int(p) for p in indices]
        missing = list(dict.fromkeys(p for p in pages if p not in self.backups))
        if missing:
            copy = allocate(len(missing))
            if copy is None:
                return None
            self._bind(missing, [HostSpan(copy, i, 1) for i in range(len(missing))])
            copy.release()
        parts = []
        for page in pages:
            span = self.backups[page]
            if parts and parts[-1].copy is span.copy and parts[-1].start + parts[-1].count == span.start:
                prev = parts.pop()
                parts.append(HostSpan(prev.copy, prev.start, prev.count + 1))
            else:
                parts.append(span)
        return HostSeries(comp, parts)

    def _bind(self, pages, spans) -> None:
        if not pages:
            return
        addresses = []
        for page, span in zip(pages, spans, strict=True):
            previous = self.backups.get(page)
            if previous is not None:
                previous.copy.release()
            elif page not in self.gpu:
                self.host_pages += 1
            span.copy.refs += 1
            self.backups[page] = span
            copy = span.copy
            base = device_ptr(copy.store.buf) + copy.where
            offset, pointers = 0, []
            for row in copy.rows:
                pointers.append(base + offset + span.start * row)
                offset += copy.units * row
            addresses.append(pointers)
        idx = upload(torch.tensor(pages, dtype=torch.int64), self.runtime.device)
        values = upload(torch.tensor(addresses, dtype=torch.int64).T.contiguous(), self.runtime.device)
        self.addresses[:self.families].index_copy_(1, idx, values)

    def restore(self, task: CopyTask) -> list[CopyTask]:
        """Install host history at new logical pages; copy only writable tails back to GPU."""
        pages = [int(p) for p in task.index]
        spans = [HostSpan(task.span.copy, task.span.start + i, 1) for i in range(len(pages))]
        self._bind(pages, spans)
        out = []
        for i, page in enumerate(pages):
            if page in self.gpu:
                out.append(CopyTask(task.comp, task.index[i:i + 1], spans[i]))
            else:
                self.immutable.add(page)
        return out

    def release(self, ids) -> None:
        pages = [int(p) for p in ids]
        copying = {self.backups[p].copy for p in pages if p in self.backups
                   and self.backups[p].copy.queued and not self.backups[p].copy.ready}
        for copy in copying:
            # A returned page id can be reused without unmapping its block. Order those
            # future writes after the backup, as well as protecting physical reclamation.
            self.runtime.stream.wait_event(copy.event)
            torch.cuda.current_stream(self.runtime.device).wait_event(copy.event)
        resident = [p for p in pages if p in self.gpu]
        self.host_pages -= sum(p in self.backups and p not in self.gpu for p in pages)
        if resident:
            self.payload.release(resident)
        self.gpu.difference_update(pages)
        self.immutable.difference_update(pages)
        copies = [self.backups.pop(p).copy for p in pages if p in self.backups]
        if copies:
            # The address table may still be read by a previously launched gather.
            event = _Release([self.runtime.stream, torch.cuda.current_stream(self.runtime.device)])
            self.pending.append((event, copies))

    def finish_restore(self, indices) -> None:
        # A restored writable tail is about to change; its old host prefix is not a backup
        # of the completed page it will eventually become.
        for page in indices.cpu().tolist():
            self.backups.pop(page).copy.release()
            self.immutable.discard(page)

    def evict_one(self) -> bool:
        for page, span in self.backups.items():
            if page in self.gpu and page in self.immutable and span.copy.queued:
                if not span.copy.ready:
                    # A completing backup is reusable capacity, not an impossible context.
                    span.copy.event.synchronize()
                    span.copy.ready = True
                # A launched gather and its attention must observe the same residency.
                torch.cuda.current_stream(self.runtime.device).wait_stream(self.runtime.stream)
                self.payload.release([page])
                self.gpu.remove(page)
                self.host_pages += 1
                self.gpu_flags[page] = 0
                return True
        return False

    def discard_unqueued(self) -> None:
        for page in list(self.backups):
            if not self.backups[page].copy.queued:
                self.backups.pop(page).copy.release()
                self.immutable.discard(page)

    def poll(self, *, wait=False) -> None:
        while self.pending:
            if wait:
                self.pending[0][0].synchronize()
            elif not self.pending[0][0].query():
                break
            _, copies = self.pending.pop(0)
            for copy in copies:
                copy.release()

    def status(self) -> dict:
        if self._read_pending and self._read_event.query():
            self._last_read_bytes = int(self._read_host[0])
            self._read_pending = False
        if not self._read_pending:
            # Observe on a side stream without making the scheduler wait for attention.
            self._read_stream.wait_stream(self.runtime.stream)
            with torch.cuda.stream(self._read_stream):
                self._read_host.copy_(self.read_bytes, non_blocking=True)
                self._read_event.record()
            self._read_pending = True
        return dict(kv_gpu_payload_pages=len(self.gpu) - 1, kv_host_payload_pages=self.host_pages,
                    kv_host_payload_bytes=self.host_pages * self.page_bytes,
                    kv_host_read_bytes=self._last_read_bytes)
