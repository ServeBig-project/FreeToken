from __future__ import annotations

import torch

from freetoken.kernel.triton.gdn_replay import gdn_replay_advance, gdn_replay_fold


def replay_shapes(n_layers, conv_dim, v_heads, k_heads, key_dim, value_dim, kernel, dtype,
                  rows, ring, draft_steps, graph_batch) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Shape and dtype of every ReplaySSM buffer; allocation and byte budgets both use it."""
    shapes = {
        "u": ((n_layers, rows, v_heads, ring, value_dim), dtype),
        "k": ((n_layers, rows, k_heads, ring, key_dim), dtype),
        "g": ((n_layers, rows, v_heads, ring), torch.float32),
        "start": ((rows,), torch.int32),   # checkpoint position per record row
        "stats": ((2,), torch.int64),      # folds, folded records
    }
    if draft_steps:
        # Raw conv inputs by absolute position: the target's kernel-1 inputs before a round
        # plus the round's draft/verify inputs.
        shapes["window"] = ((n_layers, rows, kernel - 1 + draft_steps + 1, conv_dim), dtype)
    if graph_batch:
        # Fixed per-sequence rows, slots and offsets that captured graphs read; decode, draft
        # and verify replays run in stream order, each staging its own values right before.
        shapes["graph_rows"] = ((graph_batch,), torch.int32)
        shapes["graph_slots"] = ((graph_batch,), torch.int32)
        shapes["graph_cu"] = ((graph_batch + 1,), torch.int32)
    return shapes


def _device(values, device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)


class GdnReplay:
    """Update records of each active request's GDN inputs after its checkpoint.

    A request's record row is its ``table_idx``; its full-state slot is the checkpoint at
    position ``start[row]`` (GPU), and its records ``[start, cached_len)`` complete the target
    state. Draft and verify also write records past ``cached_len``; they become target history
    only as ``cached_len`` advances over them. Folds are decided on the GPU from ``start``:
    target decode folds inside its kernel and advances ``start`` after the last GDN layer, an
    SD round folds before drafting. The host only resets ``start`` at prefill.
    """

    def __init__(self, pool, shapes, device) -> None:
        self.pool = pool
        buffers = {name: torch.zeros(shape, dtype=dtype, device=device)
                   for name, (shape, dtype) in shapes.items()}
        self.u, self.k, self.g = buffers["u"], buffers["k"], buffers["g"]
        self.start, self.stats = buffers["start"], buffers["stats"]
        self.window = buffers.get("window")
        self.graph_rows = buffers.get("graph_rows")
        self.graph_slots = buffers.get("graph_slots")
        self.graph_cu = buffers.get("graph_cu")
        self.rows, self.ring = self.u.shape[1], self.u.shape[3]
        # (first, last) positions whose conv state the window holds after a round's commit.
        self.window_span: list[tuple[int, int] | None] = [None] * self.rows
        # Logical positions per role, folds, exports and fold-kernel GPU time, for /v1/stats.
        self.counts = dict.fromkeys(("ar_tokens", "draft_tokens", "verify_tokens", "flushes",
                                     "flushed_records", "snapshot_exports"), 0)
        self.counts.update(flush_gpu_ms=0.0, export_gpu_ms=0.0)
        self._timed: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._stats_host = torch.zeros(2, dtype=torch.int64, pin_memory=True)
        self._stats_copied: torch.cuda.Event | None = None
        # Compile the fold and round-advance kernels now, not inside the first round or
        # donation: a zero-record export of the padding slot onto itself, an empty advance.
        pad = pool.padding_slot
        gdn_replay_fold(pool.recurrent_states, self.u, self.k, self.g, self.start,
                        _device([(pad, pad, 0, 0, 0)], device))
        gdn_replay_advance(self.start, self.stats, *(_device([v], device) for v in (-1, 0, 1)),
                           self.ring)

    def snapshot(self) -> dict:
        """Counters as of the previous snapshot's copy of the GPU fold counters (lagging by
        one report so reading never waits), plus the GPU time of completed fold kernels."""
        pending = []
        for key, start, end in self._timed:
            if end.query():
                self.counts[key] += start.elapsed_time(end)
            else:
                pending.append((key, start, end))
        self._timed = pending
        if self._stats_copied is None or self._stats_copied.query():
            if self._stats_copied is not None:
                self.counts["flushes"], self.counts["flushed_records"] = self._stats_host.tolist()
            self._stats_host.copy_(self.stats, non_blocking=True)
            self._stats_copied = torch.cuda.Event()
            self._stats_copied.record()
        return dict(self.counts)

    def observe(self, batch) -> None:
        """Count the real positions a forward runs through the records."""
        if batch.is_speculative_verify:
            self.counts["verify_tokens"] += sum(req.extend_len for req in batch.reqs)
        elif batch.draft_experts is not None:
            self.counts["draft_tokens"] += batch.size
        else:
            self.counts["ar_tokens"] += batch.decode_size

    @staticmethod
    def _slot(req) -> int:
        return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx

    def record_rows(self, reqs) -> list[int]:
        """Record row per request; padding rows (the graph dummy) are -1."""
        return [r.table_idx if r.table_idx < self.rows else -1 for r in reqs]

    def begin_prefill(self, reqs) -> None:
        # Prefill leaves the complete state in the slot: no records until the next decode.
        if not reqs:
            return
        device = self.pool.device
        rows = [req.table_idx for req in reqs]
        self.start[_device(rows, device)] = _device([req.device_len for req in reqs], device)
        for row in rows:
            self.window_span[row] = None

    def _fold(self, plan, key: str) -> None:
        pool = self.pool
        planned = _device(plan, pool.device)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        gdn_replay_fold(pool.recurrent_states, self.u, self.k, self.g, self.start, planned)
        end.record()
        self._timed.append((key, start, end))

    def can_export(self, req, position: int) -> bool:
        """The complete state after ``position`` inputs is reachable now: the current state,
        or a position inside the round committed since the last forward (nothing has folded
        past it yet)."""
        span = self.window_span[req.table_idx]
        return position == req.cached_len or (
            span is not None and span[1] == req.cached_len and span[0] <= position <= span[1])

    def export(self, req, position: int, dst: int) -> None:
        """Write the complete state after ``position`` inputs (see ``can_export``) into ``dst``."""
        row, slot = req.table_idx, self._slot(req)
        self._fold([(slot, dst, row, position, 0)], "export_gpu_ms")
        self.counts["snapshot_exports"] += 1
        conv = self.pool.conv_states
        if position == req.cached_len:
            conv[:, dst].copy_(conv[:, slot])
        else:
            conv[:, dst] = self.window[:, row, self._window_cols(position)].transpose(-1, -2)

    def materialize(self, req) -> None:
        """Make the request's own slot its complete current state (it is about to be donated)."""
        slot = self._slot(req)
        self._fold([(slot, slot, req.table_idx, req.cached_len, 0)], "export_gpu_ms")
        self.counts["snapshot_exports"] += 1

    def _window_cols(self, position: int) -> list[int]:
        # Conv state at ``position``: the raw inputs of the kernel-1 positions before it.
        width, km1 = self.window.shape[2], self.pool.conv_states.shape[-1]
        return [(position - km1 + j) % width for j in range(km1)]

    def begin_round(self, reqs, lengths) -> _ReplayRound:
        """Fold wherever a request's verify window would not fit, then seed the conv window
        with the target conv state; draft and verify leave the target state untouched."""
        device = self.pool.device
        rows = [req.table_idx for req in reqs]
        slots = [self._slot(req) for req in reqs]
        firsts = [req.cached_len for req in reqs]
        widths = [length + 1 for length in lengths]
        self._fold(list(zip(slots, slots, rows, firsts, widths)), "flush_gpu_ms")
        rows_t, firsts_t = _device(rows, device), _device(firsts, device)
        gdn_replay_advance(self.start, self.stats, rows_t, firsts_t, _device(widths, device),
                           self.ring)
        cols = _device([self._window_cols(p) for p in firsts], device)
        conv = self.pool.conv_states
        self.window[:, rows_t[:, None], cols] = conv[:, _device(slots, device)].transpose(-1, -2)
        for row in rows:
            self.window_span[row] = None
        return _ReplayRound(self, rows, slots, firsts)


class _ReplayRound:
    """One SD round: verify already rewrote the records; commit only picks the conv state."""

    def __init__(self, replay: GdnReplay, rows, slots, firsts) -> None:
        self.replay, self.rows, self.slots, self.firsts = replay, rows, slots, firsts

    def prepare_verify(self, batch, lengths) -> None:
        pass  # verify reads the same rows through the batch's own metadata

    def commit(self, retained) -> None:
        replay = self.replay
        kept = [(row, slot, first, first + n) for row, slot, first, n in
                zip(self.rows, self.slots, self.firsts, retained, strict=True) if n]
        if not kept:
            return
        rows, slots, firsts, ends = zip(*kept)
        device = replay.pool.device
        cols = _device([replay._window_cols(end) for end in ends], device)
        replay.pool.conv_states[:, _device(slots, device)] = (
            replay.window[:, _device(rows, device)[:, None], cols].transpose(-1, -2))
        for row, first, end in zip(rows, firsts, ends):
            replay.window_span[row] = (first, end)
