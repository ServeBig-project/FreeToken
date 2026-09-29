from __future__ import annotations

import torch

from freetoken.kernel.triton.gdn_replay import gdn_replay_fold


def replay_shapes(n_layers, conv_dim, v_heads, k_heads, key_dim, value_dim, kernel, dtype,
                  rows, ring, draft_steps) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Shape and dtype of every ReplaySSM buffer; allocation and byte budgets both use it."""
    shapes = {
        "u": ((n_layers, rows, v_heads, ring, value_dim), dtype),
        "k": ((n_layers, rows, k_heads, ring, key_dim), dtype),
        "g": ((n_layers, rows, v_heads, ring), torch.float32),
    }
    if draft_steps:
        # Raw conv inputs by absolute position: the target's kernel-1 inputs before a round
        # plus the round's draft/verify inputs.
        shapes["window"] = ((n_layers, rows, kernel - 1 + draft_steps + 1, conv_dim), dtype)
    return shapes


def _device(rows, device) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)


class GdnReplay:
    """Update records of each active request's GDN inputs after its checkpoint.

    A request's record row is its ``table_idx``; its full-state slot is the checkpoint at
    position ``start[row]``, and its records ``[start, cached_len)`` complete the target state.
    Draft and verify also write records past ``cached_len``; they become target history only
    as ``cached_len`` advances over them. The host decides every fold, so no GPU value is read
    back to schedule one.
    """

    def __init__(self, pool, shapes, device) -> None:
        self.pool = pool
        buffers = {name: torch.empty(shape, dtype=dtype, device=device)
                   for name, (shape, dtype) in shapes.items()}
        self.u, self.k, self.g = buffers["u"], buffers["k"], buffers["g"]
        self.window = buffers.get("window")
        self.rows, self.ring = self.u.shape[1], self.u.shape[3]
        self.start = [0] * self.rows
        # (first, last) positions whose conv state the window still holds after a commit.
        self.window_span: list[tuple[int, int] | None] = [None] * self.rows
        # Logical positions per role, request-level folds and their GPU time, for /v1/stats.
        self.counts = dict.fromkeys(("ar_tokens", "draft_tokens", "verify_tokens", "flushes",
                                     "flushed_records", "snapshot_exports"), 0)
        self.counts.update(flush_gpu_ms=0.0, export_gpu_ms=0.0)
        self._timed: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def snapshot(self) -> dict:
        """Counters, adding the GPU time of every fold that has already completed."""
        pending = []
        for key, start, end in self._timed:
            if end.query():
                self.counts[key] += start.elapsed_time(end)
            else:
                pending.append((key, start, end))
        self._timed = pending
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

    def cursors(self, reqs) -> list[tuple[int, int, int]]:
        """Kernel cursors (row, checkpoint position, first input position); padding rows -1."""
        return [(r.table_idx, self.start[r.table_idx], r.cached_len) if r.table_idx < self.rows
                else (-1, 0, 0) for r in reqs]

    def begin_prefill(self, reqs) -> None:
        # Prefill leaves the complete state in the slot: no records until the next decode.
        for req in reqs:
            self.start[req.table_idx] = req.device_len
            self.window_span[req.table_idx] = None

    def reserve(self, reqs, widths, keep=None) -> None:
        """Fold confirmed records into checkpoints so each request can append ``width``
        more; stop at ``keep(req)`` when that position still has a snapshot to export."""
        plan = []
        for req, width in zip(reqs, widths, strict=True):
            row, pos = req.table_idx, req.cached_len
            start = self.start[row]
            if pos + width - start <= self.ring:
                continue
            end = keep(req) if keep is not None else None
            if end is None or not (start <= end and pos + width - end <= self.ring):
                end = pos
            slot = self._slot(req)
            plan.append((slot, slot, row, start, end - start))
            self.start[row] = end
        self._fold(plan)

    def _fold(self, plan) -> None:
        if not plan:
            return
        pool = self.pool
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        gdn_replay_fold(pool.recurrent_states, self.u, self.k, self.g, _device(plan, pool.device))
        end.record()
        # A plan either folds records in place or exports one state to another slot.
        self._timed.append(("flush_gpu_ms" if plan[0][0] == plan[0][1] else "export_gpu_ms",
                            start, end))
        for src, dst, _, _, count in plan:
            if src == dst:
                self.counts["flushes"] += 1
                self.counts["flushed_records"] += count
            else:
                self.counts["snapshot_exports"] += 1

    def can_export(self, req, position: int) -> bool:
        row = req.table_idx
        if not self.start[row] <= position <= req.cached_len:
            return False
        span = self.window_span[row]
        return position == req.cached_len or (span is not None and span[0] <= position <= span[1])

    def export(self, req, position: int, dst: int) -> None:
        """Write the complete state after ``position`` inputs (see ``can_export``) into ``dst``."""
        row, slot = req.table_idx, self._slot(req)
        start = self.start[row]
        self._fold([(slot, dst, row, start, position - start)])
        conv = self.pool.conv_states
        if position == req.cached_len:
            conv[:, dst].copy_(conv[:, slot])
        else:
            conv[:, dst] = self.window[:, row, self._window_cols(position)].transpose(-1, -2)

    def materialize(self, req) -> None:
        """Fold every record into the request's own slot (it is about to be donated)."""
        row, slot = req.table_idx, self._slot(req)
        start = self.start[row]
        if req.cached_len > start:
            self._fold([(slot, slot, row, start, req.cached_len - start)])
            self.start[row] = req.cached_len

    def _window_cols(self, position: int) -> list[int]:
        # Conv state at ``position``: the raw inputs of the kernel-1 positions before it.
        width, km1 = self.window.shape[2], self.pool.conv_states.shape[-1]
        return [(position - km1 + j) % width for j in range(km1)]

    def begin_round(self, reqs, lengths) -> _ReplayRound:
        """Make room for each request's verify window and seed the conv window with the
        target conv state; draft and verify then leave the target state untouched."""
        self.reserve(reqs, [length + 1 for length in lengths])
        rows = [req.table_idx for req in reqs]
        slots = [self._slot(req) for req in reqs]
        firsts = [req.cached_len for req in reqs]
        device = self.pool.device
        cols = _device([self._window_cols(p) for p in firsts], device)
        conv = self.pool.conv_states
        self.window[:, _device(rows, device)[:, None], cols] = (
            conv[:, _device(slots, device)].transpose(-1, -2))
        for row in rows:
            self.window_span[row] = None
        return _ReplayRound(self, rows, slots, firsts)


class _ReplayRound:
    """One SD round: verify already rewrote the records; commit only picks the conv state."""

    def __init__(self, replay: GdnReplay, rows, slots, firsts) -> None:
        self.replay, self.rows, self.slots, self.firsts = replay, rows, slots, firsts

    def prepare_verify(self, batch, lengths) -> None:
        pass  # verify reads the same rows through the batch's own cursors

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
