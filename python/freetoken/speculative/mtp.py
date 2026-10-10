"""Native MTP drafting: the checkpoint's MTP layer drafts from the target's last real streams
and keeps its own attention history in the target's pages.

MTP row ``t`` pairs the target's final streams ``R[t]`` with token ``t+1`` at logical position
``t``. Its K/V and index keys are stored one slot ahead, in target slot ``t+1``, so a page only
holds rows that depend on its own tokens. Once a request has consumed ``c`` tokens, rows
``[0, c-1)`` are history and ``R[c-1]`` waits in the request's tail state. Prefill and AR
forwards add their rows right after they run; a round drafts from ``(R[c-1], x[c])`` and its
commit adds the rows the target retained, from the verify forward's real streams.
"""

from __future__ import annotations

from copy import copy

import torch

from freetoken.core import Batch, get_global_ctx

from . import DraftResult

# Rows per history write: its fusion and projection transients stay bounded on long prompts.
_HISTORY_ROWS = 1024


def _upload(values, device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)


def _slot(req) -> int:
    return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx


class MTPRuntime:
    """The engine's MTP layer and the history it writes into the target's pages."""

    def __init__(self, engine, mtp) -> None:
        self.engine, self.mtp = engine, mtp
        self.history_rows = 0
        weights = mtp.state_dict().values()
        self.weight_bytes = sum(t.numel() * t.element_size() for t in weights)
        # What the weights actually are, reported apart from the target's formats.
        self.weight_dtype = ",".join(sorted({str(t.dtype).removeprefix("torch.") for t in weights}))
        self.weight_residency = ",".join(sorted({t.device.type for t in weights}))

    def _tail(self) -> torch.Tensor:
        return get_global_ctx().linear_state_pool.slot_state(self.mtp.tail_state)

    def metadata(self, reqs, firsts, ends):
        """QSA metadata of MTP rows ``[first, end)`` per request, stored one slot ahead, and
        their logical positions and storage locations."""
        backend, device = self.engine.attn_backend, self.engine.device
        views = []
        for req, first, end in zip(reqs, firsts, ends, strict=True):
            view = copy(req)
            view.cached_len, view.device_len = first, end
            views.append(view)
        batch = Batch(views)
        batch.padded_reqs = views
        backend.prepare_metadata(batch)
        md = batch.attn_metadata
        md.shift = 1
        positions = _upload([p for f, e in zip(firsts, ends) for p in range(f, e)], device)
        rows = _upload([r.table_idx for r, f, e in zip(reqs, firsts, ends) for _ in range(f, e)],
                       device)
        out_loc = self.engine.page_table[rows, positions + 1]
        positions = positions.to(torch.int32)
        backend.plan_writes(md, out_loc, positions, _upload([_slot(r) for r in reqs], device).int())
        return md, positions, out_loc

    def advance(self, batch: Batch, features: torch.Tensor) -> None:
        """After a target prefill or AR forward of ``batch`` (its final streams ``features``):
        the rows it made history. A prefill's state captures inside the forward (page-aligned
        prefix snapshots) get the MTP state there too."""
        reqs = batch.reqs
        captures = [[(c.pos, c.slot) for c in r.state_captures if c.slot is not None
                     and c.pos is not None and r.cached_len < c.pos < r.device_len]
                    for r in reqs]
        self.extend(reqs, [r.cached_len for r in reqs], [r.extend_len for r in reqs],
                    features, batch.input_ids, captures)
        batch.draft_features = None

    def extend(self, reqs, starts, counts, streams, input_ids, captures=None) -> None:
        """Forward rows ``[start, start+count)`` of each request (``streams`` and ``input_ids``
        list every request's forward rows in order, ``count`` of them used) became target
        history: write MTP rows ``[start-1, start+count-1)``, then keep the last row's streams
        as the request's tail. ``captures`` per request: (position, slot) snapshots to fill."""
        tail = self._tail()
        captures = captures or [[] for _ in reqs]
        pieces, sources, slots, offset = [], [], [], 0
        for req, start, count, width, caps in zip(reqs, starts, counts,
                                                  (r.extend_len for r in reqs), captures,
                                                  strict=True):
            if count:
                pieces.append((req, start, count, offset, caps))
                sources.append(offset + count - 1)
                slots.append(_slot(req))
                for position, slot in caps:
                    sources.append(offset + position - 1 - start)
                    slots.append(slot)
            offset += width
        # Writes of at most _HISTORY_ROWS rows; a long prefill spans several.
        rows, budget = [], _HISTORY_ROWS
        for req, start, count, offset, caps in pieces:
            first = max(start - 1, 0)
            while first < start + count - 1:
                end = min(start + count - 1, first + budget)
                rows.append((req, first, end, start, offset, caps))
                budget -= end - first
                first = end
                if budget == 0:
                    self._write(rows, streams, input_ids, tail)
                    rows, budget = [], _HISTORY_ROWS
        if rows:
            self._write(rows, streams, input_ids, tail)
        if slots:
            device = streams.device
            tail[_upload(slots, device)] = streams[_upload(sources, device)]

    def _write(self, rows, streams, input_ids, tail) -> None:
        """MTP rows ``t`` in ``[first, end)``: streams ``R[t]`` (the tail state for ``t =
        start-1``, else forward row ``offset+t-start``) and token ``t+1``. A capture at
        position ``p`` keeps the open group after row ``p-2``."""
        sources, tokens, from_tail, tails, snapshot_rows, snapshot_slots = [], [], [], [], [], []
        for req, first, end, start, offset, caps in rows:
            if first < start:
                from_tail.append(len(sources))
                tails.append(_slot(req))
            for position, slot in caps:
                if first <= position - 2 < end:
                    snapshot_rows.append(len(sources) + position - 2 - first)
                    snapshot_slots.append(slot)
            sources += range(offset + first - start, offset + end - start)
            tokens += range(offset + first + 1 - start, offset + end + 1 - start)
        device = streams.device
        x = streams[_upload(sources, device).clamp_min(0)]
        if from_tail:
            x[_upload(from_tail, device)] = tail[_upload(tails, device)]
        tokens = _upload(tokens, device)
        reqs, firsts, ends = zip(*((r, f, e) for r, f, e, *_ in rows))
        md, positions, out_loc = self.metadata(reqs, firsts, ends)
        if snapshot_rows:
            md.snapshots = (_upload(snapshot_rows, device), _upload(snapshot_slots, device))
        backend, layer = self.engine.attn_backend, self.mtp.layer_id
        self.mtp.write_history(
            x, self.engine.model.model.embed_tokens.forward(input_ids[tokens]), positions,
            lambda k, v, index: backend.write_history(k, v, index, layer, out_loc, md))
        self.history_rows += len(positions)


class _RoundHistory:
    """One round's MTP history, advanced at the retained length from the verify's streams."""

    def __init__(self, runtime: MTPRuntime) -> None:
        self.runtime = runtime
        self.verify: Batch | None = None  # set once the round's verify batch exists

    def commit(self, retained) -> None:
        verify = self.verify
        self.runtime.extend(verify.reqs, [r.cached_len for r in verify.reqs], retained,
                            verify.draft_features, verify.input_ids)
        verify.draft_features = self.verify = None


class MTPDrafter:
    """Drafts with the MTP layer: step 1 selects its history groups, later steps reuse them."""

    uses_target_state = False

    def __init__(self, engine, table, generator: torch.Generator) -> None:
        self.engine, self.table, self.generator = engine, table, generator
        self.runtime: MTPRuntime = engine.mtp
        self.positions = 0

    def plan(self, batch, lengths, *, reserve=None):
        return reserve(lengths) if reserve is not None else lengths

    def snapshot(self) -> dict:
        # The MTP layer's history is one more layer of the target's pages, priced with them.
        layers = self.engine.config.model_config.attention_group_for_layer(
            self.runtime.mtp.layer_id).layer_ids
        return dict(mtp_draft_positions=self.positions,
                    mtp_history_rows=self.runtime.history_rows,
                    mtp_weight_bytes=self.runtime.weight_bytes,
                    mtp_weight_dtype=self.runtime.weight_dtype,
                    mtp_weight_residency=self.runtime.weight_residency,
                    mtp_history_bytes_per_token=self.engine.kv_cache.unit_bytes()[0] // len(layers))

    def propose(self, batch, views, starts, lengths) -> DraftResult:
        engine, sampler, runtime = self.engine, self.engine.sampler, self.runtime
        steps = max(lengths)
        tokens_out = torch.zeros(batch.size, steps + 1, dtype=torch.int32, device=engine.device)
        probs_out = torch.zeros(batch.size, steps + 1, sampler.vocab_size, dtype=torch.float32,
                                device=engine.device)
        drafting = [i for i, n in enumerate(lengths) if n]
        reqs = [batch.reqs[i] for i in drafting]
        # The seed x[c] sits at ``start - 1``: step 1 is MTP row c-1 = (R[c-1], x[c]).
        firsts = [starts[i] - 2 for i in drafting]
        md, positions, out_loc = runtime.metadata(reqs, firsts, [f + 1 for f in firsts])
        md.ring_rows.fill_(-1)  # the open group's history stays as it was
        device = engine.device
        rows = _upload([batch.reqs[i].table_idx for i in drafting], device)
        tokens = self.table.token_pool[rows, _upload([starts[i] - 1 for i in drafting], device)]
        streams = runtime._tail()[_upload([_slot(r) for r in reqs], device)]
        backend, layer = engine.attn_backend, runtime.mtp.layer_id
        selected = []

        def first_step(q, k, v, index):
            backend.write_history(k, v, index, layer, out_loc, md)
            selected.append(backend.select(index, md, layer))
            return backend.attend(q, selected[0], md, layer)

        for step in range(steps):
            attend = first_step if step == 0 else (
                lambda q, k, v, index: backend.attend(q, selected[0], md, layer))
            hidden, streams = runtime.mtp.forward(
                streams, engine.model.model.embed_tokens.forward(tokens), positions + step, attend)
            live = [n for n, i in enumerate(drafting) if lengths[i] > step]
            out = [drafting[n] for n in live]
            logits = engine.model.lm_head.forward_selected(hidden[live])
            probs = sampler.probabilities(logits, sampler.prepare(Batch([views[i] for i in out])))
            sampled = torch.multinomial(probs, 1, generator=self.generator).flatten().to(torch.int32)
            probs_out[out, step] = probs
            tokens_out[out, step] = sampled
            self.table.token_pool[[views[i].table_idx for i in out],
                                  [starts[i] + step for i in out]] = sampled
            tokens = tokens.clone()
            tokens[live] = sampled
            self.positions += len(out)
        return DraftResult(tokens_out, probs_out, lengths, _RoundHistory(runtime))
