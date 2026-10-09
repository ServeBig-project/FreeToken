from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from freetoken.core import Batch
from freetoken.engine import ForwardOutput
from freetoken.speculative.self_draft import SelfDrafter
from freetoken.utils import div_ceil

from .batch_composition import DecodeBatchSelector

if TYPE_CHECKING:
    from freetoken.engine import Engine
    from .cache import CacheManager
    from .forward import ForwardInput
    from .table import TableManager


@dataclass
class SpeculativeRound:
    """Drafted tokens and their provisional resources, ready for target verification."""

    batch: Batch
    verify: Batch
    proposals: torch.Tensor
    draft_probs: torch.Tensor
    lengths: list[int]
    starts: list[int]
    ends: list[int]
    state: object | None


class SpeculativeDecoder:
    """Share target history, draft into temporary KV, then verify and commit a prefix."""

    def __init__(
        self,
        engine: Engine,
        cache: CacheManager,
        table: TableManager,
        prepare: Callable[[Batch], ForwardInput],
    ) -> None:
        self.engine = engine
        self.cache = cache
        self.table = table
        self.prepare = prepare
        self.draft_views: list = []  # shared runtime: views whose draft pages are reserved
        self.cost = engine.speculative_cost
        self.phase = engine.config.speculative_phase
        # FlashInfer uses its updated default-generator offset for the current draw.
        # A separate seed keeps subsequent Torch draws from reusing that random stream.
        self.generator = torch.Generator(device=engine.device)
        self.generator.manual_seed((torch.cuda.initial_seed() + 1) % (1 << 64))
        if engine.config.speculative_draft_model_path:
            from freetoken.speculative.dflash import DFlashDrafter

            self.drafter = DFlashDrafter(engine, table, self.generator)
        else:
            self.drafter = SelfDrafter(engine, table, self._logits, self.generator)
        self.control = getattr(self.drafter, "control", None)
        self.selector = DecodeBatchSelector()
        self.draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.emitted_tokens = 0
        self.verify_positions = [0, 0]  # real and physical
        self.verify_steps = 0
        self.verify_rounds = {"inwave": 0, "outwave": 0}
        self.verify_requests = {"inwave": 0, "outwave": 0}
        # Decode requests that ran AR instead of SD, by reason.
        self.fallback_requests: dict[str, int] = {}
        # Requests that ran SD with a shorter draft: cut by their own output tail or by a resource.
        self.clipped_requests = {"tail": 0, "capacity": 0}
        self.state_slot_stops = 0
        self.draft_length_histogram = [0] * (engine.config.speculative_num_steps + 1)

    def snapshot(self) -> dict:
        result = {
            "drafter": "dflash" if self.engine.config.speculative_draft_model_path else "self",
            "phase": self.phase,
            "draft_tokens": self.draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "emitted_tokens": self.emitted_tokens,
            "verify_positions": self.verify_positions[0],
            "verify_physical_positions": self.verify_positions[1],
            "verify_steps": self.verify_steps,
            "verify_rounds": dict(self.verify_rounds),
            "verify_requests": dict(self.verify_requests),
            "fallback_requests": dict(self.fallback_requests),
            "clipped_requests": dict(self.clipped_requests),
            "state_slot_stops": self.state_slot_stops,
            "draft_length_histogram": list(self.draft_length_histogram),
        }
        result.update(self.drafter.snapshot())
        if self.cost is not None:
            result.update(self.cost.snapshot())
        return result

    def _record_lengths(self, lengths) -> None:
        for length in lengths:
            self.draft_length_histogram[length] += 1

    def _fallback(self, reason: str, batch: Batch) -> None:
        self.fallback_requests[reason] = self.fallback_requests.get(reason, 0) + batch.size

    def _draft_lengths(self, batch: Batch) -> tuple[list[int], bool]:
        """Per-request draft limits and whether free KV pages or window slots, not the tail, cut one."""
        config = self.engine.config
        budget = config.max_forward_len - batch.size
        pages = self.cache.available_size // self.cache.page_size
        page_size = self.cache.page_size
        cache, windows = self.cache, None
        if cache.swa_paged:
            # Each drafted position also binds a window slot; release what the requests have
            # moved past before shortening any draft.
            if cache.swa_available_size < batch.size * config.speculative_num_steps:
                cache.maybe_free_swa_out_of_window(batch.reqs, force=True)
            windows = cache.swa_available_size
        lengths = []
        kv_short = False
        for req in batch.reqs:
            first_page = div_ceil(req.device_len, page_size)
            capacity = (first_page + pages) * page_size - req.device_len
            if windows is not None:
                capacity = min(capacity, windows)
            wanted = min(config.speculative_num_steps, req.remain_len - 1, budget)
            length = min(wanted, capacity)
            kv_short |= length < wanted
            lengths.append(length)
            budget -= length
            pages -= div_ceil(req.device_len + length, page_size) - first_page
            if windows is not None:
                windows -= (div_ceil(req.device_len + length, page_size) - first_page) * page_size
        return lengths, kv_short

    def _logits(self, batch: Batch) -> torch.Tensor:
        batch.padded_reqs = batch.reqs
        forward_input = self.prepare(batch)
        batch.input_ids = self.table.token_pool[forward_input.input_tuple]
        return self.engine.compute_logits(batch)

    def allows(self, in_wave: bool) -> bool:
        return self.phase in ("all", "inwave" if in_wave else "outwave")

    def may_run(self, wave_active: bool) -> bool:
        """Whether this iteration can speculate; without a wave one may still open."""
        return self.allows(True) or (not wave_active and self.allows(False))

    def observe(self, accepted: list) -> None:
        """Committed outcome of the last round: per request the drafts the target accepted
        (None when unknown), before any stop truncated them."""
        self.accepted_draft_tokens += sum(a for a in accepted if a is not None)
        if self.control is not None:
            self.control.observe(accepted)

    def begin_round(self) -> None:
        """A decode-only batch was chosen: time the round from before its preparation."""
        if self.control is not None:
            self.control.begin()

    def end_round(self) -> None:
        """The last round's replies are queued: close its timing."""
        if self.control is not None:
            self.control.end()

    def forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """One decode step outside a prefill wave: SD when allowed, else AR."""
        batch = forward_input.batch
        round_ = None
        if not self.allows(False):
            self._fallback("phase", batch)
        else:
            lengths = self.admit(batch, full_width=False)
            round_ = self.start(batch, lengths) if lengths is not None else None
        if round_ is None:
            return self.engine.forward_batch(batch, forward_input.sample_args)
        logits = self.engine.compute_logits(round_.verify)
        real = batch.size + sum(round_.lengths)
        graphs = self.engine.graph_runner.speculative
        self.verify_positions[0] += real
        self.verify_positions[1] += graphs.verify_tokens(batch.size, real) if graphs is not None else real
        return self.finish(round_, logits, "outwave")

    def admit_inwave(self, batch: Batch) -> list[int] | None:
        """Draft lengths of one same-width round beside a prefill wave, or None for AR."""
        if not self.allows(True):
            self._fallback("phase", batch)
            return None
        return self.admit(batch, full_width=True)

    def admit(self, batch: Batch, *, full_width: bool) -> list[int] | None:
        """Draft lengths if this round can speculate; every limit is checked before any write.

        ``full_width`` admits only rounds where every request drafts the full
        window; otherwise each request keeps its own shorter legal window.
        """
        lengths, kv_short = self._draft_lengths(batch)
        limited = self.cache.limit_speculation(lengths)
        if any(lengths) and not any(limited):
            self.state_slot_stops += 1
        full = [self.engine.config.speculative_num_steps] * batch.size
        short = (lambda values: values != full) if full_width else (lambda values: not any(values))
        if short(lengths):
            reason = "kv_capacity" if kv_short else "tail"
        elif short(limited) or (full_width and limited != lengths):
            reason = "state_capacity"
        else:
            reason = None
            if self.cache.page_units is not None:
                # One claim for the round's pages, window slots and scratch states, before the
                # controller decides, so it records the length that can actually run.
                limited = self._reserve_round(batch, limited, full_width)
            if short(limited):
                reason = "kv_capacity"
            else:
                lengths = self.drafter.plan(batch, limited)
                if short(lengths):
                    reason = "draft_plan"
        if reason is not None:
            self._drop_round()
            self._fallback(reason, batch)
            self._record_lengths([0] * batch.size)
            return None
        steps = self.engine.config.speculative_num_steps
        for req, length in zip(batch.reqs, limited):
            if length < min(steps, req.remain_len - 1):
                self.clipped_requests["capacity"] += 1
            elif length < steps:
                self.clipped_requests["tail"] += 1
        return lengths

    def _reserve_round(self, batch: Batch, lengths: list[int], full_width: bool) -> list[int]:
        """Shared runtime: take the round's resources for ``lengths``, shortening every draft
        (a full width round: all or none) until they fit; ``start`` then runs these views."""
        self._drop_round()
        pool = self.cache.linear_state_pool
        while any(lengths):
            views = [copy(req) for req in batch.reqs]
            for view, length in zip(views, lengths, strict=True):
                view.cached_len, view.device_len = view.device_len, view.device_len + length
            states = pool.speculative_size(lengths) if pool is not None else 0
            if self.cache.reserve_round(views, states):
                self.draft_views = views
                return lengths
            if full_width:
                break
            limit = max(lengths) // 2
            lengths = [min(length, limit) for length in lengths]
        return [0] * len(lengths)

    def _drop_round(self) -> None:
        """Give back what a round that does not run had reserved."""
        for view in self.draft_views:
            self.cache._cancel_decode_reservation(view)
        self.draft_views = []
        self.cache.drop_speculative_slots()

    def start(self, batch: Batch, lengths: list[int]) -> SpeculativeRound:
        """Draft the admitted lengths and prepare the verification batch."""
        starts = [req.device_len for req in batch.reqs]
        ends = [start + length for start, length in zip(starts, lengths, strict=True)]
        # The views a shared runtime reserved the draft pages for at admission.
        views, self.draft_views = self.draft_views or [copy(req) for req in batch.reqs], []
        if self.cost is not None:
            self.cost.begin_state(0)
        state = self.cache.begin_speculation(
            batch.reqs, views, lengths, draft=self.drafter.uses_target_state)
        if self.cost is not None:
            self.cost.end_state(0, batch.size)
        # The ordinary decode query is already allocated. Reserve only the extra
        # span; the same physical slots serve drafting and target verification.
        for req, start, end in zip(views, starts, ends, strict=True):
            req.cached_len, req.device_len = start, end
        self.cache.allocate_paged(views)

        if self.control is not None:
            self.control.mark(1)
        draft = self.drafter.propose(batch, views, starts, lengths)
        if self.control is not None:
            self.control.mark(2)
        lengths = draft.lengths
        self.draft_tokens += sum(lengths)
        self._record_lengths(lengths)

        # Target attention overwrites every provisional query, layer by layer,
        # while retaining only the already verified prefix before the round.
        for req, start, length in zip(views, starts, lengths, strict=True):
            req.cached_len, req.device_len = start - 1, start + length
        verify = Batch(views, is_speculative_verify=True)
        if state is not None:
            state.prepare_verify(verify, lengths)
        verify.padded_reqs = verify.reqs
        verify_input = self.prepare(verify)
        verify.input_ids = self.table.token_pool[verify_input.input_tuple]
        return SpeculativeRound(batch, verify, draft.tokens, draft.probabilities,
                                lengths, starts, ends, state)

    def finish(self, round_: SpeculativeRound, logits: torch.Tensor, phase: str) -> ForwardOutput:
        """Accept a prefix from complete target logits; the scheduler commits it."""
        if self.control is not None:
            self.control.mark(3)
        engine, sampler = self.engine, self.engine.sampler
        batch, lengths, starts, proposals = round_.batch, round_.lengths, round_.starts, round_.proposals
        target_probs = sampler.probabilities(
            logits, sampler.prepare(round_.verify, repeats=[length + 1 for length in lengths])
        )
        self.verify_steps += 1
        self.verify_rounds[phase] += 1
        self.verify_requests[phase] += batch.size
        output = torch.full_like(proposals, -1)
        accepted_lengths = torch.empty(batch.size, dtype=torch.int64, device=engine.device) if self.cost is not None else None
        offset = 0
        for i, length in enumerate(lengths):
            p = target_probs[offset : offset + length + 1]
            q = round_.draft_probs[i, : length + 1]
            candidates = proposals[i, :length].long()
            p_chosen = p[:length].gather(1, candidates[:, None]).flatten()
            q_chosen = q[:length].gather(1, candidates[:, None]).flatten()
            uniform = torch.rand(length, device=engine.device, generator=self.generator)
            accepted = (uniform * q_chosen < p_chosen).to(torch.int32).cumprod(0).sum(0, keepdim=True)
            if accepted_lengths is not None:
                accepted_lengths[i : i + 1] = accepted
            # The zero q row after the last draft makes the all-accepted case
            # sample its bonus directly from p. Otherwise sample max(p-q, 0).
            # A 0-dim index would be read back to the host; keep ``accepted`` 1-D.
            correction = (p.index_select(0, accepted) - q.index_select(0, accepted)).clamp_min_(0)
            token = torch.multinomial(correction, 1, generator=self.generator).to(torch.int32)
            out = proposals[i, : length + 1].clone()
            out.scatter_(0, accepted, token.flatten())
            out.masked_fill_(torch.arange(length + 1, device=engine.device) > accepted, -1)
            output[i, : length + 1] = out
            req = batch.reqs[i]
            self.table.token_pool[req.table_idx, starts[i] : starts[i] + length + 1] = out
            offset += length + 1

        if accepted_lengths is not None:
            self.drafter.observe_acceptance(lengths, accepted_lengths)

        host = output.to("cpu", non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(engine.stream)
        return ForwardOutput(output, host, ready, speculative_ends=round_.ends,
                             speculative_state=round_.state)
