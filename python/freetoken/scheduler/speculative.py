from __future__ import annotations

from copy import copy
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
        self.cost = engine.speculative_cost
        # FlashInfer uses its updated default-generator offset for the current draw.
        # A separate seed keeps subsequent Torch draws from reusing that random stream.
        self.generator = torch.Generator(device=engine.device)
        self.generator.manual_seed((torch.cuda.initial_seed() + 1) % (1 << 64))
        if engine.config.speculative_draft_model_path:
            from freetoken.speculative.dflash import DFlashDrafter

            self.drafter = DFlashDrafter(engine, table, self._logits, self.generator)
        else:
            self.drafter = SelfDrafter(engine, table, self._logits, self.generator)
        self.selector = DecodeBatchSelector()
        self.draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.verify_steps = 0
        self.state_slot_stops = 0
        self.draft_length_histogram = [0] * (engine.config.speculative_num_steps + 1)

    def snapshot(self) -> dict:
        result = {
            "draft_tokens": self.draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "verify_steps": self.verify_steps,
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

    def _draft_lengths(self, batch: Batch) -> list[int]:
        config = self.engine.config
        budget = config.max_forward_len - batch.size
        pages = self.cache.available_size // self.cache.page_size
        page_size = self.cache.page_size
        lengths = []
        for req in batch.reqs:
            first_page = div_ceil(req.device_len, page_size)
            capacity = (first_page + pages) * page_size - req.device_len
            length = min(config.speculative_num_steps, req.remain_len - 1, budget, capacity)
            lengths.append(length)
            budget -= length
            pages -= div_ceil(req.device_len + length, page_size) - first_page
        return lengths

    def _logits(self, batch: Batch) -> torch.Tensor:
        batch.padded_reqs = batch.reqs
        forward_input = self.prepare(batch)
        batch.input_ids = self.table.token_pool[forward_input.input_tuple]
        return self.engine.compute_logits(batch)

    def forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch = forward_input.batch
        lengths = self._draft_lengths(batch)
        limited = self.cache.limit_speculation(lengths)
        if any(lengths) and not any(limited):
            self.state_slot_stops += 1
        lengths = limited
        if not any(lengths):
            self._record_lengths(lengths)
            return self.engine.forward_batch(batch, forward_input.sample_args)

        engine, sampler = self.engine, self.engine.sampler
        lengths = self.drafter.plan(batch, lengths)
        if not any(lengths):
            self._record_lengths(lengths)
            return engine.forward_batch(batch, forward_input.sample_args)
        starts = [req.device_len for req in batch.reqs]
        ends = [start + length for start, length in zip(starts, lengths, strict=True)]
        views = [copy(req) for req in batch.reqs]
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

        draft = self.drafter.propose(batch, views, starts, lengths)
        proposals, draft_probs, lengths = draft.tokens, draft.probabilities, draft.lengths
        self.draft_tokens += sum(lengths)
        self._record_lengths(lengths)

        # Target attention overwrites every provisional query, layer by layer,
        # while retaining only the already verified prefix before the round.
        for req, start, length in zip(views, starts, lengths, strict=True):
            req.cached_len, req.device_len = start - 1, start + length
        verify = Batch(views, is_speculative_verify=True)
        if state is not None:
            state.prepare_verify(verify, lengths)
        logits = self._logits(verify)
        target_probs = sampler.probabilities(
            logits, sampler.prepare(verify, repeats=[length + 1 for length in lengths])
        )
        self.verify_steps += 1
        output = torch.full_like(proposals, -1)
        accepted_lengths = torch.empty(batch.size, dtype=torch.int64, device=engine.device) if self.cost is not None else None
        offset = 0
        for i, length in enumerate(lengths):
            p = target_probs[offset : offset + length + 1]
            q = draft_probs[i, : length + 1]
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
        return ForwardOutput(output, host, ready, speculative_ends=ends, speculative_state=state)
