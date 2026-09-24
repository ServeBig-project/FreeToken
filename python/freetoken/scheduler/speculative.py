from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Callable

import torch
from flashlib.kernels.slot_cache import Stat

from freetoken.core import Batch
from freetoken.engine import ForwardOutput
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
        self.selector = DecodeBatchSelector()
        self.draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.verify_steps = 0
        self.residency_stops = 0
        self.state_slot_stops = 0
        self.draft_length_histogram = [0] * (engine.config.speculative_num_steps + 1)
        self.draft_loads = (
            torch.zeros((), dtype=torch.int64, device=engine.device)
            if engine.config.moe_collect_stats else None
        )

    def snapshot(self) -> dict:
        result = {
            "draft_tokens": self.draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "verify_steps": self.verify_steps,
            "residency_stops": self.residency_stops,
            "state_slot_stops": self.state_slot_stops,
            "draft_length_histogram": list(self.draft_length_histogram),
        }
        if self.draft_loads is not None:
            result["draft_expert_loads"] = int(self.draft_loads.item())
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

    def _commit_states(self, pool, states, accepted: torch.Tensor, slots: list[int]) -> None:
        """Live slot <- the scratch state after the last accepted token (position 0 when every
        draft was rejected: the token that was the ordinary decode query). Stream-ordered, so the
        slots can go back to the pool right away."""
        device = accepted.device
        scratch = torch.tensor([s[1] for s in states], dtype=torch.int64, device=device)
        src = scratch.gather(1, accepted[:, None]).squeeze(1)
        live = torch.tensor([s[0] for s in states], dtype=torch.int64, device=device)
        for layer in range(pool.num_linear_layers):
            rec, cv = pool.recurrent_states[layer], pool.conv_states[layer]
            rec.index_copy_(0, live, rec.index_select(0, src))
            cv.index_copy_(0, live, cv.index_select(0, src))
        pool.free(slots)

    def _logits(self, batch: Batch) -> torch.Tensor:
        batch.padded_reqs = batch.reqs
        forward_input = self.prepare(batch)
        batch.input_ids = self.table.token_pool[forward_input.input_tuple]
        return self.engine.compute_logits(batch)

    def forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch = forward_input.batch
        lengths = self._draft_lengths(batch)
        if not any(lengths):
            self._record_lengths(lengths)
            return self.engine.forward_batch(batch, forward_input.sample_args)

        engine, sampler = self.engine, self.engine.sampler
        expert_cache = engine.moe_offload_cache
        available = None
        resident_ok = torch.ones((), dtype=torch.bool, device=engine.device) if self.cost is not None else None
        if (engine.config.speculative_draft_residency != "off" and expert_cache is not None
                and not engine.config.speculative_draft_load_missing):
            # Draft hits cannot evict or load experts, and legacy SD never interleaves
            # target work into this loop, so this set is stable for the whole round.
            available = expert_cache.slot_for_id >= 0
            resident_ok = (available.sum(dim=1) >= engine.config.speculative_draft_experts).all()
            if self.cost is None and not bool(resident_ok.item()):
                self.residency_stops += sum(length > 0 for length in lengths)
                self._record_lengths([0] * batch.size)
                return engine.forward_batch(batch, forward_input.sample_args)
        if self.cost is not None:
            allowed, resident, limit = self.cost.admit(lengths, resident_ok)
            if not allowed or not resident:
                if allowed and not resident:
                    self.residency_stops += sum(length > 0 for length in lengths)
                self._record_lengths([0] * batch.size)
                return engine.forward_batch(batch, forward_input.sample_args)
            lengths = [min(length, limit) for length in lengths]
        loads_before = (
            expert_cache.lru_stats[:, Stat.MISS].sum()
            if self.draft_loads is not None and expert_cache is not None else None
        )
        starts = [req.device_len for req in batch.reqs]
        ends = [start + length for start, length in zip(starts, lengths, strict=True)]
        views = [copy(req) for req in batch.reqs]
        # GDN models: drafts advance a copy of each live state; verification keeps one state
        # per position so the commit can pick the accepted one. Slots come from the pool.
        pool, states, slots = engine.linear_state_pool, None, None
        if pool is not None:
            per_request = engine.config.speculative_num_steps + 2
            if pool.num_free_slots() < per_request * batch.size:
                self.state_slot_stops += 1
                self._record_lengths([0] * batch.size)
                return engine.forward_batch(batch, forward_input.sample_args)
            slots = pool.alloc(per_request * batch.size)
            states = []
            for i, (req, view) in enumerate(zip(batch.reqs, views, strict=True)):
                live = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
                own = slots[i * per_request : (i + 1) * per_request]
                pool.copy_from(live, own[0])
                view.linear_slot_idx = own[0]
                states.append((live, own[1:]))
        # The ordinary decode query is already allocated. Reserve only the extra
        # span; the same physical slots serve drafting and target verification.
        for req, start, end in zip(views, starts, ends, strict=True):
            req.cached_len, req.device_len = start, end
        self.cache.allocate_paged(views)

        steps = max(lengths)
        draft_probs = torch.zeros(
            batch.size, steps + 1, sampler.vocab_size, dtype=torch.float32, device=engine.device
        )
        proposals = torch.zeros(batch.size, steps + 1, dtype=torch.int32, device=engine.device)
        for step in range(steps):
            active = [i for i, length in enumerate(lengths) if length > step]
            draft_reqs = [views[i] for i in active]
            for i in active:
                views[i].cached_len = starts[i] + step - 1
                views[i].device_len = starts[i] + step
            draft = Batch(
                draft_reqs, decode_size=len(draft_reqs),
                draft_experts=engine.config.speculative_draft_experts,
                draft_available_experts=available,
            )
            logits = self._logits(draft)
            probs = sampler.probabilities(logits, sampler.prepare(draft))
            tokens = torch.multinomial(
                probs, 1, generator=self.generator
            ).flatten().to(torch.int32)
            draft_probs[active, step] = probs
            proposals[active, step] = tokens
            rows = [views[i].table_idx for i in active]
            positions = [starts[i] + step for i in active]
            self.table.token_pool[rows, positions] = tokens
            if self.cost is not None and step + 1 < steps:
                if not self.cost.continue_draft(lengths, step + 1, batch.size):
                    lengths = [min(length, step + 1) for length in lengths]
                    break
        self.draft_tokens += sum(lengths)
        self._record_lengths(lengths)
        if loads_before is not None:
            self.draft_loads += expert_cache.lru_stats[:, Stat.MISS].sum() - loads_before

        # Target attention overwrites every provisional query, layer by layer,
        # while retaining only the already verified prefix before the round.
        for req, start, length in zip(views, starts, lengths, strict=True):
            req.cached_len, req.device_len = start - 1, start + length
        verify = Batch(views, is_speculative_verify=True)
        if states is not None:
            verify.speculative_states = [
                (live, scratch[: length + 1]) for (live, scratch), length in zip(states, lengths, strict=True)
            ]
        logits = self._logits(verify)
        target_probs = sampler.probabilities(
            logits, sampler.prepare(verify, repeats=[length + 1 for length in lengths])
        )
        self.verify_steps += 1
        output = torch.full_like(proposals, -1)
        accepted_lengths = torch.empty(batch.size, dtype=torch.int64, device=engine.device) if self.cost is not None else None
        accepted_all = []
        offset = 0
        for i, length in enumerate(lengths):
            p = target_probs[offset : offset + length + 1]
            q = draft_probs[i, : length + 1]
            candidates = proposals[i, :length].long()
            p_chosen = p[:length].gather(1, candidates[:, None]).flatten()
            q_chosen = q[:length].gather(1, candidates[:, None]).flatten()
            uniform = torch.rand(length, device=engine.device, generator=self.generator)
            accepted = (uniform * q_chosen < p_chosen).to(torch.int32).cumprod(0).sum(0, keepdim=True)
            accepted_all.append(accepted)
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
            self.cost.observe_acceptance(lengths, accepted_lengths)
        if states is not None:
            self._commit_states(pool, states, torch.cat(accepted_all).long(), slots)

        host = output.to("cpu", non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(engine.stream)
        return ForwardOutput(output, host, ready, speculative_ends=ends)
