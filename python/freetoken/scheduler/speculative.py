from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Callable

import torch

from freetoken.core import Batch
from freetoken.engine import ForwardOutput
from freetoken.engine.speculative_policy import DraftExpansion
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
        # FlashInfer uses its updated default-generator offset for the current draw.
        # A separate seed keeps subsequent Torch draws from reusing that random stream.
        self.generator = torch.Generator(device=engine.device)
        self.generator.manual_seed((torch.cuda.initial_seed() + 1) % (1 << 64))
        self.selector = DecodeBatchSelector()
        self.draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.verify_steps = 0
        self.adaptive_stops = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "draft_tokens": self.draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "verify_steps": self.verify_steps,
            "adaptive_stops": self.adaptive_stops,
        }

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
        if not any(lengths):
            return self.engine.forward_batch(batch, forward_input.sample_args)

        engine, sampler = self.engine, self.engine.sampler
        expansion = DraftExpansion(engine, batch.size) if engine.config.draft_cost else None
        starts = [req.device_len for req in batch.reqs]
        ends = [start + length for start, length in zip(starts, lengths, strict=True)]
        views = [copy(req) for req in batch.reqs]
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
            if not active:
                break
            draft_reqs = [views[i] for i in active]
            for i in active:
                views[i].cached_len = starts[i] + step - 1
                views[i].device_len = starts[i] + step
            draft = Batch(
                draft_reqs, decode_size=len(draft_reqs),
                draft_experts=engine.config.speculative_draft_experts,
            )
            if expansion is not None:
                draft.draft_routes = torch.empty(
                    engine.config.model_config.num_moe_layers, len(active), draft.draft_experts,
                    dtype=torch.int32, device=engine.device,
                )
            logits = self._logits(draft)
            if expansion is not None:
                allowed = expansion.allows(active, draft.draft_routes, first=step == 0)
                for i, keep in zip(active, allowed, strict=True):
                    if not keep:
                        lengths[i] = step
                        self.adaptive_stops += 1
                active = [i for i, keep in zip(active, allowed, strict=True) if keep]
                if not active:
                    break
                logits = logits[allowed]
                draft = Batch([views[i] for i in active], decode_size=len(active))
            probs = sampler.probabilities(logits, sampler.prepare(draft))
            tokens = torch.multinomial(
                probs, 1, generator=self.generator
            ).flatten().to(torch.int32)
            if expansion is not None:
                expansion.record(active, logits, tokens)
            draft_probs[active, step] = probs
            proposals[active, step] = tokens
            rows = [views[i].table_idx for i in active]
            positions = [starts[i] + step for i in active]
            self.table.token_pool[rows, positions] = tokens
        self.draft_tokens += sum(lengths)

        # Target attention overwrites every provisional query, layer by layer,
        # while retaining only the already verified prefix before the round.
        for req, start, length in zip(views, starts, lengths, strict=True):
            req.cached_len, req.device_len = start - 1, start + length
        verify = Batch(views, is_speculative_verify=True)
        logits = self._logits(verify)
        target_probs = sampler.probabilities(
            logits, sampler.prepare(verify, repeats=[length + 1 for length in lengths])
        )
        self.verify_steps += 1
        output = torch.full_like(proposals, -1)
        offset = 0
        for i, length in enumerate(lengths):
            p = target_probs[offset : offset + length + 1]
            q = draft_probs[i, : length + 1]
            candidates = proposals[i, :length].long()
            p_chosen = p[:length].gather(1, candidates[:, None]).flatten()
            q_chosen = q[:length].gather(1, candidates[:, None]).flatten()
            uniform = torch.rand(length, device=engine.device, generator=self.generator)
            accepted = (uniform * q_chosen < p_chosen).to(torch.int32).cumprod(0).sum()
            # The zero q row after the last draft makes the all-accepted case
            # sample its bonus directly from p. Otherwise sample max(p-q, 0).
            correction = (p[accepted] - q[accepted]).clamp_min_(0)
            token = torch.multinomial(correction, 1, generator=self.generator).to(torch.int32)
            out = proposals[i, : length + 1].clone()
            out[accepted] = token[0]
            out.masked_fill_(torch.arange(length + 1, device=engine.device) > accepted, -1)
            output[i, : length + 1] = out
            req = batch.reqs[i]
            self.table.token_pool[req.table_idx, starts[i] : starts[i] + length + 1] = out
            offset += length + 1

        host = output.to("cpu", non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(engine.stream)
        return ForwardOutput(output, host, ready, speculative_ends=ends)
