from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from flashlib.kernels.slot_cache import Stat

from freetoken.core import Batch, Req

from . import DraftResult

if TYPE_CHECKING:
    from freetoken.engine import Engine
    from freetoken.scheduler.table import TableManager


class SelfDrafter:
    uses_target_state = True

    def __init__(
        self,
        engine: Engine,
        table: TableManager,
        logits: Callable[[Batch], torch.Tensor],
        generator: torch.Generator,
    ) -> None:
        self.engine = engine
        self.table = table
        self.logits = logits
        self.generator = generator
        self.cost = engine.speculative_cost
        self.available = None
        self.loads_before = None
        self.residency_stops = 0
        self.draft_loads = (
            torch.zeros((), dtype=torch.int64, device=engine.device)
            if engine.config.moe_collect_stats else None
        )

    def snapshot(self) -> dict:
        result = {"residency_stops": self.residency_stops}
        if self.draft_loads is not None:
            result["draft_expert_loads"] = int(self.draft_loads.item())
        return result

    def observe_acceptance(self, lengths, accepted):
        self.cost.observe_acceptance(lengths, accepted)

    def plan(self, batch: Batch, lengths: list[int]) -> list[int]:
        engine = self.engine
        expert_cache = engine.moe_offload_cache
        self.available = None
        resident_ok = torch.ones((), dtype=torch.bool, device=engine.device) if self.cost is not None else None
        if (engine.config.speculative_draft_residency != "off" and expert_cache is not None
                and not engine.config.speculative_draft_load_missing):
            # Draft hits cannot evict or load experts, and self drafting never
            # interleaves target work, so this set is stable for the whole round.
            self.available = expert_cache.slot_for_id >= 0
            resident_ok = (self.available.sum(dim=1) >= engine.config.speculative_draft_experts).all()
            if self.cost is None and not bool(resident_ok.item()):
                self.residency_stops += sum(length > 0 for length in lengths)
                return [0] * batch.size
        if self.cost is not None:
            allowed, resident, limit = self.cost.admit(lengths, resident_ok)
            if not allowed or not resident:
                if allowed and not resident:
                    self.residency_stops += sum(length > 0 for length in lengths)
                return [0] * batch.size
            lengths = [min(length, limit) for length in lengths]
        return lengths

    def propose(
        self,
        batch: Batch,
        views: list[Req],
        starts: list[int],
        lengths: list[int],
    ) -> DraftResult:
        engine, sampler = self.engine, self.engine.sampler
        # Read here, not at plan time: a layered round plans while earlier groups still run.
        self.loads_before = (
            engine.moe_offload_cache.lru_stats[:, Stat.MISS].sum()
            if self.draft_loads is not None and engine.moe_offload_cache is not None else None
        )
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
                draft_available_experts=self.available,
            )
            logits = self.logits(draft)
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
        if self.loads_before is not None:
            self.draft_loads += engine.moe_offload_cache.lru_stats[:, Stat.MISS].sum() - self.loads_before
        return DraftResult(proposals, draft_probs, lengths)
