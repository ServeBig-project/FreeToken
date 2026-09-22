from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def load_draft_cost(path: str) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    fields = ("target_token_ms", "draft_step_ms", "expert_bandwidth_gib_s")
    if not isinstance(data, dict) or any(
        type(data.get(key)) not in (float, int) or not math.isfinite(data[key]) or data[key] <= 0
        for key in fields
    ):
        raise ValueError("adaptive profile requires finite positive target_token_ms, draft_step_ms, expert_bandwidth_gib_s")
    return {key: float(data[key]) for key in fields}


class DraftExpansion:
    """Decide whether to draw another proposal using only its already-known prefix."""

    def __init__(self, engine):
        self.cost = engine.config.draft_cost
        model = engine.config.model_config
        self.resident = torch.full(
            (model.num_moe_layers, model.num_experts), engine.config.moe_backend == "fused",
            dtype=torch.bool, device=engine.device,
        )
        if engine.config.resident_experts:
            pairs = torch.tensor(engine.config.resident_experts, dtype=torch.int64, device=engine.device)
            self.resident[pairs[:, 0], pairs[:, 1]] = True
        cache = engine.moe_offload_cache
        self.expert_ms = (cache._expert_row_bytes if cache is not None else 0) / (
            self.cost["expert_bandwidth_gib_s"] * (1 << 30) / 1000
        )

    def start(self, batch_size: int) -> None:
        self.seen = torch.zeros(
            (batch_size, *self.resident.shape), dtype=torch.bool, device=self.resident.device
        )
        self.confidence = torch.ones(batch_size, device=self.resident.device)

    def allows(self, active: list[int], routes: torch.Tensor, *, first: bool) -> list[bool]:
        seen = self.seen[active]
        routed = torch.zeros_like(seen).scatter_(2, routes.transpose(0, 1).long(), True)
        new_cold = (routed & ~seen & ~self.resident).sum(dim=(1, 2))
        self.seen[active] = seen | routed
        if first:
            return [True] * len(active)
        benefit = self.confidence[active] * self.cost["target_token_ms"]
        cost = new_cold * self.expert_ms + self.cost["draft_step_ms"]
        return (benefit >= cost).tolist()

    def record(self, active: list[int], logits: torch.Tensor, tokens: torch.Tensor) -> None:
        # Confidence is the model's raw probability, not a temperature-zero delta.
        scores = logits.float()
        selected = scores.gather(1, tokens.long()[:, None]).flatten()
        probability = (selected - scores.logsumexp(dim=-1)).exp()
        self.confidence[active] *= probability
