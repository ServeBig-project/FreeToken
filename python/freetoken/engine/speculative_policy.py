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


def reuse_routing(logits, weights, ids, lengths: list[int], cap: int, renormalize: bool):
    """Official 4090 preference bias, computed separately for each request."""
    top_k = ids.shape[1]
    changed_count = torch.zeros((), dtype=torch.int64, device=logits.device)
    if top_k == logits.shape[1]:
        return weights, ids, changed_count
    result_ids, result_weights = ids.clone(), weights.clone()
    offset = 0
    for length in lengths:
        end = offset + length
        if length > 1:
            row = logits[offset:end].float()
            top = row.topk(top_k + 1, dim=-1).values
            strength = (top[:, 0] - top[:, -1]).mean()
            importance_weight = (row.max(dim=-1).values - row.mean(dim=-1)).clamp_min(0)
            maximum = importance_weight.max()
            importance = (importance_weight[:, None] * row).sum(dim=0) / maximum.clamp_min(1e-30)
            preferred = importance.argsort(descending=True, stable=True)[:cap]
            selection = row.clone()
            selection[:, preferred] += strength
            selected = selection.argsort(dim=-1, descending=True, stable=True)[:, :top_k].to(torch.int32)
            original = ids[offset:end]
            changed = (selected.sort(dim=-1).values != original.sort(dim=-1).values).any(dim=-1)
            changed &= (maximum > 0) & (strength > 0)
            # Selection bias never enters the expert mixture weights.
            selected_weights = row.softmax(dim=-1).gather(1, selected.long())
            if renormalize:
                selected_weights /= selected_weights.sum(dim=-1, keepdim=True)
            result_ids[offset:end] = torch.where(changed[:, None], selected, original)
            result_weights[offset:end] = torch.where(changed[:, None], selected_weights, weights[offset:end])
            changed_count += changed.sum()
        offset = end
    return result_weights, result_ids, changed_count
