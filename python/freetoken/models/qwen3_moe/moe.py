from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer
from freetoken.core import get_global_ctx
from freetoken.moe.fused import fused_topk

if TYPE_CHECKING:
    import torch

    from freetoken.models.config import ModelConfig


class Qwen3MoeMLP(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        self.layer_id = layer_id
        self.experts = make_moe_layer(config, layer_id=layer_id)
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        ctx = get_global_ctx()
        draft_experts = ctx.batch.draft_experts
        if draft_experts is not None or ctx.expert_counts is not None:
            weights, ids = fused_topk(
                hidden_states, router_logits, draft_experts or self.experts.top_k, self.experts.renormalize
            )
            if ctx.batch.draft_routes is not None:
                ctx.batch.draft_routes[self.layer_id].copy_(ids)
            if ctx.expert_counts is not None:
                import torch
                ctx.expert_counts[self.layer_id].scatter_add_(
                    0, ids.flatten().long(), torch.ones_like(ids.flatten(), dtype=torch.int64)
                )
            return self.experts.routed_forward(hidden_states, weights, ids)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states


__all__ = ["Qwen3MoeMLP"]
