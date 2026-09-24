from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer
from freetoken.core import get_global_ctx
from freetoken.moe.fused import fused_topk
from freetoken.moe.resident_draft import cached_draft_routing

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
        if draft_experts is not None and ctx.speculative_cost is not None:
            _, predicted = fused_topk(hidden_states, router_logits, self.experts.top_k, self.experts.renormalize)
            ctx.speculative_cost.record_prediction(self.layer_id, predicted, router_logits)
        if draft_experts is not None:
            available = (ctx.batch.draft_available_experts[self.layer_id]
                         if ctx.batch.draft_available_experts is not None else None)
            if ctx.draft_load_missing:
                available = ctx.moe_offload_cache.slot_for_id[self.layer_id] >= 0
            if available is not None:
                weights, ids = cached_draft_routing(
                    hidden_states, router_logits, draft_experts, self.experts.renormalize,
                    available, load_missing=ctx.draft_load_missing,
                )
            else:
                weights, ids = fused_topk(hidden_states, router_logits, draft_experts, self.experts.renormalize)
            return self.experts.routed_forward(hidden_states, weights, ids)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states


__all__ = ["Qwen3MoeMLP"]
