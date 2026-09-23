from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer
from freetoken.core import get_global_ctx
from freetoken.moe.fused import fused_topk
from freetoken.moe.resident_draft import cached_draft_routing
from freetoken.kernel.triton.moe_reuse import reuse_routing

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
        reuse = ctx.reuse_expert_cap and ctx.batch.is_speculative_verify
        if draft_experts is not None or ctx.expert_counts is not None or reuse:
            available = (ctx.batch.draft_available_experts[self.layer_id]
                         if ctx.batch.draft_available_experts is not None else None)
            if draft_experts is not None and ctx.draft_load_missing:
                available = ctx.moe_offload_cache.slot_for_id[self.layer_id] >= 0
            if available is not None:
                weights, ids, missing = cached_draft_routing(
                    hidden_states, router_logits, draft_experts, self.experts.renormalize,
                    available,
                    ctx.draft_affinity[self.layer_id] if ctx.draft_residency == "affinity" else None,
                    load_missing=ctx.draft_load_missing,
                )
                if ctx.batch.draft_replacement_masks is not None and missing is not None:
                    ctx.batch.draft_replacement_masks.append(missing)
            else:
                weights, ids = fused_topk(
                    hidden_states, router_logits, draft_experts or self.experts.top_k, self.experts.renormalize
                )
            if ctx.batch.draft_routes is not None:
                ctx.batch.draft_routes[self.layer_id].copy_(ids)
            if reuse:
                weights, ids = reuse_routing(
                    router_logits, weights, ids, ctx.batch.reuse_offsets,
                    ctx.reuse_expert_cap, self.experts.renormalize, ctx.reuse_changed_routes,
                )
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
