from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer
from freetoken.core import get_global_ctx
from freetoken.moe.resident_draft import draft_routing

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
        draft = draft_routing(get_global_ctx(), self.layer_id, self.experts, hidden_states, router_logits)
        if draft is not None:
            return self.experts.routed_forward(hidden_states, *draft)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states


__all__ = ["Qwen3MoeMLP"]
