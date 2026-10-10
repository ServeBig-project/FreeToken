from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearRowParallel,
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
)
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from .config import ModelConfig


class BaseLLMModel(ABC, BaseOP):
    supports_draft_features = False

    @abstractmethod
    def forward(self) -> torch.Tensor: ...


def record_draft_feature(layer_id, hidden, residual):
    from freetoken.core import get_global_ctx

    context = get_global_ctx().draft_context
    if context is not None:
        context.record(layer_id, hidden, residual)


@dataclass
class LayerGroupState:
    hidden: torch.Tensor
    residual: torch.Tensor | None
    next_layer: int = 0
    # DFlash context inputs recorded at target layers, until the last layer projects them.
    draft_features: dict[int, torch.Tensor] = field(default_factory=dict)


class ResidualLayerGroupCausalLM(BaseLLMModel):
    """Explicit opt-in for the common hidden/residual decoder-layer interface."""

    @property
    def layer_group_state_width(self) -> int:
        return 2 * self.model.embed_tokens.weight.shape[-1]

    @property
    def layer_group_num_layers(self) -> int:
        return len(self.model.layers.op_list)

    def begin_layer_group_prefill(self, input_ids: torch.Tensor) -> LayerGroupState:
        return LayerGroupState(
            hidden=self.model.embed_tokens.forward(input_ids),
            residual=None,
        )

    @staticmethod
    def layer_group_state_layer(state: LayerGroupState) -> int:
        return state.next_layer

    @staticmethod
    def layer_group_merge_states(
        decode: LayerGroupState,
        prefill: LayerGroupState,
    ) -> LayerGroupState:
        if decode.next_layer != prefill.next_layer:
            raise RuntimeError("decode and prefill states are at different layers")
        if (decode.residual is None) != (prefill.residual is None):
            raise RuntimeError("decode and prefill residual states do not match")
        residual = (
            None
            if decode.residual is None
            else torch.cat((decode.residual, prefill.residual), dim=0)
        )
        return LayerGroupState(
            hidden=torch.cat((decode.hidden, prefill.hidden), dim=0),
            residual=residual,
            next_layer=decode.next_layer,
            draft_features={layer: torch.cat((value, prefill.draft_features[layer]), dim=0)
                            for layer, value in decode.draft_features.items()},
        )

    @staticmethod
    def layer_group_split_state(
        state: LayerGroupState,
        decode_rows: int,
    ) -> tuple[LayerGroupState, LayerGroupState]:
        decode_residual = prefill_residual = None
        if state.residual is not None:
            decode_residual = state.residual[:decode_rows]
            prefill_residual = state.residual[decode_rows:]
        return (
            LayerGroupState(
                state.hidden[:decode_rows],
                decode_residual,
                state.next_layer,
                {layer: value[:decode_rows] for layer, value in state.draft_features.items()},
            ),
            LayerGroupState(
                state.hidden[decode_rows:],
                prefill_residual,
                state.next_layer,
                {layer: value[decode_rows:] for layer, value in state.draft_features.items()},
            ),
        )

    @staticmethod
    def create_layer_range_graph_inputs(
        seed: LayerGroupState,
    ) -> LayerGroupState:
        if seed.residual is None:
            raise RuntimeError(
                "layer-range graphs require residual state after the first stage"
            )
        from freetoken.core import get_global_ctx

        context = get_global_ctx().draft_context
        hidden = torch.zeros_like(seed.hidden)
        features = context.feature_buffers(hidden) if context is not None else {}
        return LayerGroupState(hidden, torch.zeros_like(seed.residual), draft_features=features)

    @staticmethod
    def make_layer_range_graph_state(
        inputs: LayerGroupState,
        start_layer: int,
        rows: int,
    ) -> LayerGroupState:
        return LayerGroupState(
            inputs.hidden[:rows],
            inputs.residual[:rows],
            start_layer,
            {layer: value[:rows] for layer, value in inputs.draft_features.items()
             if layer < start_layer},
        )

    @staticmethod
    def stage_layer_range_graph_inputs(
        inputs: LayerGroupState,
        state: LayerGroupState,
        rows: int,
        start_layer: int,
    ) -> None:
        if state.next_layer != start_layer or state.residual is None:
            raise ValueError(
                f"layer-range replay expected residual state at layer {start_layer}"
            )
        inputs.hidden[:rows].copy_(state.hidden)
        inputs.residual[:rows].copy_(state.residual)
        for layer, value in state.draft_features.items():
            inputs.draft_features[layer][:rows].copy_(value)

    @staticmethod
    def finish_layer_range_graph_replay(
        captured: LayerGroupState,
        rows: int,
        end_layer: int,
    ) -> LayerGroupState:
        return LayerGroupState(
            captured.hidden[:rows],
            (
                captured.residual[:rows]
                if captured.residual is not None
                else None
            ),
            end_layer,
            # The next range replay reuses this graph's pool and staging buffers.
            {layer: value[:rows].clone() for layer, value in captured.draft_features.items()},
        )

    def advance_layer_group_prefill(
        self,
        state: LayerGroupState,
        end_layer: int,
    ) -> LayerGroupState:
        if not state.next_layer < end_layer <= self.layer_group_num_layers:
            raise ValueError(
                f"invalid layer-group range [{state.next_layer}, {end_layer}) for "
                f"{self.layer_group_num_layers} layers"
            )
        from freetoken.core import get_global_ctx

        context = get_global_ctx().draft_context
        for layer_id in range(state.next_layer, end_layer):
            state.hidden, state.residual = self.model.layers.op_list[layer_id].forward(
                state.hidden, state.residual
            )
            if context is not None:
                context.record(layer_id, state.hidden, state.residual, state.draft_features)
        state.next_layer = end_layer
        if context is not None and end_layer == self.layer_group_num_layers:
            context.flush(get_global_ctx().batch, state.draft_features)
            state.draft_features = {}
        return state

    def finish_layer_group_prefill(
        self,
        state: LayerGroupState,
        output_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.next_layer != self.layer_group_num_layers:
            raise ValueError(
                "cannot finish layer-group prefill before every decoder layer ran"
            )
        if output_indices is None:
            hidden = self.model.norm.forward(state.hidden, state.residual)[0]
            return self.lm_head.forward(hidden)
        hidden = state.hidden[output_indices].contiguous()
        residual = (
            state.residual[output_indices].contiguous()
            if state.residual is not None
            else None
        )
        hidden = self.model.norm.forward(hidden, residual)[0]
        return self.lm_head.forward_selected(hidden)


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        fn_map = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}
        act_fn = fn_map.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


__all__ = [
    "BaseLLMModel",
    "GatedMLP",
    "LayerGroupState",
    "ResidualLayerGroupCausalLM",
]
