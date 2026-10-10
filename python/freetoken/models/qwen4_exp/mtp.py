"""Qwen3.8-Flash-Next native MTP draft layer (checkpoint ``mtp.*``: one QSA + MoE layer, BF16).

Row ``t`` pairs the target's streams before its final mixer, ``R[t]`` (``[hc_count*hidden]``),
with the embedding of token ``t+1`` at logical position ``t``, and predicts token ``t+2``::

    e   = fc_embedding(norm_e(embed(token[t+1])))                  # [T, hidden]
    R'  = fc_hidden(norm_h(R).view(T, hc_count, hidden)) + e       # one projection for every stream
    R'' = layer(R')                                                # QSA + MoE with hyper-connections
    h   = mixer.mix(R'')                                           # [T, hidden] for the target's head

``norm_h`` is one RMSNorm over all ``hc_count*hidden`` features. ``R''`` replaces ``R`` for the
next draft step. The embedding and head are the target's (one copy of each). The attention
history belongs to the caller's ``attend``. References: vLLM e3cae8d2
``models/qwen4_exp/nvidia/mtp.py`` and SGLang 8ff02f58 ``models/qwen4_exp_mtp.py``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

import torch
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm
from freetoken.layers.moe import MoELayer
from freetoken.models.quant_linear import make_replicated
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
from freetoken.moe.fused import fused_topk
from freetoken.utils import torch_dtype

from .attention import Qwen4ExpAttention
from .config import MTP_TAIL_STATE
from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .model import Qwen4ExpDecoderLayer

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class MTPMoE(Qwen3_5MoE):
    """The MTP layer's routed and shared experts. The routed experts are resident BF16 and
    routed here over all of them: the target's MoE backend may offload its own."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config, experts=MoELayer(
            num_experts=config.num_experts, top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size, intermediate_size=config.moe_intermediate_size,
            renormalize=True))

    def routed(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        weights, ids = fused_topk(hidden_states, router_logits, self.experts.top_k, renormalize=True)
        return self.experts.routed_forward(hidden_states, weights, ids)


class MTPLayer(Qwen4ExpDecoderLayer):
    """QSA attention with separate projections and the MTP's own experts; no PLE, no GDN."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self._is_linear = False
        self.ple = None
        self.self_attn = Qwen4ExpAttention(config, layer_id, split=True)
        self.mlp = MTPMoE(config)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)

    def forward(self, R: torch.Tensor, positions: torch.Tensor,
                attend: Callable[..., torch.Tensor]) -> torch.Tensor:
        return self.residual(R, lambda x: self.self_attn.attend(x, positions, attend))


class Qwen4ExpMTP(BaseOP):
    """The draft layer. ``config`` is the target's; the layer's dense projections are BF16
    whatever the target's dense plan, since its checkpoint weights are. Its history is the
    attention layer ``layer_id``; the target's last streams wait in slot state ``tail_state``."""

    tail_state = MTP_TAIL_STATE

    def __init__(self, config: ModelConfig) -> None:
        args = config.qwen4_args
        if config.mtp_layers != 1:
            raise ValueError(f"native MTP drafting needs one MTP layer; the checkpoint has {config.mtp_layers}")
        config = dataclasses.replace(config, dense_precision="bf16")
        self.hc_count, self.hidden_size = args.hc_count, config.hidden_size
        self.fc_embedding = make_replicated(config, config.hidden_size, config.hidden_size)
        self.fc_hidden = make_replicated(config, config.hidden_size, config.hidden_size)
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(args.stream_width, config.rms_norm_eps, 1)
        # logical layer id past the target's layers: the attention history's key
        self.layer_id = config.num_layers
        self.layer = MTPLayer(config, self.layer_id)
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    def forward(self, streams: torch.Tensor, embeddings: torch.Tensor, positions: torch.Tensor,
                attend: Callable[..., torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """One draft step over ``T`` rows: ``streams [T, hc_count*hidden]`` (the target's, or the
        previous step's), ``embeddings [T, hidden]`` of each row's next token, logical
        ``positions [T]``. Returns the head input ``[T, hidden]`` and the next step's streams."""
        streams = self.layer.forward(self._fuse(streams, embeddings), positions, attend)
        return self.hyper_connection_mixer.mix(streams)[0], streams

    def write_history(self, streams: torch.Tensor, embeddings: torch.Tensor,
                      positions: torch.Tensor, write: Callable[..., None]) -> None:
        """The attention history of rows built from the target's real streams: only the input
        fusion, the attention's stream mix and its K/V and index keys (``write(k, v, index)``)
        run; the attention itself, the experts and the head produce nothing to keep."""
        x, _ = self.layer.attn_hyper_connection.mix(self._fuse(streams, embeddings))
        self.layer.self_attn.write_history(x, positions, write)

    def _fuse(self, streams: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        rows = streams.shape[0]
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embeddings))
        r = self.pre_fc_norm_hidden.forward(streams).view(rows * self.hc_count, self.hidden_size)
        r = self.fc_hidden.forward(r).view(rows, self.hc_count, self.hidden_size) + e.unsqueeze(1)
        return r.view(rows, -1)


def load_mtp(model_path: str, config: ModelConfig, device: torch.device) -> Qwen4ExpMTP:
    """The draft layer with its checkpoint weights, BF16 and resident on ``device``."""
    from .weight import iter_mtp_weights

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        mtp = Qwen4ExpMTP(config)
    mtp.load_state_dict(dict(iter_mtp_weights(model_path, config, device)))
    return mtp


__all__ = ["MTPLayer", "MTPMoE", "Qwen4ExpMTP", "load_mtp"]
