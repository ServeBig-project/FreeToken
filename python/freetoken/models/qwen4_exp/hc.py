"""Hyper-connection (gated residual) blocks for Qwen3.8-Flash-Next.

Every layer reads and writes ``hc_count`` residual streams packed as ``R [T, hc_count*hidden]``
(stream outer, hidden inner -- the checkpoint layout). The mix/combine bodies are the vendored
vLLM Triton kernels (``kernel/triton/hc.py``) around two GEMMs; fp32 intermediates, cast back
at the store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.kernel.triton.hc import grouped_gemma_rmsnorm, hc_combine, hc_gate_mix, hc_silu
from freetoken.layers import BaseOP
from freetoken.models.quant_linear import make_replicated

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class GroupedPlusOneRMSNorm(BaseOP):
    """Per-stream RMSNorm of an ``[T, num_groups*group]`` tensor, scaled by ``(1 + w)``.

    HF ``Qwen4ExpTextRMSNorm(dim, group_size)``. The checkpoint weight is zero-centered and
    loaded RAW: the kernel applies (1+w) in fp32 at runtime, never folded into the bf16 weight.
    """

    def __init__(self, size: int, eps: float, num_groups: int) -> None:
        self.weight = torch.empty(size)
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grouped_gemma_rmsnorm(x, self.weight, self.eps, self.num_groups)


class GatedResidual(BaseOP):
    """One hyper-connection block (HF ``Qwen4ExpTextGatedResidual``)::

        x, s = hc.mix(R)          # R [T, hc_count*hidden] -> x [T, hidden], s [T, hc_count] or None
        y    = block(x)           # attention / GDN / MoE, [T, hidden] -> [T, hidden]
        R    = hc.combine(R, y, s)

        Rn      = groupRMSNorm(R) * (1 + hc_norm.weight)        # per stream, fp32 stats
        lora, s = input_mix_weight_down_block_inject(Rn)        # merged GEMM: [lowrank | hc_count | pad]
        gate    = input_mix_weight_up(silu(lora / hc_count))
        x       = mean_i(sigmoid(gate_i) * Rn_i)
        R'_i    = R_i + 2*sigmoid(s_i / hc_count) * y

    ``s`` is the RAW inject logit slice of the merged GEMM; ``combine`` applies the activation.
    The merged weight is ``[lowrank + hc_count + pad, hc_count*hidden]`` (Qwen3.8: 320 + 4 + 12
    rows, the zero pad keeps the skinny GEMM 16-row aligned). ``use_combine=False`` is the
    top-level mixer: it owns the unmerged ``input_mix_weight_down`` and has no ``combine``.

    Weight keys (prefix stripped): ``hc_norm.weight``, ``input_mix_weight_down_block_inject.weight``
    (loader: concat of ``input_mix_weight_down``, ``block_inject_weight`` and zero rows),
    ``input_mix_weight_up.weight``.
    """

    def __init__(self, config: ModelConfig, use_combine: bool = True) -> None:
        args = config.qwen4_args
        self.hc_count = args.hc_count
        self.lowrank = args.hc_lowrank
        self.use_combine = use_combine
        width = args.stream_width
        self.hc_norm = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        if use_combine:
            pad = (-(self.lowrank + self.hc_count)) % 16
            self.input_mix_weight_down_block_inject = make_replicated(
                config, width, self.lowrank + self.hc_count + pad
            )
        else:
            self.input_mix_weight_down = make_replicated(config, width, self.lowrank)
        self.input_mix_weight_up = make_replicated(config, self.lowrank, width)

    def mix(self, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Block input ``x [T, hidden]`` and the inject logits ``s [T, hc_count]`` (None without combine)."""
        rn = self.hc_norm.forward(R)
        if self.use_combine:
            down = self.input_mix_weight_down_block_inject.forward(rn)
            lora, s = down[:, : self.lowrank], down[:, self.lowrank : self.lowrank + self.hc_count]
        else:
            lora, s = self.input_mix_weight_down.forward(rn), None
        gate = self.input_mix_weight_up.forward(hc_silu(lora, self.hc_count))
        return hc_gate_mix(rn, gate, self.hc_count), s

    def combine(self, R: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Inject the block output ``y [T, hidden]`` back into every stream of ``R``."""
        return hc_combine(R, y, s, self.hc_count)


__all__ = ["GatedResidual", "GroupedPlusOneRMSNorm"]
