"""Dense projection precision: ``auto`` follows the checkpoint, ``bf16`` keeps (or decodes)
the weights in BF16, ``fp8`` stores them as FP8 E4M3 with one FP32 scale per output row and
runs the existing W8A16 kernels (``kernel/triton/fp8_pertensor_linear.py``). The choice is
resolved once from the user option and the checkpoint, never from the model's name.

Resolved values: ``"source"`` (auto on a checkpoint without a recorded plan: every operator
keeps the scheme its checkpoint declares), ``"bf16"`` or ``"fp8"`` (explicit, or recorded by
the FTW converter). Only the projection role is ever quantized by the plan.
"""

from __future__ import annotations

import os

import torch

DENSE_OPTIONS = ("auto", "bf16", "fp8")
FTW_META_KEY = "dense_precision"
_FP8_MAX = 448.0


def quant_fp8_per_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row fp8-e4m3 quantization: ``w ~= weight_fp8 * scale[:, None]``."""
    wf = w.float()
    scale = (wf.abs().amax(dim=1) / _FP8_MAX).clamp(min=1e-12)
    q = (wf / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def dequant_fp8_per_row(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (q.float() * scale.float()[:, None]).to(torch.bfloat16)


def plan_dense(precision: str, role: str) -> str:
    """Storage of one operator under the resolved plan: ``projection`` weights follow an
    explicit fp8 choice; embeddings and routers keep their component precision."""
    return "fp8" if precision == "fp8" and role == "projection" else "bf16"


def resolve_dense_precision(option: str, model_path: str | None) -> str:
    """``auto`` is an FTW checkpoint's recorded plan, else ``"source"``. An explicit ``bf16``
    on an FTW that stores fp8 decodes at load; an explicit ``fp8`` needs the converter."""
    from freetoken.checkpoint.ftw import INDEX_NAME, FTWReader, is_ftw_checkpoint

    if option not in DENSE_OPTIONS:
        raise ValueError(f"dense precision must be one of {DENSE_OPTIONS}, got {option!r}")
    stored = None
    if model_path is not None and is_ftw_checkpoint(model_path):
        stored = FTWReader(model_path).meta(FTW_META_KEY)
    if option == "auto":
        return stored or "source"
    if option == "fp8" and stored not in (None, "fp8"):
        raise ValueError(
            f"{os.path.join(model_path, INDEX_NAME)} stores {stored} dense weights; "
            "convert with --dense-quant fp8 instead of serving with it"
        )
    return option


def effective_dense_precision(model_config) -> str:
    """What the dense projections actually run as: the plan, or the checkpoint's own scheme."""
    plan = getattr(model_config, "dense_precision", "source")
    if plan != "source":
        return plan
    schemes = tuple(getattr(model_config, name, "none") for name in ("attn_quant", "dense_quant", "lm_head_quant"))
    if any("fp8" in s for s in schemes) or getattr(model_config, "expert_quant", "none") == "fp8_block":
        return "fp8"
    if "nvfp4" in schemes:
        return "nvfp4"
    return "bf16"


__all__ = [
    "DENSE_OPTIONS",
    "FTW_META_KEY",
    "dequant_fp8_per_row",
    "effective_dense_precision",
    "plan_dense",
    "quant_fp8_per_row",
    "resolve_dense_precision",
]
