"""Dense projection precision: ``auto`` follows the checkpoint, ``bf16`` keeps the source
weights in BF16, ``fp8`` stores them as FP8 E4M3 with one FP32 scale per output row and
runs the existing W8A16 kernels (``kernel/triton/fp8_pertensor_linear.py``). The choice is
resolved once from the user option and the checkpoint, never from the model's name.
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


def resolve_dense_precision(option: str, model_path: str | None) -> str:
    """``"bf16"`` or ``"fp8"``. ``auto`` is an FTW checkpoint's recorded precision, else the
    source's (the supported sources ship BF16 dense weights). An explicit choice on an FTW
    must match what it stores: the converter is where a precision is applied."""
    from freetoken.checkpoint.ftw import INDEX_NAME, FTWReader, is_ftw_checkpoint

    if option not in DENSE_OPTIONS:
        raise ValueError(f"dense precision must be one of {DENSE_OPTIONS}, got {option!r}")
    stored = None
    if model_path is not None and is_ftw_checkpoint(model_path):
        stored = FTWReader(model_path).meta(FTW_META_KEY, "bf16")
    if option == "auto":
        return stored or "bf16"
    if stored is not None and stored != option:
        raise ValueError(
            f"{os.path.join(model_path, INDEX_NAME)} stores {stored} dense weights; "
            f"reconvert with --dense-quant {option} instead of serving with it"
        )
    return option


__all__ = ["DENSE_OPTIONS", "FTW_META_KEY", "quant_fp8_per_row", "resolve_dense_precision"]
