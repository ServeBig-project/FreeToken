"""K/V cache encoding (``--kv-dtype``): ``auto`` keeps the pool's native format, ``bf16``
and ``int8`` ask for a codec the pool family declares (``BaseKVCachePool.kv_codecs``).

INT8 is symmetric per token and per KV head, K and V separately, scale in BF16::

    m = max(abs(x[head_dim]))
    s = BF16(m / 127)            # all-zero vector: s = 1
    q = clamp(round_half_even(x / float32(s)), -127, 127).int8
    read: float32(q) * float32(s)

Quantization divides by the stored (rounded) scale, so GPU and host copies of the same
page decode identically; the reader applies the scale inside the attention kernel.
"""

from __future__ import annotations

import torch

KV_OPTIONS = ("auto", "bf16", "int8")
INT8_MAX = 127.0


def resolve_kv_dtype(option: str, pool_cls) -> str:
    """``"bf16"`` or ``"int8"`` for ``pool_cls`` (``kv_codecs`` names what it can store)."""
    if option not in KV_OPTIONS:
        raise ValueError(f"kv dtype must be one of {KV_OPTIONS}, got {option!r}")
    codecs = pool_cls.kv_codecs
    if option == "auto":
        return codecs[0]
    if option not in codecs:
        raise ValueError(
            f"{pool_cls.__name__} stores K/V as {'/'.join(codecs)} only; --kv-dtype {option} "
            "is not available for this model"
        )
    return option


def kv_storage_name(codec: str, dtype: torch.dtype) -> str:
    """What the pool stores for ``codec``: int8, or the plain format in the model ``dtype``."""
    if codec == "int8":
        return codec
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]


def quantize_kv_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x [..., head_dim]`` -> ``(q int8 [..., head_dim], scale bf16 [...])``."""
    amax = x.float().abs().amax(dim=-1)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / INT8_MAX).to(torch.bfloat16)
    q = torch.round(x.float() / scale.float().unsqueeze(-1)).clamp_(-INT8_MAX, INT8_MAX)
    return q.to(torch.int8), scale


__all__ = ["INT8_MAX", "KV_OPTIONS", "quantize_kv_int8", "resolve_kv_dtype"]
