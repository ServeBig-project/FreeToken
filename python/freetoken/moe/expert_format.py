"""Expert weight formats: bank layouts and the bound compute method every MoE path calls.

The model's expert component states its math (:class:`ExpertMath`); a format declares its
banks here and implements the compute. Movement (resident, slot cache, streaming) stays
in the callers, which only hand the bound method bank views whose rows the routes index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import torch

# quant_format -> bank names, in registration order: the single place a format's bank
# layout is declared. The cache machinery (copy_missing, the prefill double buffers,
# bank_views) iterates banks in this order and set_bank_sources validates against it.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights
    "bf16": ("gate_up", "down"),
    # DeepSeek-V3-style 128x128 block-fp8 experts (Qwen3.5-FP8): fp8-e4m3 weights +
    # bf16 per-block weight_scale_inv. gate_up [L*E, 2I, H] fp8 + gate_up_scale
    # [L*E, 2I//128, H//128] bf16; down [L*E, H, I] fp8 + down_scale [L*E, H//128, I//128].
    # Half the host/cache footprint of bf16; the grouped GEMM (kernel/triton/fp8_blockscale_moe)
    # reads the routed fp8 rows directly and dequantizes in the K-loop (no bf16 materialization).
    "fp8_block": ("gate_up", "gate_up_scale", "down", "down_scale"),
    # native GGUF Q4_0 experts: packed block bytes per output row, dequantized inside
    # the borrowed ggml MoE kernels. gate_up [L*E, 2I, H//32*18], down [L*E, H, I//32*18].
    "q4_0": ("gate_up", "down"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp16 globals (w1/w3 carry distinct
    # globals, and folding them into the e4m3 block scales would underflow)
    "nvfp4": (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    ),
    # pre-tiled layouts for the borrowed kernels; the globals are folded into the
    # block scales at repack time and collapse to [L*E] GPU-resident alpha vectors
    # (set_alphas), so they are not banks
    "nvfp4_marlin": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    "nvfp4_b12x": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # gpt-oss mxfp4, transposed split-K layout (N innermost): per-expert blocks_t
    # [K//2, N] (uint8), scales_t [K//32, N] (uint8 e8m0), bias [N]. No folded alphas
    # (scales are a bank); split-K GEMV decode + transposed _t grouped prefill.
    "mxfp4_triton": (
        "gate_up_blocks",
        "gate_up_scales",
        "gate_up_bias",
        "down_blocks",
        "down_scales",
        "down_bias",
    ),
    # DeepSeek-V4 FP4: packed e2m1 codes + e8m0 per-32 block scales, no global scale
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels.
    "ds_fp4": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # NoWAG: three projections, each with word-major packed assignments [W, N]
    # and input/output normalizers. The model-wide codebook is a shared tensor.
    "nowag": (
        "gate_assignments",
        "gate_input_norm",
        "gate_output_norm",
        "up_assignments",
        "up_input_norm",
        "up_output_norm",
        "down_assignments",
        "down_input_norm",
        "down_output_norm",
    ),
}

# Banks a format carries only when the model has them (expert biases).
_OPTIONAL_BANKS: dict[str, tuple[str, ...]] = {"nowag": ("gate_bias", "up_bias", "down_bias")}

# Speculative-decoding capabilities by ModelConfig.expert_quant: experts whose compute
# is captured in the draft/verify CUDA graphs, and experts whose rows the measured
# missing-expert loads and verify prefetch move (they copy whole bank rows of any
# layout and cost them by measured copy time).
SPECULATIVE_GRAPH_FORMATS = ("none", "nvfp4", "nowag")
SPECULATIVE_LOAD_FORMATS = ("none", "nowag")

# Dynamic per-token, per-128-lane E4M3 quantize/dequantize with UE8M0 scales
# (DeepSeek-V4 expert inputs).
E4M3_GROUP128_UE8M0 = "dynamic_e4m3_per_token_group128_ue8m0"


@dataclass(frozen=True)
class ExpertMath:
    """The routed-expert computation the model defines, independent of weight format."""

    activation: str = "silu"
    activation_alpha: float = 1.702
    activation_limit: float | None = None
    router_weight_on_input: bool = False
    # The router weight scales the down projection's input (after the gated
    # activation, before any down input rounding) instead of its output.
    router_weight_on_down_input: bool = False
    gate_up_input_rounding: str | None = None
    down_input_rounding: str | None = None


@dataclass(frozen=True)
class ExpertLayout:
    format: str
    hidden_size: int
    intermediate_size: int
    num_experts: int


@dataclass(frozen=True)
class ExpertMethod:
    """``run(x, expert_rows, route_weights, banks, shared, *, workspace, out, ...)`` and
    ``workspace_spec(rows, top_k, *, bank_rows)`` bound to one math/layout.

    ``run`` also takes the caller's existing kernel-selection hints: ``prefill`` (the
    batch runs the extend path), ``sort_rows`` (rows a grouped kernel sorts over) and
    ``expert_map`` (``expert_rows`` are logical ids, mapped to bank rows in-kernel).
    """

    run: Callable[..., torch.Tensor]
    workspace_spec: Callable[..., dict[str, tuple[tuple[int, ...], torch.dtype]]]
    # The grouped kernel sorts logical ids and maps them through ``expert_map``,
    # so a resident group sorts over E experts rather than every cache slot.
    logical_sort: bool = False
    # Reported by /v1/cache/status: the kernels this method may dispatch to and the
    # format's own encoding parameters.
    kernel_backends: tuple[str, ...] = ()
    format_parameters: dict = field(default_factory=dict)


def expert_math(layer) -> ExpertMath:
    """Read the math a MoE layer was built with (format scalars ride on ``extra_attrs``)."""
    return ExpertMath(
        activation=layer.activation,
        activation_alpha=getattr(layer, "hidden_act_alpha", 1.702),
        activation_limit=getattr(layer, "swiglu_limit", None),
        router_weight_on_input=layer.apply_router_weight_on_input,
        router_weight_on_down_input=getattr(layer, "router_weight_on_down_input", False),
        gate_up_input_rounding=getattr(layer, "gate_up_input_rounding", None),
        down_input_rounding=getattr(layer, "down_input_rounding", None),
    )


def _into(out: torch.Tensor | None, result: torch.Tensor) -> torch.Tensor:
    if out is None or out is result:
        return result
    return out.copy_(result)


def _no_workspace(rows: int, top_k: int, *, bank_rows: int) -> dict:
    return {}


def _run_bf16(math, resident, x, rows, weights, banks, shared, *, workspace=None, out=None,
              prefill=False, sort_rows=None, expert_map=None):
    from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

    args = (x, banks["gate_up"], banks["down"], weights, rows, math.activation,
            math.router_weight_on_input)
    # Resident bf16 decode also runs the grouped kernel (it overwrites ``x``).
    if prefill or resident:
        return _into(out, fused_experts_impl(*args, expert_map=expert_map))
    return _into(out, fused_experts_decode_impl(*args))


def _run_fp8_block(math, resident, x, rows, weights, banks, shared, *, workspace=None, out=None,
                   prefill=False, sort_rows=None, expert_map=None):
    from freetoken.moe.fused_fp8_block import (
        fused_experts_decode_fp8_block,
        fused_experts_fp8_block,
    )

    w = (banks["gate_up"], banks["gate_up_scale"], banks["down"], banks["down_scale"])
    if prefill:
        return _into(out, fused_experts_fp8_block(
            x, *w, weights, rows, sort_rows, math.activation, math.router_weight_on_input,
            expert_map,
        ))
    return _into(out, fused_experts_decode_fp8_block(
        x, *w, weights, rows, math.activation, math.router_weight_on_input,
    ))


def _run_nvfp4_tiled(fmt, math, resident, x, rows, weights, banks, shared, *, workspace=None,
                     out=None, prefill=False, sort_rows=None, expert_map=None):
    # Borrowed W4A16 fused MoE -- Marlin (vLLM, sm_80-99) or b12x (flashinfer, sm_120)
    # over their pre-tiled banks; one kernel serves prefill and decode, with the
    # movement-matched per-row global scales passed as the *_alpha banks.
    from freetoken.moe.nvfp4_backends import b12x_fused_experts, marlin_fused_experts

    fused = marlin_fused_experts if fmt == "nvfp4_marlin" else b12x_fused_experts
    return _into(out, fused(
        x, banks["gate_up_packed"], banks["gate_up_scale"], banks["gate_up_alpha"],
        banks["down_packed"], banks["down_scale"], banks["down_alpha"],
        weights, rows, math.activation, math.router_weight_on_input,
    ))


def _run_nvfp4(math, resident, x, rows, weights, banks, shared, *, workspace=None, out=None,
               prefill=False, sort_rows=None, expert_map=None):
    # Triton inline-dequant kernels over the native ModelOpt rows: no BF16 copy of the
    # experts. None == "no clamp" everywhere else in the repo (mxfp4 maps it to +inf).
    limit = float("inf") if math.activation_limit is None else math.activation_limit
    w = tuple(banks[name] for name in _BANK_SCHEMAS["nvfp4"])
    if prefill:
        from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

        return _into(out, fused_experts_nvfp4(
            x, *w, weights, rows, sort_rows, math.activation, math.router_weight_on_input,
            math.activation_alpha, limit, expert_map,
        ))
    from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

    return _into(out, fused_experts_decode_nvfp4_marlin(
        x, *w, weights, rows, math.activation, math.router_weight_on_input,
        math.activation_alpha, limit,
    ))


def _run_q4_0(math, resident, x, rows, weights, banks, shared, *, workspace=None, out=None,
              prefill=False, sort_rows=None, expert_map=None):
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

    return _into(out, fused_experts_gguf_q4_0(
        x, banks["gate_up"], banks["down"], weights, rows, math.activation
    ))


def _run_mxfp4(math, resident, x, rows, weights, banks, shared, *, workspace=None, out=None,
               prefill=False, sort_rows=None, expert_map=None):
    from freetoken.moe.fused_mxfp4 import (
        MXFP4_DECODE_MAX_TOKENS,
        run_mxfp4_prefill_experts_t,
        run_mxfp4_splitk_decode_experts,
    )

    # The resident layer picks the kernel by token count; offload by batch phase.
    grouped = x.shape[0] > MXFP4_DECODE_MAX_TOKENS if resident else prefill
    run = run_mxfp4_prefill_experts_t if grouped else run_mxfp4_splitk_decode_experts
    return _into(out, run(
        x, weights, rows, *(banks[name] for name in _BANK_SCHEMAS["mxfp4_triton"]),
        top_k=rows.shape[1],
        hidden_act_alpha=math.activation_alpha,
        swiglu_limit=math.activation_limit,
        **({"expert_map": expert_map} if grouped else {}),
    ))


def _run_ds_fp4(math, resident, x, rows, weights, banks, shared, *, workspace=None, out=None,
                prefill=False, sort_rows=None, expert_map=None):
    # Grouped inline-dequant GEMM for streaming prefill chunks; per-route dequant GEMV
    # for decode and the sparse small-chunk slot path (sorting the whole slot cache
    # would drown in padding). These kernels apply the router weight to the down
    # output, not to its input as DSV4 defines; kept as is.
    w = tuple(banks[name] for name in _BANK_SCHEMAS["ds_fp4"])
    if prefill and sort_rows is not None:
        from freetoken.moe.fused_ds_fp4 import routed_experts_fp4_prefill

        return _into(out, routed_experts_fp4_prefill(
            x, rows, weights, *w, math.activation_limit, sort_rows, expert_map,
        ))
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    return _into(out, routed_experts_fp4(x, rows, weights, *w, math.activation_limit))


_RUNS = {
    "bf16": _run_bf16,
    "fp8_block": _run_fp8_block,
    "nvfp4": _run_nvfp4,
    "nvfp4_marlin": partial(_run_nvfp4_tiled, "nvfp4_marlin"),
    "nvfp4_b12x": partial(_run_nvfp4_tiled, "nvfp4_b12x"),
    "q4_0": _run_q4_0,
    "mxfp4_triton": _run_mxfp4,
    "ds_fp4": _run_ds_fp4,
}


_LOGICAL_SORT = ("bf16", "nvfp4", "fp8_block", "mxfp4_triton", "ds_fp4")


def bind_expert_method(
    math: ExpertMath,
    layout: ExpertLayout,
    format_state,
    *,
    device: torch.device,
    backend: str,
) -> ExpertMethod:
    """Check that ``layout.format`` can compute ``math`` on ``device`` and bind it.

    ``backend`` is FreeToken's resolved ``--moe-backend``; ``"fused"`` means the
    experts are resident in GPU memory rather than streamed through a slot cache.
    """
    if layout.format == "nowag":
        from freetoken.moe.nowag.method import bind_nowag_method

        return bind_nowag_method(math, layout, format_state, device=device, backend=backend)
    if layout.format not in _RUNS:
        raise ValueError(f"no expert compute method for format {layout.format!r}")
    run = partial(_RUNS[layout.format], math, backend == "fused")
    return ExpertMethod(run, _no_workspace, logical_sort=layout.format in _LOGICAL_SORT,
                        kernel_backends=(layout.format,))


__all__ = [
    "E4M3_GROUP128_UE8M0",
    "ExpertLayout",
    "ExpertMath",
    "ExpertMethod",
    "bind_expert_method",
    "expert_math",
]
