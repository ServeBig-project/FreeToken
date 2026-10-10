"""NoWAG expert compute bound to the model's expert math."""

from __future__ import annotations

import os
from functools import partial

import torch

from freetoken.moe.expert_format import (
    E4M3_GROUP128_UE8M0,
    ExpertLayout,
    ExpertMath,
    ExpertMethod,
)

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.nowag import cuda_ops
from freetoken.kernel.nowag.moe_ops import (
    MAX_STRUCTURAL_DOWN_BLOCK_M,
    moe_middle_workspace_layout,
    nowag_fused_moe,
)
from freetoken.kernel.triton.dsv4.fp8_linear import (
    act_quant_fp8_inplace,
    act_quant_fp8_roundtrip,
)
# Larger route sets use the in-tree Triton aligner, which has no native limit on
# the number of physical expert rows.
from freetoken.kernel.triton.moe_align import (
    moe_align_block_size,
    moe_align_block_size_adaptive,
    moe_align_block_size_adaptive_tail64,
    uses_large_moe_align,
)

from .weights import RUNTIME_ASSIGNMENT_LAYOUT, NowagState

# Where the down input normalizer is applied: folded into the Gate/Up epilogue, or
# after the down input rounding (it must not move across that rounding point).
_GATE_UP_EPILOGUE_NORM = "gate_up_epilogue"
_DOWN_PROLOGUE_NORM = "down_prologue"


# Model activation name -> the NoWAG kernels' gated activation.
_ACTIVATION_KINDS = {
    "silu": "silu_mul",
    "swish": "silu_mul",
    "gelu": "gelu_mul",
    "gelu_tanh": "gelu_tanh_mul",
    "gelu_pytorch_tanh": "gelu_tanh_mul",
    "gpt_oss_swiglu": "swigluoai_mul",
    "swigluoai": "swigluoai_mul",
}


def check_nowag_math(math: ExpertMath) -> None:
    """Raise for expert math the NoWAG kernels cannot compute yet."""
    if math.activation not in _ACTIVATION_KINDS:
        raise NotImplementedError(
            f"NoWAG experts cannot compute activation {math.activation!r} yet"
        )
    if math.router_weight_on_input:
        raise NotImplementedError(
            "NoWAG experts cannot apply router weights on the expert input yet"
        )
    for name in ("gate_up_input_rounding", "down_input_rounding"):
        if getattr(math, name) not in (None, E4M3_GROUP128_UE8M0):
            raise NotImplementedError(
                f"NoWAG experts cannot compute {name}={getattr(math, name)!r}"
            )


def _align_routes(
    topk_ids: torch.Tensor,
    block_size: int,
    physical_expert_rows: int,
    *,
    alignment_storage: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alignment_storage is None and topk_ids.numel() <= 256:
        return cuda_ops.moe_sparse_route_align(
            topk_ids=topk_ids, block_size=block_size, num_experts=physical_expert_rows
        )
    return moe_align_block_size(
        topk_ids, block_size, physical_expert_rows, alignment_storage=alignment_storage
    )


def _round_down_input(middle: torch.Tensor) -> torch.Tensor:
    return act_quant_fp8_inplace(middle, 128)


def bind_nowag_method(
    math: ExpertMath,
    layout: ExpertLayout,
    state: NowagState,
    *,
    device: torch.device,
    backend: str,
) -> ExpertMethod:
    from .cpu import prepare_cpu_weights

    check_nowag_math(math)
    kernel_backend = os.environ.get("FREETOKEN_NOWAG_BACKEND", "auto")
    if kernel_backend not in ("triton", "auto"):
        raise ValueError("FREETOKEN_NOWAG_BACKEND must be 'triton' or 'auto'")
    if device.type == "cuda":
        cuda_ops._extension()  # compile before warmup and graph capture, not mid-forward
    run = partial(_run, math, layout, state, kernel_backend)
    # Auto picks the CUDA Exact-K48 kernels from measured profiles, which exist only for
    # D6 SiLU experts.
    exact = (kernel_backend == "auto" and state.d == 6 and math.activation in ("silu", "swish")
             and not math.router_weight_on_down_input)
    return ExpertMethod(
        run=run,
        workspace_spec=partial(_workspace_spec, layout, state),
        kernel_backends=("triton", "cuda_exact_k48") if exact else ("triton",),
        format_parameters={"d": state.d, "assignment_bits": state.assignment_bits},
        speculative_graphs=True,
        speculative_loads=True,
        prepare_cpu=partial(prepare_cpu_weights, math, state),
    )


def _workspace_spec(
    layout: ExpertLayout, state: NowagState, rows: int, top_k: int, *, bank_rows: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Scratch for ``rows`` physical token rows over ``bank_rows`` addressable bank rows:
    the bound over every backend the kernel may pick for this geometry (two compute
    slabs, the largest Down alignment tile, either adaptive task queue)."""
    if rows == 0:
        return {}
    routes = rows * top_k
    middle_rows = max(
        moe_middle_workspace_layout(
            num_routes=routes,
            num_experts=bank_rows,
            alignment_block_m=MAX_STRUCTURAL_DOWN_BLOCK_M,
            physical_intermediate_size=state.intermediate_size,
            structural_down=True,
            compute_slabs=2,
            adaptive_m_tiles=True,
            caller_owned_alignment_storage=False,
            adaptive_residual_policy=policy,
        ).total_rows
        for policy in ("bm16", "tail64")
    )
    return {
        "middle": ((middle_rows, state.intermediate_size), torch.bfloat16),
        "route_output": ((routes, layout.hidden_size), torch.bfloat16),
    }


def _run(
    math: ExpertMath,
    layout: ExpertLayout,
    state: NowagState,
    kernel_backend: str,
    x: torch.Tensor,
    slots: torch.Tensor,
    topk_weights: torch.Tensor,
    banks: dict[str, torch.Tensor],
    shared: dict[str, torch.Tensor],
    *,
    workspace=None,
    out: torch.Tensor | None = None,
    prefill: bool = False,
    sort_rows: int | None = None,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty_like(x) if out is None else out
    # The aligners drop negative (padding) routes, leaving their outputs unwritten:
    # send them to row 0 with zero weight instead.
    padding = slots < 0
    slots = slots.clamp_min(0)
    topk_weights = topk_weights.masked_fill(padding, 0.0)
    # The rounded input is staged in ``out`` until the final sum overwrites it.
    if (math.gate_up_input_rounding is not None and out is not None
            and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()):
        raise ValueError("NoWAG out must not share storage with x")
    gate_up_input_transform = None
    if math.gate_up_input_rounding is not None:
        gate_up_input_transform = partial(act_quant_fp8_roundtrip, block=128, output=out)
    codebook = shared["codebook"].unsqueeze(0)
    return nowag_fused_moe(
        hidden_states=x,
        gate_codebook=codebook,
        gate_packed_assignments=banks["gate_assignments"],
        gate_input_norm=banks["gate_input_norm"],
        gate_output_norm=banks["gate_output_norm"],
        gate_in_features=x.shape[1],
        up_codebook=codebook,
        up_packed_assignments=banks["up_assignments"],
        up_input_norm=banks["up_input_norm"],
        up_output_norm=banks["up_output_norm"],
        up_in_features=x.shape[1],
        down_codebook=codebook,
        down_packed_assignments=banks["down_assignments"],
        down_input_norm=banks["down_input_norm"],
        down_output_norm=banks["down_output_norm"],
        down_in_features=banks["gate_output_norm"].shape[1],
        down_input_group_start_lane=state.down_start_lane,
        topk_weights=topk_weights,
        topk_ids=slots,
        model_num_experts=layout.num_experts,
        group_size=state.d,
        assignment_bits=state.assignment_bits,
        assignment_layout=RUNTIME_ASSIGNMENT_LAYOUT,
        validate_route_ids=False,
        structural_down=True,
        gate_up_backend=kernel_backend,
        down_backend=kernel_backend,
        output=out,
        middle_workspace=workspace["middle"] if workspace else None,
        route_output_workspace=(
            workspace["route_output"][: slots.numel()] if workspace else None
        ),
        swiglu_limit=math.activation_limit,
        router_weight_on_middle=math.router_weight_on_down_input,
        activation_kind=_ACTIVATION_KINDS[math.activation],
        activation_alpha=math.activation_alpha,
        gate_bias=banks.get("gate_bias"),
        up_bias=banks.get("up_bias"),
        down_bias=banks.get("down_bias"),
        gate_up_input_rounding=math.gate_up_input_rounding or "none",
        down_input_rounding=math.down_input_rounding or "none",
        down_norm_placement=(
            _DOWN_PROLOGUE_NORM if math.down_input_rounding else _GATE_UP_EPILOGUE_NORM
        ),
        gate_up_input_transform=gate_up_input_transform,
        middle_transform=_round_down_input if math.down_input_rounding is not None else None,
        align_routes=_align_routes,
        align_routes_adaptive=(
            moe_align_block_size_adaptive
            if uses_large_moe_align(slots.numel())
            else None
        ),
        align_routes_adaptive_tail64=(
            moe_align_block_size_adaptive_tail64
            if uses_large_moe_align(slots.numel())
            else None
        ),
        caller_owned_alignment_storage=False,
        sum_routes=moe_sum_reduce_triton,
    )
