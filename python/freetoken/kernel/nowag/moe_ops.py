"""Expert-routed NoWag lookup kernels for vLLM FusedMoE.

This module implements the TP-local serving contract directly from
NoWag codebooks and row-local packed assignments.  It never reconstructs a
dense expert weight.  vLLM performs the token-to-expert alignment once; both
lookup kernels consume the same aligned tickets.  The structural path aligns
at Down's macro-tile while Gate/Up subdivides it:

* kernel A computes independent gate/up projections and fused SiLU-multiply;
* kernel B computes the down projection and applies the router weight.

The Triton path accepts the configured codeword width and pads its reduction
lanes to the next power of two.  The specialized CUDA Exact-K48 path retains
its audited D=6 wire format.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal

import torch

from .assignment_layout import AssignmentLayout, assignment_layout_info
from .moe_activation import (
    DOWN_PROLOGUE_NORM,
    GATE_UP_EPILOGUE_NORM,
    SILU_MUL,
    ActivationKind,
    ActivationRounding,
    DownNormPlacement,
    MoeActivationMath,
    NO_ACTIVATION_ROUNDING,
)

if TYPE_CHECKING:
    from .moe_tuning import MoeCudaLaunchPlan

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in CPU-only envs.
    triton = None
    tl = None


GROUP_SIZE: Final = 6
ASSIGNMENT_BITS: Final = 12
PAD_D: Final = 8
EXACT_K48: Final = 48

# Legacy A and B share these launch parameters.  The structural path aligns at
# its independent Down macro-tile and lets Gate/Up subdivide that alignment.
BLOCK_M: Final = 16
MAX_BLOCK_M: Final = 64
BLOCK_N: Final = 32
BLOCK_G: Final = 8
GROUP_SIZE_M: Final = 8
MAX_STRUCTURAL_DOWN_BLOCK_M: Final = 128
MAX_STRUCTURAL_DOWN_BLOCK_N: Final = 128
ROUTE_ALIGNMENT_SMALL_CAP: Final = 1024

MoeSchedule = Literal["baseline", "route_density_grouped"]
GateUpBackend = Literal["triton", "cuda_pipeline", "cuda_exact_k48", "auto"]
DownBackend = Literal["triton", "cuda_exact_k48", "auto"]
BASELINE_SCHEDULE: Final[MoeSchedule] = "baseline"
ROUTE_DENSITY_GROUPED_SCHEDULE: Final[MoeSchedule] = "route_density_grouped"
TRITON_GATE_UP_BACKEND: Final[GateUpBackend] = "triton"
CUDA_PIPELINE_GATE_UP_BACKEND: Final[GateUpBackend] = "cuda_pipeline"
CUDA_EXACT_K48_GATE_UP_BACKEND: Final[GateUpBackend] = "cuda_exact_k48"
TRITON_DOWN_BACKEND: Final[DownBackend] = "triton"
CUDA_EXACT_K48_DOWN_BACKEND: Final[DownBackend] = "cuda_exact_k48"
AUTO_GATE_UP_BACKEND: Final[GateUpBackend] = "auto"
AUTO_DOWN_BACKEND: Final[DownBackend] = "auto"

UINT32_MASK: Final = (1 << 32) - 1
_ACTIVATION_IDS: Final = {"silu_mul": 0, "gelu_mul": 1, "gelu_tanh_mul": 2, "swigluoai_mul": 3}


@lru_cache(maxsize=4096)
def _resolve_cuda_moe_auto_decision(
    requested_gate_backend: GateUpBackend,
    requested_down_backend: DownBackend,
    device_index: int,
    dtype: torch.dtype,
    group_size: int,
    assignment_bits: int,
    codebook_size: int,
    model_num_experts: int,
    physical_expert_rows: int,
    hidden_size: int,
    intermediate_size: int,
    physical_intermediate_size: int,
    pad_down_to_k48: bool,
    down_input_group_start_lane: int,
    assignment_layout: AssignmentLayout,
    structural_down: bool,
    top_k: int,
    num_tokens: int,
    activation_math: MoeActivationMath,
) -> tuple[GateUpBackend, DownBackend, MoeCudaLaunchPlan | None]:
    """Resolve one complete automatic backend and launch-plan decision."""
    from .execution_profile import cuda_hardware_key, select_cuda_moe_backend
    from .moe_tuning import select_cuda_moe_launch_plan

    device = torch.device("cuda", device_index)
    activation_profile = activation_math.profile_identity()
    hardware_key = cuda_hardware_key(device)
    selected_backend = select_cuda_moe_backend(
        device=device,
        dtype=dtype,
        group_size=group_size,
        assignment_bits=assignment_bits,
        codebook_size=codebook_size,
        num_experts=model_num_experts,
        physical_expert_rows=physical_expert_rows,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        physical_intermediate_size=physical_intermediate_size,
        pad_down_to_k48=pad_down_to_k48,
        down_input_group_start_lane=down_input_group_start_lane,
        assignment_layout=assignment_layout,
        structural_down=structural_down,
        top_k=top_k,
        num_tokens=num_tokens,
        **activation_profile,
    )
    resolved_gate_backend = (
        selected_backend
        if requested_gate_backend == AUTO_GATE_UP_BACKEND
        else requested_gate_backend
    )
    resolved_down_backend = (
        selected_backend
        if requested_down_backend == AUTO_DOWN_BACKEND
        else requested_down_backend
    )
    if (
        resolved_gate_backend != CUDA_EXACT_K48_GATE_UP_BACKEND
        and resolved_down_backend != CUDA_EXACT_K48_DOWN_BACKEND
    ):
        return resolved_gate_backend, resolved_down_backend, None

    launch_plan = select_cuda_moe_launch_plan(
        hardware_key=hardware_key,
        dtype=dtype,
        group_size=group_size,
        assignment_bits=assignment_bits,
        codebook_size=codebook_size,
        num_experts=model_num_experts,
        physical_expert_rows=physical_expert_rows,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        physical_intermediate_size=physical_intermediate_size,
        pad_down_to_k48=pad_down_to_k48,
        down_input_group_start_lane=down_input_group_start_lane,
        assignment_layout=assignment_layout,
        structural_down=structural_down,
        top_k=top_k,
        activation_kind=activation_profile["activation_kind"],
        gate_up_input_rounding=activation_profile[
            "gate_up_input_rounding"
        ],
        swiglu_limit=activation_profile["swiglu_limit"],
        down_input_rounding=activation_profile["down_input_rounding"],
        down_norm_placement=activation_profile["down_norm_placement"],
        num_tokens=num_tokens,
    )
    return resolved_gate_backend, resolved_down_backend, launch_plan


def _validate_padded_backend_pair(
    gate_up_backend: GateUpBackend,
    down_backend: DownBackend,
) -> None:
    """Require one coherent producer/consumer contract for a padded middle.

    A physical-K48 middle has only two supported runtime interpretations:
    Triton/Triton consumes the logical view, while Exact/Exact consumes the
    complete padded storage.  A mixed pair would make the producer and
    consumer disagree about the prefix/tail contract.  ``auto`` must be
    resolved before this check; keeping that invariant here avoids relying on
    today's profile entries selecting both stages together by coincidence.
    """
    supported = {
        (TRITON_GATE_UP_BACKEND, TRITON_DOWN_BACKEND),
        (CUDA_EXACT_K48_GATE_UP_BACKEND, CUDA_EXACT_K48_DOWN_BACKEND),
    }
    pair = (gate_up_backend, down_backend)
    if pair not in supported:
        raise ValueError(
            "pad_down_to_k48 requires a coherent Triton/Triton or "
            "cuda_exact_k48/cuda_exact_k48 backend pair after auto "
            f"resolution, got {pair!r}"
        )


@dataclass(frozen=True)
class StructuralDownConfig:
    """Shape-derived launch configuration for the structural Down path."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int = 2


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def structural_down_block_m(
    num_tokens: int,
    top_k: int,
    num_experts: int,
) -> int:
    """Choose a Down macro-tile without consulting runtime routing results.

    The ceiling average bounds uniform-routing padding to less than one extra
    power-of-two block.  A minimum of 16 keeps the dot tile Tensor-Core
    friendly, while 128 caps per-program register pressure.  This is a shape
    rule, not a per-model tuning table.
    """
    if num_tokens <= 0 or top_k <= 0 or num_experts <= 0:
        raise ValueError("num_tokens, top_k, and num_experts must be positive")
    routes_per_expert = (num_tokens * top_k + num_experts - 1) // num_experts
    return min(
        MAX_STRUCTURAL_DOWN_BLOCK_M,
        max(BLOCK_M, _next_power_of_two(routes_per_expert)),
    )


def structural_down_config(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    out_features: int,
) -> StructuralDownConfig:
    """Return generic, stage-specific Down BM/BN/warp parameters."""
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    block_m = structural_down_block_m(num_tokens, top_k, num_experts)
    # Bound the FP32 accumulator footprint generically instead of attaching a
    # lucky BN to a known model shape.  In particular BM=128 pairs with BN=64,
    # while BM<=64 can use BN=128.
    max_block_n = min(MAX_STRUCTURAL_DOWN_BLOCK_N, 8192 // block_m)
    block_n = max(
        16,
        _next_power_of_two(min(out_features, max_block_n)),
    )
    num_warps = 8 if block_m * block_n >= 8192 else 4
    return StructuralDownConfig(
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
    )


def required_structural_middle_rows(
    num_routes: int,
    num_experts: int,
    down_block_m: int,
) -> int:
    """Return vLLM alignment's static upper bound for sorted middle rows.

    This mirrors ``moe_align_block_size`` without reading its device-side
    ``num_tokens_post_padded`` scalar, so callers can preallocate a graph-safe
    workspace without a CPU synchronization.
    """
    if num_routes <= 0 or num_experts <= 0 or down_block_m <= 0:
        raise ValueError(
            "num_routes, num_experts, and down_block_m must be positive"
        )
    max_rows = num_routes + num_experts * (down_block_m - 1)
    if num_routes < num_experts:
        max_rows = min(num_routes * down_block_m, max_rows)
    return max_rows


def route_alignment_workspace_int32_elements(
    *,
    num_routes: int,
    num_experts: int,
    block_m: int,
) -> int:
    """Return storage for graph-static external route alignment.

    External aligners reserve one extra sentinel expert.  The returned flat
    int32 capacity covers the three live outputs plus all small/large-path
    scratch.  The route-count threshold is part of this public callback
    contract, so sizing and execution cannot select different layouts.
    """
    if num_routes <= 0 or num_experts <= 0 or block_m <= 0:
        raise ValueError("num_routes, num_experts, and block_m must be positive")
    effective_experts = num_experts + 1
    if num_routes < effective_experts:
        sorted_capacity = num_routes * block_m
    else:
        sorted_capacity = num_routes + effective_experts * (block_m - 1)
    expert_id_capacity = (
        sorted_capacity + block_m - 1
    ) // block_m
    # sorted, expert_ids, num_post_pad, fill_counter, cumsum; the large path
    # additionally owns counts[effective_experts].
    elements = (
        sorted_capacity
        + expert_id_capacity
        + 2 * effective_experts
        + 2
    )
    if num_routes > ROUTE_ALIGNMENT_SMALL_CAP:
        elements += effective_experts
    return elements


@dataclass(frozen=True)
class MoeMiddleWorkspaceLayout:
    """Row layout of one persistent ``[rows, physical_intermediate]`` slab."""

    middle_rows: int
    compute_rows: int
    task_metadata_rows: int
    alignment_storage_rows: int
    alignment_storage_int32_elements: int

    @property
    def task_metadata_start(self) -> int:
        return self.compute_rows

    @property
    def alignment_storage_start(self) -> int:
        return self.compute_rows + self.task_metadata_rows

    @property
    def total_rows(self) -> int:
        return self.alignment_storage_start + self.alignment_storage_rows


def moe_middle_workspace_layout(
    *,
    num_routes: int,
    num_experts: int,
    alignment_block_m: int,
    physical_intermediate_size: int,
    structural_down: bool,
    compute_slabs: int,
    adaptive_m_tiles: bool,
    caller_owned_alignment_storage: bool,
    adaptive_residual_policy: str = "bm16",
) -> MoeMiddleWorkspaceLayout:
    """Return the single authoritative middle-workspace layout.

    Both supported middle dtypes (BF16 and FP16) occupy two bytes.  Tail
    regions are rounded to whole middle rows so every returned slice remains
    graph-static and naturally aligned.
    """
    if physical_intermediate_size <= 0:
        raise ValueError("physical_intermediate_size must be positive")
    if compute_slabs not in (1, 2):
        raise ValueError("compute_slabs must be 1 or 2")
    middle_rows = (
        required_structural_middle_rows(
            num_routes,
            num_experts,
            alignment_block_m,
        )
        if structural_down
        else num_routes
    )
    task_metadata_rows = 0
    if adaptive_m_tiles:
        from .cuda_ops import adaptive_task_metadata_rows

        task_metadata_rows = adaptive_task_metadata_rows(
            num_routes=num_routes,
            num_experts=num_experts,
            physical_intermediate_size=physical_intermediate_size,
            adaptive_residual_policy=adaptive_residual_policy,
        )
    alignment_elements = 0
    alignment_rows = 0
    if caller_owned_alignment_storage:
        alignment_elements = route_alignment_workspace_int32_elements(
            num_routes=num_routes,
            num_experts=num_experts,
            block_m=alignment_block_m,
        )
        row_bytes = physical_intermediate_size * torch.bfloat16.itemsize
        alignment_rows = (
            alignment_elements * torch.int32.itemsize + row_bytes - 1
        ) // row_bytes
    return MoeMiddleWorkspaceLayout(
        middle_rows=middle_rows,
        compute_rows=middle_rows * compute_slabs,
        task_metadata_rows=task_metadata_rows,
        alignment_storage_rows=alignment_rows,
        alignment_storage_int32_elements=alignment_elements,
    )


def _structural_num_m_blocks(
    *,
    expert_id_capacity: int,
    num_routes: int,
    middle_rows: int,
    block_m: int,
) -> int:
    """Bound a structural launch by every statically allocated row buffer."""
    return min(
        expert_id_capacity,
        num_routes,
        middle_rows // block_m,
    )


if triton is not None:

    @triton.jit
    def _moe_program_ids(
        NUM_PID_N: tl.constexpr,
        GROUPED_SCHEDULE: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Map a launch ID to one aligned-route tile and one output tile."""
        if GROUPED_SCHEDULE:
            pid = tl.program_id(axis=0)
            num_pid_m = tl.num_programs(axis=0) // NUM_PID_N
            num_pid_in_group: tl.constexpr = GROUP_SIZE_M * NUM_PID_N
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_in_group = pid % num_pid_in_group
            pid_m = first_pid_m + pid_in_group % group_size_m
            pid_n = pid_in_group // group_size_m
            return pid_m, pid_n
        return tl.program_id(axis=0), tl.program_id(axis=1)

    @triton.jit
    def _decode_8x12bit_ids(
        assignments_ptr,
        expert,
        offsets_n,
        n_size,
        group_block,
        num_groups,
        num_words,
        stride_ae,
        stride_an,
        stride_aw,
        BLOCK_N: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """Decode BLOCK_G IDs with exactly three uint32 loads per 8 IDs."""
        tl.static_assert(BLOCK_G % 8 == 0)
        packets_per_block: tl.constexpr = BLOCK_G // 8

        packet_offsets = (
            group_block * packets_per_block + tl.arange(0, packets_per_block)
        )
        packet_group_bases = packet_offsets * 8
        word_bases = packet_offsets * 3
        valid_packet = (offsets_n[:, None] < n_size) & (
            packet_group_bases[None, :] < num_groups
        )
        row_offsets = (
            expert * stride_ae + offsets_n[:, None].to(tl.int64) * stride_an
        )

        word0 = tl.load(
            assignments_ptr + row_offsets + word_bases[None, :] * stride_aw,
            mask=valid_packet & (word_bases[None, :] < num_words),
            other=0,
        ).to(tl.uint32)
        word1 = tl.load(
            assignments_ptr
            + row_offsets
            + (word_bases[None, :] + 1) * stride_aw,
            mask=valid_packet & ((word_bases[None, :] + 1) < num_words),
            other=0,
        ).to(tl.uint32)
        word2 = tl.load(
            assignments_ptr
            + row_offsets
            + (word_bases[None, :] + 2) * stride_aw,
            mask=valid_packet & ((word_bases[None, :] + 2) < num_words),
            other=0,
        ).to(tl.uint32)

        id_lane = tl.arange(0, 8)
        bit_offset = id_lane * 12
        word_lane = bit_offset // 32
        shift = bit_offset % 32
        low = tl.where(
            word_lane[None, None, :] == 0,
            word0[:, :, None],
            tl.where(
                word_lane[None, None, :] == 1,
                word1[:, :, None],
                word2[:, :, None],
            ),
        )
        high = tl.where(
            word_lane[None, None, :] == 0,
            word1[:, :, None],
            tl.where(
                word_lane[None, None, :] == 1,
                word2[:, :, None],
                0,
            ),
        )
        crosses_word = shift + 12 > 32
        safe_left_shift = tl.where(crosses_word, 32 - shift, 0)
        high_part = tl.where(
            crosses_word[None, None, :],
            high << safe_left_shift[None, None, :],
            0,
        )
        ids = ((low >> shift[None, None, :]) | high_part) & 0xFFF
        return tl.reshape(ids, BLOCK_N, BLOCK_G)

    @triton.jit
    def _decode_generic_12bit_ids(
        assignments_ptr,
        expert,
        offsets_n,
        n_size,
        group_block,
        num_groups,
        num_words,
        stride_ae,
        stride_an,
        stride_aw,
        BLOCK_N: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """Decode each ID independently from its low/crossing word.

        This is the controlled ablation for the 8-ID/3-word packet decoder.
        Storage, lookup, dot shape, and launch parameters remain identical.
        """
        offsets_g = group_block * BLOCK_G + tl.arange(0, BLOCK_G)
        bit_positions = offsets_g * 12
        word_indices = bit_positions // 32
        shifts = bit_positions % 32
        valid = (offsets_n[:, None] < n_size) & (
            offsets_g[None, :] < num_groups
        )
        row_offsets = (
            expert * stride_ae + offsets_n[:, None].to(tl.int64) * stride_an
        )
        low = tl.load(
            assignments_ptr + row_offsets + word_indices[None, :] * stride_aw,
            mask=valid & (word_indices[None, :] < num_words),
            other=0,
        ).to(tl.uint32)
        crosses_word = shifts + 12 > 32
        high = tl.load(
            assignments_ptr
            + row_offsets
            + (word_indices[None, :] + 1) * stride_aw,
            mask=valid
            & crosses_word[None, :]
            & ((word_indices[None, :] + 1) < num_words),
            other=0,
        ).to(tl.uint32)
        safe_left_shift = tl.where(crosses_word, 32 - shifts, 0)
        high_part = tl.where(
            crosses_word[None, :],
            high << safe_left_shift[None, :],
            0,
        )
        return ((low >> shifts[None, :]) | high_part) & 0xFFF

    @triton.jit
    def _nowag_moe_gate_up_kernel(
        hidden_ptr,
        gate_codebook_ptr,
        gate_assignments_ptr,
        gate_input_norm_ptr,
        gate_output_norm_ptr,
        up_codebook_ptr,
        up_assignments_ptr,
        up_input_norm_ptr,
        up_output_norm_ptr,
        down_input_norm_ptr,
        gate_bias_ptr,
        up_bias_ptr,
        sorted_ticket_ids_ptr,
        expert_ids_ptr,
        num_tickets_post_padded_ptr,
        middle_ptr,
        num_routes,
        top_k,
        n_size,
        k_size,
        num_groups,
        gate_codebook_size,
        up_codebook_size,
        num_assignment_words,
        stride_hm,
        stride_hk,
        stride_gce,
        stride_gcc,
        stride_gcd,
        stride_gae,
        stride_gan,
        stride_gaw,
        stride_gine,
        stride_gink,
        stride_gone,
        stride_gonn,
        stride_uce,
        stride_ucc,
        stride_ucd,
        stride_uae,
        stride_uan,
        stride_uaw,
        stride_uine,
        stride_uink,
        stride_uone,
        stride_uonn,
        stride_dine,
        stride_dink,
        stride_mm,
        stride_mn,
        COMPUTE_TYPE: tl.constexpr,
        PACKED_12_BLOCK_DECODE: tl.constexpr,
        CLAMP_SWIGLU: tl.constexpr,
        SWIGLU_LIMIT: tl.constexpr,
        ACTIVATION: tl.constexpr,
        ACTIVATION_ALPHA: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        PREAPPLY_DOWN_NORM: tl.constexpr,
        SORTED_MIDDLE_LAYOUT: tl.constexpr,
        ALIGNMENT_BLOCK_RATIO: tl.constexpr,
        GROUPED_SCHEDULE: tl.constexpr,
        NUM_PID_N: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_G: tl.constexpr,
        D: tl.constexpr,
        PAD_D: tl.constexpr,
    ):
        """Expert-aware gate/up lookup-dot followed by act(gate) * up.

        ``ACTIVATION``: 0 SiLU, 1 GELU (erf), 2 GELU (tanh), 3 SwiGLU-OAI
        ``gate * sigmoid(alpha * gate) * (up + 1)``.  Bias is ``[E, N]``.
        """
        pid_m, pid_n = _moe_program_ids(
            NUM_PID_N=NUM_PID_N,
            GROUPED_SCHEDULE=GROUPED_SCHEDULE,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )
        num_tickets_post_padded = tl.load(num_tickets_post_padded_ptr)
        if pid_m * BLOCK_M >= num_tickets_post_padded:
            return

        ticket_positions = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        tickets = tl.load(sorted_ticket_ids_ptr + ticket_positions).to(tl.int64)
        valid_ticket = tickets < num_routes
        token_ids = tickets // top_k
        # Structural Down aligns at a larger macro-tile.  Every Gate/Up
        # sub-tile maps back to the owning macro-tile's expert ID.
        expert = tl.load(
            expert_ids_ptr + pid_m // ALIGNMENT_BLOCK_RATIO
        ).to(tl.int64)

        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_n = offsets_n < n_size
        lanes = tl.arange(0, PAD_D)
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for group_block in range(0, tl.cdiv(num_groups, BLOCK_G)):
            offsets_g = group_block * BLOCK_G + tl.arange(0, BLOCK_G)
            offsets_k = offsets_g[:, None] * D + lanes[None, :]
            valid_gd = (
                (offsets_g[:, None] < num_groups)
                & (lanes[None, :] < D)
                & (offsets_k < k_size)
            )
            activation = tl.load(
                hidden_ptr
                + token_ids[:, None, None] * stride_hm
                + offsets_k[None, :, :].to(tl.int64) * stride_hk,
                mask=valid_ticket[:, None, None] & valid_gd[None, :, :],
                other=0.0,
            )

            if PACKED_12_BLOCK_DECODE:
                gate_ids = _decode_8x12bit_ids(
                    gate_assignments_ptr,
                    expert,
                    offsets_n,
                    n_size,
                    group_block,
                    num_groups,
                    num_assignment_words,
                    stride_gae,
                    stride_gan,
                    stride_gaw,
                    BLOCK_N=BLOCK_N,
                    BLOCK_G=BLOCK_G,
                ).to(tl.int64)
            else:
                gate_ids = _decode_generic_12bit_ids(
                    gate_assignments_ptr,
                    expert,
                    offsets_n,
                    n_size,
                    group_block,
                    num_groups,
                    num_assignment_words,
                    stride_gae,
                    stride_gan,
                    stride_gaw,
                    BLOCK_N=BLOCK_N,
                    BLOCK_G=BLOCK_G,
                ).to(tl.int64)
            gate_scale = tl.load(
                gate_input_norm_ptr
                + expert * stride_gine
                + offsets_k.to(tl.int64) * stride_gink,
                mask=valid_gd,
                other=0.0,
            )
            gate_weight = tl.load(
                gate_codebook_ptr
                + expert * stride_gce
                + gate_ids[:, :, None] * stride_gcc
                + lanes[None, None, :].to(tl.int64) * stride_gcd,
                mask=valid_n[:, None, None]
                & (offsets_g[None, :, None] < num_groups)
                & (lanes[None, None, :] < D)
                & (offsets_k[None, :, :] < k_size)
                & (gate_ids[:, :, None] < gate_codebook_size),
                other=0.0,
            )
            gate_activation = tl.reshape(
                (activation * gate_scale[None, :, :]).to(COMPUTE_TYPE),
                BLOCK_M,
                BLOCK_G * PAD_D,
            )
            gate_weight = tl.reshape(
                gate_weight, BLOCK_N, BLOCK_G * PAD_D
            )
            gate_acc += tl.dot(
                gate_activation,
                tl.trans(gate_weight),
                input_precision="ieee",
            )

            if PACKED_12_BLOCK_DECODE:
                up_ids = _decode_8x12bit_ids(
                    up_assignments_ptr,
                    expert,
                    offsets_n,
                    n_size,
                    group_block,
                    num_groups,
                    num_assignment_words,
                    stride_uae,
                    stride_uan,
                    stride_uaw,
                    BLOCK_N=BLOCK_N,
                    BLOCK_G=BLOCK_G,
                ).to(tl.int64)
            else:
                up_ids = _decode_generic_12bit_ids(
                    up_assignments_ptr,
                    expert,
                    offsets_n,
                    n_size,
                    group_block,
                    num_groups,
                    num_assignment_words,
                    stride_uae,
                    stride_uan,
                    stride_uaw,
                    BLOCK_N=BLOCK_N,
                    BLOCK_G=BLOCK_G,
                ).to(tl.int64)
            up_scale = tl.load(
                up_input_norm_ptr
                + expert * stride_uine
                + offsets_k.to(tl.int64) * stride_uink,
                mask=valid_gd,
                other=0.0,
            )
            up_weight = tl.load(
                up_codebook_ptr
                + expert * stride_uce
                + up_ids[:, :, None] * stride_ucc
                + lanes[None, None, :].to(tl.int64) * stride_ucd,
                mask=valid_n[:, None, None]
                & (offsets_g[None, :, None] < num_groups)
                & (lanes[None, None, :] < D)
                & (offsets_k[None, :, :] < k_size)
                & (up_ids[:, :, None] < up_codebook_size),
                other=0.0,
            )
            up_activation = tl.reshape(
                (activation * up_scale[None, :, :]).to(COMPUTE_TYPE),
                BLOCK_M,
                BLOCK_G * PAD_D,
            )
            up_weight = tl.reshape(up_weight, BLOCK_N, BLOCK_G * PAD_D)
            up_acc += tl.dot(
                up_activation,
                tl.trans(up_weight),
                input_precision="ieee",
            )

        gate_output_norm = tl.load(
            gate_output_norm_ptr + expert * stride_gone + offsets_n * stride_gonn,
            mask=valid_n,
            other=0.0,
        )
        up_output_norm = tl.load(
            up_output_norm_ptr + expert * stride_uone + offsets_n * stride_uonn,
            mask=valid_n,
            other=0.0,
        )

        gate = gate_acc * gate_output_norm[None, :]
        up = up_acc * up_output_norm[None, :]
        if HAS_BIAS:
            bias_offsets = expert * n_size + offsets_n
            gate += tl.load(gate_bias_ptr + bias_offsets, mask=valid_n, other=0.0)[None, :]
            up += tl.load(up_bias_ptr + bias_offsets, mask=valid_n, other=0.0)[None, :]
        # Match dense FusedMoE's GEMM1 output precision before activation.
        gate_fp32 = gate.to(COMPUTE_TYPE).to(tl.float32)
        up_fp32 = up.to(COMPUTE_TYPE).to(tl.float32)
        if CLAMP_SWIGLU:
            gate_fp32 = tl.minimum(gate_fp32, SWIGLU_LIMIT)
            up_fp32 = tl.minimum(
                tl.maximum(up_fp32, -SWIGLU_LIMIT), SWIGLU_LIMIT
            )
        if ACTIVATION == 0:
            middle = gate_fp32 * tl.sigmoid(gate_fp32) * up_fp32
        elif ACTIVATION == 1:
            middle = 0.5 * gate_fp32 * (1.0 + tl.erf(gate_fp32 * 0.7071067811865476)) * up_fp32
        elif ACTIVATION == 2:
            inner = 0.7978845608028654 * (gate_fp32 + 0.044715 * gate_fp32 * gate_fp32 * gate_fp32)
            # tanh(inner) == 2 * sigmoid(2 * inner) - 1
            middle = gate_fp32 * tl.sigmoid(2.0 * inner) * up_fp32
        else:
            middle = gate_fp32 * tl.sigmoid(ACTIVATION_ALPHA * gate_fp32) * (up_fp32 + 1.0)
        if PREAPPLY_DOWN_NORM:
            # Match the legacy sequence exactly: first round the SiLU*up
            # result as if it had been stored/reloaded, then multiply the
            # BF16/FP16 Down input norm and round again before the Down dot.
            down_input_norm = tl.load(
                down_input_norm_ptr
                + expert * stride_dine
                + offsets_n * stride_dink,
                mask=valid_n,
                other=0.0,
            )
            middle = (
                middle.to(COMPUTE_TYPE) * down_input_norm[None, :]
            ).to(COMPUTE_TYPE)
        if SORTED_MIDDLE_LAYOUT:
            middle_rows = ticket_positions
        else:
            middle_rows = tickets
        tl.store(
            middle_ptr
            + middle_rows[:, None] * stride_mm
            + offsets_n[None, :].to(tl.int64) * stride_mn,
            middle,
            mask=valid_ticket[:, None] & valid_n[None, :],
        )

    @triton.jit
    def _nowag_moe_down_kernel(
        middle_ptr,
        down_codebook_ptr,
        down_assignments_ptr,
        down_input_norm_ptr,
        down_output_norm_ptr,
        down_bias_ptr,
        topk_weights_ptr,
        sorted_ticket_ids_ptr,
        expert_ids_ptr,
        num_tickets_post_padded_ptr,
        route_output_ptr,
        num_routes,
        n_size,
        k_size,
        num_groups,
        codebook_size,
        num_assignment_words,
        stride_mm,
        stride_mk,
        stride_dce,
        stride_dcc,
        stride_dcd,
        stride_dae,
        stride_dan,
        stride_daw,
        stride_dine,
        stride_dink,
        stride_done,
        stride_donn,
        stride_rom,
        stride_ron,
        COMPUTE_TYPE: tl.constexpr,
        PACKED_12_BLOCK_DECODE: tl.constexpr,
        PREAPPLIED_DOWN_NORM: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        SORTED_MIDDLE_LAYOUT: tl.constexpr,
        GROUPED_SCHEDULE: tl.constexpr,
        NUM_PID_N: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_G: tl.constexpr,
        D: tl.constexpr,
        PAD_D: tl.constexpr,
        INPUT_GROUP_START_LANE: tl.constexpr = 0,
    ):
        """Expert-aware down lookup-dot with router weight in FP32."""
        pid_m, pid_n = _moe_program_ids(
            NUM_PID_N=NUM_PID_N,
            GROUPED_SCHEDULE=GROUPED_SCHEDULE,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )
        num_tickets_post_padded = tl.load(num_tickets_post_padded_ptr)
        if pid_m * BLOCK_M >= num_tickets_post_padded:
            return

        ticket_positions = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        tickets = tl.load(sorted_ticket_ids_ptr + ticket_positions).to(tl.int64)
        valid_ticket = tickets < num_routes
        expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if SORTED_MIDDLE_LAYOUT:
            middle_rows = ticket_positions
        else:
            middle_rows = tickets

        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_n = offsets_n < n_size
        lanes = tl.arange(0, PAD_D)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for group_block in range(0, tl.cdiv(num_groups, BLOCK_G)):
            offsets_g = group_block * BLOCK_G + tl.arange(0, BLOCK_G)
            offsets_k = (
                offsets_g[:, None] * D
                + lanes[None, :]
                - INPUT_GROUP_START_LANE
            )
            valid_gd = (
                (offsets_g[:, None] < num_groups)
                & (lanes[None, :] < D)
                & (offsets_k >= 0)
                & (offsets_k < k_size)
            )
            activation = tl.load(
                middle_ptr
                + middle_rows[:, None, None] * stride_mm
                + offsets_k[None, :, :].to(tl.int64) * stride_mk,
                mask=valid_ticket[:, None, None] & valid_gd[None, :, :],
                other=0.0,
            )
            if PACKED_12_BLOCK_DECODE:
                ids = _decode_8x12bit_ids(
                    down_assignments_ptr,
                    expert,
                    offsets_n,
                    n_size,
                    group_block,
                    num_groups,
                    num_assignment_words,
                    stride_dae,
                    stride_dan,
                    stride_daw,
                    BLOCK_N=BLOCK_N,
                    BLOCK_G=BLOCK_G,
                ).to(tl.int64)
            else:
                ids = _decode_generic_12bit_ids(
                    down_assignments_ptr,
                    expert,
                    offsets_n,
                    n_size,
                    group_block,
                    num_groups,
                    num_assignment_words,
                    stride_dae,
                    stride_dan,
                    stride_daw,
                    BLOCK_N=BLOCK_N,
                    BLOCK_G=BLOCK_G,
                ).to(tl.int64)
            weight = tl.load(
                down_codebook_ptr
                + expert * stride_dce
                + ids[:, :, None] * stride_dcc
                + lanes[None, None, :].to(tl.int64) * stride_dcd,
                mask=valid_n[:, None, None]
                & (offsets_g[None, :, None] < num_groups)
                & (lanes[None, None, :] < D)
                & (offsets_k[None, :, :] >= 0)
                & (offsets_k[None, :, :] < k_size)
                & (ids[:, :, None] < codebook_size),
                other=0.0,
            )
            if PREAPPLIED_DOWN_NORM:
                activation = tl.reshape(
                    activation,
                    BLOCK_M,
                    BLOCK_G * PAD_D,
                )
            else:
                input_norm = tl.load(
                    down_input_norm_ptr
                    + expert * stride_dine
                    + offsets_k.to(tl.int64) * stride_dink,
                    mask=valid_gd,
                    other=0.0,
                )
                activation = tl.reshape(
                    (activation * input_norm[None, :, :]).to(COMPUTE_TYPE),
                    BLOCK_M,
                    BLOCK_G * PAD_D,
                )
            weight = tl.reshape(weight, BLOCK_N, BLOCK_G * PAD_D)
            accumulator += tl.dot(
                activation,
                tl.trans(weight),
                input_precision="ieee",
            )

        output_norm = tl.load(
            down_output_norm_ptr + expert * stride_done + offsets_n * stride_donn,
            mask=valid_n,
            other=0.0,
        )
        router_weight = tl.load(
            topk_weights_ptr + tickets,
            mask=valid_ticket,
            other=0.0,
        )
        accumulator *= output_norm[None, :]
        if HAS_BIAS:
            accumulator += tl.load(
                down_bias_ptr + expert * n_size + offsets_n, mask=valid_n, other=0.0
            )[None, :]
        accumulator *= router_weight[:, None]
        tl.store(
            route_output_ptr
            + tickets[:, None] * stride_rom
            + offsets_n[None, :].to(tl.int64) * stride_ron,
            accumulator,
            mask=valid_ticket[:, None] & valid_n[None, :],
        )


# Static weights are range-checked once when C < 4096.  A weak reference
# prevents a recycled CUDA address from inheriting another tensor's result.
_validated_assignments: dict[tuple[object, ...], weakref.ReferenceType[torch.Tensor]] = {}
_validated_routes: dict[tuple[object, ...], weakref.ReferenceType[torch.Tensor]] = {}


def _packed_words(
    in_features: int,
    input_group_start_lane: int = 0,
    group_size: int = GROUP_SIZE,
) -> int:
    groups = (
        input_group_start_lane + in_features + group_size - 1
    ) // group_size
    return (groups * ASSIGNMENT_BITS + 31) // 32


def padded_down_k48_width(
    in_features: int,
    input_group_start_lane: int = 0,
) -> int:
    """Return the physical Down K that preserves a split D6 codeword.

    The logical local shard begins ``input_group_start_lane`` values into its
    first global D6 codeword.  Prefixing that many zero activations restores
    the global codeword boundary; rounding the resulting row to K48 lets the
    exact kernel consume only complete eight-codeword packets.  This is a
    shape rule, not a model-specific table.
    """
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if not 0 <= input_group_start_lane < GROUP_SIZE:
        raise ValueError(
            "input_group_start_lane must be in "
            f"[0, {GROUP_SIZE - 1}], got {input_group_start_lane}"
        )
    occupied = input_group_start_lane + in_features
    return ((occupied + EXACT_K48 - 1) // EXACT_K48) * EXACT_K48


def route_density_block_m(
    num_tokens: int,
    top_k: int,
    num_experts: int,
) -> int:
    """Choose a power-of-two M tile from average routes per expert.

    The decision uses shape metadata only, so it does not synchronize routing
    results back to the host.  Sparse/decode workloads remain at the baseline
    tile while dense prefill workloads may grow to 32 or 64 rows.
    """
    if num_tokens <= 0 or top_k <= 0 or num_experts <= 0:
        raise ValueError("num_tokens, top_k, and num_experts must be positive")
    routes_per_expert = max(1, (num_tokens * top_k) // num_experts)
    block_m = 1 << (routes_per_expert.bit_length() - 1)
    return min(MAX_BLOCK_M, max(BLOCK_M, block_m))


def _launch_grid(
    num_m_blocks: int,
    num_pid_n: int,
    grouped_schedule: bool,
) -> tuple[int, ...]:
    if grouped_schedule:
        return (num_m_blocks * num_pid_n,)
    return (num_m_blocks, num_pid_n)


def _validate_assignment_range(
    packed: torch.Tensor,
    *,
    in_features: int,
    codebook_size: int,
    name: str,
    assignment_layout: AssignmentLayout,
    input_group_start_lane: int = 0,
    group_size: int = GROUP_SIZE,
) -> None:
    """Validate a sub-4096 codebook once without constructing dense weights."""
    if codebook_size == 1 << ASSIGNMENT_BITS:
        return

    groups = (
        input_group_start_lane + in_features + group_size - 1
    ) // group_size
    key = (
        packed.device.type,
        packed.device.index,
        packed.data_ptr(),
        packed._version,
        tuple(packed.shape),
        assignment_layout,
        codebook_size,
        groups,
        group_size,
        input_group_start_lane,
    )
    cached = _validated_assignments.get(key)
    if cached is not None and cached() is packed:
        return

    layout_info = assignment_layout_info(packed, assignment_layout)
    logical = (
        packed
        if assignment_layout == "row_major"
        else packed.permute(0, 2, 1)
    )
    rows = logical.reshape(-1, layout_info.num_words)
    words = rows.to(torch.int64) & UINT32_MASK
    # Padding one zero word makes the following-word load valid for the final
    # non-crossing ID as well; only crossing lanes consume it.
    words = torch.cat((words, torch.zeros_like(words[:, :1])), dim=1)
    bit_positions = (
        torch.arange(groups, device=packed.device, dtype=torch.int64)
        * ASSIGNMENT_BITS
    )
    word_indices = torch.div(bit_positions, 32, rounding_mode="floor")
    shifts = bit_positions % 32
    low = words.index_select(1, word_indices)
    high = words.index_select(1, word_indices + 1)
    values = low >> shifts
    crosses = shifts + ASSIGNMENT_BITS > 32
    values |= torch.where(
        crosses,
        high << (32 - shifts),
        torch.zeros_like(high),
    )
    maximum = int((values & 0xFFF).max().item())
    if maximum >= codebook_size:
        raise ValueError(
            f"{name} assignment ID {maximum} exceeds codebook size "
            f"{codebook_size}"
        )
    _validated_assignments[key] = weakref.ref(packed)


def _validate_route_range(topk_ids: torch.Tensor, num_experts: int) -> None:
    """Synchronize only once for an unchanged routing tensor.

    vLLM's router already guarantees this range in serving.  The explicit
    check keeps the standalone black-box API safe, while the identity/version
    cache prevents repeated benchmark calls from paying two device reductions
    and a host synchronization.
    """
    key = (
        topk_ids.device.type,
        topk_ids.device.index,
        topk_ids.data_ptr(),
        topk_ids._version,
        tuple(topk_ids.shape),
        num_experts,
    )
    cached = _validated_routes.get(key)
    if cached is not None and cached() is topk_ids:
        return

    min_expert = int(topk_ids.min().item())
    max_expert = int(topk_ids.max().item())
    if min_expert < 0 or max_expert >= num_experts:
        raise ValueError(
            f"topk_ids must be in [0, {num_experts - 1}], got "
            f"min={min_expert}, max={max_expert}"
        )
    _validated_routes[key] = weakref.ref(topk_ids)


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_projection(
    *,
    name: str,
    codebook: torch.Tensor,
    packed_assignments: torch.Tensor,
    input_norm: torch.Tensor,
    output_norm: torch.Tensor,
    in_features: int,
    expected_experts: int,
    expected_out_features: int | None,
    dtype: torch.dtype,
    device: torch.device,
    assignment_layout: AssignmentLayout,
    group_size: int,
    input_group_start_lane: int = 0,
    expected_assignment_words: int | None = None,
) -> int:
    if codebook.ndim != 3 or codebook.shape[0] not in (1, expected_experts):
        raise ValueError(
            f"{name}_codebook must be [B, C, d] with B=1 or "
            f"B=E={expected_experts}, "
            f"got {tuple(codebook.shape)}"
        )
    if codebook.shape[2] != group_size:
        raise ValueError(
            f"{name}_codebook codeword width must be {group_size}, "
            f"got {codebook.shape[2]}"
        )
    codebook_size = codebook.shape[1]
    if not 1 <= codebook_size <= 1 << ASSIGNMENT_BITS:
        raise ValueError(f"{name} codebook size must be in [1, 4096]")
    if in_features <= 0:
        raise ValueError(f"{name}_in_features must be positive")
    if not 0 <= input_group_start_lane < group_size:
        raise ValueError(
            f"{name}_input_group_start_lane must be in [0, "
            f"{group_size - 1}], got {input_group_start_lane}"
        )
    if output_norm.ndim != 2 or output_norm.shape[0] != expected_experts:
        raise ValueError(
            f"{name}_output_norm must be [E, N], got {tuple(output_norm.shape)}"
        )
    out_features = output_norm.shape[1]
    if expected_out_features is not None and out_features != expected_out_features:
        raise ValueError(
            f"{name} out_features must be {expected_out_features}, "
            f"got {out_features}"
        )

    _require_tensor(
        codebook, name=f"{name}_codebook", device=device, dtype=dtype
    )
    expected_words = (
        _packed_words(in_features, input_group_start_lane, group_size)
        if expected_assignment_words is None
        else expected_assignment_words
    )
    if expected_words < _packed_words(
        in_features, input_group_start_lane, group_size
    ):
        raise ValueError(
            f"{name} expected_assignment_words cannot truncate the logical "
            "assignment stream"
        )
    expected_assignment_shape = (
        (expected_experts, out_features, expected_words)
        if assignment_layout == "row_major"
        else (expected_experts, expected_words, out_features)
    )
    _require_tensor(
        packed_assignments,
        name=f"{name}_packed_assignments",
        device=device,
        dtype=torch.int32,
        shape=expected_assignment_shape,
    )
    _require_tensor(
        input_norm,
        name=f"{name}_input_norm",
        device=device,
        dtype=dtype,
        shape=(expected_experts, in_features),
    )
    _require_tensor(
        output_norm,
        name=f"{name}_output_norm",
        device=device,
        dtype=dtype,
        shape=(expected_experts, out_features),
    )
    _validate_assignment_range(
        packed_assignments,
        in_features=in_features,
        codebook_size=codebook_size,
        name=name,
        assignment_layout=assignment_layout,
        input_group_start_lane=input_group_start_lane,
        group_size=group_size,
    )
    return out_features


@torch.inference_mode()
def nowag_fused_moe(
    *,
    hidden_states: torch.Tensor,
    gate_codebook: torch.Tensor,
    gate_packed_assignments: torch.Tensor,
    gate_input_norm: torch.Tensor,
    gate_output_norm: torch.Tensor,
    gate_in_features: int,
    up_codebook: torch.Tensor,
    up_packed_assignments: torch.Tensor,
    up_input_norm: torch.Tensor,
    up_output_norm: torch.Tensor,
    up_in_features: int,
    down_codebook: torch.Tensor,
    down_packed_assignments: torch.Tensor,
    down_input_norm: torch.Tensor,
    down_output_norm: torch.Tensor,
    down_in_features: int,
    down_input_group_start_lane: int = 0,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    model_num_experts: int | None = None,
    group_size: int = GROUP_SIZE,
    assignment_bits: int = ASSIGNMENT_BITS,
    assignment_layout: AssignmentLayout = "row_major",
    output: torch.Tensor | None = None,
    middle_workspace: torch.Tensor | None = None,
    route_output_workspace: torch.Tensor | None = None,
    validate_route_ids: bool = True,
    use_block12_decoder: bool = False,
    schedule: MoeSchedule = BASELINE_SCHEDULE,
    structural_down: bool = True,
    gate_up_backend: GateUpBackend = TRITON_GATE_UP_BACKEND,
    down_backend: DownBackend = TRITON_DOWN_BACKEND,
    pad_down_to_k48: bool = False,
    gate_up_debug_trace: torch.Tensor | None = None,
    cuda_launch_plan: MoeCudaLaunchPlan | None = None,
    swiglu_limit: float | None = None,
    activation_kind: ActivationKind = SILU_MUL,
    activation_alpha: float = 1.702,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
    gate_up_input_rounding: ActivationRounding = NO_ACTIVATION_ROUNDING,
    down_input_rounding: ActivationRounding = NO_ACTIVATION_ROUNDING,
    down_norm_placement: DownNormPlacement | None = None,
    gate_up_input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    middle_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    align_routes: Callable[
        [torch.Tensor, int, int],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    | None = None,
    align_routes_adaptive: Callable[
        [
            torch.Tensor,
            int,
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    | None = None,
    align_routes_adaptive_tail64: Callable[
        [
            torch.Tensor,
            int,
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    | None = None,
    caller_owned_alignment_storage: bool = False,
    sum_routes: Callable[[torch.Tensor, torch.Tensor], object] | None = None,
) -> torch.Tensor:
    """Run NoWag FusedMoE and return ``[tokens, hidden]``.

    The keyword API deliberately mirrors the test-owned ``PackedMoERequest``
    adapter.  Gate, up, and down use independent codebooks, packed assignments,
    and norms. Packed assignments may use row-major ``[E,N,W]`` or word-major
    ``[E,W,N]`` physical storage; conversion belongs in checkpoint/loading
    preparation, never this timed call. ``schedule="baseline"`` retains the
    original ``BM=16`` 2-D
    launch. ``schedule="route_density_grouped"`` selects ``BM`` from average
    routes per expert and groups neighboring M programs for weight locality.
    ``structural_down=True`` aligns once at an independent Down macro-tile,
    stores the intermediate in expert-sorted order, and can fold the Down
    input norm into Gate/Up's epilogue.  Set it to ``False`` for the legacy
    A/B path.
    ``gate_up_backend="cuda_pipeline"`` selects the experimental SM80+
    two-stage shared-memory lookup/MMA pipeline for route-dense tiles.  Small-M
    tiles retain the Triton kernel because they cannot hide pipeline startup.
    Under TP, Gate/Up are already output-sharded and Down returns this rank's
    partial output for vLLM to all-reduce. ``down_input_group_start_lane``
    preserves a global D-wide codeword when a row-shard boundary cuts through
    it. EP and zero-point variants remain outside this contract.

    ``pad_down_to_k48=True`` stores the logical expert middle after a zero
    prefix at ``down_input_group_start_lane`` and rounds its physical width to
    K48.  It is intended for an exact Gate/Up + exact Down pair when TP cuts a
    global D6 codeword; no dense weight or inference-time repacking is added.

    ``swiglu_limit`` clamps gate (max) and up (both signs) before the
    activation.  ``activation_kind`` other than ``silu_mul`` and the optional
    contiguous ``[E, N]`` biases run on the Triton kernels only; ``auto``
    resolves to Triton for them.  The two rounding fields describe
    optional storage round-trips around the expert math; their callables do
    the actual conversion without tying this kernel to one serving runtime.
    When Down is rounded, its normalizer belongs in ``down_prologue`` so the
    order is activation, rounding, normalizer, matrix multiplication.
    ``align_routes`` and ``sum_routes`` let another serving runtime provide
    the two routing primitives without changing the lookup kernels.  When an
    adaptive launch plan is active, ``align_routes_adaptive`` may replace the
    align call and fill the supplied BM64 queue, residual BM16 queue, and two
    device counts in place while returning the ordinary align three-tuple.
    ``align_routes_adaptive_tail64`` is the explicit A/B callback that folds
    residuals of 49..63 rows into the same BM64 queue.
    With ``caller_owned_alignment_storage=True``, the external callback also
    receives an ``alignment_storage=`` keyword naming a flat int32 slice
    carved from the persistent middle-workspace tail.  The default preserves
    the original callback contract and allocation behavior.

    ``model_num_experts`` separates the model's expert count from the number
    of physical rows in a serving cache.  Profiles key both counts because
    cache strides affect measured performance; generic tile choices use the
    model count.  Tensor validation, route IDs, and alignment continue to use
    the physical row count.  They are identical without cache indirection.

    ``gate_up_debug_trace`` is an internal profiling hook for the CUDA
    pipeline.  Supplying an int64 CUDA buffer selects a separately compiled
    instrumented kernel; the normal ``None`` path has no tracing branches.
    """
    if triton is None:
        raise RuntimeError("Triton is required for nowag_fused_moe")
    if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda:
        raise ValueError("hidden_states must be a CUDA tensor")
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be [M, H], got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("hidden_states must use bfloat16 or float16")
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")
    if not 1 <= group_size <= 16:
        raise ValueError(f"group_size must be in [1, 16], got {group_size}")
    if assignment_bits != ASSIGNMENT_BITS:
        raise ValueError(
            f"assignment_bits must be {ASSIGNMENT_BITS}, got {assignment_bits}"
        )
    if not 0 <= down_input_group_start_lane < group_size:
        raise ValueError(
            "down_input_group_start_lane must be in "
            f"[0, {group_size - 1}], got {down_input_group_start_lane}"
        )
    if not isinstance(pad_down_to_k48, bool):
        raise ValueError("pad_down_to_k48 must be a bool")
    if not isinstance(use_block12_decoder, bool):
        raise ValueError("use_block12_decoder must be a bool")
    if not isinstance(structural_down, bool):
        raise ValueError("structural_down must be a bool")
    if not isinstance(caller_owned_alignment_storage, bool):
        raise ValueError("caller_owned_alignment_storage must be a bool")
    if down_norm_placement is None:
        down_norm_placement = (
            GATE_UP_EPILOGUE_NORM if structural_down else DOWN_PROLOGUE_NORM
        )
    activation_math = MoeActivationMath(
        activation_kind=activation_kind,
        activation_alpha=activation_alpha,
        gate_up_input_rounding=gate_up_input_rounding,
        swiglu_limit=swiglu_limit,
        down_input_rounding=down_input_rounding,
        down_norm_placement=down_norm_placement,
    )
    if (
        activation_math.gate_up_input_rounding == NO_ACTIVATION_ROUNDING
    ) != (gate_up_input_transform is None):
        raise ValueError(
            "gate_up_input_transform must be provided exactly when Gate/Up "
            "input rounding is enabled"
        )
    if (
        activation_math.down_input_rounding == NO_ACTIVATION_ROUNDING
    ) != (middle_transform is None):
        raise ValueError(
            "middle_transform must be provided exactly when Down input "
            "rounding is enabled"
        )
    if gate_up_input_transform is not None:
        transformed_input = gate_up_input_transform(hidden_states)
        if not isinstance(transformed_input, torch.Tensor):
            raise ValueError("gate_up_input_transform must return a tensor")
        if (
            transformed_input.shape != hidden_states.shape
            or transformed_input.device != hidden_states.device
            or transformed_input.dtype != hidden_states.dtype
            or not transformed_input.is_contiguous()
        ):
            raise ValueError(
                "gate_up_input_transform must preserve shape, CUDA device, "
                "dtype, and contiguity"
            )
        hidden_states = transformed_input
    if (gate_bias is None) != (up_bias is None):
        raise ValueError("gate_bias and up_bias must be provided together")
    if activation_kind != SILU_MUL or gate_bias is not None or down_bias is not None:
        if gate_up_backend == AUTO_GATE_UP_BACKEND:
            gate_up_backend = TRITON_GATE_UP_BACKEND
        if down_backend == AUTO_DOWN_BACKEND:
            down_backend = TRITON_DOWN_BACKEND
        if (gate_up_backend, down_backend) != (TRITON_GATE_UP_BACKEND, TRITON_DOWN_BACKEND):
            raise ValueError("non-SiLU activations and expert biases require the Triton backends")
    if (align_routes is None) != (sum_routes is None):
        raise ValueError("align_routes and sum_routes must be provided together")
    if caller_owned_alignment_storage and align_routes is None:
        raise ValueError(
            "caller_owned_alignment_storage requires an external align_routes"
        )
    if gate_up_backend not in (
        TRITON_GATE_UP_BACKEND,
        CUDA_PIPELINE_GATE_UP_BACKEND,
        CUDA_EXACT_K48_GATE_UP_BACKEND,
        AUTO_GATE_UP_BACKEND,
    ):
        raise ValueError(
            "gate_up_backend must be 'triton', 'cuda_pipeline', "
            "'cuda_exact_k48', or 'auto', "
            f"got {gate_up_backend!r}"
        )
    if down_backend not in (
        TRITON_DOWN_BACKEND,
        CUDA_EXACT_K48_DOWN_BACKEND,
        AUTO_DOWN_BACKEND,
    ):
        raise ValueError(
            "down_backend must be 'triton', 'cuda_exact_k48', or 'auto', "
            f"got {down_backend!r}"
        )
    if gate_up_backend in (
        CUDA_PIPELINE_GATE_UP_BACKEND,
        CUDA_EXACT_K48_GATE_UP_BACKEND,
        AUTO_GATE_UP_BACKEND,
    ) and not structural_down:
        raise ValueError(
            f"gate_up_backend={gate_up_backend!r} requires structural_down=True"
        )
    if down_backend in (
        CUDA_EXACT_K48_DOWN_BACKEND,
        AUTO_DOWN_BACKEND,
    ) and not structural_down:
        raise ValueError(
            "down_backend='cuda_exact_k48'/'auto' requires structural_down=True"
        )
    if (
        gate_up_debug_trace is not None
        and gate_up_backend != CUDA_PIPELINE_GATE_UP_BACKEND
    ):
        raise ValueError(
            "gate_up_debug_trace requires gate_up_backend='cuda_pipeline'"
        )
    if (
        gate_up_backend == CUDA_PIPELINE_GATE_UP_BACKEND
        and activation_math.down_norm_placement != GATE_UP_EPILOGUE_NORM
    ):
        raise ValueError(
            "cuda_pipeline requires down_norm_placement="
            "'gate_up_epilogue'"
        )
    if (
        gate_up_backend == CUDA_PIPELINE_GATE_UP_BACKEND
        and activation_math.swiglu_limit is not None
    ):
        raise ValueError("cuda_pipeline does not implement clamped SwiGLU")
    if assignment_layout not in ("row_major", "word_major"):
        raise ValueError(
            "assignment_layout must be 'row_major' or 'word_major', "
            f"got {assignment_layout!r}"
        )
    if schedule not in (BASELINE_SCHEDULE, ROUTE_DENSITY_GROUPED_SCHEDULE):
        raise ValueError(
            "schedule must be 'baseline' or 'route_density_grouped', "
            f"got {schedule!r}"
        )

    num_tokens, hidden_size = hidden_states.shape
    if num_tokens <= 0 or hidden_size <= 0:
        raise ValueError("hidden_states dimensions must be positive")
    if gate_in_features != hidden_size or up_in_features != hidden_size:
        raise ValueError(
            "gate_in_features and up_in_features must equal hidden size "
            f"{hidden_size}"
        )
    if gate_output_norm.ndim != 2:
        raise ValueError("gate_output_norm must be [E, N]")
    num_experts = gate_output_norm.shape[0]
    if num_experts <= 0:
        raise ValueError("at least one expert is required")
    if model_num_experts is None:
        model_num_experts = num_experts
    elif (
        not isinstance(model_num_experts, int)
        or isinstance(model_num_experts, bool)
        or model_num_experts <= 0
    ):
        raise ValueError("model_num_experts must be a positive integer")
    device = hidden_states.device
    dtype = hidden_states.dtype

    intermediate_size = _validate_projection(
        name="gate",
        codebook=gate_codebook,
        packed_assignments=gate_packed_assignments,
        input_norm=gate_input_norm,
        output_norm=gate_output_norm,
        in_features=gate_in_features,
        expected_experts=num_experts,
        expected_out_features=None,
        dtype=dtype,
        device=device,
        assignment_layout=assignment_layout,
        group_size=group_size,
    )
    _validate_projection(
        name="up",
        codebook=up_codebook,
        packed_assignments=up_packed_assignments,
        input_norm=up_input_norm,
        output_norm=up_output_norm,
        in_features=up_in_features,
        expected_experts=num_experts,
        expected_out_features=intermediate_size,
        dtype=dtype,
        device=device,
        assignment_layout=assignment_layout,
        group_size=group_size,
    )
    if down_in_features != intermediate_size:
        raise ValueError(
            f"down_in_features must equal intermediate size {intermediate_size}"
        )
    physical_intermediate_size = (
        padded_down_k48_width(
            intermediate_size,
            down_input_group_start_lane,
        )
        if pad_down_to_k48
        else intermediate_size
    )
    physical_down_assignment_words = (
        _packed_words(physical_intermediate_size, group_size=group_size)
        if pad_down_to_k48
        else None
    )
    _validate_projection(
        name="down",
        codebook=down_codebook,
        packed_assignments=down_packed_assignments,
        input_norm=down_input_norm,
        output_norm=down_output_norm,
        in_features=down_in_features,
        expected_experts=num_experts,
        expected_out_features=hidden_size,
        dtype=dtype,
        device=device,
        assignment_layout=assignment_layout,
        group_size=group_size,
        input_group_start_lane=down_input_group_start_lane,
        expected_assignment_words=physical_down_assignment_words,
    )

    for name, bias, width in (
        ("gate_bias", gate_bias, intermediate_size),
        ("up_bias", up_bias, intermediate_size),
        ("down_bias", down_bias, hidden_size),
    ):
        if bias is not None:
            _require_tensor(bias, name=name, device=device, dtype=dtype, shape=(num_experts, width))
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be [M, top_k], got {tuple(topk_ids.shape)}")
    top_k = topk_ids.shape[1]
    if topk_ids.shape[0] != num_tokens or not 1 <= top_k <= model_num_experts:
        raise ValueError(
            f"topk_ids must be [{num_tokens}, top_k] with 1 <= top_k <= "
            f"{model_num_experts}, got {tuple(topk_ids.shape)}"
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must use int32 or int64")
    _require_tensor(topk_ids, name="topk_ids", device=device)
    _require_tensor(
        topk_weights,
        name="topk_weights",
        device=device,
        shape=tuple(topk_ids.shape),
    )
    if not topk_weights.is_floating_point():
        raise ValueError("topk_weights must use a floating-point dtype")
    # The standalone API validates caller-provided routes.  vLLM's router
    # already guarantees this invariant, so its adapter disables the check to
    # avoid a GPU-to-CPU synchronization for every newly produced route tensor.
    if validate_route_ids:
        _validate_route_range(topk_ids, num_experts)

    # The vLLM modular-experts adapter invokes this function inside its opaque
    # custom-op implementation, so this profile lookup sees concrete M rather
    # than a symbolic torch.compile dimension.  Standalone callers get the same
    # deterministic behavior.
    if gate_up_backend == AUTO_GATE_UP_BACKEND or down_backend == AUTO_DOWN_BACKEND:
        device_index = device.index
        if device_index is None:
            raise RuntimeError("CUDA tensor device must have a concrete index")
        (
            gate_up_backend,
            down_backend,
            automatic_launch_plan,
        ) = _resolve_cuda_moe_auto_decision(
            gate_up_backend,
            down_backend,
            device_index,
            dtype,
            group_size,
            assignment_bits,
            gate_codebook.shape[1],
            model_num_experts,
            num_experts,
            hidden_size,
            intermediate_size,
            physical_intermediate_size,
            pad_down_to_k48,
            down_input_group_start_lane,
            assignment_layout,
            structural_down,
            top_k,
            num_tokens,
            activation_math,
        )
        if cuda_launch_plan is None:
            cuda_launch_plan = automatic_launch_plan

    if group_size != GROUP_SIZE and (
        gate_up_backend != TRITON_GATE_UP_BACKEND
        or down_backend != TRITON_DOWN_BACKEND
    ):
        raise ValueError(
            "non-D6 NoWag MoE codebooks currently require the generic "
            "Triton/Triton backend pair"
        )

    if pad_down_to_k48:
        _validate_padded_backend_pair(gate_up_backend, down_backend)

    if output is not None:
        _require_tensor(
            output,
            name="output",
            device=device,
            dtype=dtype,
            shape=(num_tokens, hidden_size),
        )

    if align_routes is None or sum_routes is None:
        raise ValueError("align_routes and sum_routes are required")

    num_routes = num_tokens * top_k
    uses_exact_cuda = (
        gate_up_backend == CUDA_EXACT_K48_GATE_UP_BACKEND
        or down_backend == CUDA_EXACT_K48_DOWN_BACKEND
    )
    uses_profiled_triton_plan = (
        cuda_launch_plan is not None
        and gate_up_backend == TRITON_GATE_UP_BACKEND
        and down_backend == TRITON_DOWN_BACKEND
        and cuda_launch_plan.source == "profile"
        and bool(cuda_launch_plan.profile_name)
    )
    if (
        cuda_launch_plan is not None
        and not uses_exact_cuda
        and not uses_profiled_triton_plan
    ):
        raise ValueError(
            "cuda_launch_plan requires an exact-K48 backend or a named "
            "offline profile for a Triton/Triton pair"
        )
    if uses_exact_cuda and cuda_launch_plan is None:
        from .execution_profile import cuda_hardware_key
        from .moe_tuning import select_cuda_moe_launch_plan

        cuda_launch_plan = select_cuda_moe_launch_plan(
            hardware_key=cuda_hardware_key(device),
            dtype=dtype,
            group_size=group_size,
            assignment_bits=assignment_bits,
            codebook_size=gate_codebook.shape[1],
            num_experts=model_num_experts,
            physical_expert_rows=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            physical_intermediate_size=physical_intermediate_size,
            pad_down_to_k48=pad_down_to_k48,
            down_input_group_start_lane=down_input_group_start_lane,
            assignment_layout=assignment_layout,
            structural_down=structural_down,
            top_k=top_k,
            activation_kind="silu_mul",
            gate_up_input_rounding=(
                activation_math.gate_up_input_rounding
            ),
            swiglu_limit=activation_math.swiglu_limit,
            down_input_rounding=activation_math.down_input_rounding,
            down_norm_placement=activation_math.down_norm_placement,
            num_tokens=num_tokens,
        )
    adaptive_m_tiles = bool(
        cuda_launch_plan is not None and cuda_launch_plan.adaptive_m_tiles
    )
    adaptive_residual_policy = (
        getattr(cuda_launch_plan, "adaptive_residual_policy", "bm16")
        if adaptive_m_tiles and cuda_launch_plan is not None
        else "bm16"
    )
    if cuda_launch_plan is not None:
        if (
            cuda_launch_plan.source == "manual"
            and down_backend == CUDA_EXACT_K48_DOWN_BACKEND
            and cuda_launch_plan.down.block_m == 32
        ):
            raise ValueError(
                "manual Exact-K48 Down plans do not implement block_m=32"
            )
        if adaptive_m_tiles and (
            gate_up_backend != CUDA_EXACT_K48_GATE_UP_BACKEND
            or down_backend != CUDA_EXACT_K48_DOWN_BACKEND
        ):
            raise ValueError(
                "adaptive_m_tiles requires the Exact-K48 Gate/Up and Down pair"
            )
        if cuda_launch_plan.down.block_m % cuda_launch_plan.gate_up.block_m:
            raise ValueError(
                "CUDA launch plan requires Down BM divisible by Gate/Up BM"
            )
        if cuda_launch_plan.gate_up.num_warps != (
            cuda_launch_plan.gate_up.block_n // 16
        ):
            raise ValueError(
                "current profiled Gate/Up plans require one warp per N=16"
            )
        if cuda_launch_plan.down.num_warps != (
            cuda_launch_plan.down.block_n // 16
        ):
            raise ValueError(
                "current profiled Down plans require one warp per N=16"
            )
        if (
            cuda_launch_plan.down.shared_codebook_entries
            or cuda_launch_plan.down.assignment_l2_only
        ):
            raise ValueError(
                "Down does not support a codebook cache policy"
            )
        gate_uses_codebook_cache_policy = bool(
            cuda_launch_plan.gate_up.shared_codebook_entries
            or cuda_launch_plan.gate_up.assignment_l2_only
        )
        if (
            cuda_launch_plan.gate_up.shared_codebook_entries
            and not cuda_launch_plan.gate_up.assignment_l2_only
        ):
            raise ValueError(
                "shared Gate/Up codebook entries require "
                "assignment_l2_only=True"
            )
        if gate_uses_codebook_cache_policy and (
            gate_up_backend != CUDA_EXACT_K48_GATE_UP_BACKEND
        ):
            raise ValueError(
                "Gate/Up codebook cache policies require the Exact-K48 "
                "backend"
            )
        if gate_uses_codebook_cache_policy and (
            group_size != GROUP_SIZE
            or assignment_bits != ASSIGNMENT_BITS
            or dtype != torch.bfloat16
            or assignment_layout != "word_major"
            or not structural_down
            or tuple(gate_codebook.shape) != (1, 4096, GROUP_SIZE)
            or tuple(up_codebook.shape) != (1, 4096, GROUP_SIZE)
        ):
            raise ValueError(
                "Gate/Up codebook cache policies require structural BF16 "
                "D6/B12 Exact-K48 with word-major assignments and shared "
                "[1,4096,6] Gate/Up codebooks"
            )
    else:
        gate_uses_codebook_cache_policy = False

    grouped_schedule = schedule == ROUTE_DENSITY_GROUPED_SCHEDULE
    preapply_down_norm = (
        activation_math.down_norm_placement == GATE_UP_EPILOGUE_NORM
    )
    if cuda_launch_plan is not None:
        gate_block_m = cuda_launch_plan.gate_up.block_m
    else:
        gate_block_m = (
            route_density_block_m(num_tokens, top_k, model_num_experts)
            if grouped_schedule
            else BLOCK_M
        )
    if structural_down:
        if cuda_launch_plan is not None:
            down_config = StructuralDownConfig(
                block_m=cuda_launch_plan.down.block_m,
                block_n=cuda_launch_plan.down.block_n,
                num_warps=cuda_launch_plan.down.num_warps,
                num_stages=cuda_launch_plan.down.num_stages,
            )
        else:
            down_config = structural_down_config(
                num_tokens,
                top_k,
                model_num_experts,
                hidden_size,
            )
        down_block_m = down_config.block_m
        if down_block_m < gate_block_m or down_block_m % gate_block_m != 0:
            raise RuntimeError(
                "structural Down BM must be an integer multiple of Gate/Up BM"
            )
    else:
        down_config = StructuralDownConfig(
            block_m=gate_block_m,
            block_n=BLOCK_N,
            num_warps=4,
        )
        down_block_m = gate_block_m
    alignment_block_m = 16 if adaptive_m_tiles else down_block_m
    alignment_block_ratio = alignment_block_m // gate_block_m

    adaptive_align_fills_tasks = adaptive_m_tiles and (
        (
            adaptive_residual_policy == "bm16"
            and align_routes_adaptive is not None
        )
        or (
            adaptive_residual_policy == "tail64"
            and align_routes_adaptive_tail64 is not None
        )
    )
    sorted_tickets: torch.Tensor | None = None
    expert_ids: torch.Tensor | None = None
    num_tickets_post_padded: torch.Tensor | None = None
    # Structural mode aligns once at Down's macro-tile. Gate/Up subdivides
    # each aligned expert block, so its sorted positions map directly to the
    # middle rows consumed by Down. A ticket is the flattened
    # [token, top-k slot] index produced by vLLM.  The adaptive callback is
    # deferred until its graph-stable task queues have been carved below.
    if (
        not adaptive_align_fills_tasks
        and not caller_owned_alignment_storage
    ):
        sorted_tickets, expert_ids, num_tickets_post_padded = align_routes(
            topk_ids,
            alignment_block_m,
            num_experts,
        )
    required_middle_rows = (
        required_structural_middle_rows(
            num_routes,
            num_experts,
            alignment_block_m,
        )
        if structural_down
        else num_routes
    )
    supports_exact_k48_device = torch.cuda.get_device_capability(device) >= (8, 0)
    pipeline_block_m = (
        cuda_launch_plan.gate_up.block_m
        if cuda_launch_plan is not None
        and cuda_launch_plan.gate_up.block_m in (16, 32, 64, 128)
        else 64
        if down_block_m >= 64
        else 32
        if down_block_m >= 32
        else None
    )
    use_cuda_pipeline = (
        gate_up_backend == CUDA_PIPELINE_GATE_UP_BACKEND
        and pipeline_block_m is not None
    )
    exact_gate_up_block_n = (
        cuda_launch_plan.gate_up.block_n
        if cuda_launch_plan is not None
        else 128
        if intermediate_size % 128 == 0
        else 64
        if intermediate_size % 64 == 0
        else None
    )
    exact_gate_up_candidate = (
        gate_up_backend == CUDA_EXACT_K48_GATE_UP_BACKEND
        and supports_exact_k48_device
        and structural_down
        and dtype == torch.bfloat16
        and assignment_layout == "word_major"
        and pipeline_block_m is not None
        and exact_gate_up_block_n in (64, 128)
        and intermediate_size % exact_gate_up_block_n == 0
        and hidden_size % 2 == 0
        and (
            cuda_launch_plan is None
            or cuda_launch_plan.gate_up.num_stages == 2
        )
    )
    if gate_uses_codebook_cache_policy and not exact_gate_up_candidate:
        raise ValueError(
            "the requested Gate/Up codebook cache policy does not match a "
            "supported Exact-K48 launch"
        )
    exact_down_candidate = (
        down_backend == CUDA_EXACT_K48_DOWN_BACKEND
        and supports_exact_k48_device
        and structural_down
        and dtype == torch.bfloat16
        and assignment_layout == "word_major"
        and down_block_m in (16, 64, 128)
        and down_config.block_n in (64, 128)
        and hidden_size % down_config.block_n == 0
        and physical_intermediate_size % 2 == 0
        and down_codebook.shape[1] == 4096
        and (
            (not pad_down_to_k48 and down_input_group_start_lane == 0)
            or (pad_down_to_k48 and exact_gate_up_candidate)
        )
        and (
            not pad_down_to_k48
            or physical_intermediate_size % EXACT_K48 == 0
        )
        and (
            cuda_launch_plan is None
            or cuda_launch_plan.down.num_stages == 2
        )
    )
    if adaptive_m_tiles and not (
        exact_gate_up_candidate and exact_down_candidate
    ):
        raise ValueError(
            "adaptive_m_tiles requires the supported BF16 D6/B12, "
            "word-major, structural Exact-K48 shape with BN128"
        )
    if pad_down_to_k48:
        # A padded middle is one atomic producer/consumer contract.  If a
        # fixed Exact request misses either stage's structural preflight, both
        # stages fall back to the already-audited logical Triton view; never
        # execute a half-Exact pair accidentally.
        use_padded_exact_pair = (
            exact_gate_up_candidate and exact_down_candidate
        )
        use_cuda_exact_gate_up = use_padded_exact_pair
        use_cuda_exact_down = use_padded_exact_pair
    else:
        # Preserve the explicit unpadded research A/B modes.
        use_cuda_exact_gate_up = exact_gate_up_candidate
        use_cuda_exact_down = exact_down_candidate
    use_adaptive_m_tiles = adaptive_m_tiles
    workspace_layout = moe_middle_workspace_layout(
        num_routes=num_routes,
        num_experts=num_experts,
        alignment_block_m=alignment_block_m,
        physical_intermediate_size=physical_intermediate_size,
        structural_down=structural_down,
        compute_slabs=(
            2 if use_cuda_pipeline or use_cuda_exact_gate_up else 1
        ),
        adaptive_m_tiles=use_adaptive_m_tiles,
        caller_owned_alignment_storage=caller_owned_alignment_storage,
        adaptive_residual_policy=adaptive_residual_policy,
    )
    if workspace_layout.middle_rows != required_middle_rows:
        raise RuntimeError("middle-workspace layout disagrees with alignment")
    required_workspace_rows = workspace_layout.total_rows
    task_capacity64 = 0
    task_capacity16 = 0
    task_metadata_rows = workspace_layout.task_metadata_rows
    if use_adaptive_m_tiles:
        if adaptive_residual_policy == "tail64":
            from .cuda_ops import adaptive_tail64_task_capacities

            task_capacity64, task_capacity16 = (
                adaptive_tail64_task_capacities(
                    num_routes=num_routes,
                    num_experts=num_experts,
                )
            )
        else:
            from .cuda_ops import adaptive_task_capacities

            task_capacity64, task_capacity16 = adaptive_task_capacities(
                num_routes=num_routes,
                num_experts=num_experts,
            )
    if middle_workspace is None:
        middle_storage = torch.empty(
            (required_workspace_rows, physical_intermediate_size),
            device=device,
            dtype=dtype,
        )
    else:
        _require_tensor(
            middle_workspace,
            name="middle_workspace",
            device=device,
            dtype=dtype,
        )
        if (
            middle_workspace.ndim != 2
            or middle_workspace.shape[0] < required_workspace_rows
            or middle_workspace.shape[1] != physical_intermediate_size
        ):
            raise ValueError(
                "middle_workspace must be [rows, physical_intermediate_size] "
                f"with rows >= {required_workspace_rows} and "
                f"physical_intermediate_size = {physical_intermediate_size}, "
                f"got {tuple(middle_workspace.shape)}"
            )
        middle_storage = middle_workspace
    # The CUDA backend stores Gate and Up in two contiguous row slabs.  Its
    # epilogue overwrites the first slab with the expert-sorted middle, which
    # remains contiguous for the existing structural Down kernel.
    middle_physical = middle_storage[:required_middle_rows]
    middle_offset = down_input_group_start_lane if pad_down_to_k48 else 0
    middle = middle_physical[
        :, middle_offset : middle_offset + intermediate_size
    ]
    tasks64 = None
    tasks16 = None
    task_counts = None
    if use_adaptive_m_tiles:
        metadata_start = workspace_layout.task_metadata_start
        metadata = middle_storage[
            metadata_start : metadata_start + task_metadata_rows
        ].view(torch.int32).flatten()
        tasks64_elements = 2 * task_capacity64
        tasks16_elements = 2 * task_capacity16
        tasks64 = metadata[:tasks64_elements].view(task_capacity64, 2)
        tasks16 = metadata[
            tasks64_elements : tasks64_elements + tasks16_elements
        ].view(task_capacity16, 2)
        task_counts = metadata[
            tasks64_elements + tasks16_elements :
            tasks64_elements + tasks16_elements + 2
        ]

    alignment_storage = None
    if caller_owned_alignment_storage:
        alignment_start = workspace_layout.alignment_storage_start
        alignment_storage = middle_storage[
            alignment_start :
            alignment_start + workspace_layout.alignment_storage_rows
        ].view(torch.int32).flatten()[
            : workspace_layout.alignment_storage_int32_elements
        ]

    if adaptive_align_fills_tasks:
        adaptive_callback = (
            align_routes_adaptive_tail64
            if adaptive_residual_policy == "tail64"
            else align_routes_adaptive
        )
        assert adaptive_callback is not None
        assert tasks64 is not None
        assert tasks16 is not None
        assert task_counts is not None
        if caller_owned_alignment_storage:
            assert alignment_storage is not None
            sorted_tickets, expert_ids, num_tickets_post_padded = (
                adaptive_callback(
                    topk_ids,
                    alignment_block_m,
                    num_experts,
                    tasks64,
                    tasks16,
                    task_counts,
                    alignment_storage=alignment_storage,
                )
            )
        else:
            sorted_tickets, expert_ids, num_tickets_post_padded = (
                adaptive_callback(
                    topk_ids,
                    alignment_block_m,
                    num_experts,
                    tasks64,
                    tasks16,
                    task_counts,
                )
            )
    elif caller_owned_alignment_storage:
        assert alignment_storage is not None
        sorted_tickets, expert_ids, num_tickets_post_padded = align_routes(
            topk_ids,
            alignment_block_m,
            num_experts,
            alignment_storage=alignment_storage,
        )
    assert sorted_tickets is not None
    assert expert_ids is not None
    assert num_tickets_post_padded is not None

    # Some serving runtimes reserve one extra sentinel entry in expert_ids.
    # The device scalar still selects the live prefix, but the static launch
    # must never include a block whose rows do not fit the middle workspace.
    down_num_m_blocks = (
        _structural_num_m_blocks(
            expert_id_capacity=expert_ids.numel(),
            num_routes=num_routes,
            middle_rows=required_middle_rows,
            block_m=down_block_m,
        )
        if structural_down
        else min(expert_ids.numel(), num_routes)
    )
    gate_num_m_blocks = down_num_m_blocks * alignment_block_ratio

    if route_output_workspace is None:
        route_output = torch.empty(
            (num_routes, hidden_size), device=device, dtype=dtype
        )
    else:
        _require_tensor(
            route_output_workspace,
            name="route_output_workspace",
            device=device,
            dtype=dtype,
            shape=(num_routes, hidden_size),
        )
        route_output = route_output_workspace
    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

    padded_group_size = _next_power_of_two(group_size)
    gate_groups = (hidden_size + group_size - 1) // group_size
    gate_assignment_info = assignment_layout_info(
        gate_packed_assignments, assignment_layout
    )
    up_assignment_info = assignment_layout_info(
        up_packed_assignments, assignment_layout
    )
    down_assignment_info = assignment_layout_info(
        down_packed_assignments, assignment_layout
    )
    gate_words = gate_assignment_info.num_words
    gate_num_pid_n = triton.cdiv(intermediate_size, BLOCK_N)
    gate_grid = _launch_grid(
        gate_num_m_blocks,
        gate_num_pid_n,
        grouped_schedule,
    )
    if use_adaptive_m_tiles:
        from .cuda_ops import moe_gate_up_exact_k48_adaptive

        assert tasks64 is not None
        assert tasks16 is not None
        assert task_counts is not None
        assert cuda_launch_plan is not None
        if not adaptive_align_fills_tasks:
            if adaptive_residual_policy == "tail64":
                from .cuda_ops import moe_build_adaptive_tasks_tail64

                adaptive_builder = moe_build_adaptive_tasks_tail64
            else:
                from .cuda_ops import moe_build_adaptive_tasks

                adaptive_builder = moe_build_adaptive_tasks
            adaptive_builder(
                sorted_tickets=sorted_tickets,
                expert_ids16=expert_ids,
                num_tickets_post_padded=num_tickets_post_padded,
                num_routes=num_routes,
                num_experts=num_experts,
                tasks64=tasks64,
                tasks16=tasks16,
                task_counts=task_counts,
            )
        moe_gate_up_exact_k48_adaptive(
            hidden_states=hidden_states,
            gate_codebook=gate_codebook,
            gate_packed_assignments=gate_packed_assignments,
            gate_input_norm=gate_input_norm,
            gate_output_norm=gate_output_norm,
            up_codebook=up_codebook,
            up_packed_assignments=up_packed_assignments,
            up_input_norm=up_input_norm,
            up_output_norm=up_output_norm,
            down_input_norm=down_input_norm,
            sorted_tickets=sorted_tickets,
            expert_ids16=expert_ids,
            num_tickets_post_padded=num_tickets_post_padded,
            gate_up_workspace=middle_storage[: 2 * required_middle_rows],
            tasks64=tasks64,
            tasks16=tasks16,
            task_counts=task_counts,
            num_routes=num_routes,
            top_k=top_k,
            block_n=cuda_launch_plan.gate_up.block_n,
            tasks_per_cta=cuda_launch_plan.gate_up.tasks_per_cta,
            output_start_lane=middle_offset,
            swiglu_limit=activation_math.swiglu_limit,
            preapply_down_norm=preapply_down_norm,
        )
    elif use_cuda_exact_gate_up:
        assert pipeline_block_m is not None
        assert exact_gate_up_block_n is not None
        pipeline_alignment_ratio = down_block_m // pipeline_block_m
        pipeline_num_m_blocks = down_num_m_blocks * pipeline_alignment_ratio
        gate_up_kwargs = dict(
            hidden_states=hidden_states,
            gate_codebook=gate_codebook,
            gate_packed_assignments=gate_packed_assignments,
            gate_input_norm=gate_input_norm,
            gate_output_norm=gate_output_norm,
            up_codebook=up_codebook,
            up_packed_assignments=up_packed_assignments,
            up_input_norm=up_input_norm,
            up_output_norm=up_output_norm,
            down_input_norm=down_input_norm,
            sorted_tickets=sorted_tickets,
            expert_ids=expert_ids,
            num_tickets_post_padded=num_tickets_post_padded,
            gate_up_workspace=middle_storage[: 2 * required_middle_rows],
            num_routes=num_routes,
            top_k=top_k,
            num_m_blocks=pipeline_num_m_blocks,
            alignment_block_ratio=pipeline_alignment_ratio,
            block_m=pipeline_block_m,
            block_n=exact_gate_up_block_n,
            output_start_lane=middle_offset,
            swiglu_limit=activation_math.swiglu_limit,
            preapply_down_norm=preapply_down_norm,
        )
        if gate_uses_codebook_cache_policy:
            from .cuda_ops import moe_gate_up_exact_k48_codebook_cache

            assert cuda_launch_plan is not None
            moe_gate_up_exact_k48_codebook_cache(
                **gate_up_kwargs,
                shared_codebook_entries=(
                    cuda_launch_plan.gate_up.shared_codebook_entries
                ),
                assignment_l2_only=(
                    cuda_launch_plan.gate_up.assignment_l2_only
                ),
            )
        else:
            from .cuda_ops import moe_gate_up_exact_k48

            moe_gate_up_exact_k48(**gate_up_kwargs)
    elif use_cuda_pipeline:
        from .cuda_ops import aligned_codebook, moe_gate_up_pipeline

        assert pipeline_block_m is not None
        pipeline_alignment_ratio = down_block_m // pipeline_block_m
        pipeline_num_m_blocks = down_num_m_blocks * pipeline_alignment_ratio
        moe_gate_up_pipeline(
            hidden_states=hidden_states,
            gate_codebook=aligned_codebook(gate_codebook),
            gate_packed_assignments=gate_packed_assignments,
            gate_input_norm=gate_input_norm,
            gate_output_norm=gate_output_norm,
            up_codebook=aligned_codebook(up_codebook),
            up_packed_assignments=up_packed_assignments,
            up_input_norm=up_input_norm,
            up_output_norm=up_output_norm,
            down_input_norm=down_input_norm,
            sorted_tickets=sorted_tickets,
            expert_ids=expert_ids,
            num_tickets_post_padded=num_tickets_post_padded,
            gate_up_workspace=middle_storage[: 2 * required_middle_rows],
            num_routes=num_routes,
            top_k=top_k,
            num_m_blocks=pipeline_num_m_blocks,
            alignment_block_ratio=pipeline_alignment_ratio,
            block_m=pipeline_block_m,
            word_major_assignments=assignment_layout == "word_major",
            use_block12_decoder=use_block12_decoder,
            debug_trace=gate_up_debug_trace,
        )
    else:
        _nowag_moe_gate_up_kernel[gate_grid](
            hidden_states,
            gate_codebook,
            gate_packed_assignments,
            gate_input_norm,
            gate_output_norm,
            up_codebook,
            up_packed_assignments,
            up_input_norm,
            up_output_norm,
            down_input_norm,
            gate_output_norm if gate_bias is None else gate_bias,
            up_output_norm if up_bias is None else up_bias,
            sorted_tickets,
            expert_ids,
            num_tickets_post_padded,
            middle,
            num_routes,
            top_k,
            intermediate_size,
            hidden_size,
            gate_groups,
            gate_codebook.shape[1],
            up_codebook.shape[1],
            gate_words,
            hidden_states.stride(0),
            hidden_states.stride(1),
            0 if gate_codebook.shape[0] == 1 else gate_codebook.stride(0),
            gate_codebook.stride(1),
            gate_codebook.stride(2),
            *gate_assignment_info.kernel_strides,
            gate_input_norm.stride(0),
            gate_input_norm.stride(1),
            gate_output_norm.stride(0),
            gate_output_norm.stride(1),
            0 if up_codebook.shape[0] == 1 else up_codebook.stride(0),
            up_codebook.stride(1),
            up_codebook.stride(2),
            *up_assignment_info.kernel_strides,
            up_input_norm.stride(0),
            up_input_norm.stride(1),
            up_output_norm.stride(0),
            up_output_norm.stride(1),
            down_input_norm.stride(0),
            down_input_norm.stride(1),
            middle.stride(0),
            middle.stride(1),
            COMPUTE_TYPE=compute_type,
            PACKED_12_BLOCK_DECODE=use_block12_decoder,
            CLAMP_SWIGLU=swiglu_limit is not None,
            SWIGLU_LIMIT=(0.0 if swiglu_limit is None else float(swiglu_limit)),
            ACTIVATION=_ACTIVATION_IDS[activation_math.activation_kind],
            ACTIVATION_ALPHA=float(activation_math.activation_alpha),
            HAS_BIAS=gate_bias is not None,
            PREAPPLY_DOWN_NORM=preapply_down_norm,
            SORTED_MIDDLE_LAYOUT=structural_down,
            ALIGNMENT_BLOCK_RATIO=alignment_block_ratio,
            GROUPED_SCHEDULE=grouped_schedule,
            NUM_PID_N=gate_num_pid_n,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_M=gate_block_m,
            BLOCK_N=BLOCK_N,
            BLOCK_G=BLOCK_G,
            D=group_size,
            PAD_D=padded_group_size,
            num_warps=4,
            num_stages=2,
        )

    if middle_transform is not None:
        transformed = middle_transform(middle)
        if transformed is not middle:
            raise RuntimeError("middle_transform must update and return its input tensor")

    down_groups = (
        down_input_group_start_lane + intermediate_size + group_size - 1
    ) // group_size
    down_num_pid_n = triton.cdiv(hidden_size, down_config.block_n)
    down_grid = _launch_grid(
        down_num_m_blocks,
        down_num_pid_n,
        grouped_schedule,
    )
    if use_adaptive_m_tiles:
        from .cuda_ops import moe_down_exact_k48_adaptive

        assert tasks64 is not None
        assert tasks16 is not None
        assert task_counts is not None
        moe_down_exact_k48_adaptive(
            sorted_middle=(middle_physical if pad_down_to_k48 else middle),
            codebook=down_codebook,
            packed_assignments=down_packed_assignments,
            output_norm=down_output_norm,
            topk_weights=topk_weights,
            sorted_tickets=sorted_tickets,
            expert_ids16=expert_ids,
            num_tickets_post_padded=num_tickets_post_padded,
            tasks64=tasks64,
            tasks16=tasks16,
            task_counts=task_counts,
            route_output=route_output,
            num_routes=num_routes,
            tasks_per_cta=cuda_launch_plan.down.tasks_per_cta,
            input_norm=down_input_norm,
            logical_in_features=intermediate_size,
            input_group_start_lane=middle_offset,
            preapplied_input_norm=preapply_down_norm,
        )
    elif use_cuda_exact_down:
        from .cuda_ops import moe_down_padded64

        moe_down_padded64(
            sorted_middle=(middle_physical if pad_down_to_k48 else middle),
            codebook=down_codebook,
            packed_assignments=down_packed_assignments,
            output_norm=down_output_norm,
            topk_weights=topk_weights,
            sorted_tickets=sorted_tickets,
            expert_ids=expert_ids,
            num_tickets_post_padded=num_tickets_post_padded,
            route_output=route_output,
            num_routes=num_routes,
            num_m_blocks=down_num_m_blocks,
            block_m=down_block_m,
            block_n=down_config.block_n,
            use_exact_k48=True,
            use_coalesced_k48=True,
            input_norm=down_input_norm,
            logical_in_features=intermediate_size,
            input_group_start_lane=middle_offset,
            preapplied_input_norm=preapply_down_norm,
        )
    else:
        _nowag_moe_down_kernel[down_grid](
            middle,
            down_codebook,
            down_packed_assignments,
            down_input_norm,
            down_output_norm,
            down_output_norm if down_bias is None else down_bias,
            topk_weights,
            sorted_tickets,
            expert_ids,
            num_tickets_post_padded,
            route_output,
            num_routes,
            hidden_size,
            intermediate_size,
            down_groups,
            down_codebook.shape[1],
            down_assignment_info.num_words,
            middle.stride(0),
            middle.stride(1),
            0 if down_codebook.shape[0] == 1 else down_codebook.stride(0),
            down_codebook.stride(1),
            down_codebook.stride(2),
            *down_assignment_info.kernel_strides,
            down_input_norm.stride(0),
            down_input_norm.stride(1),
            down_output_norm.stride(0),
            down_output_norm.stride(1),
            route_output.stride(0),
            route_output.stride(1),
            COMPUTE_TYPE=compute_type,
            PACKED_12_BLOCK_DECODE=use_block12_decoder,
            PREAPPLIED_DOWN_NORM=preapply_down_norm,
            HAS_BIAS=down_bias is not None,
            SORTED_MIDDLE_LAYOUT=structural_down,
            GROUPED_SCHEDULE=grouped_schedule,
            NUM_PID_N=down_num_pid_n,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_M=down_block_m,
            BLOCK_N=down_config.block_n,
            BLOCK_G=BLOCK_G,
            D=group_size,
            PAD_D=padded_group_size,
            INPUT_GROUP_START_LANE=down_input_group_start_lane,
            num_warps=down_config.num_warps,
            num_stages=down_config.num_stages,
        )

    if output is None:
        output = torch.empty_like(hidden_states)
    sum_routes(route_output.view(num_tokens, top_k, hidden_size), output)
    return output


