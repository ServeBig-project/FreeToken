"""Python boundary around the NoWAG CUDA kernels in ``csrc/``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


ADAPTIVE_RESIDUAL_BM16 = "bm16"
ADAPTIVE_RESIDUAL_TAIL64 = "tail64"

_CSRC = Path(__file__).with_name("csrc")


@lru_cache(maxsize=1)
def _extension():
    """Build the kernels on first use (cached under TORCH_EXTENSIONS_DIR) and load them."""
    from torch.utils.cpp_extension import load

    nvidia = Path(torch.__file__).resolve().parent.parent / "nvidia"
    return load(
        name="freetoken_nowag_cuda",
        sources=sorted(str(p) for p in _CSRC.iterdir() if p.suffix in (".cpp", ".cu")),
        extra_include_paths=[str(p) for p in sorted(nvidia.glob("*/include")) if p.is_dir()],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
    )


def adaptive_task_capacities(
    *,
    num_routes: int,
    num_experts: int,
) -> tuple[int, int]:
    """Return the fixed BM64 and residual BM16 capacities for one route shape.

    The BM16 bound uses at most ``min(num_routes, num_experts)`` non-empty
    experts and is ``min(R, 4*P, (R + 15*min(R, P)) // 16)``.
    """
    if num_routes <= 0:
        raise ValueError("num_routes must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    active_experts = min(num_routes, num_experts)
    residual_bound = (num_routes + 15 * active_experts) // 16
    capacity16 = min(num_routes, 4 * num_experts, residual_bound)
    return max(1, num_routes // 64), max(1, capacity16)


def adaptive_tail64_task_capacities(
    *,
    num_routes: int,
    num_experts: int,
) -> tuple[int, int]:
    """Return fixed BM64/BM16 capacities for the dense-tail policy."""
    _, capacity16 = adaptive_task_capacities(
        num_routes=num_routes,
        num_experts=num_experts,
    )
    return max(1, num_routes // 49), capacity16


def adaptive_task_metadata_rows(
    *,
    num_routes: int,
    num_experts: int,
    physical_intermediate_size: int,
    adaptive_residual_policy: str = ADAPTIVE_RESIDUAL_BM16,
) -> int:
    """Return BF16 workspace rows needed by both queues and their counts."""
    if physical_intermediate_size <= 0:
        raise ValueError("physical_intermediate_size must be positive")
    if adaptive_residual_policy == ADAPTIVE_RESIDUAL_BM16:
        capacities = adaptive_task_capacities
    elif adaptive_residual_policy == ADAPTIVE_RESIDUAL_TAIL64:
        capacities = adaptive_tail64_task_capacities
    else:
        raise ValueError(
            "adaptive_residual_policy must be 'bm16' or 'tail64'"
        )
    capacity64, capacity16 = capacities(
        num_routes=num_routes, num_experts=num_experts
    )
    int32_elements = 2 * capacity64 + 2 * capacity16 + 2
    row_bytes = physical_intermediate_size * torch.bfloat16.itemsize
    return (int32_elements * torch.int32.itemsize + row_bytes - 1) // row_bytes


def moe_gate_up_exact_k48(
    *,
    hidden_states: torch.Tensor,
    gate_codebook: torch.Tensor,
    gate_packed_assignments: torch.Tensor,
    gate_input_norm: torch.Tensor,
    gate_output_norm: torch.Tensor,
    up_codebook: torch.Tensor,
    up_packed_assignments: torch.Tensor,
    up_input_norm: torch.Tensor,
    up_output_norm: torch.Tensor,
    down_input_norm: torch.Tensor,
    sorted_tickets: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tickets_post_padded: torch.Tensor,
    gate_up_workspace: torch.Tensor,
    num_routes: int,
    top_k: int,
    num_m_blocks: int,
    alignment_block_ratio: int,
    block_m: int,
    block_n: int,
    precompute_tokens: bool = True,
    output_start_lane: int = 0,
    fuse_projections: bool = True,
    swiglu_limit: float | None = None,
    preapply_down_norm: bool = True,
) -> None:
    """Run Gate and Up with the shared D6/12-bit exact-K48 MMA core.

    ``gate_output_norm`` keeps the logical projection width.  A wider
    ``gate_up_workspace`` is a physical Down-input row: valid values start at
    ``output_start_lane`` and the fused epilogue clears both padding regions.
    The defaults preserve the original unpadded ``[2*rows, N]`` layout.
    """
    _extension().moe_gate_up_exact_k48(
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
        sorted_tickets,
        expert_ids,
        num_tickets_post_padded,
        gate_up_workspace,
        num_routes,
        top_k,
        num_m_blocks,
        alignment_block_ratio,
        block_m,
        block_n,
        precompute_tokens,
        output_start_lane,
        fuse_projections,
        0.0 if swiglu_limit is None else float(swiglu_limit),
        preapply_down_norm,
    )


def moe_gate_up_exact_k48_adaptive(
    *,
    hidden_states: torch.Tensor,
    gate_codebook: torch.Tensor,
    gate_packed_assignments: torch.Tensor,
    gate_input_norm: torch.Tensor,
    gate_output_norm: torch.Tensor,
    up_codebook: torch.Tensor,
    up_packed_assignments: torch.Tensor,
    up_input_norm: torch.Tensor,
    up_output_norm: torch.Tensor,
    down_input_norm: torch.Tensor,
    sorted_tickets: torch.Tensor,
    expert_ids16: torch.Tensor,
    num_tickets_post_padded: torch.Tensor,
    gate_up_workspace: torch.Tensor,
    tasks64: torch.Tensor,
    tasks16: torch.Tensor,
    task_counts: torch.Tensor,
    num_routes: int,
    top_k: int,
    block_n: int = 128,
    tasks_per_cta: int = 2,
    output_start_lane: int = 0,
    fuse_projections: bool = True,
    swiglu_limit: float | None = None,
    preapply_down_norm: bool = True,
) -> None:
    """Run exact-K48 Gate/Up over device-built BM64 and BM16 queues."""
    if (
        not isinstance(tasks_per_cta, int)
        or isinstance(tasks_per_cta, bool)
        or tasks_per_cta not in (1, 2, 4, 8)
    ):
        raise ValueError("tasks_per_cta must be one of 1, 2, 4, or 8")
    _extension().moe_gate_up_exact_k48_adaptive(
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
        sorted_tickets,
        expert_ids16,
        num_tickets_post_padded,
        gate_up_workspace,
        tasks64,
        tasks16,
        task_counts,
        num_routes,
        top_k,
        block_n,
        tasks_per_cta,
        output_start_lane,
        fuse_projections,
        0.0 if swiglu_limit is None else float(swiglu_limit),
        preapply_down_norm,
    )


def moe_gate_up_exact_k48_codebook_cache(
    *,
    hidden_states: torch.Tensor,
    gate_codebook: torch.Tensor,
    gate_packed_assignments: torch.Tensor,
    gate_input_norm: torch.Tensor,
    gate_output_norm: torch.Tensor,
    up_codebook: torch.Tensor,
    up_packed_assignments: torch.Tensor,
    up_input_norm: torch.Tensor,
    up_output_norm: torch.Tensor,
    down_input_norm: torch.Tensor,
    sorted_tickets: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tickets_post_padded: torch.Tensor,
    gate_up_workspace: torch.Tensor,
    num_routes: int,
    top_k: int,
    num_m_blocks: int,
    alignment_block_ratio: int,
    block_m: int,
    block_n: int,
    shared_codebook_entries: int,
    assignment_l2_only: bool = False,
    precompute_tokens: bool = True,
    output_start_lane: int = 0,
    fuse_projections: bool = True,
    swiglu_limit: float | None = None,
    preapply_down_norm: bool = True,
) -> None:
    """Run the BM16 Gate/Up assignment/codebook cache policy.

    ``(shared_codebook_entries, assignment_l2_only)`` selects one of three
    explicit controls.  ``(0, False)`` calls the existing production function;
    ``(0, True)`` keeps the codebook global while making assignment loads
    L2-only; ``(4096, True)`` places the complete codebook in shared memory.
    The non-default paths require shared BF16 D6/B12 codebooks with shape
    ``[1,4096,6]``.  They preserve assignment IDs and do not allocate another
    assignment tensor at runtime.
    """
    if shared_codebook_entries not in (0, 4096):
        raise ValueError(
            "shared_codebook_entries must be 0 or 4096"
        )
    if not isinstance(assignment_l2_only, bool):
        raise TypeError("assignment_l2_only must be a bool")
    if shared_codebook_entries and not assignment_l2_only:
        raise ValueError(
            "shared codebook candidates require assignment_l2_only=True"
        )
    if shared_codebook_entries == 0 and not assignment_l2_only:
        return moe_gate_up_exact_k48(
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
            gate_up_workspace=gate_up_workspace,
            num_routes=num_routes,
            top_k=top_k,
            num_m_blocks=num_m_blocks,
            alignment_block_ratio=alignment_block_ratio,
            block_m=block_m,
            block_n=block_n,
            precompute_tokens=precompute_tokens,
            output_start_lane=output_start_lane,
            fuse_projections=fuse_projections,
            swiglu_limit=swiglu_limit,
            preapply_down_norm=preapply_down_norm,
        )
    _extension().moe_gate_up_exact_k48_codebook_cache(
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
        sorted_tickets,
        expert_ids,
        num_tickets_post_padded,
        gate_up_workspace,
        num_routes,
        top_k,
        num_m_blocks,
        alignment_block_ratio,
        block_m,
        block_n,
        precompute_tokens,
        output_start_lane,
        fuse_projections,
        0.0 if swiglu_limit is None else float(swiglu_limit),
        preapply_down_norm,
        shared_codebook_entries,
        assignment_l2_only,
    )


def moe_down_padded64(
    *,
    sorted_middle: torch.Tensor,
    codebook: torch.Tensor,
    packed_assignments: torch.Tensor,
    output_norm: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_tickets: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tickets_post_padded: torch.Tensor,
    route_output: torch.Tensor,
    num_routes: int,
    num_m_blocks: int,
    block_m: int,
    block_n: int,
    use_exact_k48: bool = False,
    use_coalesced_k48: bool = False,
    input_norm: torch.Tensor | None = None,
    logical_in_features: int | None = None,
    input_group_start_lane: int = 0,
    preapplied_input_norm: bool = True,
) -> None:
    """Launch the experimental padded-K64 Down control kernel.

    With ``preapplied_input_norm=False``, the CUDA path multiplies the private
    Down input norm after any caller-provided middle round-trip and before the
    lookup matrix multiplication.  It consumes the original D6 codebook and
    word-major block8 packed IDs; it never constructs a dense weight matrix or
    an aligned D8 HBM replica.
    """
    if not preapplied_input_norm and input_norm is None:
        raise ValueError(
            "input_norm is required when preapplied_input_norm=False"
        )
    if (
        sorted_tickets.ndim != 1
        or sorted_tickets.numel() < sorted_middle.shape[0]
    ):
        raise ValueError(
            "sorted_tickets must cover the sorted_middle row capacity"
        )
    # The extension ignores this tensor when the norm was already applied.
    input_norm_tensor = sorted_middle if input_norm is None else input_norm
    logical_width = (
        sorted_middle.shape[1]
        if logical_in_features is None
        else int(logical_in_features)
    )
    _extension().moe_down_padded64(
        sorted_middle,
        codebook,
        packed_assignments,
        input_norm_tensor,
        output_norm,
        topk_weights,
        sorted_tickets,
        expert_ids,
        num_tickets_post_padded,
        route_output,
        num_routes,
        num_m_blocks,
        block_m,
        block_n,
        use_exact_k48,
        use_coalesced_k48,
        logical_width,
        input_group_start_lane,
        preapplied_input_norm,
    )


def moe_down_exact_k48_adaptive(
    *,
    sorted_middle: torch.Tensor,
    codebook: torch.Tensor,
    packed_assignments: torch.Tensor,
    output_norm: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_tickets: torch.Tensor,
    expert_ids16: torch.Tensor,
    num_tickets_post_padded: torch.Tensor,
    tasks64: torch.Tensor,
    tasks16: torch.Tensor,
    task_counts: torch.Tensor,
    route_output: torch.Tensor,
    num_routes: int,
    tasks_per_cta: int = 2,
    input_norm: torch.Tensor | None = None,
    logical_in_features: int | None = None,
    input_group_start_lane: int = 0,
    preapplied_input_norm: bool = True,
) -> None:
    """Run exact-K48 Down over device-built BM64 and BM16 queues."""
    if (
        not isinstance(tasks_per_cta, int)
        or isinstance(tasks_per_cta, bool)
        or tasks_per_cta not in (1, 2, 4, 8)
    ):
        raise ValueError("tasks_per_cta must be one of 1, 2, 4, or 8")
    if not preapplied_input_norm and input_norm is None:
        raise ValueError(
            "input_norm is required when preapplied_input_norm=False"
        )
    input_norm_tensor = sorted_middle if input_norm is None else input_norm
    logical_width = (
        sorted_middle.shape[1]
        if logical_in_features is None
        else int(logical_in_features)
    )
    _extension().moe_down_exact_k48_adaptive(
        sorted_middle,
        codebook,
        packed_assignments,
        input_norm_tensor,
        output_norm,
        topk_weights,
        sorted_tickets,
        expert_ids16,
        num_tickets_post_padded,
        tasks64,
        tasks16,
        task_counts,
        route_output,
        num_routes,
        tasks_per_cta,
        logical_width,
        input_group_start_lane,
        preapplied_input_norm,
    )


def moe_sparse_route_align(
    *,
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align at most 256 routes without an expert-sized workspace.

    Valid routes are ordered by ``(expert, flattened_ticket)``.  Each expert
    run is padded to ``block_size`` with the sentinel ``topk_ids.numel()``.
    The returned buffers have capture-stable capacities ``R * block_size``
    and ``R``; the final CUDA scalar contains the live sorted-ticket prefix.
    Expert IDs outside ``[0, num_experts)`` are ignored.
    """
    sorted_tickets, expert_ids, num_tickets_post_padded = (
        _extension().moe_sparse_route_align(
            topk_ids,
            block_size,
            num_experts,
        )
    )
    return sorted_tickets, expert_ids, num_tickets_post_padded


def moe_build_adaptive_tasks(
    *,
    sorted_tickets: torch.Tensor,
    expert_ids16: torch.Tensor,
    num_tickets_post_padded: torch.Tensor,
    num_routes: int,
    num_experts: int,
    tasks64: torch.Tensor | None = None,
    tasks16: torch.Tensor | None = None,
    task_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-capacity ``int2{row_start, expert}`` task queues.

    Passing all three output tensors lets a graph-captured caller carve the
    queues from a persistent workspace.  Omitting all three allocates tensors
    with the same fixed capacities.  Queue order is intentionally unspecified;
    every task covers a disjoint interval in the BM16-aligned route buffer.
    """
    capacity64, capacity16 = adaptive_task_capacities(
        num_routes=num_routes,
        num_experts=num_experts,
    )
    provided = (tasks64 is not None, tasks16 is not None, task_counts is not None)
    if any(provided) and not all(provided):
        raise ValueError(
            "tasks64, tasks16, and task_counts must be provided together"
        )
    if not any(provided):
        tasks64 = torch.empty(
            (capacity64, 2), device=sorted_tickets.device, dtype=torch.int32
        )
        tasks16 = torch.empty(
            (capacity16, 2), device=sorted_tickets.device, dtype=torch.int32
        )
        task_counts = torch.empty(
            (2,), device=sorted_tickets.device, dtype=torch.int32
        )
    assert tasks64 is not None and tasks16 is not None and task_counts is not None
    if tuple(tasks64.shape) != (capacity64, 2):
        raise ValueError(f"tasks64 must have shape ({capacity64}, 2)")
    if tuple(tasks16.shape) != (capacity16, 2):
        raise ValueError(
            "tasks16 must have the adaptive residual-task shape "
            f"({capacity16}, 2)"
        )
    if tuple(task_counts.shape) != (2,):
        raise ValueError("task_counts must have shape (2,)")
    _extension().moe_build_adaptive_tasks(
        sorted_tickets,
        expert_ids16,
        num_tickets_post_padded,
        tasks64,
        tasks16,
        task_counts,
        num_routes,
        num_experts,
    )
    return tasks64, tasks16, task_counts


def moe_build_adaptive_tasks_tail64(
    *,
    sorted_tickets: torch.Tensor,
    expert_ids16: torch.Tensor,
    num_tickets_post_padded: torch.Tensor,
    num_routes: int,
    num_experts: int,
    tasks64: torch.Tensor | None = None,
    tasks16: torch.Tensor | None = None,
    task_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build BM64/BM16 queues with residual rows 49..63 folded into BM64."""
    capacity64, capacity16 = adaptive_tail64_task_capacities(
        num_routes=num_routes,
        num_experts=num_experts,
    )
    provided = (tasks64 is not None, tasks16 is not None, task_counts is not None)
    if any(provided) and not all(provided):
        raise ValueError(
            "tasks64, tasks16, and task_counts must be provided together"
        )
    if not any(provided):
        tasks64 = torch.empty(
            (capacity64, 2), device=sorted_tickets.device, dtype=torch.int32
        )
        tasks16 = torch.empty(
            (capacity16, 2), device=sorted_tickets.device, dtype=torch.int32
        )
        task_counts = torch.empty(
            (2,), device=sorted_tickets.device, dtype=torch.int32
        )
    assert tasks64 is not None and tasks16 is not None and task_counts is not None
    if tuple(tasks64.shape) != (capacity64, 2):
        raise ValueError(f"tasks64 must have shape ({capacity64}, 2)")
    if tuple(tasks16.shape) != (capacity16, 2):
        raise ValueError(f"tasks16 must have shape ({capacity16}, 2)")
    if tuple(task_counts.shape) != (2,):
        raise ValueError("task_counts must have shape (2,)")
    _extension().moe_build_adaptive_tasks_tail64(
        sorted_tickets,
        expert_ids16,
        num_tickets_post_padded,
        tasks64,
        tasks16,
        task_counts,
        num_routes,
        num_experts,
    )
    return tasks64, tasks16, task_counts


