"""Physical layouts for packed MoE assignment words.

The logical coordinate order is always ``(expert, output, word)``.  The
existing row-major storage materializes those axes as ``[E, N, W]``; the
word-major storage uses ``[E, W, N]`` so adjacent output rows for one packed
word are contiguous.  Layout conversion materializes a new contiguous tensor
and is intended for checkpoint loading or other preparation work, never a
timed inference call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch


AssignmentLayout: TypeAlias = Literal["row_major", "word_major"]


@dataclass(frozen=True)
class AssignmentLayoutInfo:
    """Logical dimensions and kernel strides for one physical layout."""

    num_experts: int
    out_features: int
    num_words: int
    stride_expert: int
    stride_output: int
    stride_word: int

    @property
    def kernel_strides(self) -> tuple[int, int, int]:
        """Return ``(expert, output, word)`` strides expected by the kernel."""
        return self.stride_expert, self.stride_output, self.stride_word


def _require_packed_assignments(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("packed assignments must be a torch.Tensor")
    if tensor.ndim != 3:
        raise ValueError(
            "packed assignments must have three dimensions, "
            f"got shape {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.int32:
        raise TypeError(
            "packed assignments must use torch.int32, "
            f"got {tensor.dtype}"
        )


def assignment_layout_info(
    packed_assignments: torch.Tensor,
    layout: AssignmentLayout,
) -> AssignmentLayoutInfo:
    """Describe logical dimensions and strides without copying the tensor.

    Triton addresses assignments as ``expert * se + output * sn + word * sw``.
    This helper maps either physical axis order onto that stable semantic
    contract, keeping layout-specific indexing out of the kernel.
    """
    _require_packed_assignments(packed_assignments)
    if layout == "row_major":
        num_experts, out_features, num_words = packed_assignments.shape
        stride_expert, stride_output, stride_word = packed_assignments.stride()
    elif layout == "word_major":
        num_experts, num_words, out_features = packed_assignments.shape
        stride_expert, stride_word, stride_output = packed_assignments.stride()
    else:
        raise ValueError(
            "assignment layout must be 'row_major' or 'word_major', "
            f"got {layout!r}"
        )

    return AssignmentLayoutInfo(
        num_experts=num_experts,
        out_features=out_features,
        num_words=num_words,
        stride_expert=stride_expert,
        stride_output=stride_output,
        stride_word=stride_word,
    )


