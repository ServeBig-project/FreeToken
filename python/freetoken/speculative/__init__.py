from dataclasses import dataclass

import torch


@dataclass
class DraftResult:
    tokens: torch.Tensor
    probabilities: torch.Tensor
    lengths: list[int]
    # The drafter's own history for this round: ``commit(retained)`` with the target's state.
    history: object | None = None
