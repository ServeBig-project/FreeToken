from dataclasses import dataclass

import torch


@dataclass
class DraftResult:
    tokens: torch.Tensor
    probabilities: torch.Tensor
    lengths: list[int]
