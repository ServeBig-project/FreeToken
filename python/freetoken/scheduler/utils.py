from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import Req, SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    # Layer-major prefill must cut the next original-token chunk before the
    # previous chunk reaches the model's final layer.  This cursor records only
    # that admission boundary; it is not request completion state.
    layered_cached_len: int | None = None
    mm_embeds: torch.Tensor | None = None
    cache_group: str = ""
    # (since, ready length, restorable length) while waiting for a host restore
    restore_wait: tuple[float, int, int] | None = None
    # Order the scheduler received the request in; a paused request keeps it.
    arrival: int = 0
    # A paused request being recomputed: its tokens are the prompt and committed outputs.
    paused: Req | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        if self.paused is not None:
            return self.paused.max_device_len - self.input_len
        return self.sampling_params.max_tokens

    @property
    def prompt_len(self) -> int:
        return self.paused.prompt_len if self.paused is not None else self.input_len


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
