from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from freetoken.core import Batch, Req

from .forward import ForwardInput
from .prefill import PrefillManager

if TYPE_CHECKING:
    from .decode import DecodeManager
    from .table import TableManager


@dataclass
class ResidentWaveMember:
    uid: int
    planned_chunks: int
    admitted_chunks: int = 0
    latest_req: Req | None = None
    aborted: bool = False


@dataclass
class ResidentWaveAdmission:
    """FIFO membership and complete-request accounting for one resident wave.

    ``soft_chunk_cap`` applies to the sum of complete requests admitted after
    the first member. A first request larger than the cap makes membership
    exclusive; layered-pipeline separately bounds its materialized token range.
    """

    soft_chunk_cap: int
    chunk_token_limit: int
    members: dict[int, ResidentWaveMember] = field(default_factory=dict)
    frozen: bool = False
    exclusive: bool = False

    @property
    def uids(self) -> set[int]:
        return set(self.members)

    @property
    def reserved_chunks(self) -> int:
        return sum(member.planned_chunks for member in self.members.values())

    def refresh_members(self, prefill_manager: PrefillManager) -> list[int]:
        """Refresh exact continuations, then admit complete FIFO requests that fit."""
        if self.frozen:
            return []

        candidates = prefill_manager.pending_wave_candidates(self.chunk_token_limit)
        remaining_by_uid = {uid: chunks for uid, chunks, _ in candidates}
        for member in self.members.values():
            member.planned_chunks = member.admitted_chunks + remaining_by_uid.get(
                member.uid, 0
            )

        added: list[int] = []
        for uid, chunks, multimodal in candidates:
            if uid in self.members:
                continue
            if self.exclusive:
                break
            if not self.members:
                self.members[uid] = ResidentWaveMember(uid, chunks)
                added.append(uid)
                self.exclusive = multimodal or chunks > self.soft_chunk_cap
                if self.exclusive:
                    break
                continue
            if multimodal or self.reserved_chunks + chunks > self.soft_chunk_cap:
                break
            self.members[uid] = ResidentWaveMember(uid, chunks)
            added.append(uid)
        return added

    def freeze(self) -> None:
        self.frozen = True

    def retain_uids(self, uids: set[int]) -> None:
        """Keep the FIFO prefix that was actually materialized for this wave."""
        self.members = {
            uid: member for uid, member in self.members.items() if uid in uids
        }

    def record_materialized_requests(self, reqs: list[Req]) -> None:
        """Attach the one logical request range materialized for each member."""
        for req in reqs:
            member = self.members.get(req.uid)
            if member is None:
                raise RuntimeError(f"resident wave contains unadmitted uid {req.uid}")
            member.latest_req = req
            member.admitted_chunks += 1


def abort_resident_admission(
    admission: ResidentWaveAdmission, uid: int
) -> Req | None:
    member = admission.members.get(uid)
    if member is None:
        return None
    owner = member.latest_req
    if owner is None:
        del admission.members[uid]
        return None
    member.aborted = True
    owner.aborted = True
    return owner


def write_and_filter(
    forward_input: ForwardInput,
    next_tokens_gpu: torch.Tensor,
    table_manager: TableManager,
    decode_manager: DecodeManager,
) -> None:
    table_manager.token_pool[forward_input.write_tuple] = next_tokens_gpu
    decode_manager.filter_reqs(forward_input.batch.reqs)


def request_output_view(
    forward_input: ForwardInput, request_indices: list[int]
) -> ForwardInput:
    """Build the request-aligned output view after logits were selected by row."""
    from freetoken.engine.sample import BatchSamplingArgs

    batch = Batch(
        reqs=[forward_input.batch.reqs[index] for index in request_indices],
        decode_size=0,
    )
    args = forward_input.sample_args
    selected_args = BatchSamplingArgs(
        temperatures=(
            args.temperatures[request_indices]
            if args.temperatures is not None
            else None
        ),
        top_k=(args.top_k[request_indices] if args.top_k is not None else None),
        top_p=(args.top_p[request_indices] if args.top_p is not None else None),
    )
    write_tuple = (
        forward_input.write_tuple[0][request_indices],
        forward_input.write_tuple[1][request_indices],
    )
    return ForwardInput(
        batch=batch,
        sample_args=selected_args,
        input_tuple=forward_input.input_tuple,
        write_tuple=write_tuple,
    )


__all__ = [
    "ResidentWaveAdmission",
    "ResidentWaveMember",
    "abort_resident_admission",
    "request_output_view",
    "write_and_filter",
]
