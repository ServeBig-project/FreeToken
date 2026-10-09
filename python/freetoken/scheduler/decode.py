from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from freetoken.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)
    # Shared runtime: requests kept out of new batches until they drained and are paused;
    # admission then sets aside one step per running request, not whole outputs.
    held: Set[Req] = field(default_factory=set)
    reserve_outputs: bool = True

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs)
                             if req.can_decode and req not in self.held}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)
        self.held.discard(req)

    def hold(self, req: Req) -> None:
        self.running_reqs.discard(req)
        self.held.add(req)

    def unhold(self, req: Req) -> None:
        self.held.discard(req)
        self.running_reqs.add(req)

    def abort_req(self, uid: int) -> Req | None:
        for reqs in (self.running_reqs, self.held):
            for req in reqs:
                if req.uid == uid:
                    reqs.remove(req)
                    return req
        return None

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        if not self.reserve_outputs:
            return self.page_size * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        reqs = sorted(self.running_reqs, key=lambda req: req.uid)
        return Batch(reqs=reqs, decode_size=len(reqs))

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
