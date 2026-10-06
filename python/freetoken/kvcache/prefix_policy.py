"""Which prefix positions to keep a recurrent state at, and what to give up first.

A policy only decides from host metadata the tree already keeps (positions, purposes, access
times); the cache manager applies the decision and owns every slot and reference.
"""
from __future__ import annotations

from typing import List, Tuple

# Why a state was kept: the prompt end of a round, the committed end of a round, a shared
# boundary another request was seen to diverge from, or an explicitly requested anchor.
INPUT, OUTPUT, FORK, ANCHOR = "input", "output", "fork", "anchor"


class BaselinePolicy:
    """The existing behavior: one state per prompt near its end, LRU eviction, no pruning."""

    def checkpoint_positions(self, prompt_len: int, cached_len: int,
                             matched_len: int) -> List[Tuple[int, str]]:
        # The last prompt token is recomputed on reuse, so the reusable end is before it.
        return [(prompt_len - 1, INPUT)]

    def prune_after_commit(self, published, chain) -> list:
        return []

    def eviction_order(self, candidates, *, tier: str, kind: str, required: int) -> list:
        """``candidates``: evictable tree nodes of one ``kind`` (kv, window or state) on one
        ``tier``; the manager evicts in this order until ``required`` units are freed."""
        return sorted(candidates, key=lambda n: n.tic)


class ContinuationPolicy(BaselinePolicy):
    """Keep each round's prompt end and committed end plus observed fork boundaries; drop the
    states a newer round replaces on an unbranched chain."""

    def checkpoint_positions(self, prompt_len, cached_len, matched_len):
        out = super().checkpoint_positions(prompt_len, cached_len, matched_len)
        if cached_len < matched_len < prompt_len - 1:
            # The tokens matched deeper than any state: another history diverges here.
            out.append((matched_len, FORK))
        return out

    def prune_after_commit(self, published, chain) -> list:
        """``chain``: state nodes above the round's deepest boundary, up to the first branch.
        Only replaced round ends go; this round's own states and fork/anchor states stay."""
        return [n for n in chain if n not in published and n.purpose in (INPUT, OUTPUT)]

    def eviction_order(self, candidates, *, tier, kind, required):
        if kind != "state":
            return super().eviction_order(candidates, tier=tier, kind=kind, required=required)
        # By the state's own last real reuse: a round's resume point stays fresh while the
        # session keeps coming back to it, and an abandoned branch's end ages out. (A deeper
        # state is no sign of replacement: the next round usually forks right above it.)
        return sorted(candidates, key=lambda n: n.state_tic)


POLICIES = {"baseline": BaselinePolicy, "continuation": ContinuationPolicy}
