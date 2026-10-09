"""One ``RadixCache`` plus slot ledgers, driven through the public interface only.

Slot ids are globally unique and never reused, so every location or state slot the tree hands
back names exactly one earlier hand-out: the ledgers turn "what came back" into exact assertions
and catch double frees and leaks. ``check()`` runs the tree's own ``check_integrity`` and the
conservation checks after a scenario step.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from freetoken.kvcache.prefix_policy import POLICIES
from freetoken.kvcache.radix_cache import RadixCache

STATS_KEYS = ("checkpoint_created", "checkpoint_deduplicated", "checkpoint_pruned",
              "checkpoint_evicted", "gpu_checkpoint_peak", "host_checkpoint_peak")


def make_tree(page_size: int, **kw) -> RadixCache:
    return RadixCache(torch.device("cpu"), page_size, POLICIES["baseline"](),
                      dict.fromkeys(STATS_KEYS, 0), **kw)


def ids_tensor(ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(ids), dtype=torch.int64)


def slots_tensor(slots: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(slots), dtype=torch.int32)


class Ledger:
    """Never-reused slot ids, and the book of which ones came back."""

    def __init__(self, base: int) -> None:
        self._next = base
        self.handed: List[int] = []
        self.free: set = set()

    def take(self, n: int) -> List[int]:
        out = list(range(self._next, self._next + n))
        self._next += n
        self.handed.extend(out)
        return out

    def release(self, slots: Sequence[int]) -> None:
        for s in slots:
            assert s in self.handed, f"slot {s} came back but was never handed out"
            assert s not in self.free, f"slot {s} came back twice"
            self.free.add(s)

    def in_use(self) -> set:
        return set(self.handed) - self.free


class Evicted:
    def __init__(self, ev) -> None:
        self.kv: List[int] = ev.kv.tolist()
        self.window: List[int] = ev.window.tolist()
        self.states: List[int] = [int(s) for s in ev.states]


class Session:
    def __init__(self, page_size: int, *, window: Optional[int] = None,
                 has_state: bool = False) -> None:
        self.P = page_size
        self.W = window
        self.has_state = has_state
        self.tree = make_tree(page_size, window=window, has_state=has_state)
        self.kv = Ledger(1_000_000)
        self.states = Ledger(9_000_000)
        self.window_freed: List[int] = []
        self.held: List[int] = []      # locations an early-stopped insert left with the caller
        self.last_end = None
        self.last_window: List[int] = []

    # -- ops ------------------------------------------------------------------
    def match(self, ids: Sequence[int]):
        return self.tree.match(ids_tensor(ids))

    def insert(self, ids: Sequence[int], slots: Optional[Sequence[int]] = None, *,
               update_after: int = 0, window_freed_before: int = 0):
        """Returns ``(prefix_len, freed, state_taken, slots, state)`` with ``freed`` the returned
        duplicates (``Evicted.kv``); ``last_window`` keeps the subset whose window binding the
        caller still holds (``Evicted.window``). Everything the caller gives up -- the freed duplicates, the ragged tail, an untaken state -- goes back to the ledgers.
        An early stop (end node ``None``) leaves the full pages from ``prefix_len`` on with the
        caller: they go to ``held`` until ``release_held``."""
        if slots is None:
            slots = self.kv.take(len(ids))
        state = self.states.take(1)[0] if self.has_state else None
        prefix_len, freed, taken, self.last_end = self.tree.insert(
            ids_tensor(ids), slots_tensor(slots), state=state, update_after=update_after,
            window_freed_before=window_freed_before)
        assert list(freed.states) == []
        self.last_window = freed.window.tolist()
        freed = freed.kv.tolist()
        live = set(slots[window_freed_before:]) if self.W is not None else set()
        assert sorted(self.last_window) == sorted(set(freed) & live)
        full = (len(ids) // self.P) * self.P
        if self.last_end is None:
            self.held.extend(slots[int(prefix_len): full])
        self.kv.release(freed + list(slots[full:]))
        if state is not None and not taken:
            self.states.release([state])
        return int(prefix_len), freed, bool(taken), list(slots), state

    def release_held(self) -> None:
        self.kv.release(self.held)
        self.held = []

    def lock(self, ids: Sequence[int]):
        h = self.match(ids)
        self.tree.lock(h)
        return h

    def unlock(self, handle) -> None:
        self.tree.unlock(handle)

    def _evicted(self, ev) -> Evicted:
        out = Evicted(ev)
        self.kv.release(out.kv)
        self.states.release(out.states)
        assert not set(out.window) & set(self.window_freed), "a window was freed twice"
        self.window_freed.extend(out.window)
        return out

    def evict_kv(self, n: int) -> Evicted:
        return self._evicted(self.tree.evict_kv(n))

    def evict_window(self, n: int) -> Evicted:
        return self._evicted(self.tree.evict_window(n))

    def evict_states(self, n: int) -> Evicted:
        return self._evicted(self.tree.evict_states(n))

    def request(self, ids: Sequence[int], prompt_len: int):
        """Match a prompt, lock it, commit the extended sequence on top, unlock: the only way a
        scenario produces ``update_after > 0`` -- the slots below it ARE the tree's own."""
        h = self.match(ids[:prompt_len])
        self.tree.lock(h)
        slots = h.kv_indices.tolist() + self.kv.take(len(ids) - h.cached_len)
        out = self.insert(ids, slots, update_after=h.cached_len)
        self.tree.unlock(h)
        return out

    # -- reading --------------------------------------------------------------
    def evictable(self, kind: str) -> int:
        return self.tree.evictable[kind]

    def protected(self, kind: str) -> int:
        return self.tree.protected[kind]

    def check(self) -> None:
        self.tree.check_integrity()
        kv_live = len(self.kv.in_use() - set(self.held))
        assert self.tree.kv_tokens == kv_live, (self.tree.kv_tokens, kv_live)
        assert self.evictable("kv") + self.protected("kv") == kv_live
        if self.has_state:
            assert self.tree.state_count == len(self.states.in_use())
            assert self.evictable("state") + self.protected("state") == self.tree.state_count
