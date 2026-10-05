"""One prefix tree for every cache layout.

A node is a run of committed tokens. Its paged KV lives at ``value``: full-pool locations that
every paged tensor family (attention KV, draft history KV, DSV4 compressed rows) indexes the same
way. Two optional components ride on the same nodes:

* window: sliding-window KV reached through the pool's full->window mapping. ``window_freed``
  marks a node whose window KV slid out (its full KV survives). A position is resumable for the
  window only with ``window`` live tokens behind it, or a path with no freed node back to root.
  ``window_ref`` locks just the trailing window, bounded by ``window_uuid`` (mirrors sglang).
* state: a recurrent state slot belonging to the node's END position, with its own lock count.
  Splitting never creates a state; it stays on the node that still ends at its position.

A position is resumable when every component the tree carries can resume there. Leaves that are
not resumable hold nothing reusable and are reclaimed whenever eviction exposes them.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import torch
from freetoken.utils import align_down

KEY_FN = Callable[[torch.Tensor], Any]

# A match/insert event's clock value is multiplied by this so the event can stamp its path with
# depth offsets that decrease toward the root without colliding with the next event: LRU then
# reclaims near-root nodes of one path first.
_EVENT_STRIDE = 1 << 24


class TreeNode:
    def __init__(self, key: torch.Tensor, value: torch.Tensor, tic: int) -> None:
        self.key = key
        self.value = value
        self.parent: TreeNode | None = None
        self.children: Dict[Any, TreeNode] = {}
        self.tic = tic
        self.ref = 0
        self.window_freed = False
        self.window_ref = 0
        self.window_uuid: int | None = None
        self.state: int | None = None
        self.state_ref = 0

    @property
    def length(self) -> int:
        return len(self.key)

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return not self.children

    def match_len(self, ids: torch.Tensor) -> int:
        from freetoken.kernel import fast_compare_key

        return fast_compare_key(self.key, ids)

    def __lt__(self, other: TreeNode) -> bool:
        return self.tic < other.tic


@dataclass(eq=False)
class CacheHandle:
    """A matched (or committed) prefix: the lock target and the reusable KV locations."""

    cached_len: int
    node: TreeNode | None
    kv_indices: torch.Tensor
    state: int | None = None        # state slot to restore the request's live state from
    window_uuid: int | None = None  # set by lock: where the window lock stops
    state_locked: bool = False

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class Evicted(NamedTuple):
    kv: torch.Tensor      # full-pool locations to free
    window: torch.Tensor  # full-pool locations whose window slots to free
    states: List[int]     # state slots to free


def _key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())


class RadixCache:
    def __init__(self, device: torch.device, page_size: int, *, window: int | None = None,
                 has_state: bool = False) -> None:
        if has_state:
            from freetoken.kernel.fla.chunk import CHUNK_SIZE

            # States land on CHUNK_SIZE boundaries; page-aligned so KV and state boundaries meet.
            assert CHUNK_SIZE % page_size == 0, (
                f"state caching needs CHUNK_SIZE({CHUNK_SIZE}) % page_size({page_size}) == 0"
            )
        self.device = device
        self.page_size = page_size
        self.window = window
        self.has_state = has_state
        self.key_fn = _key_fn(page_size)
        # int32 like the page table, so an empty result never promotes a concatenation.
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.roots: Dict[str, TreeNode] = {}
        self.evictable = dict.fromkeys(("kv", "window", "state"), 0)
        self.protected = dict.fromkeys(("kv", "window", "state"), 0)
        self._clk = 0
        self._uuid = 0

    # ---------------------------------------------------------------- match / insert
    def match(self, ids: torch.Tensor, group: str = "") -> CacheHandle:
        """Deepest resumable position on the matched path, stamping the walked path."""
        path = self._walk(ids, group)
        best, pos, best_pos, live, freed_seen = -1, 0, 0, 0, False
        for i, node in enumerate(path):
            pos += node.length
            if self.window is not None:
                if node.window_freed:
                    freed_seen, live = True, 0
                else:
                    live += node.length
                if freed_seen and live < self.window:
                    continue
            if self.has_state and node.state is None:
                continue
            best, best_pos = i, pos
        self._stamp(path[-1] if path else self._root(group))
        if best < 0:
            return CacheHandle(0, self._root(group), self.empty)
        node = path[best]
        kv = torch.cat([n.value for n in path[: best + 1]])
        return CacheHandle(best_pos, node, kv, state=node.state)

    def insert(self, ids: torch.Tensor, kv: torch.Tensor, *, group: str = "",
               state: int | None = None, update_after: int = 0,
               window_freed_before: int = 0) -> Tuple[int, torch.Tensor, bool]:
        """Insert the committed page-aligned prefix of ``ids`` with locations ``kv``.

        ``update_after`` is the request's reused-prefix length: matched nodes past it carry the
        request's own fresh locations, which become duplicates to free -- unless the node's window
        was freed and the request holds it live, in which case the node adopts the request's
        locations (revive). Positions below ``window_freed_before`` had their window freed by the
        request, so a new suffix there is inserted window-freed. ``state`` is donated to the end
        node when that node has none. Returns (matched length before insertion, locations the
        caller frees from every pool, whether ``state`` was taken)."""
        n = align_down(len(ids), self.page_size)
        ids, kv = ids[:n], kv[:n]
        freed: List[torch.Tensor] = []
        node, total = self._root(group), 0
        while total < n:
            child = node.children.get(self.key_fn(ids[total:]))
            if child is None:
                break
            m = align_down(child.match_len(ids[total:]), self.page_size)
            if m == 0:
                break
            partial = m < child.length
            if partial:
                child = self._split(child, m)
            seg = kv[total : total + m]
            if update_after < total + m:
                if child.window_freed:
                    child = self._revive(child, seg, total, window_freed_before, freed)
                else:
                    freed.append(seg.clone())
            total += m
            node = child
            if partial:
                break
        if total < n:
            suffix_ids, suffix_kv = ids[total:], kv[total:].clone()
            if self.window is not None:
                head = max(0, min(window_freed_before, n) - total)
                # Never a window-freed leaf: the request keeps at least its last page live.
                head = min(head, max(0, len(suffix_ids) - self.page_size))
                if head > 0:
                    node = self._add_child(node, suffix_ids[:head], suffix_kv[:head], freed=True)
                    suffix_ids, suffix_kv = suffix_ids[head:], suffix_kv[head:]
            if len(suffix_ids):
                node = self._add_child(node, suffix_ids, suffix_kv, freed=False)
        taken = False
        if state is not None and not node.is_root() and node.state is None:
            node.state = state
            self._account("state", 1, locked=node.state_ref > 0)
            taken = True
        return total, (torch.cat(freed) if freed else self.empty), taken

    def _revive(self, child: TreeNode, seg: torch.Tensor, total: int, freed_before: int,
                freed: List[torch.Tensor]) -> TreeNode:
        assert child.window_ref == 0, "a window-freed node cannot hold a window lock"
        end = total + child.length
        if child.ref > 0 or freed_before >= end:
            # A locked reader still gathers the node's current locations through its own row, or
            # the request freed this window too: keep the node as is, drop the request's copy.
            freed.append(seg.clone())
            return child
        if freed_before > total:
            # The request's freed frontier falls inside: the head stays freed, revive the tail.
            start = freed_before - total
            self._split(child, start)
            freed.append(seg[:start].clone())
            seg = seg[start:]
        freed.append(child.value)
        child.value = seg.clone()
        child.window_freed = False
        child.tic = self._tick()
        self._account("window", child.length, locked=False)
        return child

    def _add_child(self, parent: TreeNode, ids: torch.Tensor, kv: torch.Tensor, *,
                   freed: bool) -> TreeNode:
        child = TreeNode(ids, kv, self._tick())
        self._link(child, parent)
        child.window_freed = freed
        self._account("kv", child.length, locked=False)
        if self.window is not None and not freed:
            self._account("window", child.length, locked=False)
        return child

    # ---------------------------------------------------------------- locking
    def lock(self, handle: CacheHandle) -> None:
        """Protect the handle's path KV, its trailing window and its state."""
        node = handle.node
        if node.state is not None:
            if node.state_ref == 0:
                self._move("state", 1, to_protected=True)
            node.state_ref += 1
            handle.state_locked = True
        window_locked, cur = 0, node
        while not cur.is_root():
            if cur.ref == 0:
                self._move("kv", cur.length, to_protected=True)
            cur.ref += 1
            if (self.window is not None and window_locked < self.window
                    and not cur.window_freed):
                if cur.window_ref == 0:
                    self._move("window", cur.length, to_protected=True)
                cur.window_ref += 1
                window_locked += cur.length
                if window_locked >= self.window:
                    if cur.window_uuid is None:
                        self._uuid += 1
                        cur.window_uuid = self._uuid
                    handle.window_uuid = cur.window_uuid
            cur = cur.parent

    def unlock(self, handle: CacheHandle) -> None:
        node = handle.node
        if handle.state_locked:
            node.state_ref -= 1
            if node.state_ref == 0 and node.state is not None:
                self._move("state", 1, to_protected=False)
            handle.state_locked = False
        dec_window, cur = self.window is not None, node
        while not cur.is_root():
            cur.ref -= 1
            assert cur.ref >= 0
            if cur.ref == 0:
                self._move("kv", cur.length, to_protected=False)
            if dec_window and not cur.window_freed and cur.window_ref > 0:
                cur.window_ref -= 1
                if cur.window_ref == 0:
                    self._move("window", cur.length, to_protected=False)
                if handle.window_uuid is not None and cur.window_uuid == handle.window_uuid:
                    dec_window = False
            cur = cur.parent

    # ---------------------------------------------------------------- eviction
    def evict_kv(self, num_tokens: int) -> Evicted:
        """Free paged KV by LRU over unlocked leaves (an inner node is every descendant's
        prefix), with everything else the leaf holds."""
        out = Evicted([], [], [])
        heap = [n for n in self._nodes() if n.is_leaf() and n.ref == 0]
        heapq.heapify(heap)
        freed = 0
        while freed < num_tokens and heap:
            node = heapq.heappop(heap)
            if node.ref != 0 or not node.is_leaf():
                continue
            freed += self._remove(node, out)
            parent, cascaded = self._reclaim_dead(node.parent, out)
            freed += cascaded
            if parent.is_leaf() and parent.ref == 0 and not parent.is_root():
                heapq.heappush(heap, parent)
        return self._evicted(out)

    def evict_window(self, num_tokens: int) -> Evicted:
        """Free window KV by LRU over unlocked live windows, inner nodes included: an inner or
        KV-locked node only loses its window; a free leaf is removed whole."""
        return self._evict_component(
            num_tokens, lambda n: not n.window_freed and n.window_ref == 0,
            lambda n: n.length, self._drop_window)

    def evict_states(self, num: int) -> Evicted:
        """Free state slots by LRU over unlocked states, inner nodes included."""
        return self._evict_component(
            num, lambda n: n.state is not None and n.state_ref == 0,
            lambda n: 1, self._drop_state)

    def _evict_component(self, amount: int, eligible, size, drop) -> Evicted:
        out = Evicted([], [], [])
        heap = [n for n in self._nodes() if eligible(n)]
        heapq.heapify(heap)
        freed = 0
        while freed < amount and heap:
            node = heapq.heappop(heap)
            if not eligible(node):
                continue
            freed += size(node)
            if node.is_leaf() and node.ref == 0:
                self._remove(node, out)
                self._reclaim_dead(node.parent, out)
            else:
                drop(node, out)
        return self._evicted(out)

    def trim_head_window(self, ids: torch.Tensor, keep_from: int, group: str = "") -> torch.Tensor:
        """Free the window of the path strictly below ``keep_from`` (page-aligned), keeping full
        KV: only the trailing window before a resume point needs to stay live. Locked, freed and
        leaf nodes are left alone. Returns the locations whose window slots to free."""
        if keep_from <= 0:
            return self.empty
        self.match(ids[:keep_from], group)  # splits a node boundary at keep_from
        out = Evicted([], [], [])
        node, pos = self._root(group), 0
        while pos < keep_from:
            child = node.children.get(self.key_fn(ids[pos:]))
            if child is None or pos + child.length > keep_from:
                break
            if not child.window_freed and child.window_ref == 0 and not child.is_leaf():
                self._drop_window(child, out)
            node, pos = child, pos + child.length
        return self._evicted(out).window

    # ---------------------------------------------------------------- accounting / checks
    def check_integrity(self) -> None:
        for n in self._nodes():
            assert n.ref >= n.window_ref >= 0 and n.state_ref >= 0
            if n.window_freed:
                assert n.window_ref == 0, "a window-freed node cannot hold a window lock"

    @property
    def kv_tokens(self) -> int:
        return self.evictable["kv"] + self.protected["kv"]

    # ---------------------------------------------------------------- helpers
    def _root(self, group: str) -> TreeNode:
        root = self.roots.get(group)
        if root is None:
            root = self.roots[group] = TreeNode(self.empty, self.empty, 0)
            root.ref = 1  # never evicted
        return root

    def _walk(self, ids: torch.Tensor, group: str) -> List[TreeNode]:
        node, pos, path = self._root(group), 0, []
        while pos < len(ids):
            child = node.children.get(self.key_fn(ids[pos:]))
            if child is None:
                break
            m = align_down(child.match_len(ids[pos:]), self.page_size)
            if m == 0:
                break
            if m < child.length:
                path.append(self._split(child, m))
                break
            path.append(child)
            node, pos = child, pos + m
        return path

    def _split(self, node: TreeNode, pos: int) -> TreeNode:
        """Cut ``node`` at ``pos``; returns the new prefix node. Locks and the window state cover
        both halves; the window-lock boundary moves to the root-side half; the state stays."""
        assert 0 < pos < node.length
        parent = node.parent
        del parent.children[self.key_fn(node.key)]
        head = TreeNode(node.key[:pos], node.value[:pos], node.tic)
        head.ref = node.ref
        head.window_freed, head.window_ref = node.window_freed, node.window_ref
        head.window_uuid, node.window_uuid = node.window_uuid, None
        self._link(head, parent)
        node.key, node.value = node.key[pos:], node.value[pos:]
        self._link(node, head)
        return head

    def _link(self, child: TreeNode, parent: TreeNode) -> None:
        child.parent = parent
        parent.children[self.key_fn(child.key)] = child

    def _unlink(self, node: TreeNode) -> None:
        del node.parent.children[self.key_fn(node.key)]

    def _resumable_end(self, node: TreeNode) -> bool:
        if self.has_state and node.state is None:
            return False
        return not (self.window is not None and node.window_freed)

    def _remove(self, node: TreeNode, out: Evicted) -> int:
        """Unlink an unlocked leaf and hand back everything it holds."""
        out.kv.append(node.value)
        self._account("kv", -node.length, locked=False)
        if self.window is not None and not node.window_freed:
            self._drop_window(node, out)
        if node.state is not None:
            self._drop_state(node, out)
        self._unlink(node)
        return node.length

    def _reclaim_dead(self, node: TreeNode, out: Evicted) -> Tuple[TreeNode, int]:
        """Remove the non-resumable unlocked leaves a removal exposed, walking up. Returns the
        surviving ancestor and the tokens removed."""
        freed = 0
        while (node.is_leaf() and node.ref == 0 and not node.is_root()
               and not self._resumable_end(node)):
            parent = node.parent
            freed += self._remove(node, out)
            node = parent
        return node, freed

    def _drop_window(self, node: TreeNode, out: Evicted) -> None:
        out.window.append(node.value)
        node.window_freed = True
        self._account("window", -node.length, locked=False)

    def _drop_state(self, node: TreeNode, out: Evicted) -> None:
        out.states.append(node.state)
        node.state = None
        self._account("state", -1, locked=False)

    def _evicted(self, out: Evicted) -> Evicted:
        cat = lambda xs: torch.cat(xs) if xs else self.empty
        return Evicted(cat(out.kv), cat(out.window), out.states)

    def _account(self, kind: str, amount: int, *, locked: bool) -> None:
        (self.protected if locked else self.evictable)[kind] += amount

    def _move(self, kind: str, amount: int, *, to_protected: bool) -> None:
        sign = 1 if to_protected else -1
        self.protected[kind] += sign * amount
        self.evictable[kind] -= sign * amount

    def _nodes(self) -> List[TreeNode]:
        out, stack = [], list(self.roots.values())
        while stack:
            n = stack.pop()
            if not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _stamp(self, node: TreeNode) -> None:
        self._clk += 1
        base, off = self._clk * _EVENT_STRIDE, 0
        while not node.is_root():
            node.tic = base - off
            off += 1
            node = node.parent

    def _tick(self) -> int:
        self._clk += 1
        return self._clk * _EVENT_STRIDE


__all__ = ["RadixCache", "CacheHandle", "Evicted", "TreeNode"]
