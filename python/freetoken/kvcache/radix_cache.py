"""One prefix tree for every cache layout, over GPU and host copies.

A node is a run of committed tokens. Its paged KV lives at ``value``: full-pool locations that
every paged tensor family (attention KV, draft history KV, DSV4 compressed rows) indexes the same
way. Two optional components ride on the same nodes:

* window: sliding-window KV reached through the pool's full->window mapping. ``window_freed``
  marks a node whose GPU window KV is gone (its full KV survives). A position is resumable for
  the window only with ``window`` live tokens behind it, or a path with no freed node back to
  root. ``window_ref`` locks just the trailing window, bounded by ``window_uuid`` (sglang).
* state: a recurrent state slot belonging to the node's END position, with its own lock count.
  Splitting never creates a state; it stays on the node that still ends at its position.

Each kind of data may also have a host copy (``host``, ``host_window``, ``host_state``: spans of
``prefix_store.HostCopy``), alone or next to the GPU copy. GPU data stays prefix-closed: it
leaves for the host leafward and comes back rootward. A position is *ready* when every
component is on the GPU there, *restorable* when each is on the GPU or the host.

Leaves that are not restorable hold nothing reusable and are reclaimed whenever eviction exposes
them. Locks protect GPU data; ``busy`` marks nodes a copy is reading or filling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import torch
from freetoken.utils import align_down

KEY_FN = Callable[[torch.Tensor], Any]

# A match/insert event's clock value is multiplied by this so the event can stamp its path with
# depth offsets that decrease toward the root without colliding with the next event: LRU then
# reclaims near-root nodes of one path first.
_EVENT_STRIDE = 1 << 24


class TreeNode:
    def __init__(self, key: torch.Tensor, value: torch.Tensor | None, tic: int) -> None:
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
        self.purpose: str | None = None  # why the state is kept (prefix_policy)
        self.state_tic = 0               # the state's last real reuse
        self.state_doomed = False        # dropped once its last lock goes
        self.host: list | None = None         # host spans of the paged components
        self.host_window: list | None = None  # host spans of the window component
        self.host_state: list | None = None   # host spans of the state component
        self.busy = False                     # a copy reads or fills this node

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
    state: int | None = None         # state slot to restore the request's live state from
    matched_len: int = 0             # tokens shared with the tree, past what is resumable
    restore: TreeNode | None = None  # deeper resume point whose missing data is on the host
    restore_len: int = 0
    window_uuid: int | None = None   # set by lock: where the window lock stops
    state_locked: bool = False

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class Evicted(NamedTuple):
    kv: torch.Tensor      # full-pool locations to free
    window: torch.Tensor  # full-pool locations whose window slots to free
    states: List[int]     # state slots to free


@dataclass(eq=False)
class CopyPlan:
    """Data to move for one resume point: ``(node, units)`` of paged KV and window, and
    whether the point's state moves. ``handle`` locks the path while the copy runs."""

    node: TreeNode
    kv: List[Tuple[TreeNode, int]]
    window: List[Tuple[TreeNode, int]]
    state: bool
    handle: CacheHandle
    nodes: List[TreeNode] = field(default_factory=list)  # every node marked busy


def _key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())


class RadixCache:
    def __init__(self, device: torch.device, page_size: int, policy, stats: dict, *,
                 window: int | None = None, has_state: bool = False) -> None:
        self.device = device
        self.page_size = page_size
        self.policy = policy
        self.stats = stats  # cumulative counters, owned by the manager across rebuilds
        self.window = window
        self.has_state = has_state
        self.key_fn = _key_fn(page_size)
        # int32 like the page table, so an empty result never promotes a concatenation.
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.roots: Dict[str, TreeNode] = {}
        self.evictable = dict.fromkeys(("kv", "window", "state"), 0)
        self.protected = dict.fromkeys(("kv", "window", "state"), 0)
        self.host_states = 0
        self._clk = 0
        self._uuid = 0
        self._released = Evicted([], [], [])  # GPU data dropped outside an eviction call

    # ---------------------------------------------------------------- match / insert
    def match(self, ids: torch.Tensor, group: str = "", *, reuse: bool = False) -> CacheHandle:
        """Deepest ready position on the matched path, and the deepest restorable one, stamping
        the walked path. ``reuse`` marks the returned state as really reused (not ancestors)."""
        path = self._walk(ids, group)
        ready, ready_pos, restorable, restorable_pos, pos = -1, 0, -1, 0, 0
        on_gpu = on_host = True
        runs = {"gpu": [0, False], "host": [0, False]}  # live window run, freed node seen
        for i, node in enumerate(path):
            pos += node.length
            on_gpu = on_gpu and node.value is not None
            on_host = on_host and (node.value is not None or node.host is not None)
            if not on_host:
                break
            gpu_window = node.value is not None and not node.window_freed
            ok = {}
            for tier, has_window, has_state in (
                ("gpu", gpu_window, node.state is not None),
                ("host", gpu_window or node.host_window is not None,
                 node.state is not None or node.host_state is not None),
            ):
                run = runs[tier]
                if has_window:
                    run[0] += node.length
                else:
                    run[:] = [0, True]
                ok[tier] = ((self.window is None or not run[1] or run[0] >= self.window)
                            and (not self.has_state or has_state))
            if on_gpu and ok["gpu"]:
                ready, ready_pos = i, pos
            if ok["host"]:
                restorable, restorable_pos = i, pos
        self._stamp(path[-1] if path else self._root(group))
        restore = path[restorable] if restorable > ready else None
        if ready < 0:
            return CacheHandle(0, self._root(group), self.empty, matched_len=pos,
                               restore=restore, restore_len=restorable_pos)
        node = path[ready]
        if reuse and node.state is not None:
            node.state_tic = self._tick()
        kv = torch.cat([n.value for n in path[: ready + 1]])
        return CacheHandle(ready_pos, node, kv, state=node.state, matched_len=pos,
                           restore=restore, restore_len=restorable_pos)

    def insert(self, ids: torch.Tensor, kv: torch.Tensor, *, group: str = "",
               state: int | None = None, purpose: str | None = None, update_after: int = 0,
               window_freed_before: int = 0) -> Tuple[int, torch.Tensor, bool, TreeNode]:
        """Insert the committed page-aligned prefix of ``ids`` with locations ``kv``.

        ``update_after`` is the request's reused-prefix length: matched nodes past it carry the
        request's own fresh locations, which become duplicates to free -- unless the node lacks
        that data on the GPU (window freed, or only a host copy) and the request holds it, in
        which case the node adopts the request's locations. Positions below
        ``window_freed_before`` had their window freed by the request, so a new suffix there is
        inserted window-freed. ``state`` is donated to the end node when that node has none.
        Returns (matched length before insertion, locations the caller frees from every pool,
        whether ``state`` was taken, the end node)."""
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
                if child.value is None or child.window_freed:
                    child = self._adopt(child, seg, total, window_freed_before, freed)
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
            node.state, node.purpose, node.state_tic = state, purpose, self._tick()
            node.state_doomed = False
            self._account("state", 1, locked=node.state_ref > 0)
            taken = True
            if node.host_state is None:
                self.stats["checkpoint_created"] += 1
            self.stats["gpu_checkpoint_peak"] = max(
                self.stats["gpu_checkpoint_peak"], self.state_count)
        elif state is not None and node.state is not None:
            self.stats["checkpoint_deduplicated"] += 1
        return total, (torch.cat(freed) if freed else self.empty), taken, node

    def _adopt(self, child: TreeNode, seg: torch.Tensor, total: int, freed_before: int,
               freed: List[torch.Tensor]) -> TreeNode:
        """Give ``child`` the request's GPU copy of what it lacks on the GPU (its window, or
        everything when only the host has it), where the request still holds it."""
        assert child.window_ref == 0, "a window-freed node cannot hold a window lock"
        end = total + child.length
        live = self.window is None or freed_before <= total  # request's window covers it all
        if child.ref > 0 or child.busy or (
                child.value is not None and freed_before >= end):
            # A reader still uses the node's current data, a copy is filling it, or the
            # request freed this window too: keep the node, drop the request's copy.
            freed.append(seg.clone())
            return child
        if not live and freed_before < end:
            # The request's freed frontier falls inside: the head keeps what it has.
            start = freed_before - total
            self._split(child, start)
            freed.append(seg[:start].clone())
            seg, live = seg[start:], True
        if child.value is not None:
            freed.append(child.value)
        else:
            self._account("kv", child.length, locked=False)
        child.value = seg.clone()
        child.window_freed = not live
        child.tic = self._tick()
        if self.window is not None and live:
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
                self._move("kv", self._gpu_len(cur), to_protected=True)
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
            self._drop_if_doomed(node)
            handle.state_locked = False
        dec_window, cur = self.window is not None, node
        while not cur.is_root():
            cur.ref -= 1
            assert cur.ref >= 0
            if cur.ref == 0:
                self._move("kv", self._gpu_len(cur), to_protected=False)
            if dec_window and not cur.window_freed and cur.window_ref > 0:
                cur.window_ref -= 1
                if cur.window_ref == 0:
                    self._move("window", cur.length, to_protected=False)
                if handle.window_uuid is not None and cur.window_uuid == handle.window_uuid:
                    dec_window = False
            cur = cur.parent

    # ---------------------------------------------------------------- GPU eviction
    def evict_kv(self, num_tokens: int) -> Evicted:
        """Free paged KV leafward in the policy's order: a node with a host copy just drops its
        GPU copy; one without is removable only as a leaf, with everything it holds."""
        out = Evicted([], [], [])
        freed = 0

        def eligible(n: TreeNode) -> bool:
            return (n.value is not None and n.ref == 0 and not n.busy
                    and (n.host is not None or n.is_leaf())
                    and all(c.value is None for c in n.children.values()))

        while freed < num_tokens:
            cands = [n for n in self._nodes() if eligible(n)]
            if not cands:
                break
            for node in self.policy.eviction_order(cands, tier="gpu", kind="kv",
                                                   required=num_tokens - freed):
                if freed >= num_tokens:
                    break
                if not (self._attached(node) and eligible(node)):
                    continue
                freed += node.length
                if node.host is not None:
                    self._offload(node, out)
                    freed += self._reclaim_dead(node, out)[1]
                else:
                    self._remove(node, out)
                    freed += self._reclaim_dead(node.parent, out)[1]
        return self._evicted(out)

    def evict_window(self, num_tokens: int) -> Evicted:
        """Free GPU window KV from unlocked live windows, inner nodes included."""
        return self._evict_component(
            num_tokens, "window",
            lambda n: n.value is not None and not n.window_freed and n.window_ref == 0,
            lambda n: n.length, self._drop_window)

    def evict_states(self, num: int) -> Evicted:
        """Free GPU state slots from unlocked states, inner nodes included."""
        return self._evict_component(
            num, "state", lambda n: n.state is not None and n.state_ref == 0,
            lambda n: 1, self._drop_gpu_state)

    def _evict_component(self, amount: int, kind: str, eligible, size, drop) -> Evicted:
        out = Evicted([], [], [])
        cands = [n for n in self._nodes() if eligible(n) and not n.busy]
        freed = 0
        for node in self.policy.eviction_order(cands, tier="gpu", kind=kind, required=amount):
            if freed >= amount:
                break
            if not (self._attached(node) and eligible(node)):
                continue
            freed += size(node)
            drop(node, out)
            self._reclaim_dead(node, out)
        return self._evicted(out)

    # ---------------------------------------------------------------- host tier
    def evict_host(self, nbytes: int, enough: Callable[[], bool]) -> None:
        """Release host copies in the policy's order until ``enough()`` (the host store can
        place the request: shared copies free memory only with their last span). Paged host
        KV goes only where the GPU still has it or no descendant needs it."""
        cands = [n for n in self._nodes() if not n.busy and n.ref == 0
                 and (n.host or n.host_window or n.host_state)]
        for node in self.policy.eviction_order(cands, tier="host", kind="host", required=nbytes):
            if enough():
                break
            if not self._attached(node) or node.busy or node.ref:
                continue
            node.host_window = _release(node.host_window)
            self._drop_host_state(node)
            if node.host is not None and node.value is not None:
                node.host = _release(node.host)
            elif node.host is not None and node.is_leaf():
                self._remove(node, self._released)
                self._reclaim_dead(node.parent, self._released)
            else:
                self._reclaim_dead(node, self._released)

    def plan_restore(self, node: TreeNode) -> CopyPlan | None:
        """Lock ``node``'s path and list what comes back from the host to make it ready; None
        while another copy works on any of it."""
        path = self._path(node)
        kv = [(n, n.length // self.page_size) for n in path if n.value is None]
        window = [(n, n.length // self.page_size) for n in self._window_nodes(node)
                  if n.value is None or n.window_freed]
        state = self.has_state and node.state is None
        nodes = list(dict.fromkeys([n for n, _ in kv + window] + ([node] if state else [])))
        if any(n.busy for n in nodes):
            return None
        return self._plan(node, kv, window, state, nodes)

    def finish_restore(self, plan: CopyPlan, kv: List[torch.Tensor], state: int | None) -> None:
        """Publish restored locations onto whatever nodes now cover each planned range."""
        ps = self.page_size
        for (n, units), value in zip(plan.kv, kv, strict=True):
            for piece, start, count in self._pieces(n, units):
                piece.value, piece.window_freed = value[start * ps : (start + count) * ps], True
                self._account("kv", piece.length, locked=True)
        for n, units in plan.window:
            for piece, _, _ in self._pieces(n, units):
                piece.window_freed = False
                self._account("window", piece.length, locked=False)
        if plan.state:
            plan.node.state = state
            self._account("state", 1, locked=False)
            self.stats["gpu_checkpoint_peak"] = max(
                self.stats["gpu_checkpoint_peak"], self.state_count)
        self._end_plan(plan)

    def plan_backup(self, node: TreeNode) -> CopyPlan | None:
        """Lock ``node``'s path and list its GPU data that has no host copy yet."""
        kv = [(n, n.length // self.page_size) for n in self._path(node)
              if n.value is not None and n.host is None and not n.busy]
        window = [(n, n.length // self.page_size) for n in self._window_nodes(node)
                  if n.value is not None and not n.window_freed and n.host_window is None
                  and not n.busy]
        state = (self.has_state and node.state is not None and node.host_state is None
                 and not node.busy)
        nodes = list(dict.fromkeys([n for n, _ in kv + window] + ([node] if state else [])))
        if not nodes:
            return None
        return self._plan(node, kv, window, state, nodes)

    def finish_backup(self, plan: CopyPlan, kv: list, window: list, state: list | None) -> None:
        """Attach the finished host copies; a node split meanwhile shares its copy's spans."""
        from .prefix_store import HostSpan

        for (n, units), copies in zip(plan.kv, kv, strict=True):
            self._attach(n, units, copies, "host")
        for (n, units), copies in zip(plan.window, window, strict=True):
            self._attach(n, units, copies, "host_window")
        if state is not None:
            plan.node.host_state = [HostSpan(c, 0, 1) for c in state]
            self.host_states += 1
            self.stats["host_checkpoint_peak"] = max(
                self.stats["host_checkpoint_peak"], self.host_states)
        self._end_plan(plan)

    def drop_gpu(self) -> None:
        """Idle GPU re-allocation: forget every GPU copy (its locations are about to become
        invalid) and keep only what the host still holds."""
        for node in sorted(self._nodes(), key=lambda n: -len(self._path(n))):  # leaves first
            node.value, node.window_freed, node.window_ref, node.window_uuid = None, True, 0, None
            if node.state is not None:
                node.state = None
                if node.host_state is None:
                    node.purpose = None
            if node.is_leaf() and not self._resumable_end(node):
                self._remove(node, Evicted([], [], []))
        self.evictable = dict.fromkeys(self.evictable, 0)
        self.protected = dict.fromkeys(self.protected, 0)

    def abandon(self, plan: CopyPlan) -> None:
        self._end_plan(plan)

    def _plan(self, node, kv, window, state, nodes) -> CopyPlan:
        handle = CacheHandle(0, node, self.empty)
        self.lock(handle)
        for n in nodes:
            n.busy = True
        return CopyPlan(node, kv, window, state, handle, nodes)

    def _end_plan(self, plan: CopyPlan) -> None:
        cur = plan.node
        while not cur.is_root():  # split heads created meanwhile inherited busy
            cur.busy = False
            cur = cur.parent
        for n in plan.nodes:
            n.busy = False
        self.unlock(plan.handle)
        for n in plan.nodes:
            self._drop_if_doomed(n)

    def _attach(self, node: TreeNode, units: int, copies: list, attr: str) -> None:
        from .prefix_store import HostSpan

        pieces = self._pieces(node, units)
        for c in copies:
            c.refs += len(pieces) - 1  # one reference per span
        for piece, start, count in pieces:
            setattr(piece, attr, [HostSpan(c, start, count) for c in copies])

    def _pieces(self, node: TreeNode, units: int) -> List[Tuple[TreeNode, int, int]]:
        """The nodes that now cover what was ``node``'s ``units`` units when a copy was planned:
        a split keeps ``node`` as the tail and puts the head on new parents. Yields
        (node, first unit, units), tail first."""
        out = []
        while units > 0:
            own = node.length // self.page_size
            units -= own
            out.append((node, units, own))
            node = node.parent
        return out

    # ---------------------------------------------------------------- pruning
    def state_chain(self, node: TreeNode) -> List[TreeNode]:
        """State nodes above ``node`` on its unbranched path (stopping at the first branch)."""
        out, cur = [], node.parent
        while cur is not None and not cur.is_root() and len(cur.children) == 1:
            if cur.state is not None or cur.host_state is not None:
                out.append(cur)
            cur = cur.parent
        return out

    def drop_states(self, nodes: List[TreeNode]) -> None:
        """Drop these states from both tiers now, or when their last lock goes."""
        for node in nodes:
            if node.state_doomed:
                continue
            self.stats["checkpoint_pruned"] += 1
            node.state_doomed = True
            self._drop_if_doomed(node)

    def _drop_if_doomed(self, node: TreeNode) -> None:
        """Carry out a pruning decision once no lock or copy uses the state."""
        if node.state_doomed and node.state_ref == 0 and not node.busy:
            if node.state is not None:
                self._drop_gpu_state(node, self._released, count=False)
            self._drop_host_state(node, count=False)
            node.state_doomed = False

    def take_released(self) -> Evicted:
        out, self._released = self._released, Evicted([], [], [])
        return self._evicted(out)

    def trim_head_window(self, ids: torch.Tensor, keep_from: int, group: str = "") -> torch.Tensor:
        """Free the GPU window of the path strictly below ``keep_from`` (page-aligned), keeping
        full KV: only the trailing window before a resume point needs to stay live. Locked,
        freed and leaf nodes are left alone. Returns the locations whose window slots to free."""
        if keep_from <= 0:
            return self.empty
        self.match(ids[:keep_from], group)  # splits a node boundary at keep_from
        out = Evicted([], [], [])
        node, pos = self._root(group), 0
        while pos < keep_from:
            child = node.children.get(self.key_fn(ids[pos:]))
            if child is None or pos + child.length > keep_from:
                break
            if (child.value is not None and not child.window_freed and child.window_ref == 0
                    and not child.is_leaf() and not child.busy):
                self._drop_window(child, out)
            node, pos = child, pos + child.length
        return self._evicted(out).window

    # ---------------------------------------------------------------- accounting / checks
    def check_integrity(self) -> None:
        for n in self._nodes():
            assert n.ref >= n.window_ref >= 0 and n.state_ref >= 0
            if n.window_freed:
                assert n.window_ref == 0, "a window-freed node cannot hold a window lock"
            assert n.value is not None or n.window_freed, "a host-only node has no GPU window"

    @property
    def kv_tokens(self) -> int:
        return self.evictable["kv"] + self.protected["kv"]

    @property
    def state_count(self) -> int:
        return self.evictable["state"] + self.protected["state"]

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

    def _path(self, node: TreeNode) -> List[TreeNode]:
        out = []
        while not node.is_root():
            out.append(node)
            node = node.parent
        return out[::-1]

    def _window_nodes(self, node: TreeNode) -> List[TreeNode]:
        """The nodes holding the window that ends at ``node``."""
        out, covered = [], 0
        while self.window is not None and not node.is_root() and covered < self.window:
            out.append(node)
            covered += node.length
            node = node.parent
        return out

    def _split(self, node: TreeNode, pos: int) -> TreeNode:
        """Cut ``node`` at ``pos``; returns the new prefix node. Locks, the window state, host
        copies and ``busy`` cover both halves; the window-lock boundary moves to the root-side
        half; the state stays."""
        assert 0 < pos < node.length
        parent = node.parent
        del parent.children[self.key_fn(node.key)]
        value = None if node.value is None else node.value[:pos]
        head = TreeNode(node.key[:pos], value, node.tic)
        head.ref, head.busy = node.ref, node.busy
        head.window_freed, head.window_ref = node.window_freed, node.window_ref
        head.window_uuid, node.window_uuid = node.window_uuid, None
        units = pos // self.page_size
        for attr in ("host", "host_window"):
            spans = getattr(node, attr)
            if spans is not None:
                halves = [s.split(units) for s in spans]
                setattr(head, attr, [h for h, _ in halves])
                setattr(node, attr, [t for _, t in halves])
        self._link(head, parent)
        node.key = node.key[pos:]
        if node.value is not None:
            node.value = node.value[pos:]
        self._link(node, head)
        return head

    def _link(self, child: TreeNode, parent: TreeNode) -> None:
        child.parent = parent
        parent.children[self.key_fn(child.key)] = child

    def _unlink(self, node: TreeNode) -> None:
        del node.parent.children[self.key_fn(node.key)]

    def _attached(self, node: TreeNode) -> bool:
        return node.parent is not None and (
            node.parent.children.get(self.key_fn(node.key)) is node)

    def _gpu_len(self, node: TreeNode) -> int:
        return node.length if node.value is not None else 0

    def _resumable_end(self, node: TreeNode) -> bool:
        if node.value is None and node.host is None:
            return False
        if self.has_state and node.state is None and node.host_state is None:
            return False
        gpu_window = node.value is not None and not node.window_freed
        return self.window is None or gpu_window or node.host_window is not None

    def _offload(self, node: TreeNode, out: Evicted) -> None:
        """Drop the GPU copy of a node whose paged KV is also on the host."""
        out.kv.append(node.value)
        self._account("kv", -node.length, locked=False)
        if self.window is not None and not node.window_freed:
            self._drop_window(node, out)
        node.window_freed = True
        if node.state is not None:
            self._drop_gpu_state(node, out)
        node.value = None

    def _remove(self, node: TreeNode, out: Evicted) -> int:
        """Unlink an unlocked leaf and hand back everything it holds."""
        if node.value is not None:
            out.kv.append(node.value)
            self._account("kv", -node.length, locked=False)
            if self.window is not None and not node.window_freed:
                self._drop_window(node, out)
        if node.state is not None:
            self._drop_gpu_state(node, out)
        self._drop_host_state(node)
        node.host = _release(node.host)
        node.host_window = _release(node.host_window)
        self._unlink(node)
        return node.length

    def _reclaim_dead(self, node: TreeNode, out: Evicted) -> Tuple[TreeNode, int]:
        """Remove the non-restorable unlocked leaves a removal exposed, walking up. Returns the
        surviving ancestor and the tokens removed."""
        freed = 0
        while (node.is_leaf() and node.ref == 0 and not node.is_root() and not node.busy
               and not self._resumable_end(node)):
            parent = node.parent
            freed += self._remove(node, out)
            node = parent
        return node, freed

    def _drop_window(self, node: TreeNode, out: Evicted) -> None:
        out.window.append(node.value)
        node.window_freed = True
        self._account("window", -node.length, locked=False)

    def _drop_gpu_state(self, node: TreeNode, out: Evicted, count: bool = True) -> None:
        """Drop the GPU copy of a state; the state is gone once no host copy remains."""
        out.states.append(node.state)
        node.state = None
        self._account("state", -1, locked=False)
        if node.host_state is None:
            self.stats["checkpoint_evicted"] += count
            node.purpose, node.state_doomed = None, False

    def _drop_host_state(self, node: TreeNode, count: bool = True) -> None:
        if node.host_state is None:
            return
        node.host_state = _release(node.host_state)
        self.host_states -= 1
        if node.state is None:
            self.stats["checkpoint_evicted"] += count
            node.purpose, node.state_doomed = None, False

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


def _release(spans: list | None) -> None:
    for s in spans or ():
        s.copy.release()
    return None



__all__ = ["RadixCache", "CacheHandle", "CopyPlan", "Evicted", "TreeNode"]
