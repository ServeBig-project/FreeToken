"""``RadixCache(has_state=True)``: a recurrent-state slot attached at a node's END boundary.

  * a state is ONE opaque slot id at a node's end, so a *prefix* of that node is not reusable;
  * ``match`` truncates the reusable prefix to the DEEPEST state on the path;
  * splitting a node leaves the state on the suffix half;
  * ``insert`` dedups an existing state and refills a node whose state was evicted;
  * ``evict_states`` counts STATES, keeps an internal node's KV, and frees the KV of a leaf it
    takes, cascading through the stateless leaves it exposes.

Scenarios run at page_size 4 (big enough that page keying differs from token keying).
"""
from __future__ import annotations

from typing import Tuple

import pytest

from .harness import Session

PAGE = 4


def page_ids(P: int, *labels: int) -> Tuple[int, ...]:
    """Whole pages that deliberately share leading tokens."""
    out = []
    for lab in labels:
        out.extend([lab] if P == 1 else [(lab % 3) + 1] + [7] * (P - 2) + [lab])
    return tuple(out)


def ids(*labels: int) -> Tuple[int, ...]:
    return page_ids(PAGE, *labels)


@pytest.fixture
def hyb() -> Session:
    return Session(PAGE, has_state=True)


def hit(s: Session, labels) -> tuple:
    m = s.match(ids(*labels))
    return m.cached_len, m.state


def two_node_tree(s: Session):
    """X=[page1,page2] with state mx, its child Y=[page3,page4] with state my, built through the
    request lifecycle so Y's insert reuses X's own slots and X is strictly older than Y."""
    _, _, _, x_slots, mx = s.insert(ids(1, 2))
    _, _, _, y_slots, my = s.request(ids(1, 2, 3, 4), prompt_len=2 * PAGE)
    s.check()
    return x_slots, y_slots[2 * PAGE:], mx, my


# --------------------------------------------------------------------------- insert / match
def test_insert_attaches_a_state_and_match_restores_it(hyb):
    prefix_len, freed, taken, slots, st = hyb.insert(ids(1, 2))
    assert (prefix_len, freed, taken) == (0, [], True)
    hyb.check()

    m = hyb.match(ids(1, 2))
    assert (m.cached_len, m.kv_indices.tolist(), m.state) == (2 * PAGE, slots, st)
    assert hyb.tree.state_count == 1


def test_deepest_state_wins_and_the_unmatched_suffix_is_dropped(hyb):
    _, _, mx, my = two_node_tree(hyb)
    assert mx != my
    assert hit(hyb, (1, 2, 3, 4)) == (4 * PAGE, my)
    assert hit(hyb, (1, 2, 3, 4, 5)) == (4 * PAGE, my)   # a query past the tree's end
    hyb.check()


def test_split_leaves_the_state_on_the_suffix_half(hyb):
    _, _, mx, my = two_node_tree(hyb)

    assert hit(hyb, (1, 2, 3, 5)) == (2 * PAGE, mx)      # diverges inside Y -> back to X's state
    hyb.check()
    assert hit(hyb, (1, 2, 3, 4)) == (4 * PAGE, my)      # the suffix half kept Y's state
    assert hyb.tree.state_count == 2
    assert hyb.tree.kv_tokens == 4 * PAGE                # a split moves no KV


def test_prefix_of_a_state_node_is_not_reusable(hyb):
    _, _, _, slots, st = hyb.insert(ids(1, 2, 3, 4))
    hyb.check()

    m = hyb.match(ids(1, 2))                             # interior boundary: no state there
    assert (m.cached_len, m.kv_indices.tolist(), m.state) == (0, [], None)
    m = hyb.match(ids(1, 2, 3, 4))
    assert (m.cached_len, m.kv_indices.tolist(), m.state) == (4 * PAGE, slots, st)
    assert hyb.tree.kv_tokens == 4 * PAGE
    hyb.check()


def test_insert_dedups_the_state_and_hands_back_the_duplicates(hyb):
    _, _, _, _, mx = hyb.insert(ids(1, 2))

    prefix_len, freed, taken, slots, loser = hyb.insert(ids(1, 2))
    assert (prefix_len, taken) == (2 * PAGE, False)
    assert freed == slots                                # the caller's duplicate pages
    assert hyb.tree.state_count == 1
    hyb.check()
    assert hit(hyb, (1, 2))[1] == mx                     # the original state is kept
    assert loser in hyb.states.free                      # the loser stayed the caller's


@pytest.mark.parametrize("P", [1, 4, 64], ids=["p1", "p4", "p64"])
def test_insert_stores_whole_pages_only(P):
    """Fewer tokens than a page store nothing (and take no state); a ragged tail stays the
    caller's."""
    s = Session(P, has_state=True)
    prefix_len, _, taken, _, _ = s.insert(page_ids(P, 1)[: P - 1])
    assert (prefix_len, taken) == (0, False)
    assert s.tree.kv_tokens == 0 and s.tree.state_count == 0
    s.check()

    prefix_len, _, taken, slots, _ = s.insert(page_ids(P, 1, 2) + page_ids(P, 3)[: P - 1])
    assert (prefix_len, taken) == (0, True)
    s.check()

    m = s.match(page_ids(P, 1, 2))
    assert (m.cached_len, m.kv_indices.tolist()) == (2 * P, slots[: 2 * P])
    assert s.tree.kv_tokens == 2 * P


# --------------------------------------------------------------------------- evict_states
def test_evict_states_frees_an_internal_state_and_keeps_its_kv(hyb):
    _, _, mx, my = two_node_tree(hyb)

    ev = hyb.evict_states(1)                             # X is the LRU state and is internal
    assert (ev.states, ev.kv) == ([mx], [])
    assert hyb.tree.kv_tokens == 4 * PAGE and hyb.tree.state_count == 1
    hyb.check()

    assert hit(hyb, (1, 2)) == (0, None)                 # no longer resumable there
    assert hit(hyb, (1, 2, 3, 4)) == (4 * PAGE, my)      # the descendant state is untouched


def test_insert_refills_a_node_whose_state_was_evicted(hyb):
    two_node_tree(hyb)
    hyb.evict_states(1)
    hyb.check()

    prefix_len, freed, taken, slots, st = hyb.insert(ids(1, 2))
    assert (prefix_len, taken) == (2 * PAGE, True)       # attaches, does not dedup
    assert freed == slots                                # the KV stays the tree's
    hyb.check()
    assert hit(hyb, (1, 2)) == (2 * PAGE, st)
    assert hyb.tree.state_count == 2


def test_evict_states_on_a_leaf_frees_its_kv_and_cascades(hyb):
    x_slots, y_slots, _, my = two_node_tree(hyb)
    hyb.evict_states(1)                                  # X -> KV only
    ev = hyb.evict_states(1)                             # Y: the only state left, and a leaf
    assert ev.states == [my]
    assert sorted(ev.kv) == sorted(x_slots + y_slots)    # X reclaimed in the same call
    hyb.check()

    assert hyb.tree.kv_tokens == 0 and hyb.tree.state_count == 0
    assert hyb.kv.in_use() == set() and hyb.states.in_use() == set()
    assert hit(hyb, (1, 2, 3, 4)) == (0, None)


def test_evict_states_counts_states_not_tokens(hyb):
    """A one-page node holds PAGE tokens but one state; ``evict_states(2)`` takes two."""
    hyb.insert(ids(1))
    hyb.request(ids(1, 2), prompt_len=PAGE)
    hyb.request(ids(1, 2, 3), prompt_len=2 * PAGE)
    hyb.check()
    assert hyb.tree.state_count == 3 and hyb.tree.kv_tokens == 3 * PAGE

    ev = hyb.evict_states(2)
    assert len(ev.states) == 2 and ev.kv == []           # both were internal
    assert hyb.tree.state_count == 1 and hyb.tree.kv_tokens == 3 * PAGE
    hyb.check()


def test_state_slots_are_conserved_across_eviction_waves(hyb):
    """Every donated slot comes back exactly once, and draining the states drains the KV with
    them through the leaf cascade."""
    n = 8
    hyb.insert(ids(1))
    for k in range(2, n + 1):
        hyb.request(ids(*range(1, k + 1)), prompt_len=(k - 1) * PAGE)
    hyb.check()
    assert hyb.tree.state_count == n and hyb.tree.kv_tokens == n * PAGE

    for _ in range(10):
        if hyb.tree.state_count == 0:
            break
        hyb.evict_states(3)
        hyb.check()
    else:
        pytest.fail(f"evict_states did not converge: {hyb.tree.state_count} left")

    assert hyb.states.in_use() == set()
    assert hyb.kv.in_use() == set() and hyb.tree.kv_tokens == 0
    assert hit(hyb, range(1, n + 1))[0] == 0


# --------------------------------------------------------------------------- evict_kv / locks
def test_evict_kv_takes_the_leaf_state_and_leaves_the_ancestor_usable(hyb):
    _, y_slots, mx, my = two_node_tree(hyb)

    ev = hyb.evict_kv(2 * PAGE)                          # the only unlocked leaf is Y
    assert (ev.kv, ev.states) == (y_slots, [my])         # X keeps its state: no cascade
    assert hyb.tree.kv_tokens == 2 * PAGE and hyb.tree.state_count == 1
    hyb.check()
    assert hit(hyb, (1, 2, 3, 4)) == (2 * PAGE, mx)


def test_evict_kv_cascades_through_an_exposed_stateless_leaf(hyb):
    """A stateless leaf is reclaimed in the same ``evict_kv`` that exposes it."""
    x_slots, y_slots, _, _ = two_node_tree(hyb)
    hyb.evict_states(1)                                  # X -> no state
    assert sorted(hyb.evict_kv(2 * PAGE).kv) == sorted(x_slots + y_slots)
    hyb.check()
    assert hyb.tree.kv_tokens == 0 and hyb.tree.state_count == 0
    assert hyb.kv.in_use() == set() and hyb.states.in_use() == set()


def test_lock_pins_the_state_and_the_whole_kv_path(hyb):
    """A lock protects the matched node's state and the KV up to the root -- but not an
    ancestor's state."""
    _, _, mx, _ = two_node_tree(hyb)
    held = hyb.lock(ids(1, 2, 3, 4))
    hyb.check()

    assert hyb.evict_kv(4 * PAGE).kv == []               # Y locked, X internal
    assert hyb.evictable("kv") == 0

    assert hyb.evict_states(1).states == [mx]            # X's state is unprotected
    assert hyb.evict_states(1).states == []              # Y's state is pinned
    hyb.check()

    hyb.unlock(held)
    assert hyb.evictable("kv") == 4 * PAGE
    hyb.evict_kv(2 * PAGE)                               # Y goes; X cascades behind it
    assert hyb.kv.in_use() == set() and hyb.states.in_use() == set()
    hyb.check()
