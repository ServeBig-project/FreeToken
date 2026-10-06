"""Plain KV prefix cache: ``RadixCache`` without a window or recurrent state.

Page size is parametrized over {1, 4} wherever it is load-bearing. Slot ids are globally unique and
never reused, so ``s.kv.free`` is an exact record of what the cache handed back.
"""
from __future__ import annotations

import pytest

from .harness import Session, ids_tensor, slots_tensor


@pytest.fixture(params=(1, 4), ids=("p1", "p4"))
def s(request) -> Session:
    return Session(request.param)


@pytest.fixture
def s4() -> Session:
    return Session(4)


def page(P: int, tag: int) -> tuple:
    """One page. At page_size > 1 every page starts with the same token: distinct pages that
    share leading tokens are exactly the shape a token-keyed tree gets wrong."""
    return (tag,) if P == 1 else (1,) + (7,) * (P - 2) + (tag,)


def seq(P: int, *tags: int) -> tuple:
    return tuple(t for tag in tags for t in page(P, tag))


def kv(s: Session, kind: str = "kv") -> dict:
    return {"evictable": s.evictable(kind), "protected": s.protected(kind)}


# --------------------------------------------------------------------------- match / insert
def test_cold_match_is_empty_and_a_commit_round_trips(s):
    P = s.P
    ids = seq(P, 1, 2, 3)

    cold = s.match(ids)
    assert (cold.cached_len, cold.kv_indices.tolist()) == (0, [])
    assert kv(s) == {"evictable": 0, "protected": 0}

    prefix_len, freed, _, slots, _ = s.insert(ids)
    assert prefix_len == 0 and freed == []    # nothing cached, nothing duplicated

    warm = s.match(ids)
    assert (warm.cached_len, warm.kv_indices.tolist()) == (len(ids), slots)
    assert kv(s) == {"evictable": len(ids), "protected": 0}
    s.check()


def test_cold_handle_get_matched_indices_is_empty(s4):
    ids = seq(4, 1)
    cold = s4.match(ids)
    assert cold.cached_len == 0
    assert cold.get_matched_indices().tolist() == []

    _, _, _, slots, _ = s4.insert(ids)
    assert s4.match(ids).get_matched_indices().tolist() == slots
    s4.check()


def test_insert_drops_a_trailing_partial_page(s):
    P = s.P
    ids = seq(P, 1, 2) + (9,) * (P - 1)      # a ragged tail at page_size 4, nothing at 1
    kept = 2 * P
    _, _, _, slots, _ = s.insert(ids)

    m = s.match(ids)
    assert (m.cached_len, m.kv_indices.tolist()) == (kept, slots[:kept])
    assert s.tree.kv_tokens == kept           # the partial page stays the caller's
    s.check()


@pytest.mark.parametrize("P", (1, 4, 128))
@pytest.mark.parametrize("window", (None, "page"))
def test_match_takes_a_ragged_length_and_stops_at_the_page_boundary(P, window):
    """``match`` accepts a raw length and rounds the reuse down to the page below; a windowed
    tree whose path has no window-freed node behaves the same."""
    s = Session(P, window=P if window else None)
    ids = seq(P, 1, 2)
    _, _, _, slots, _ = s.insert(ids)

    m = s.match(ids[: 2 * P - 1])            # one token short of the second page
    assert (m.cached_len, m.kv_indices.tolist()) == (P, slots[:P])
    full = s.match(ids)                      # a split inside the node is transparent
    assert (full.cached_len, full.kv_indices.tolist()) == (2 * P, slots)
    s.check()


def test_mid_node_divergence_rounds_down_to_the_page_boundary(s):
    P = s.P
    ids = seq(P, 1, 2)
    _, _, _, slots, _ = s.insert(ids)

    alt = ids[:-1] + (999,)                  # identical for 2P-1 tokens, differs in the last
    m = s.match(alt)
    assert (m.cached_len, m.kv_indices.tolist()) == (P, slots[:P])
    s.check()


@pytest.mark.parametrize("kw", ({}, {"window": 4}, {"has_state": True}),
                         ids=("plain", "window", "state"))
def test_pages_sharing_leading_tokens_do_not_share_a_prefix(kw):
    """The reuse unit is a whole page: two different pages with a common token prefix share
    nothing, in every kind of tree."""
    s = Session(4, **kw)
    one, two = (1, 3, 1, 5, 3, 2, 2, 9), (1, 3, 1, 6, 0, 0, 0, 0)
    _, _, _, s1, _ = s.insert(one)
    prefix_len, _, _, s2, _ = s.insert(two)
    assert prefix_len == 0
    for ids, slots in ((one, s1), (two, s2)):
        m = s.match(ids)
        assert (m.cached_len, m.kv_indices.tolist()) == (8, slots)
    s.check()


def test_reinserting_a_cached_prefix_hands_back_the_duplicates(s4):
    P = s4.P
    ids = seq(P, 1, 2)
    _, _, _, first, _ = s4.insert(ids)

    prefix_len, freed, _, second, _ = s4.insert(ids)
    assert prefix_len == len(ids)
    assert freed == second                   # the caller's duplicates come back to be freed
    assert s4.match(ids).kv_indices.tolist() == first   # the tree kept its own slots
    assert s4.tree.kv_tokens == len(ids)
    s4.check()


def test_request_lifecycle_reuses_the_cached_prefix(s):
    P = s.P
    full = seq(P, 1, 2, 3)
    _, _, _, s_head, _ = s.insert(full[:P])

    _, freed, _, _, _ = s.request(full, prompt_len=P)
    assert freed == [] and s.kv.free == set()   # a reused prefix duplicates and leaks nothing

    m = s.match(full)
    assert m.cached_len == 3 * P
    assert m.kv_indices.tolist()[:P] == s_head
    assert kv(s) == {"evictable": 3 * P, "protected": 0}
    s.check()


def test_insert_owns_its_indices_and_survives_caller_mutation(s4):
    """The tree must COPY the locations it stores: the caller reuses its staging buffers."""
    ids = list(range(1, 9))
    slots = s4.kv.take(len(ids))
    buf = slots_tensor(slots)
    s4.tree.insert(ids_tensor(ids), buf)
    buf.fill_(-1)
    assert s4.match(ids).kv_indices.tolist() == slots


# --------------------------------------------------------------------------- locking
def test_lock_survives_a_split(s):
    P = s.P
    ids = seq(P, 1, 2)
    s.insert(ids)
    held = s.lock(ids)
    assert kv(s) == {"evictable": 0, "protected": 2 * P}

    s.match(ids[:P])                         # splits the locked node in two
    assert kv(s) == {"evictable": 0, "protected": 2 * P}
    s.check()

    s.unlock(held)
    assert kv(s) == {"evictable": 2 * P, "protected": 0}
    s.check()


def test_lock_accounting_walks_to_the_root(s):
    P = s.P
    head, tail = seq(P, 1), seq(P, 1, 2)
    s.insert(head)
    s.insert(tail)
    assert kv(s) == {"evictable": 2 * P, "protected": 0}

    deep = s.lock(tail)                      # protects the leaf AND everything up to the root
    assert kv(s) == {"evictable": 0, "protected": 2 * P}
    again = s.lock(tail)
    assert kv(s) == {"evictable": 0, "protected": 2 * P}
    s.unlock(again)
    assert kv(s) == {"evictable": 0, "protected": 2 * P}
    s.unlock(deep)
    assert kv(s) == {"evictable": 2 * P, "protected": 0}

    shallow = s.lock(head)                   # a lock protects the root path, not descendants
    assert kv(s) == {"evictable": P, "protected": P}
    s.unlock(shallow)
    assert kv(s) == {"evictable": 2 * P, "protected": 0}
    s.check()


# --------------------------------------------------------------------------- eviction
def test_eviction_is_lru_ordered_over_leaves(s):
    P = s.P
    a, b, c = seq(P, 1), seq(P, 2), seq(P, 3)
    sa, sb, sc = (s.insert(x)[3] for x in (a, b, c))
    s.match(a)                               # a becomes the most recently used

    assert s.evict_kv(P).kv == sb            # b was the oldest
    assert s.evict_kv(P).kv == sc
    still = s.match(a)
    assert (still.cached_len, still.kv_indices.tolist()) == (P, sa)
    assert s.match(b).cached_len == 0

    assert s.evict_kv(P).kv == sa
    assert s.tree.kv_tokens == 0 and kv(s) == {"evictable": 0, "protected": 0}
    s.check()


def test_eviction_takes_the_leaf_before_the_parent(s):
    P = s.P
    head, tail = seq(P, 1), seq(P, 1, 2)
    s_head = s.insert(head)[3]
    s_tail = s.insert(tail)[3]

    assert s.evict_kv(P).kv == s_tail[P:]    # only leaves are collectible
    assert kv(s) == {"evictable": P, "protected": 0}
    m = s.match(head)
    assert (m.cached_len, m.kv_indices.tolist()) == (P, s_head)

    assert s.evict_kv(P).kv == s_head        # now the parent is a leaf itself
    assert s.tree.kv_tokens == 0
    s.check()


def test_eviction_cascades_to_a_newly_childless_parent(s):
    P = s.P
    s_head = s.insert(seq(P, 1))[3]
    s_tail = s.insert(seq(P, 1, 2))[3]

    # ONE call: the parent only becomes collectible once its last child is gone.
    assert sorted(s.evict_kv(2 * P).kv) == sorted(s_tail[P:] + s_head)
    assert s.tree.kv_tokens == 0 and kv(s) == {"evictable": 0, "protected": 0}
    s.check()


def test_locked_nodes_are_never_evicted(s):
    P = s.P
    a, b = seq(P, 1), seq(P, 2)
    sa, sb = s.insert(a)[3], s.insert(b)[3]

    held = s.lock(a)
    assert kv(s) == {"evictable": P, "protected": P}
    assert s.evict_kv(10 ** 6).kv == sb      # clamped to what is evictable
    m = s.match(a)
    assert (m.cached_len, m.kv_indices.tolist()) == (P, sa)

    s.unlock(held)
    assert kv(s) == {"evictable": P, "protected": 0}
    assert s.evict_kv(P).kv == sa
    s.check()


def test_evict_zero_is_a_no_op_and_evict_all_empties_the_tree(s4):
    P = s4.P
    assert s4.evict_kv(0).kv == []           # safe on an empty tree

    ids = seq(P, 1, 2)
    slots = s4.insert(ids)[3]
    assert s4.evict_kv(0).kv == []
    assert s4.match(ids).kv_indices.tolist() == slots

    assert s4.evict_kv(len(ids)).kv == slots
    assert kv(s4) == {"evictable": 0, "protected": 0}
    assert s4.match(ids).cached_len == 0
    s4.check()

