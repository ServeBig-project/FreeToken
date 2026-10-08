"""Windowed ``RadixCache``: what the sliding window adds on top of plain KV reuse.

Window-freed nodes (their window KV gone, their full KV kept), the three insert-side revive cases,
windowed match truncation, the window lock, the two evictors with their cascade, finish-time
restamping and ``trim_head_window``.

Three geometries: ``p1-w4`` (window = 4 pages), ``p4-w8`` (2 pages) and ``p128-w128`` (1 page --
the DSV4 shape). Scenarios are written in pages and in ``wp(s)``, never in literal token counts.
Window frees are observed through the locations the tree returns: slot ids are unique, so the
returned locations name exactly which pages lost their window.
"""
from __future__ import annotations

import pytest

from .harness import Session

GEOMETRIES = ((1, 4), (4, 8), (128, 128))


@pytest.fixture(params=GEOMETRIES, ids=[f"p{p}-w{w}" for p, w in GEOMETRIES])
def s(request) -> Session:
    P, W = request.param
    return Session(P, window=W)


def seq(s: Session, n_pages: int, start: int = 0) -> tuple:
    """Token ids for consecutive pages; distinct pages never share a token."""
    return tuple(t for i in range(start, start + n_pages) for t in [i + 1] * s.P)


def wp(s: Session) -> int:
    """How many whole pages the window spans (1 for the DSV4 page == window shape)."""
    return -(-s.W // s.P)


def chain(s: Session, n_pages: int):
    """root -> N0 -> N1 -> ..., one single-page node each (incremental commits). Returns the ids
    and each page's tree-owned slots."""
    ids = seq(s, n_pages)
    pages = []
    for k in range(1, n_pages + 1):
        slots = s.insert(ids[: k * s.P])[3]
        pages.append(slots[(k - 1) * s.P: k * s.P])
    s.check()
    return ids, pages


def flat(*pages) -> list:
    return [x for p in pages for x in p]


def win(s: Session) -> dict:
    return {"evictable": s.evictable("window"), "protected": s.protected("window")}


# --------------------------------------------------------------------------- insert: window-freed
def test_insert_marks_the_out_of_window_head_window_freed(s):
    """``window_freed_before`` splits a fresh commit into a window-freed head (full KV only) and a
    live tail, on a page boundary."""
    P = s.P
    ids = seq(s, 3)
    s.insert(ids, window_freed_before=P)
    s.check()

    assert s.evictable("kv") == 3 * P
    assert win(s) == {"evictable": 2 * P, "protected": 0}


@pytest.mark.parametrize("n_pages", [1, 2, 3])
def test_suffix_clamp_never_creates_a_window_freed_leaf(s, n_pages):
    """The window-freed suffix is clamped to leave >= one live page."""
    P = s.P
    s.insert(seq(s, n_pages), window_freed_before=(n_pages + 3) * P)
    s.check()
    assert s.evictable("kv") == n_pages * P
    assert win(s)["evictable"] == P


# --------------------------------------------------------------------------- insert: revive
def test_insert_revives_a_whole_window_freed_node(s):
    """The request's window is live over the node's whole span -> adopt the request's locations,
    hand the stale tree locations back."""
    P = s.P
    ids, pages = chain(s, wp(s) + 1)
    s.match(ids)                             # stamps decrease toward the root -> N0 is the LRU
    assert s.evict_window(1).window == pages[0]

    _, freed, _, fresh, _ = s.insert(ids, window_freed_before=0)
    s.check()

    assert set(pages[0]) <= set(freed)       # the stale tree locations came back
    assert set(fresh[P:]) <= set(freed)      # duplicates of the still-live nodes came back
    assert s.match(ids).kv_indices.tolist()[:P] == fresh[:P]
    assert win(s)["evictable"] == (wp(s) + 1) * P


def test_insert_splits_and_revives_the_live_tail(s):
    """The request's window frontier falls inside a multi-page window-freed node -> split at the
    frontier, keep the head window-freed, revive only the tail past it."""
    P = s.P
    ids = seq(s, 4)
    first = s.insert(ids, window_freed_before=3 * P)[3]       # freed [0,3P) + live [3P,4P)
    s.check()
    assert win(s)["evictable"] == P

    _, freed, _, second, _ = s.insert(ids, window_freed_before=2 * P)
    s.check()

    assert set(first[2 * P: 3 * P]) <= set(freed)
    assert set(second[: 2 * P]) <= set(freed)
    assert win(s)["evictable"] == 2 * P                       # the revived page + the live tail
    # what the tree now owns: the head kept its locations, page 2 adopted the request's
    held = first[: 2 * P] + second[2 * P: 3 * P] + first[3 * P:]
    assert sorted(s.evict_kv(10 ** 6).kv) == sorted(held)


def test_insert_keeps_a_still_out_of_window_node(s):
    """The node lies wholly below the request's frontier -> nothing to revive; only the request's
    duplicate comes back and the tree's locations are untouched."""
    P = s.P
    ids, pages = chain(s, wp(s) + 1)
    s.match(ids)
    assert s.evict_window(1).window == pages[0]

    _, freed, _, fresh, _ = s.insert(ids, window_freed_before=P)
    s.check()

    assert set(fresh[:P]) <= set(freed)
    assert s.match(ids).kv_indices.tolist()[:P] == pages[0]
    assert win(s)["evictable"] == wp(s) * P


def test_insert_refuses_to_revive_a_locked_window_freed_node(s):
    """A locked reader still gathers the node's CURRENT locations through its own row, so reviving
    under a lock would hand live KV to the next allocation. The node keeps its locations; the
    caller's window there is still live, so insert stops at the node and the caller keeps its own
    locations from there on (nothing returned as duplicates). Reviving works once the lock clears."""
    P = s.P
    n = wp(s) + 2
    ids, pages = chain(s, n)
    held = s.lock(ids)                       # pins the path, and the trailing window
    assert sorted(s.evict_window(10 ** 6).window) == sorted(pages[0] + pages[1])
    s.check()

    prefix_len, freed, taken, fresh, _ = s.insert(ids, window_freed_before=0)
    s.check()                                # caller-held locations are neither leaked nor freed

    assert s.last_end is None and not taken  # early stop at N0
    assert prefix_len == 0 and freed == []
    assert s.held == fresh                   # the caller still owns every location it passed
    assert s.match(ids).kv_indices.tolist() == flat(*pages)
    s.release_held()                         # the caller frees them itself, exactly once
    s.check()

    s.unlock(held)
    s.insert(ids, window_freed_before=0)
    s.check()
    assert s.last_end is not None
    assert win(s)["evictable"] == n * P      # lock cleared -> reviving is safe again


def test_insert_into_a_locked_window_freed_node_the_caller_also_freed_returns_duplicates(s):
    """The caller's window is freed over the locked node's whole span too -> nothing to revive and
    nothing the caller still reads there: the node keeps its locations and the caller's copy comes
    back as duplicates."""
    P = s.P
    n = wp(s) + 2
    ids, pages = chain(s, n)
    held = s.lock(ids)
    assert sorted(s.evict_window(10 ** 6).window) == sorted(pages[0] + pages[1])

    prefix_len, freed, _, fresh, _ = s.insert(ids, window_freed_before=2 * P)
    s.check()

    assert s.last_end is not None and s.held == []
    assert prefix_len == n * P
    assert set(freed) == set(fresh)          # every incoming location is a duplicate
    assert s.match(ids).kv_indices.tolist() == flat(*pages)
    assert win(s)["evictable"] + win(s)["protected"] == (n - 2) * P

    s.unlock(held)
    s.check()


def test_insert_within_the_reused_prefix_frees_nothing(s):
    """``update_after``: nodes inside the request's reused prefix hold the tree's own locations,
    so insert must neither free them (double free) nor revive them."""
    ids, _ = chain(s, 3)
    held = s.lock(ids)
    m = s.match(ids)
    assert m.cached_len == len(ids)

    prefix_len, freed, _, _, _ = s.insert(ids, m.kv_indices.tolist(), update_after=m.cached_len)
    s.check()
    assert prefix_len == len(ids) and freed == []
    s.unlock(held)
    s.check()


def test_a_split_keeps_both_halves_window_freed(s):
    """Splitting a window-freed node must leave both halves window-freed -- a live half would
    advertise window KV that has already been freed."""
    P = s.P
    ids = seq(s, 3)
    s.insert(ids, window_freed_before=2 * P)               # freed [0,2P) + live [2P,3P)
    other = ids[:P] + seq(s, 1, start=9)                   # shares page 0, diverges at page 1

    assert s.match(other).cached_len == 0
    s.check()
    assert win(s)["evictable"] == P
    assert s.match(ids[:P]).cached_len == 0


def test_a_lock_survives_a_split_at_its_window_boundary(s):
    """A lock taken before a split still releases exactly what it protected."""
    P = s.P
    n = max(2, wp(s))
    ids = seq(s, n)
    s.insert(ids)                            # a single node of n pages
    held = s.lock(ids)
    assert win(s)["protected"] == wp(s) * P  # exactly the trailing window, page-rounded

    s.match(ids[:P] + seq(s, 1, start=9))    # diverges after page 0 -> splits it
    s.check()

    s.unlock(held)
    s.check()
    assert win(s)["protected"] == 0 and s.protected("kv") == 0


# --------------------------------------------------------------------------- windowed match
def test_match_reuses_a_short_prefix_without_freed_windows(s):
    """A path with no window-freed node is reusable however short it is."""
    ids = seq(s, 1)
    s.insert(ids)
    assert s.match(ids).cached_len == s.P   # even though P may be < the window
    s.check()


def test_match_truncates_until_the_live_run_covers_the_window(s):
    """After a window-freed node nothing is reusable until the contiguous live run behind the
    query end reaches the window; the run accumulates across nodes."""
    P, W = s.P, s.W
    ids = seq(s, wp(s) + 1)
    for k in range(2, wp(s) + 2):            # freed page 0 + live pages, one node each
        s.insert(ids[: k * P], window_freed_before=P)
    s.check()
    assert win(s)["evictable"] == wp(s) * P

    for k in range(1, wp(s) + 2):
        live_run = (k - 1) * P
        assert s.match(ids[: k * P]).cached_len == (k * P if live_run >= W else 0)
    s.check()


def test_a_live_run_of_exactly_one_window_between_two_freed_nodes_is_reusable(s):
    """The boundary at a window-freed node is ``>=`` the window: a run of EXACTLY the window is
    covered. The head comes from ``trim_head_window`` (LRU can never age a root-side node past
    its descendants); the second has to be an internal node, so a live node stays below it."""
    P, n_w = s.P, wp(s)
    if n_w == 1:
        pytest.skip("page == window: the live node below the second freed node already covers "
                    "the window on its own, so the boundary cannot be isolated")
    ids, pages = chain(s, n_w + 3)            # freed | n_w live (== W) | freed | live

    assert s.trim(ids, P) == pages[0]
    s.match(ids[: (1 + n_w) * P])             # restamp 1..n_w -> node n_w+1 is the oldest live
    assert s.evict_window(1).window == pages[n_w + 1]
    s.check()

    assert s.match(ids).cached_len == (1 + n_w) * P


# --------------------------------------------------------------------------- window lock
def test_lock_protects_the_window_it_has(s):
    """A live path shorter than the window is protected as far as it goes; a full window is
    protected in full and released by unlock."""
    P, n_w = s.P, wp(s)
    ids = seq(s, n_w)
    if n_w > 1:
        short = ids[: (n_w - 1) * P]
        s.insert(short)
        held = s.lock(short)
        assert win(s)["protected"] == (n_w - 1) * P
        s.unlock(held)
        s.check()

    s.insert(ids)
    held = s.lock(ids)
    assert win(s)["protected"] >= s.W
    s.unlock(held)
    s.check()
    assert win(s)["protected"] == 0


def test_unlock_releases_only_its_own_window(s):
    """Two readers whose windows sit at different depths: releasing the deep one leaves the
    shallow reader's window protected."""
    P, n_w = s.P, wp(s)
    n = n_w + 2
    ids, _ = chain(s, n)

    deep = s.lock(ids)                        # window = the trailing n_w nodes
    shallow = s.lock(ids[: n_w * P])          # window = the leading n_w nodes
    pinned = set(range(n - n_w, n)) | set(range(n_w))
    assert win(s)["protected"] == len(pinned) * P
    assert s.protected("kv") == n * P

    s.unlock(deep)
    s.check()
    assert win(s)["protected"] == n_w * P
    assert s.protected("kv") == n_w * P

    s.unlock(shallow)
    s.check()
    assert win(s)["protected"] == 0 and s.protected("kv") == 0


# --------------------------------------------------------------------------- eviction: KV
def test_evict_kv_takes_unlocked_leaves_only(s):
    """KV eviction takes LEAVES only and never touches a locked path, even when the locked or
    internal nodes are the least recently used."""
    P = s.P
    ids, pages = chain(s, 3)
    held = s.lock(ids)
    assert s.evict_kv(10 ** 6).kv == []
    s.check()

    s.unlock(held)
    s.match(ids)                              # N0 (root-most) is now the oldest node
    assert s.evict_kv(P).kv == pages[2]       # ... yet the leaf N2 must be the victim
    s.check()
    assert {"kv": s.evictable("kv"), "window": s.evictable("window")} == {"kv": 2 * P,
                                                                          "window": 2 * P}


def test_evict_kv_cascades_through_exposed_window_freed_leaves(s):
    """Window-freed internals left as leaves hold full KV nothing can match through again, so
    the KV pass that exposes them reclaims them in the same call."""
    P = s.P
    ids, pages = chain(s, 3)
    s.match(ids)
    assert sorted(s.evict_window(2 * P).window) == sorted(pages[0] + pages[1])
    s.check()

    assert sorted(s.evict_kv(P).kv) == sorted(flat(*pages))
    s.check()
    assert s.tree.kv_tokens == 0 and s.evictable("window") == 0
    assert s.kv.in_use() == set()


# --------------------------------------------------------------------------- eviction: window
def test_evict_window_frees_an_internal_node_in_place(s):
    """Window eviction may take internal nodes: it frees only the window and keeps the full KV,
    so the prefix stays matchable."""
    P, n_w = s.P, wp(s)
    ids, pages = chain(s, n_w + 1)
    s.match(ids)

    ev = s.evict_window(1)
    s.check()
    assert ev.window == pages[0] and ev.kv == []
    assert win(s)["evictable"] == n_w * P

    got = s.match(ids)                        # the live tail covers the window ...
    assert got.cached_len == (n_w + 1) * P
    assert got.kv_indices.tolist()[:P] == pages[0]   # ... so the freed node's full KV is served


def test_evict_window_frees_an_unlocked_leaf_and_cascades(s):
    """An unlocked leaf has nothing to keep its full KV alive for: both are freed, and the
    window-freed ancestor it exposes goes with it."""
    P = s.P
    ids, pages = chain(s, 2)
    s.match(ids)

    ev = s.evict_window(2 * P)
    s.check()
    assert sorted(ev.kv) == sorted(flat(*pages))
    assert s.tree.kv_tokens == 0 and s.kv.in_use() == set()


def test_evict_window_victim_order_follows_match_recency(s):
    """``match`` restamps the matched path with times decreasing toward the root, so window
    eviction takes the root-most STALE node first -- never the freshly matched head."""
    P = s.P
    ids, pages = chain(s, 4)
    s.match(ids[: 2 * P])
    assert s.evict_window(1).window == pages[2]


def test_a_locked_window_survives_both_evictors(s):
    P, n_w = s.P, wp(s)
    n = n_w + 2
    ids, pages = chain(s, n)
    held = s.lock(ids)

    assert s.evict_kv(10 ** 6).kv == []
    ev = s.evict_window(10 ** 6)
    s.check()
    assert ev.kv == []
    assert sorted(ev.window) == sorted(pages[0] + pages[1])
    assert win(s) == {"evictable": 0, "protected": n_w * P}

    s.unlock(held)
    s.check()


def test_window_pressure_drains_every_slot_exactly_once(s):
    """Repeated window eviction converges and, once the window is exhausted, KV eviction drains
    the rest: every location handed out comes back exactly once (the ledger raises otherwise)."""
    P = s.P
    ids, _ = chain(s, 6)
    s.match(ids)

    for guard in range(40):
        if not s.evictable("window"):
            break
        s.evict_window(P)
    else:
        pytest.fail("window eviction is not converging")
    for guard in range(40):
        if not s.evictable("kv"):
            break
        s.evict_kv(P)
    else:
        pytest.fail("KV eviction is not converging")
    s.check()
    assert s.tree.kv_tokens == 0 and s.kv.in_use() == set()


# --------------------------------------------------------------------------- retention
@pytest.mark.parametrize("P, W", GEOMETRIES, ids=[f"p{p}-w{w}" for p, w in GEOMETRIES])
def test_finish_time_restamp_soft_pins_the_prompt_window(P, W):
    """Decode never re-matches the prompt, so at finish the prompt window carries the stamp of
    the prefill-boundary commit and is the first window victim. A finish-time re-match soft-pins
    it and sends the pressure to the idle tail."""
    def run(restamp: bool) -> int:
        s = Session(P, window=W)
        prompt_pages = wp(s)
        ids = seq(s, prompt_pages + 2)
        prompt = ids[: prompt_pages * P]
        s.insert(prompt)                      # prefill-boundary commit
        s.request(ids, prompt_len=len(prompt))
        if restamp:
            s.match(prompt)                   # the finish-time soft pin
        s.evict_window(1)
        s.check()
        return s.match(prompt).cached_len

    assert run(restamp=False) == 0
    assert run(restamp=True) == -(-W // P) * P


def test_trim_head_window_reclaims_the_head_and_keeps_the_window(s):
    """Only the trailing window has to stay live for a next-turn cut, so the head below
    ``keep_from`` loses its window eagerly; its full KV stays and keeps being served."""
    P, n_w = s.P, wp(s)
    n = n_w + 2
    ids, pages = chain(s, n)

    assert sorted(s.trim(ids, 2 * P)) == sorted(pages[0] + pages[1])
    s.check()
    assert win(s)["evictable"] == n_w * P
    assert s.evictable("kv") == n * P

    got = s.match(ids)
    assert got.cached_len == n * P
    assert got.kv_indices.tolist()[: 2 * P] == pages[0] + pages[1]


def test_trim_head_window_skips_locked_leaf_and_freed_nodes(s):
    """A reader still holding the head's window keeps it, a leaf never loses its window, and an
    already window-freed node is not reported twice."""
    P = s.P
    n = wp(s) + 2
    ids, pages = chain(s, n)
    held = s.lock(ids)

    assert s.trim(ids, 0) == []
    assert sorted(s.trim(ids, n * P)) == sorted(pages[0] + pages[1])
    s.check()
    assert s.trim(ids, n * P) == []

    s.unlock(held)
    s.check()
