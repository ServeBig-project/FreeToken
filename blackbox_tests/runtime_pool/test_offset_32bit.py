"""Legal slot-major addresses past 2^31 elements from the buffer base (real 30-layer dims).

Each buffer is a single dense allocation; selected (row, layer) blocks hold random data and
everything else is zero. The same operator runs on a small compact pool holding only the
selected rows, and results must match bit for bit; nothing outside the written blocks may
change. Tests skip when the GPU has no room for the allocation. Replay positions are per
input token.
"""

import math

import pytest
import torch

from freetoken.kernel import causal_conv1d as sgl
from freetoken.kernel import store_cache
from freetoken.kernel.triton import causal_conv1d_triton as tri
from freetoken.kernel.triton.gdn_replay import (gdn_replay, gdn_replay_conv, gdn_replay_fold)
from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla
from rp_common import BF16, D, DEV, HK, HV, K, KW, REAL_LAYERS, SCALE, V, gen, ints, randn

L = REAL_LAYERS
LAST = L - 1
I32 = 2 ** 31


class BigPool:
    def __init__(self, g, n, inner, dtype, live, scale=1.0, layers=L):
        nbytes = n * layers * math.prod(inner) * torch.tensor([], dtype=dtype).element_size()
        free, _ = torch.cuda.mem_get_info()
        if free < nbytes + (2 << 30):
            pytest.skip(f"needs {nbytes / 2**30:.1f} GiB + margin, {free / 2**30:.1f} GiB free")
        self.storage = torch.zeros(n, layers, *inner, dtype=dtype, device=DEV)
        self.live = {}
        for r, l in live:
            self.storage[r, l] = randn(g, *inner, dtype=dtype, scale=scale)
            self.live[(r, l)] = self.storage[r, l].clone()
        self.lm = self.storage.transpose(0, 1)
        self.row_elems = layers * math.prod(inner)
        self.block_elems = math.prod(inner)

    def offset(self, r, l):
        return r * self.row_elems + l * self.block_elems

    def compact(self, rows, layer):
        return torch.stack([self.storage[r, layer] for r in rows])

    def assert_untouched(self, written):
        expected = 0
        for key, saved in self.live.items():
            if key not in written:
                assert torch.equal(self.storage[key], saved), key
        for r, l in set(self.live) | set(written):
            expected += int(torch.count_nonzero(self.storage[r, l]))
        total = sum(int(torch.count_nonzero(c)) for c in self.storage.view(-1).split(1 << 28))
        assert total == expected

    def free(self):
        del self.storage, self.lm, self.live
        torch.cuda.empty_cache()


def test_recurrent_state_past_2pow31():
    """decode / prefill / replay / fold with checkpoint slots past 2^31 fp32 elements."""
    g = gen(61)
    n = 138  # slot stride 30 x 32 x 128 x 128 = 15,728,640 elements
    hi, mid, lo = n - 1, n - 2, 0
    live = [(hi, LAST), (mid, LAST), (lo, LAST), (hi, LAST - 1), (mid, 0), (1, LAST)]
    pool = BigPool(g, n, (HV, V, K), torch.float32, live, scale=0.05)
    assert pool.offset(hi, 0) >= I32 and pool.offset(mid, LAST) >= I32 > pool.offset(mid, 0)
    view = pool.lm[LAST]
    written = set()

    # decode: last layer of the three slots, plus padding
    sel = [hi, mid, lo]
    comp = pool.compact(sel, LAST)
    q, k = randn(g, 1, 4, HK, K, dtype=BF16), randn(g, 1, 4, HK, K, dtype=BF16)
    v, a, b = randn(g, 1, 4, HV, V, dtype=BF16), randn(g, 4, HV), randn(g, 4, HV)
    A_log, dt_bias = randn(g, HV, scale=0.5), randn(g, HV, scale=0.5)
    cu = ints([0, 1, 2, 3, 4])
    kw = dict(A_log=A_log, dt_bias=dt_bias, cu_seqlens=cu, scale=SCALE)
    o_big = gdn_decode_fla(q, k, v, a, b, state_source=view, indices=ints(sel + [-1]), **kw)
    o_cmp = gdn_decode_fla(q, k, v, a, b, state_source=comp, indices=ints([0, 1, 2, -1]), **kw)
    assert torch.equal(o_big[:3], o_cmp[:3])
    assert torch.equal(view[sel], comp)
    written |= {(s, LAST) for s in sel}
    pool.assert_untouched(written)

    # prefill continuing the same slots (chunk boundaries at 64)
    lengths = [70, 3, 65]
    T = sum(lengths)
    pq, pk = randn(g, 1, T, HK, K, dtype=BF16), randn(g, 1, T, HK, K, dtype=BF16)
    pv = randn(g, 1, T, HV, V, dtype=BF16)
    pg, pb = -randn(g, 1, T, HV).abs() * 0.3, torch.sigmoid(randn(g, 1, T, HV))
    pcu = ints([0, 70, 73, 138])
    o_big, h_big = gdn_prefill_chunk_fla(pq, pk, pv, pg, pb, state_source=view, indices=ints(sel),
                                         cu_seqlens=pcu, scale=SCALE, return_h=True)
    o_cmp, h_cmp = gdn_prefill_chunk_fla(pq, pk, pv, pg, pb, state_source=comp, indices=ints([0, 1, 2]),
                                         cu_seqlens=pcu, scale=SCALE, return_h=True)
    assert torch.equal(o_big, o_cmp) and torch.equal(h_big, h_cmp)
    assert torch.equal(view[sel], comp)
    pool.assert_untouched(written)

    # replay reads checkpoints past 2^31; fold exports into / folds within high slots
    rows, R, N = 3, 16, 4
    rec = dict(u=randn(g, rows, L, HV, R, V, dtype=BF16, scale=0.1),
               k=randn(g, rows, L, HK, R, K, dtype=BF16, scale=0.1),
               g=randn(g, rows, L, HV, R, scale=0.1))
    rec = {n_: t.transpose(0, 1) for n_, t in rec.items()}
    start = ints([5, 9, 2])
    rqkv, ra, rb = randn(g, 4 * N, D, dtype=BF16), randn(g, 4 * N, HV), randn(g, 4 * N, HV)
    rcu, rrows, rpos = ints([0, N, 2 * N, 3 * N, 4 * N]), ints([2, 0, 1, -1]), ints([p + j for p in (11, 9, 12, 0) for j in range(N)])
    rargs = lambda st, slots: (rqkv, ra, rb, A_log, dt_bias, st, rec["u"][LAST].clone(),
                               rec["k"][LAST].clone(), rec["g"][LAST].clone(), start.clone(), rcu,
                               ints(slots), rrows, rpos, SCALE)
    assert torch.equal(gdn_replay(*rargs(view, sel + [-1])), gdn_replay(*rargs(comp, [0, 1, 2, -1])))

    full_cmp = torch.stack([pool.storage[s] for s in (hi, mid, lo, 1)]).transpose(0, 1).contiguous()
    def fold(st, rows_, src, dst, ends, w):
        gdn_replay_fold(st, rec["u"], rec["k"], rec["g"], start, ints(rows_), ints(src), ints(dst),
                        ints(ends), w)

    # export (width 0) from high slots onto other slots; then an in-place fold where rows 0/1
    # trigger (end + 8 - b > R) and row 2 does not
    fold(pool.lm, [0, 1], [hi, lo], [mid, 1], [9, 11], 0)
    fold(full_cmp, [0, 1], [0, 2], [1, 3], [9, 11], 0)
    fold(pool.lm, [0, 1, 2], [hi, mid, lo], [hi, mid, lo], [14, 18, 7], 8)
    fold(full_cmp, [0, 1, 2], [0, 1, 2], [0, 1, 2], [14, 18, 7], 8)
    for i, s in enumerate((hi, mid, lo, 1)):
        assert torch.equal(pool.lm[:, s], full_cmp[:, i]), s
    written |= {(s, l) for s in (hi, mid, 1) for l in range(L)}
    pool.assert_untouched(written)
    pool.free()


@pytest.mark.parametrize("path", ["sgl", "triton"])
def test_conv_state_past_2pow31(path):
    g = gen(62)
    n = 2914  # slot stride 30 x 8192 x 3 = 737,280 elements
    hi, mid, lo = n - 1, n - 2, 0
    live = [(hi, LAST), (mid, LAST), (lo, LAST), (hi, LAST - 1), (mid, 0)]
    pool = BigPool(g, n, (D, KW - 1), BF16, live)
    assert pool.offset(hi, 0) >= I32 and pool.offset(mid, LAST) >= I32 > pool.offset(mid, 0)
    mod = sgl if path == "sgl" else tri
    view = pool.lm[LAST]
    sel = [hi, mid, lo]
    comp = pool.compact(sel, LAST)
    weight = randn(g, D, KW, dtype=BF16, scale=0.5)

    x = randn(g, 4, D, dtype=BF16)
    o_big = mod.causal_conv1d_decode(x.clone(), view, weight, ints(sel + [-1]))
    o_cmp = mod.causal_conv1d_decode(x.clone(), comp, weight, ints([0, 1, 2, -1]))
    assert torch.equal(o_big[:3], o_cmp[:3]) and torch.equal(view[sel], comp)

    lengths = [6, 1, 3]
    xs = randn(g, sum(lengths), D, dtype=BF16).T
    cu, init = ints([0, 6, 7, 10]), torch.tensor([True, True, False], device=DEV)
    o_big = mod.causal_conv1d_varlen(xs.clone(), weight, view, cu, ints(sel), init)
    o_cmp = mod.causal_conv1d_varlen(xs.clone(), weight, comp, cu, ints([0, 1, 2]), init)
    assert torch.equal(o_big, o_cmp) and torch.equal(view[sel], comp)
    pool.assert_untouched({(s, LAST) for s in sel})
    pool.free()


@pytest.mark.parametrize("big", ["u", "k"])
def test_replay_records_past_2pow31(big):
    """One record buffer uses 30 layers per row so its high rows cross 2^31; the other
    record buffer uses one layer per row (a different, still legal geometry)."""
    g = gen(63)
    R, N = 16, 4
    inner = dict(u=(HV, R, V), k=(HK, R, K))
    per_layer = math.prod(inner[big])
    n = I32 // (L * per_layer) + 2  # u: 1094 rows, k: 2186 rows
    hi, mid, lo = n - 1, n - 2, 0
    live = [(hi, LAST), (mid, LAST), (lo, LAST), (hi, LAST - 1), (mid, 0)]
    pool = BigPool(g, n, inner[big], BF16, live, scale=0.1)
    assert pool.offset(hi, 0) >= I32 and pool.offset(mid, LAST) >= I32 > pool.offset(mid, 0)
    other = "k" if big == "u" else "u"
    small = randn(g, n, 1, *inner[other], dtype=BF16, scale=0.1)[:, 0]
    gl = randn(g, n, HV, R, scale=0.1)
    state = randn(g, 3, HV, V, K, scale=0.05)
    sel = [hi, mid, lo]
    A_log, dt_bias = randn(g, HV, scale=0.5), randn(g, HV, scale=0.5)
    qkv, a, b = randn(g, 4 * N, D, dtype=BF16), randn(g, 4 * N, HV), randn(g, 4 * N, HV)
    cu, pos = ints([0, N, 2 * N, 3 * N, 4 * N]), ints([p + j for p in (20, 7, 13, 0) for j in range(N)])
    start_vals = [14, 0, 13]  # history lengths 6 / 7 / 0 records

    def run(bigbuf, rows, start_rows, n_rows):
        bufs = {big: bigbuf, other: small[:n_rows] if n_rows < n else small}
        start = torch.zeros(n_rows, device=DEV, dtype=torch.int32)
        start[start_rows] = ints(start_vals)
        return gdn_replay(qkv, a, b, A_log, dt_bias, state, bufs["u"], bufs["k"], gl[:n_rows], start, cu,
                          ints([0, 1, 2, -1]), ints(rows + [-1]), pos, SCALE)

    small0, gl0 = small.clone(), gl.clone()
    o_big = run(pool.lm[LAST], sel, sel, n)
    small_big, gl_big = small[sel].clone(), gl[sel].clone()
    # The compact run holds the same rows of every buffer at rows 0..2.
    small[:3], gl[:3] = small0[sel], gl0[sel]
    comp = pool.compact(sel, LAST)
    o_cmp = run(comp, [0, 1, 2], [0, 1, 2], 3)
    assert torch.equal(o_big, o_cmp)
    assert torch.equal(small[:3], small_big) and torch.equal(gl[:3], gl_big)
    assert torch.equal(pool.lm[LAST][sel], comp)
    pool.assert_untouched({(s, LAST) for s in sel})
    pool.free()


def test_replay_conv_window_past_2pow31():
    g = gen(64)
    W, N = 8, 4  # W >= KW - 1 + N
    n = I32 // (L * W * D) + 2  # 1094 rows; row stride 30 x 8 x 8192 elements
    hi, mid, lo = n - 1, n - 2, 0
    live = [(hi, LAST), (mid, LAST), (lo, LAST), (hi, LAST - 1), (mid, 0)]
    pool = BigPool(g, n, (W, D), BF16, live)
    assert pool.offset(hi, 0) >= I32 and pool.offset(mid, LAST) >= I32 > pool.offset(mid, 0)
    sel = [hi, mid, lo]
    comp = pool.compact(sel, LAST)
    weight = randn(g, D, KW, dtype=BF16, scale=0.5)
    x = randn(g, 4 * N, D, dtype=BF16)
    cu, pos = ints([0, N, 2 * N, 3 * N, 4 * N]), ints([p + j for p in (21, 6, 0, 0) for j in range(N)])
    o_big = gdn_replay_conv(x, weight, pool.lm[LAST], cu, ints(sel + [-1]), pos)
    o_cmp = gdn_replay_conv(x, weight, comp, cu, ints([0, 1, 2, -1]), pos)
    assert torch.equal(o_big[:3 * N], o_cmp[:3 * N])
    assert torch.equal(pool.lm[LAST][sel], comp)
    pool.assert_untouched({(s, LAST) for s in sel})
    pool.free()


def test_store_cache_past_2pow31():
    """K and V caches share one geometry (store_cache requires equal row strides)."""
    g = gen(65)
    layers, row = 10, 512  # full-attention layers interleaved per token row
    n = I32 // (layers * row) + 2  # 419,432 token rows
    hi, mid, lo = n - 1, n - 2, 0
    live = [(hi, layers - 1), (mid, layers - 1), (hi, layers - 2), (mid, 0)]
    kp = BigPool(g, n, (row,), BF16, live, layers=layers)
    vp = BigPool(g, n, (row,), BF16, live, layers=layers)
    assert kp.offset(hi, 0) >= I32 and kp.offset(mid, layers - 1) >= I32 > kp.offset(mid, 0)
    loc = [hi, lo, mid]
    k, v = randn(g, 3, row, dtype=BF16), randn(g, 3, row, dtype=BF16)
    store_cache(kp.lm[layers - 1], vp.lm[layers - 1], ints(loc, torch.int64), k, v)
    assert torch.equal(kp.lm[layers - 1][loc], k) and torch.equal(vp.lm[layers - 1][loc], v)
    for p in (kp, vp):
        p.assert_untouched({(s, layers - 1) for s in loc})
        p.free()
