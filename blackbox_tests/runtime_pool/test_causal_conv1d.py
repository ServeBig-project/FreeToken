"""causal_conv1d varlen/decode (SGL default and public Triton path) on compact vs slot-major
conv-state pools."""

import pytest
import torch

from freetoken.kernel import causal_conv1d as sgl
from freetoken.kernel.triton import causal_conv1d_triton as tri
from rp_common import BF16, D, KW, REAL_LAYERS, Pool, assert_close, conv_ref, gen, ints, randn

PATHS = {"sgl": sgl, "triton": tri}
CONV = (D, KW - 1)
TOL = dict(atol=3e-2, rtol=2e-2)  # bf16 output; Triton path documents allclose(2e-2) vs CUDA


def run_decode(path, x, state, weight, idx):
    x = x.clone()
    out = PATHS[path].causal_conv1d_decode(x, state, weight, idx)
    return x, out


def run_varlen(path, x, state, weight, lengths, idx, has_init):
    x = x.clone()
    cu = ints([0] + torch.tensor(lengths).cumsum(0).tolist())
    out = PATHS[path].causal_conv1d_varlen(x, weight, state, cu, idx, has_init)
    return x, out


def assert_same_call(res_c, res_s, pick):
    """Same output on real (non-padding) tokens, same in-place behaviour on x as the compact
    call. Padding outputs are not specified (the Triton path leaves them unwritten)."""
    (x_c, o_c), (x_s, o_s) = res_c, res_s
    assert torch.equal(pick(o_s), pick(o_c))
    assert torch.equal(x_s, x_c)
    assert (o_s.data_ptr() == x_s.data_ptr()) == (o_c.data_ptr() == x_c.data_ptr())


@pytest.mark.parametrize("path", ["sgl", "triton"])
@pytest.mark.parametrize("layers,layer", [(4, 1), (REAL_LAYERS, REAL_LAYERS - 1)])
def test_decode_strided_equals_compact(path, layers, layer):
    g = gen(21)
    slots = 12
    pool = Pool(g, slots, layers, layer, CONV, BF16)
    weight = randn(g, D, KW, dtype=BF16, scale=0.5)
    idx_list = [slots - 1, -1, 0, 7, 4, -1, -1, -1]
    x = randn(g, len(idx_list), D, dtype=BF16)
    idx = ints(idx_list)

    res_c = run_decode(path, x, pool.compact, weight, idx)
    res_s = run_decode(path, x, pool.strided, weight, idx)
    real = [i for i, s in enumerate(idx_list) if s >= 0]
    assert_same_call(res_c, res_s, lambda o: o[real])
    sel = [idx_list[i] for i in real]
    pool.assert_same(sel)
    pool.assert_untouched(sel)

    out = res_s[1]
    for i in real:
        s = idx_list[i]
        old = pool.snapshot[s, layer]  # [D, KW-1]
        assert_close(out[i], conv_ref(old.T, x[i:i + 1], weight)[0], **TOL)
        assert torch.equal(pool.strided[s], torch.cat([old[:, 1:], x[i][:, None]], 1))


@pytest.mark.parametrize("path", ["sgl", "triton"])
@pytest.mark.parametrize("layers,layer", [(4, 3), (REAL_LAYERS, 0)])
def test_varlen_strided_equals_compact(path, layers, layer):
    g = gen(22)
    slots = 10
    pool = Pool(g, slots, layers, layer, CONV, BF16)
    weight = randn(g, D, KW, dtype=BF16, scale=0.5)
    # Lengths shorter / longer than the kernel tail; one padding request (-1).
    lengths = [5, 1, 2, 3, 9]
    idx_list = [slots - 1, 0, 6, -1, 3]
    has_init = [True, False, True, True, False]
    x = randn(g, sum(lengths), D, dtype=BF16).T  # channels-first [D, T]
    args = (lengths, ints(idx_list), torch.tensor(has_init, device="cuda"))

    res_c = run_varlen(path, x, pool.compact, weight, *args)
    res_s = run_varlen(path, x, pool.strided, weight, *args)
    cu = [0] + torch.tensor(lengths).cumsum(0).tolist()
    real = [t for i, s in enumerate(idx_list) if s >= 0 for t in range(cu[i], cu[i + 1])]
    assert_same_call(res_c, res_s, lambda o: o[:, real])
    sel = [s for s in idx_list if s >= 0]
    pool.assert_same(sel)
    pool.assert_untouched(sel)

    out = res_s[1]
    start = 0
    for n, s, init in zip(lengths, idx_list, has_init):
        seg = x[:, start:start + n].T
        if s >= 0:
            hist = pool.snapshot[s, layer].T if init else torch.zeros(KW - 1, D, device="cuda")
            assert_close(out[:, start:start + n].T, conv_ref(hist, seg, weight), **TOL)
            tail = torch.cat([hist.to(BF16), seg])[-(KW - 1):].T
            assert torch.equal(pool.strided[s], tail)
        start += n


@pytest.mark.parametrize("path", ["sgl", "triton"])
def test_varlen_then_decode_continuation(path):
    g = gen(23)
    slots, layers, layer = 8, 4, 2
    pool = Pool(g, slots, layers, layer, CONV, BF16)
    weight = randn(g, D, KW, dtype=BF16, scale=0.5)
    lengths, slot_ids = [6, 2], [slots - 1, 1]
    pre = randn(g, sum(lengths), D, dtype=BF16).T
    steps = [randn(g, 3, D, dtype=BF16) for _ in range(4)]
    idx_dec = ints(slot_ids + [-1])

    outs = {}
    for name, st in (("c", pool.compact), ("s", pool.strided)):
        o = [run_varlen(path, pre, st, weight, lengths, ints(slot_ids),
                        torch.tensor([False, False], device="cuda"))[1]]
        o += [run_decode(path, x, st, weight, idx_dec)[1][:2] for x in steps]
        outs[name] = o
    for oc, os_ in zip(outs["c"], outs["s"]):
        assert torch.equal(oc, os_)
    pool.assert_same(slot_ids)
    pool.assert_untouched(slot_ids)

    start = 0
    for i, n in enumerate(lengths):
        stream = torch.cat([pre[:, start:start + n].T] + [x[i:i + 1] for x in steps])
        ref = conv_ref(torch.zeros(KW - 1, D, device="cuda"), stream, weight)
        got = torch.stack([o[i] for o in outs["s"][1:]])
        assert_close(got, ref[n:], **TOL)
        start += n


def test_sgl_and_triton_agree_on_strided_pool():
    g = gen(24)
    slots, layers, layer = 8, REAL_LAYERS, 5
    pools = {p: Pool(gen(25), slots, layers, layer, CONV, BF16) for p in PATHS}
    weight = randn(g, D, KW, dtype=BF16, scale=0.5)
    lengths, idx_list = [4, 7], [slots - 1, 2]
    x = randn(g, sum(lengths), D, dtype=BF16).T
    dec = randn(g, 2, D, dtype=BF16)
    out = {}
    for p, pool in pools.items():
        v = run_varlen(p, x, pool.strided, weight, lengths, ints(idx_list),
                       torch.tensor([True, True], device="cuda"))[1]
        d = run_decode(p, dec, pool.strided, weight, ints(idx_list))[1]
        out[p] = (v, d, pool.strided[idx_list])
    for a, b in zip(out["sgl"], out["triton"]):
        assert_close(a, b, **TOL)
