"""gdn_decode_fla / gdn_prefill_chunk_fla on compact vs slot-major recurrent-state pools."""

import pytest
import torch

from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla
from rp_common import (BF16, HK, HV, K, REAL_LAYERS, SCALE, V, Pool, assert_close, gates,
                       gdn_ref, gen, ints, randn)

STATE = (HV, V, K)


def decode_inputs(g, B):
    return dict(
        q=randn(g, 1, B, HK, K, dtype=BF16), k=randn(g, 1, B, HK, K, dtype=BF16),
        v=randn(g, 1, B, HV, V, dtype=BF16), a=randn(g, B, HV), b=randn(g, B, HV))


def run_decode(pool_state, inp, idx, A_log, dt_bias):
    B = idx.shape[0]
    return gdn_decode_fla(
        inp["q"], inp["k"], inp["v"], inp["a"], inp["b"], A_log=A_log, dt_bias=dt_bias,
        state_source=pool_state, indices=idx, cu_seqlens=ints(list(range(B + 1))), scale=SCALE)


def prefill_inputs(g, T):
    return dict(
        q=randn(g, 1, T, HK, K, dtype=BF16), k=randn(g, 1, T, HK, K, dtype=BF16),
        v=randn(g, 1, T, HV, V, dtype=BF16), g=-randn(g, 1, T, HV).abs() * 0.3,
        beta=torch.sigmoid(randn(g, 1, T, HV)))


def run_prefill(pool_state, inp, idx, lengths, return_h=False):
    cu = ints([0] + torch.tensor(lengths).cumsum(0).tolist())
    return gdn_prefill_chunk_fla(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], state_source=pool_state,
        indices=idx, cu_seqlens=cu, scale=SCALE, return_h=return_h)


@pytest.mark.parametrize("layers,layer", [(4, 2), (REAL_LAYERS, REAL_LAYERS - 1)])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_decode_strided_equals_compact(layers, layer, idx_dtype):
    g = gen(11)
    slots = 10
    pool = Pool(g, slots, layers, layer, STATE, torch.float32, scale=0.05)
    # Non-full batch padded to 8: last slot, first slot, non-contiguous ids, padding (-1).
    idx_list = [slots - 1, -1, 0, 6, 3, -1, -1, -1]
    inp = decode_inputs(g, len(idx_list))
    A_log, dt_bias = randn(g, HV, scale=0.5), randn(g, HV, scale=0.5)
    idx = ints(idx_list, idx_dtype)

    o_c = run_decode(pool.compact, inp, idx, A_log, dt_bias)
    o_s = run_decode(pool.strided, inp, idx, A_log, dt_bias)

    real = [i for i, s in enumerate(idx_list) if s >= 0]
    sel = [idx_list[i] for i in real]
    assert o_s.shape == (len(idx_list), HV, V)
    assert torch.equal(o_s[real], o_c[real])
    pool.assert_same(sel)
    pool.assert_untouched(sel)

    gg, beta = gates(inp["a"], inp["b"], A_log, dt_bias)
    for i in real:
        s = idx_list[i]
        o_r, S_r, _, _ = gdn_ref(pool.snapshot[s, layer], inp["q"][0, i:i + 1], inp["k"][0, i:i + 1],
                                 inp["v"][0, i:i + 1], gg[i:i + 1], beta[i:i + 1])
        assert_close(o_s[i], o_r[0], atol=2e-3, rtol=2e-2)
        assert_close(pool.strided[s], S_r, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("layers,layer", [(4, 0), (REAL_LAYERS, 17)])
def test_prefill_strided_equals_compact(layers, layer):
    g = gen(12)
    slots = 9
    pool = Pool(g, slots, layers, layer, STATE, torch.float32, scale=0.05)
    # Chunk granularity is 64: lengths straddle 0/1/2 chunk boundaries; slot order is
    # non-contiguous and includes the first and last slot.
    lengths = [70, 1, 129, 5]
    slot_ids = [slots - 1, 0, 5, 2]
    fresh = [0, 5]  # caller pre-zeroes fresh slots; the others continue from their state
    for pool_state in (pool.storage[:, layer], pool.compact):
        pool_state[fresh] = 0
    pool.snapshot[fresh, layer] = 0
    inp = prefill_inputs(g, sum(lengths))
    idx = ints(slot_ids)

    o_c, h_c = run_prefill(pool.compact, inp, idx, lengths, return_h=True)
    o_s, h_s = run_prefill(pool.strided, inp, idx, lengths, return_h=True)

    assert o_s.shape == (sum(lengths), HV, V)
    assert torch.equal(o_s, o_c)
    assert torch.equal(h_s, h_c)
    pool.assert_same(slot_ids)
    pool.assert_untouched(slot_ids)

    start = 0
    for n, s in zip(lengths, slot_ids):
        sl = slice(start, start + n)
        o_r, S_r, _, _ = gdn_ref(pool.snapshot[s, layer], inp["q"][0, sl], inp["k"][0, sl],
                                 inp["v"][0, sl], inp["g"][0, sl], inp["beta"][0, sl])
        assert_close(o_s[sl], o_r, atol=2e-2, rtol=5e-2)
        assert_close(pool.strided[s], S_r, atol=2e-2, rtol=5e-2)
        start += n


def test_prefill_then_decode_continuation():
    g = gen(13)
    slots, layers, layer = 8, 4, 3
    pool = Pool(g, slots, layers, layer, STATE, torch.float32, scale=0.05)
    slot_ids = [slots - 1, 2]
    for pool_state in (pool.storage[:, layer], pool.compact):
        pool_state[slot_ids] = 0
    pool.snapshot[slot_ids, layer] = 0
    lengths = [7, 64]
    pre = prefill_inputs(g, sum(lengths))
    A_log, dt_bias = randn(g, HV, scale=0.5), randn(g, HV, scale=0.5)
    steps = [decode_inputs(g, 3) for _ in range(3)]
    idx_dec = ints(slot_ids + [-1])

    outs = {}
    for name, st in (("c", pool.compact), ("s", pool.strided)):
        o = [run_prefill(st, pre, ints(slot_ids), lengths)]
        o += [run_decode(st, inp, idx_dec, A_log, dt_bias)[:2] for inp in steps]
        outs[name] = o
    for oc, os_ in zip(outs["c"], outs["s"]):
        assert torch.equal(oc, os_)
    pool.assert_same(slot_ids)
    pool.assert_untouched(slot_ids)

    start = 0
    for i, (n, s) in enumerate(zip(lengths, slot_ids)):
        sl = slice(start, start + n)
        dq = torch.cat([st_["q"][0, i:i + 1] for st_ in steps])
        dk = torch.cat([st_["k"][0, i:i + 1] for st_ in steps])
        dv = torch.cat([st_["v"][0, i:i + 1] for st_ in steps])
        dg, db = zip(*[gates(st_["a"][i:i + 1], st_["b"][i:i + 1], A_log, dt_bias) for st_ in steps])
        o_r, S_r, _, _ = gdn_ref(
            pool.snapshot[s, layer], torch.cat([pre["q"][0, sl], dq]), torch.cat([pre["k"][0, sl], dk]),
            torch.cat([pre["v"][0, sl], dv]), torch.cat([pre["g"][0, sl], *dg]),
            torch.cat([pre["beta"][0, sl], *db]))
        dec_out = torch.stack([o[i] for o in outs["s"][1:]])
        assert_close(dec_out, o_r[n:], atol=2e-2, rtol=5e-2)
        assert_close(pool.strided[s], S_r, atol=2e-2, rtol=5e-2)
        start += n
