"""Compact-layout calls of every P0 operator, run in-process against the implementation and
in a subprocess against the pre-change release (``python parity_cases.py OUT``)."""

import sys

import torch

from rp_common import BF16, D, DEV, HK, HV, K, KW, SCALE, V, gen, ints, randn


def run_all():
    from freetoken.kernel import causal_conv1d as sgl
    from freetoken.kernel import store_cache
    from freetoken.kernel.triton import causal_conv1d_triton as tri
    from freetoken.kernel.triton.gdn_replay import (gdn_replay, gdn_replay_advance, gdn_replay_conv,
                                                    gdn_replay_fold)
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla

    g = gen(71)
    out = {}
    A_log, dt_bias = randn(g, HV, scale=0.5), randn(g, HV, scale=0.5)
    weight = randn(g, D, KW, dtype=BF16, scale=0.5)

    state = randn(g, 6, HV, V, K, scale=0.05)
    idx = ints([5, -1, 0, 3])
    q, k = randn(g, 1, 4, HK, K, dtype=BF16), randn(g, 1, 4, HK, K, dtype=BF16)
    v, a, b = randn(g, 1, 4, HV, V, dtype=BF16), randn(g, 4, HV), randn(g, 4, HV)
    out["decode"] = gdn_decode_fla(q, k, v, a, b, A_log=A_log, dt_bias=dt_bias, state_source=state,
                                   indices=idx, cu_seqlens=ints([0, 1, 2, 3, 4]), scale=SCALE)[[0, 2, 3]]
    T = 140
    pq, pk = randn(g, 1, T, HK, K, dtype=BF16), randn(g, 1, T, HK, K, dtype=BF16)
    pv = randn(g, 1, T, HV, V, dtype=BF16)
    pg, pb = -randn(g, 1, T, HV).abs() * 0.3, torch.sigmoid(randn(g, 1, T, HV))
    out["prefill"], out["prefill_h"] = gdn_prefill_chunk_fla(
        pq, pk, pv, pg, pb, state_source=state, indices=ints([1, 5, 2]), cu_seqlens=ints([0, 70, 71, 140]),
        scale=SCALE, return_h=True)
    out["gdn_state"] = state

    for name, mod in (("sgl", sgl), ("tri", tri)):
        cs = randn(gen(72), 6, D, KW - 1, dtype=BF16)
        x = randn(g, 4, D, dtype=BF16)
        out[f"{name}_decode"] = mod.causal_conv1d_decode(x, cs, weight, ints([4, 0, -1, 2]))[[0, 1, 3]]
        xs = randn(g, 10, D, dtype=BF16).T
        out[f"{name}_varlen"] = mod.causal_conv1d_varlen(
            xs, weight, cs, ints([0, 6, 7, 10]), ints([5, 1, 3]), torch.tensor([True, False, True], device=DEV))
        out[f"{name}_state"] = cs

    R, N, rows = 16, 4, 3
    ck = randn(g, 1, 6, HV, V, K, scale=0.05)
    u, kr = randn(g, 1, rows, HV, R, V, dtype=BF16, scale=0.1), randn(g, 1, rows, HK, R, K, dtype=BF16, scale=0.1)
    gr = randn(g, 1, rows, HV, R, scale=0.1)
    start = ints([5, 9, 2])
    qkv, ra, rb = randn(g, 4 * N, D, dtype=BF16), randn(g, 4 * N, HV), randn(g, 4 * N, HV)
    cu, rws = ints([0, N, 2 * N, 3 * N, 4 * N]), ints([2, 0, 1, -1])
    pos = ints([p + j for p in (11, 9, 12, 0) for j in range(N)])  # per input token
    out["replay"] = gdn_replay(qkv, ra, rb, A_log, dt_bias, ck[0], u[0], kr[0], gr[0], start, cu,
                               ints([3, 0, 1, -1]), rws, pos, SCALE)
    win = randn(g, rows, 8, D, dtype=BF16)
    out["replay_conv"] = gdn_replay_conv(qkv, weight, win, cu, rws, pos)[:3 * N]
    gdn_replay_fold(ck, u, kr, gr, start, ints([0, 1, 2]), ints([0, 1, 3]), ints([2, 4, 5]), ints([13, 16, 15]), 0)
    gdn_replay_fold(ck, u, kr, gr, start, ints([0, 1, 2]), ints([0, 1, 3]), ints([0, 1, 3]), ints([13, 16, 15]), 8)
    stats = torch.zeros(2, device=DEV, dtype=torch.int64)
    gdn_replay_advance(start, stats, ints([0, 1, 2]), ints([13, 16, 15]), 8, R)
    out.update(replay_u=u, replay_k=kr, replay_g=gr, replay_ck=ck, replay_start=start, replay_stats=stats,
               replay_win=win)

    kc, vc = randn(g, 32, 512, dtype=BF16), randn(g, 32, 512, dtype=BF16)
    store_cache(kc, vc, ints([31, 0, 7], torch.int64), randn(g, 3, 512, dtype=BF16), randn(g, 3, 512, dtype=BF16))
    out.update(kc=kc, vc=vc)
    torch.cuda.synchronize()
    return {n: t.cpu() for n, t in out.items()}


if __name__ == "__main__":
    torch.save(run_all(), sys.argv[1])
