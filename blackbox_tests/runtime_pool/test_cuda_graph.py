"""Decode paths captured once in a CUDA graph on slot-major pools, replayed with changing
slot ids / padding / positions, must match eager calls on the compact layout bit for bit."""

import pytest
import torch

from freetoken.kernel import causal_conv1d as sgl
from freetoken.kernel import store_cache
from freetoken.kernel.triton import causal_conv1d_triton as tri
from freetoken.kernel.triton.gdn_replay import (gdn_replay, gdn_replay_advance, gdn_replay_conv,
                                                gdn_replay_fold)
from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla
from rp_common import BF16, D, DEV, HK, HV, K, KW, SCALE, V, Pool, gen, ints, randn

LAYERS, LAYER = 4, 3
SLOTS, ROWS, TOKENS = 10, 6, 64
B = 8
RING, WIN = 16, 16
ROW = 512
# Per iteration: decode slot ids (padding -1 varies), replay rows/slots, KV rows.
DEC_IDX = [[SLOTS - 1, 0, 4, -1, -1, -1, -1, -1],
           [2, 7, -1, 0, 5, SLOTS - 1, -1, -1],
           [3, -1, -1, -1, -1, -1, -1, 8]]
REPLAY = [[(ROWS - 1, SLOTS - 1), (0, 0), (2, 6)],
          [(1, 4), (ROWS - 1, SLOTS - 1), (3, 2)],
          [(0, 0), (4, 5), (1, 4)]]
START0 = [10, 3, 7, 6, 5, 18]  # checkpoint position per record row
KV_LOC = [[63, 0, 10, 11, 12, 30, 31, 40], [1, 2, 3, 4, 5, 6, 7, 62], [50, 51, 0, 63, 20, 21, 22, 23]]


def make_pools(g):
    return dict(
        dec=Pool(g, SLOTS, LAYERS, LAYER, (HV, V, K), torch.float32, scale=0.05),
        conv_sgl=Pool(g, SLOTS, LAYERS, LAYER, (D, KW - 1), BF16),
        conv_tri=Pool(g, SLOTS, LAYERS, LAYER, (D, KW - 1), BF16),
        ck=Pool(g, SLOTS, LAYERS, LAYER, (HV, V, K), torch.float32, scale=0.05),
        u=Pool(g, ROWS, LAYERS, LAYER, (HV, RING, V), BF16, scale=0.1),
        k=Pool(g, ROWS, LAYERS, LAYER, (HK, RING, K), BF16, scale=0.1),
        g=Pool(g, ROWS, LAYERS, LAYER, (HV, RING), torch.float32, scale=0.1),
        win=Pool(g, ROWS, LAYERS, LAYER, (WIN, D), BF16),
        kc=Pool(g, TOKENS, LAYERS, LAYER, (ROW,), BF16),
        vc=Pool(g, TOKENS, LAYERS, LAYER, (ROW,), BF16))


def make_inputs(g, N, it):
    T = 4 * N  # three requests + one padding entry, N tokens each
    rows = [r for r, _ in REPLAY[it]]
    return dict(
        q=randn(g, 1, B, HK, K, dtype=BF16), k=randn(g, 1, B, HK, K, dtype=BF16),
        v=randn(g, 1, B, HV, V, dtype=BF16), a=randn(g, B, HV), b=randn(g, B, HV),
        dec_idx=ints(DEC_IDX[it]), cu=ints(list(range(B + 1))),
        x_sgl=randn(g, B, D, dtype=BF16), x_tri=randn(g, B, D, dtype=BF16),
        qkv=randn(g, T, D, dtype=BF16), ra=randn(g, T, HV), rb=randn(g, T, HV),
        rcu=ints([0, N, 2 * N, 3 * N, 4 * N]), rows=ints(rows + [-1]),
        rows_real=ints(rows), slots_real=ints([s for _, s in REPLAY[it]]),
        slots=ints([s for _, s in REPLAY[it]] + [-1]),
        pos=ints([START0[r] + 4 + 3 * it + j for r in rows for j in range(N)] + list(range(N))),
        ends=ints([START0[r] + 4 + 3 * it for r in rows]),
        loc=ints(KV_LOC[it], torch.int64), kv_k=randn(g, B, ROW, dtype=BF16), kv_v=randn(g, B, ROW, dtype=BF16))


def step(P, S, params, start, stats, N):
    """One decode iteration over every decode-path operator; returns its outputs.
    Replay `pos` holds the absolute position of every input token."""
    A_log, dt_bias, weight = params
    out = dict(dec=gdn_decode_fla(S["q"], S["k"], S["v"], S["a"], S["b"], A_log=A_log, dt_bias=dt_bias,
                                  state_source=P["dec"][LAYER], indices=S["dec_idx"], cu_seqlens=S["cu"],
                                  scale=SCALE))
    out["sgl"] = sgl.causal_conv1d_decode(S["x_sgl"], P["conv_sgl"][LAYER], weight, S["dec_idx"])
    out["tri"] = tri.causal_conv1d_decode(S["x_tri"], P["conv_tri"][LAYER], weight, S["dec_idx"])
    gdn_replay_fold(P["ck"], P["u"], P["k"], P["g"], start, S["rows_real"], S["slots_real"],
                    S["slots_real"], S["ends"], N)
    gdn_replay_advance(start, stats, S["rows_real"], S["ends"], N, RING)
    out["rconv"] = gdn_replay_conv(S["qkv"], weight, P["win"][LAYER], S["rcu"], S["rows"], S["pos"])
    out["replay"] = gdn_replay(S["qkv"], S["ra"], S["rb"], A_log, dt_bias, P["ck"][LAYER], P["u"][LAYER],
                               P["k"][LAYER], P["g"][LAYER], start, S["rcu"], S["slots"], S["rows"],
                               S["pos"], SCALE)
    store_cache(P["kc"][LAYER], P["vc"][LAYER], S["loc"], S["kv_k"], S["kv_v"])
    return out


def initial_start():
    return ints(START0)


@pytest.mark.parametrize("N", [1, 4, 8])
def test_decode_paths_graph_replay(N):
    g = gen(51 + N)
    pools = make_pools(g)
    params = (randn(g, HV, scale=0.5), randn(g, HV, scale=0.5), randn(g, D, KW, dtype=BF16, scale=0.5))
    iters = [make_inputs(g, N, it) for it in range(len(DEC_IDX))]
    strided = {n: p.strided_lm for n, p in pools.items()}
    compact = {n: p.compact_lm for n, p in pools.items()}

    static = {n: t.clone() for n, t in iters[0].items()}
    start_g, stats_g = initial_start(), torch.zeros(2, device=DEV, dtype=torch.int64)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm-up compiles; the pools are restored afterwards
        step(strided, static, params, start_g.clone(), stats_g.clone(), N)
    torch.cuda.current_stream().wait_stream(side)
    for p in pools.values():
        p.storage.copy_(p.snapshot)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = step(strided, static, params, start_g, stats_g, N)

    start_e, stats_e = initial_start(), torch.zeros(2, device=DEV, dtype=torch.int64)
    for inp in iters:
        for n, t in inp.items():
            static[n].copy_(t)
        graph.replay()
        eager = step(compact, {n: t.clone() for n, t in inp.items()}, params, start_e, stats_e, N)
        dec_real = [i for i, s in enumerate(inp["dec_idx"].tolist()) if s >= 0]
        for n, t in eager.items():
            real = dec_real if n in ("dec", "sgl", "tri") else list(range(3 * N))
            assert torch.equal(static_out[n][real], t[real]), n
        assert torch.count_nonzero(static_out["replay"][3 * N:]) == 0
        assert torch.equal(start_g, start_e) and torch.equal(stats_g, stats_e)
    torch.cuda.synchronize()
    for p in pools.values():
        assert torch.equal(p.strided_lm, p.compact_lm)
