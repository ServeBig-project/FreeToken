"""ReplaySSM entrypoints on compact vs slot-major checkpoint / record / window pools.

Each case drives a draft/verify-like cycle per request: before every call, fold + advance
with width N; then one replay (and conv) call per layer over N inputs; only a prefix of
the N inputs is accepted, so the next call starts at p + accepted and overwrites the
rejected tail's records. Several requests with distinct rows/slots plus one padding entry share each call; the
first call gives each request a different length. `positions` holds the absolute position
of every input token; rows/slots are per sequence.
"""

import pytest
import torch

from freetoken.kernel.triton.gdn_replay import (gdn_replay, gdn_replay_advance, gdn_replay_conv,
                                                gdn_replay_fold)
from rp_common import (BF16, D, DEV, HK, HV, K, KW, REAL_LAYERS, SCALE, V, Pool, assert_close,
                       conv_ref, gates, gdn_ref, gen, ints, randn)

RING = 16
WIN = 16  # >= KW - 1 + 8
SLOTS, ROWS = 8, 7
# (row, slot, first position); last row/slot, first row/slot, others
REQS = [(ROWS - 1, SLOTS - 1, 37), (0, 0, 0), (4, 3, 100)]
FIRST = [5, 2, 9]  # first-call length per request
PAD = 2  # padding entry tokens per call
EXPORT_DST = [1, 5, 2]
STEPS = 12
REF_TOL = dict(atol=2e-3, rtol=2e-2)


def accepted(N, req, step):
    return [N, max(1, N // 2), 1][(req + step) % 3]


def split_qkv(qkv):
    T = qkv.shape[0]
    return (qkv[:, :HK * K].view(T, HK, K), qkv[:, HK * K:2 * HK * K].view(T, HK, K),
            qkv[:, 2 * HK * K:].view(T, HV, V))


def make_pools(g, layers):
    return dict(
        state=Pool(g, SLOTS, layers, 0, (HV, V, K), torch.float32, scale=0.05),
        u=Pool(g, ROWS, layers, 0, (HV, RING, V), BF16, scale=0.1),
        k=Pool(g, ROWS, layers, 0, (HK, RING, K), BF16, scale=0.1),
        g=Pool(g, ROWS, layers, 0, (HV, RING), torch.float32, scale=0.1),
        win=Pool(g, ROWS, layers, 0, (WIN, D), BF16))


def make_calls(g, layers, N):
    """Per call: per-layer inputs for every request, then a padding entry of equal length."""
    calls = []
    for step in range(STEPS + 1):
        lens = FIRST if step == 0 else [N] * len(REQS)
        T = sum(lens) + PAD
        calls.append(dict(lens=lens, layers=[dict(
            qkv=randn(g, T, D, dtype=BF16), a=randn(g, T, HV), b=randn(g, T, HV)) for _ in range(layers)]))
    return calls


def run_cycle(pools, view, params, calls, N):
    """Runs every call on one layout (`view` = 'compact_lm' or 'strided_lm')."""
    P = {name: getattr(p, view) for name, p in pools.items()}
    layers = P["state"].shape[0]
    rows_real = ints([r[0] for r in REQS])
    slots_real = ints([r[1] for r in REQS])
    start = ints(list(range(50, 50 + ROWS)))  # unselected rows keep these
    start[rows_real] = ints([r[2] for r in REQS])
    stats = torch.zeros(2, device=DEV, dtype=torch.int64)
    pos = [r[2] for r in REQS]
    outs, convs = [], []
    for step, call in enumerate(calls):
        if step:
            ends = ints(pos)
            gdn_replay_fold(P["state"], P["u"], P["k"], P["g"], start, rows_real, slots_real,
                            slots_real, ends, N)
            gdn_replay_advance(start, stats, rows_real, ends, N, RING)
        lens = call["lens"]
        cu = ints([0] + torch.tensor(lens + [PAD]).cumsum(0).tolist())
        rows_t = ints([r[0] for r in REQS] + [-1])
        slots_t = ints([r[1] for r in REQS] + [-1])
        pos_t = ints([p + j for p, n in zip(pos, lens) for j in range(n)] + list(range(PAD)))
        o_layers, c_layers = [], []
        for l in range(layers):
            inp, (A_log, dt_bias, weight) = call["layers"][l], params[l]
            c_layers.append(gdn_replay_conv(inp["qkv"], weight, P["win"][l], cu, rows_t, pos_t))
            o_layers.append(gdn_replay(inp["qkv"], inp["a"], inp["b"], A_log, dt_bias, P["state"][l],
                                       P["u"][l], P["k"][l], P["g"][l], start, cu, slots_t, rows_t,
                                       pos_t, SCALE))
        outs.append(o_layers)
        convs.append(c_layers)
        if step:
            pos = [p + accepted(N, i, step) for i, p in enumerate(pos)]
        else:
            pos = [p + n for p, n in zip(pos, lens)]
    # Export each request's current state into a different slot (width 0 always runs).
    gdn_replay_fold(P["state"], P["u"], P["k"], P["g"], start, rows_real, slots_real,
                    ints(EXPORT_DST), ints(pos), 0)
    return outs, convs, start, stats, pos


@pytest.mark.parametrize("layers,N", [(3, 1), (3, 4), (3, 8), (REAL_LAYERS, 4)])
def test_replay_cycle_strided_equals_compact(layers, N):
    g = gen(31 + N + layers)
    pools = make_pools(g, layers)
    params = [(randn(g, HV, scale=0.5), randn(g, HV, scale=0.5), randn(g, D, KW, dtype=BF16, scale=0.5))
              for _ in range(layers)]
    calls = make_calls(g, layers, N)

    res_c = run_cycle(pools, "compact_lm", params, calls, N)
    res_s = run_cycle(pools, "strided_lm", params, calls, N)

    for a, b in zip(res_c[0] + res_c[1], res_s[0] + res_s[1]):
        for x, y in zip(a, b):
            assert torch.equal(x[:-PAD], y[:-PAD])  # padding outputs are checked separately
    assert torch.equal(res_c[2], res_s[2]) and torch.equal(res_c[3], res_s[3])
    all_layers = range(layers)
    rows = [r[0] for r in REQS]
    slots = [r[1] for r in REQS] + EXPORT_DST
    for name in ("u", "k", "g", "win"):
        pools[name].assert_same(rows, all_layers)
        pools[name].assert_untouched(rows, all_layers)
    pools["state"].assert_same(slots, all_layers)
    pools["state"].assert_untouched(slots, all_layers)
    start = res_s[2]
    assert start[[r for r in range(ROWS) if r not in rows]].tolist() == \
        [50 + r for r in range(ROWS) if r not in rows]

    check_reference(pools, params, calls, N, res_s, [0, layers - 1])


def check_reference(pools, params, calls, N, res, layers):
    outs, convs, start, stats, pos = res
    snap_state, snap_win = pools["state"].snapshot, pools["win"].snapshot
    for l in layers:
        A_log, dt_bias, weight = params[l]
        for i, (row, slot, p0) in enumerate(REQS):
            S = snap_state[slot, l]
            hist = snap_win[row, l][[(p0 - j) % WIN for j in range(KW - 1, 0, -1)]]
            for step, call in enumerate(calls):
                lens = call["lens"]
                off = sum(lens[:i])
                sl = slice(off, off + lens[i])
                inp = call["layers"][l]
                q, k, v = split_qkv(inp["qkv"][sl])
                gg, beta = gates(inp["a"][sl], inp["b"][sl], A_log, dt_bias)
                o_r, _, _, _ = gdn_ref(S, q, k, v, gg, beta)
                assert_close(outs[step][l][sl], o_r, **REF_TOL)
                assert_close(convs[step][l][sl], conv_ref(hist, inp["qkv"][sl], weight),
                             atol=3e-2, rtol=2e-2)
                n_acc = accepted(N, i, step) if step else lens[i]
                _, S, _, _ = gdn_ref(S, q[:n_acc], k[:n_acc], v[:n_acc], gg[:n_acc], beta[:n_acc])
                hist = torch.cat([hist, inp["qkv"][sl][:n_acc]])[-(KW - 1):]
            # Exported state = full accepted history; padding output is zero.
            assert_close(pools["state"].strided_lm[l][EXPORT_DST[i]], S, atol=2e-3, rtol=2e-2)
        for step, call in enumerate(calls):
            assert torch.count_nonzero(outs[step][l][sum(call["lens"]):]) == 0
    # b moves to the accepted end exactly when end + N - b > RING (checked before each call).
    for i, (row, _, p0) in enumerate(REQS):
        b, p = p0, p0 + FIRST[i]
        for step in range(1, len(calls)):
            if p + N - b > RING:
                b = p
            p += accepted(N, i, step)
        assert int(start[row]) == b and p == pos[i]
