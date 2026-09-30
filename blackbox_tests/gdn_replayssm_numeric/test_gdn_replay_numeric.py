"""Black-box numeric acceptance of gdn_replay / gdn_replay_advance / gdn_replay_fold (public contract
section 5.1), production head geometry: H=16, HV=32, K=V=128, bf16 activations, fp32 state.

Tolerances (fixed before any comparison ran; derivation in gdn_numeric_ref.py):
  out, u, k_hat : |x-x*| <= 2^-8|x*| + 2^-12 rms(x*)      (bf16 roundoff + fp32/3xTF32 margin)
  g             : |g-g*| <= 1e-5|g*| + 1e-6 exp(A_log)
  written state : |S-S*| <= 2^-20|S*| + 2^-12 rms_head(S*)
  vs ideal fp64 (records never rounded): rel-RMS <= 2^-6
Non-finite values, a position off by one and cross-request reads are failures in every test;
negative controls assert the tolerance rejects them. start/stats and untouched tensors are exact.
"""
import random

import pytest
import torch

from gdn_numeric_ref import (BF16, DEV, F64, H, HV, I32, K, OUT_ATOL, OUT_RTOL, STATE_ATOL, STATE_RTOL, V, Seq,
                             World, assert_rejects, assert_within_tolerance, compare, gdn_replay_fold, l2norm,
                             pad_spec, replay_records, rms, tol_ratio)


def ar_step(w, s, x, T=1, tag=""):
    """Target AR as the model runs it: gdn_replay(fold=True), then advance."""
    out = w.call([s.spec(x, phase="ar")], fold=True, tag=tag or s.name)
    s.p += T
    return out


def window_step(w, s, x, T, tag=""):
    """A window as SD runs it: conditional in-place fold with width T, then gdn_replay(fold=False)."""
    w.prefold([(s, T)])
    out = w.call([s.spec(x, phase="window")], tag=tag or s.name)
    s.p += T
    return out


@pytest.mark.parametrize("R", [16, 64])
def test_ar_and_windows_follow_fp64_recurrence(R, log):
    """Detects: wrong gating, normalisation, scale or GQA head mapping; a wrong in-window
    recurrence; wrong record values or ring positions; the fold=True write-back or advance at the
    wrong position; T=1 and T=2..9 disagreeing on one trajectory; a fresh request (zero state,
    position 0) or large positions mishandled."""
    w = World(log, L=1, slots=4, rows=4, R=R, seed=100 + R)
    w.warm_slot(0)
    w.copy_slot(0, 1)
    w.copy_slot(0, 2)  # slot 3 stays zero: a fresh request
    n = 3 * R + 11
    X = w.x(n)
    p0 = 2 * R - 5  # windows cross the ring end early
    runs = [(Seq(0, 0, p0, "ar"), [1], ar_step), (Seq(1, 1, 200001, "win2-9"), list(range(2, 10)), window_step),
            (Seq(2, 2, p0, "win9-fold"), [9], ar_step), (Seq(3, 3, 0, "fresh"), list(range(1, 10)), window_step)]
    for s, cycle, step in runs:
        w.assign(s.m, s.p)
        t = i = 0
        while t < n:
            T = min(cycle[i % len(cycle)], n - t)
            step(w, s, [tuple(c[t: t + T] for c in X)], T)
            t, i = t + T, i + 1
    assert w.expect_stats[0] >= 4 * 2, "trajectory never flushed as planned"
    assert_within_tolerance(log)


@pytest.mark.parametrize("R,N", [(16, 1), (16, 4), (16, 8), (64, 8)])
def test_verify_ignores_draft_records(R, N, log):
    """Detects: verify reading the discarded draft records at [p, p+N) or anything outside
    [b, p); the pre-draft fold running (or not) against the rule; states exported at acceptance
    positions off by one; stale draft records leaking into the continuation or into a request
    restarted from an exported prefix."""
    w = World(log, L=1, slots=N + 4, rows=2, R=R, seed=200 + 10 * R + N)
    w.warm_slot(0)
    w.warm_slot(N + 3)  # an unrelated request's state for the cross-request control
    s = Seq(0, 0, 3 * R - 2, "sd")
    w.assign(s.m, s.p)
    # history before drafting: none; the most that needs no fold; one more, so the fold runs
    for target in (0, R - N - 1, R - N):
        while s.p - w.b[s.m] < target:
            T = min(3, target - (s.p - w.b[s.m]))
            window_step(w, s, w.inputs(T), T, tag="history")
        ran = w.prefold([(s, N + 1)], tag=f"prefold-hist{target}")
        assert ran == [target > R - N - 1]
        b, p = w.b[s.m], s.p
        for i in range(N):  # draft: other inputs, one token per call, chained on draft records
            w.call([s.spec(w.inputs(1), p=p + i, phase="draft")], tag=f"draft-hist{target}")
        draft_snap = [t.clone() for t in w.snap]
        X = w.inputs(N + 1)
        out = w.call([s.spec(X, phase="verify")], tag=f"verify-hist{target}")[0][0]
        assert_rejects(w, out, w.start_ref(0, s.m, s.slot, b, p + 1, snap=draft_snap), X[0], 0,
                       "verify also reads the draft record at p")
        assert_rejects(w, out, w.start_ref(0, s.m, N + 3, b, p), X[0], 0, "another request's state")
        if p > b:
            assert_rejects(w, out, w.start_ref(0, s.m, s.slot, b, p - 1), X[0], 0, "misses record p-1")
            assert_rejects(w, out, w.start_ref(0, s.m, s.slot, b + 1, p), X[0], 0,
                           "state position b off by one")
        # the state after accepting a drafts (+1 target token) for every a, exported in one plan
        w.fold([(s.slot, 1 + a, s.m, p + a + 1, 0) for a in range(N + 1)], tag=f"export-hist{target}")
        a = N // 2
        r = Seq(1, 1 + a, p + a + 1, "restart")  # row 1 holds stale records from earlier use
        w.assign(r.m, r.p)
        window_step(w, r, w.inputs(2), 2, tag=f"restart-hist{target}")
        s.p = p + a + 1
        window_step(w, s, w.inputs(1), 1, tag=f"continue-hist{target}")
    assert_within_tolerance(log)


def test_fold_in_place_then_continue_matches_unfolded(log, extra):
    """Detects: an in-place fold that loses, repeats or mis-decays records or leaves the slot at
    a position other than `end`, seen as the folded request diverging from an unfolded twin fed
    identical inputs at different ring positions; a conditional row that should not run but does;
    per-layer view errors (L=3)."""
    w = World(log, L=3, slots=2, rows=2, R=32, seed=300)
    w.warm_slot(0)
    w.copy_slot(0, 1)
    A, B = Seq(0, 0, 90, "folded"), Seq(1, 1, 7, "unfolded")
    w.assign(A.m, A.p)
    w.assign(B.m, B.p)

    def both(T):
        x = w.inputs(T)
        oa, ob = w.call([A.spec(x)], tag="A"), w.call([B.spec(x)], tag="B")
        A.p, B.p = A.p + T, B.p + T
        return oa, ob

    for T in (1, 5, 9, 9):  # 24 records each
        both(T)
    assert w.prefold([(A, 9), (B, 8)], tag="in-place") == [True, False]  # 24+9 > 32, 24+8 = 32
    bitwise = []
    for T in (1, 4, 3):
        oa, ob = both(T)
        for l in range(w.L):
            ya, yb = oa[l][0], ob[l][0].to(F64)
            ratio = tol_ratio(ya, yb, 2 * OUT_RTOL, 2 * OUT_ATOL * rms(yb, -1))
            log.append(dict(kind="folded_vs_unfolded", layer=l, T=T, tol_ratio=ratio))
            bitwise.append(bool((oa[l][0] == ob[l][0]).all()))
    extra["folded_vs_unfolded_bitwise_equal"] = all(bitwise)
    assert_within_tolerance(log)


def test_fold_exports_prefix_states(log):
    """Detects: an export written for the wrong prefix (end or b off by one), records, start or
    the source slot modified, other slots touched, a conditional row run against the rule,
    per-layer indexing errors (L=3), empty (end=b) and full (end=b+R) exports, and an exported
    state that does not work as the start of a new request."""
    R = 16
    w = World(log, L=3, slots=8, rows=4, R=R, seed=310)
    for slot in (0, 6, 7):
        w.warm_slot(slot)
    s, t, v = Seq(0, 0, 5 * R + 3, "src"), Seq(2, 6, 11, "other"), Seq(3, 7, 40, "idle")
    for q, chunks in ((s, (1, 9, 6)), (t, (4, 3, 6)), (v, (2,))):  # s: exactly R records
        w.assign(q.m, q.p)
        for T in chunks:
            w.call([q.spec(w.inputs(T))], tag=f"{q.name}-history")
            q.p += T
    b0 = w.b[s.m]
    ran = w.fold([(0, 1, 0, b0, 0), (0, 2, 0, b0 + 1, 0), (0, 3, 0, b0 + 7, 0), (0, 4, 0, b0 + R, 0),
                  (6, 6, 2, t.p, 4), (7, 7, 3, v.p, 3)], tag="exports+conditional")
    assert ran == [True, True, True, True, True, False]  # 13+4 > 16 runs, 2+3 does not
    w.advance([t.m, v.m], [t.p, v.p], [4, 3])
    assert w.prefold([(s, 1)], tag="in-place-full-ring") == [True]
    r = Seq(1, 3, b0 + 7, "from-export")  # row 1 still holds the initial garbage records
    w.assign(r.m, r.p)
    for T in (3, 1):
        w.call([r.spec(w.inputs(T))], tag="from-export")
        r.p += T
    for T in (2, 9):
        w.call([s.spec(w.inputs(T)), t.spec(w.inputs(T)), v.spec(w.inputs(T))], tag="continue")
        s.p, t.p, v.p = s.p + T, t.p + T, v.p + T
    assert_within_tolerance(log)


def test_batch_rows_permuted_mixed_T_padding(log, extra):
    """Detects: a request reading another request's slot, records or tokens; dependence on the
    order of requests in a batch; wrong cu_seqlens/positions segmentation with mixed T; padding
    rows producing output or writing anything; strided qkv/a/b rows; fold=True writing back or
    advancing only the requests whose window overflows."""
    R = 32
    w = World(log, L=1, slots=8, rows=8, R=R, seed=500)
    slots, rows = [3, 7, 0, 5, 1, 6], [5, 2, 7, 0, 3, 6]
    hist, Ts = [0, 5, 27, 1, 12, 30], [1, 9, 3, 5, 7, 2]
    starts = [0, 45, 2 * R - 3, 1000, 123457, 31]
    seqs = [Seq(rows[i], slots[i], starts[i], f"r{i}") for i in range(6)]
    for sl in slots[1:]:
        w.warm_slot(sl)  # slot 3 (r0) stays zero: a fresh request at position 0
    for s in seqs:
        w.assign(s.m, s.p)
    rng = random.Random(5)
    while any(s.p - w.b[s.m] < h for s, h in zip(seqs, hist)):
        specs = [s.spec(w.inputs(min(4, h - (s.p - w.b[s.m]))), phase="history")
                 for s, h in zip(seqs, hist) if s.p - w.b[s.m] < h]
        specs.insert(rng.randrange(len(specs) + 1), pad_spec(w.inputs(rng.randint(1, 3))))
        w.call(specs, qkv_pad=64, tag="history")
        for sp in specs:
            for s in seqs:
                if sp["m"] == s.m:
                    s.p += sp["x"][0][0].shape[0]
    X = [w.inputs(T) for T in Ts]
    pads = [w.inputs(2), w.inputs(1)]
    saved = w.save()
    orders = {"order1": [0, "p0", 1, 2, "p1", 3, 4, 5], "order2": ["p1", 5, 3, 1, 4, 2, 0, "p0"]}
    got = {}
    for name, order in orders.items():
        w.restore(saved)
        specs = [pad_spec(pads[int(o[1])]) if isinstance(o, str) else seqs[o].spec(X[o], phase=name)
                 for o in order]
        outs = w.call(specs, qkv_pad=64, tag=name)[0]
        got[name] = {o: outs[j] for j, o in enumerate(order) if not isinstance(o, str)}
    for i in range(6):  # each request alone
        w.restore(saved)
        got.setdefault("solo", {})[i] = w.call([seqs[i].spec(X[i], phase="solo")], tag="solo")[0][0]
    bitwise = {}
    for i in range(6):
        ref = got["order1"][i].to(F64)
        for name in ("order2", "solo"):
            ratio = tol_ratio(got[name][i], ref, 2 * OUT_RTOL, 2 * OUT_ATOL * rms(ref, -1))
            log.append(dict(kind=f"order1_vs_{name}", name=f"r{i}", tol_ratio=ratio))
            bitwise[f"r{i}_{name}"] = bool((got[name][i] == got["order1"][i]).all())
    extra["bitwise_equal_across_orders_and_solo"] = bitwise
    # cross-request control on the widest window: another request's state/records must be rejected
    o = seqs[2]
    assert_rejects(w, got["order1"][1], w.start_ref(0, o.m, o.slot, w.b[o.m], o.p), X[1][0], 0,
                   "another request's slot and records")
    # fold=True batch: r2 (27+9) and r5 (30+3) overflow the ring and are written back, others not
    w.restore(saved)
    T2 = [2, 9, 9, 1, 9, 3]
    specs = [seqs[i].spec(w.inputs(T2[i]), phase="fold-batch") for i in (4, 2, 0, 5, 1, 3)]
    specs.insert(3, pad_spec(w.inputs(2)))
    w.call(specs, fold=True, qkv_pad=64, tag="fold-batch")
    assert w.expect_stats[0] == 2, w.expect_stats
    assert_within_tolerance(log)


@pytest.mark.parametrize("R", [16, 32, 64])
def test_long_trajectory_many_wraps_and_folds(R, log, extra):
    """Detects: errors that only show after the ring wraps many times or after repeated folds
    (stale positions, drift, error growth with steps), with AR steps (fold=True), SD rounds
    (pre-draft folds, drafts, verify with random acceptance) and random windows sharing batches,
    padding rows and prefix exports along the way (L=2)."""
    rng = random.Random(R)
    w = World(log, L=2, slots=5, rows=3, R=R, seed=400 + R)
    for slot in range(3):
        w.warm_slot(slot)
    ar, sd, win = Seq(2, 0, 5, "ar"), Seq(0, 1, 3 * R + 7, "sd"), Seq(1, 2, 200001, "win")
    reqs = (ar, sd, win)
    for s in reqs:
        w.assign(s.m, s.p)
    total = max(8 * R, 256)
    draft = None  # [N, drafts done] while sd is drafting
    counts = dict(calls=0, ar_steps=0, sd_steps=0, verifies=0, exports=0)
    while min(s.p - s.p0 for s in reqs) < total:
        T_win = rng.randint(1, 9)
        if draft is None and rng.random() < 0.4:  # an AR step for everyone
            specs = [ar.spec(w.inputs(1), phase="ar"), sd.spec(w.inputs(1), phase="ar"),
                     win.spec(w.inputs(T_win), phase="ar")]
            fold, advance = True, {ar: 1, sd: 1, win: T_win}
            counts["ar_steps"] += 1
        else:  # an SD step: pre-folds for everyone not mid-draft, then fold=False
            if draft is None:
                draft = [rng.randint(1, min(8, R - 1)), 0]
            N, i = draft
            w.prefold([(ar, 1), (win, T_win)] + ([(sd, N + 1)] if i == 0 else []))
            specs = [ar.spec(w.inputs(1), phase="ar1"), win.spec(w.inputs(T_win), phase="win"),
                     sd.spec(w.inputs(1), p=sd.p + i, phase="draft") if i < N else
                     sd.spec(w.inputs(N + 1), phase="verify")]
            fold, advance = False, {ar: 1, win: T_win}
            counts["sd_steps"] += 1
        if rng.random() < 0.1:
            e = rng.choice(reqs)
            w.fold([(e.slot, 3 + counts["exports"] % 2, e.m, rng.randint(w.b[e.m], e.p), 0)], tag="export")
            counts["exports"] += 1
        rng.shuffle(specs)
        if rng.random() < 0.3:
            specs.insert(rng.randrange(len(specs) + 1), pad_spec(w.inputs(rng.randint(1, 4))))
        w.call(specs, fold=fold, tag="traj")
        counts["calls"] += 1
        for s, T in advance.items():
            s.p += T
        if not fold:
            N, i = draft
            if i < N:
                draft[1] += 1
            else:
                sd.p += rng.randint(0, N) + 1
                draft = None
                counts["verifies"] += 1
    counts["flushes"], counts["flushed_records"] = w.expect_stats
    counts["ring_wraps_per_request"] = {s.name: (s.p - s.p0) / R for s in reqs}
    extra["counts"] = counts
    # error as a function of steps: 8 equal bins over each request's own trajectory
    series = {}
    for name in ("ar", "sd", "win"):
        es = [e for e in log if e.get("tag") == "traj" and e.get("name") == name]
        last = max(e["step"] for e in es) + 1
        bins = [dict(first_step=j * last // 8, out_max_tol_ratio=0.0, out_max_abs=0.0, out_max_rel=0.0,
                     out_max_rel_rms=0.0, ideal_max_rel_rms=0.0, ideal_mean_rel_rms=[]) for j in range(8)]
        for e in es:
            bn = bins[e["step"] * 8 // last]
            if e["kind"] == "out":
                bn["out_max_tol_ratio"] = max(bn["out_max_tol_ratio"], e["tol_ratio"])
                bn["out_max_abs"] = max(bn["out_max_abs"], e["max_abs"])
                bn["out_max_rel"] = max(bn["out_max_rel"], e["max_rel"])
                bn["out_max_rel_rms"] = max(bn["out_max_rel_rms"], e["rel_rms"])
            elif e["kind"] == "out_vs_ideal":
                bn["ideal_max_rel_rms"] = max(bn["ideal_max_rel_rms"], e["rel_rms"])
                bn["ideal_mean_rel_rms"].append(e["rel_rms"])
        for bn in bins:
            vals = bn["ideal_mean_rel_rms"]
            bn["ideal_mean_rel_rms"] = sum(vals) / len(vals) if vals else None
        series[name] = bins
    extra["error_by_step_bins"] = series
    assert_within_tolerance(log)


@pytest.mark.parametrize("count", [4, 16, 64])
def test_fold_precision_with_strong_varying_decay(count, log, extra):
    """Detects: fold accuracy far from fp32 when the folded records carry large per-token decays
    that vary between tokens (the checkpoint has heads with g near -92 per token at a=0 in
    layer 0, and |g| > 8 in six layers). Records are written directly: fold's published input."""
    R = 64
    gen = torch.Generator(device=DEV).manual_seed(700 + count)
    randn = lambda *shape: torch.randn(*shape, generator=gen, device=DEV)
    state = torch.zeros(1, 2, HV, V, K, device=DEV)
    state[0, 0] = 0.1 * randn(HV, V, K)
    u = (0.5 * randn(1, 1, HV, R, V)).to(BF16)
    k = l2norm(randn(1, 1, H, R, K)).to(BF16)
    typical = 2.0 ** (torch.arange(HV, device=DEV) / 2 - 6)  # per-head |g|: 2^-6 .. 2^9.5
    g = -typical[None, None, :, None] * (0.5 + torch.rand(1, 1, HV, R, generator=gen, device=DEV))
    b = 1000
    start = torch.tensor([b], dtype=I32, device=DEV)
    gdn_replay_fold(state, u, k, g, start, torch.tensor([[0, 1, 0, b + count, 0]], dtype=I32, device=DEV))
    idx = torch.arange(b, b + count, device=DEV) & (R - 1)
    ref = replay_records(state[0, 0].to(F64), u[0, 0][:, idx].transpose(0, 1).to(F64),
                         k[0, 0][:, idx].transpose(0, 1).to(F64), g[0, 0][:, idx].T.to(F64))
    cum = g[0, 0][:, idx].to(F64).sum(-1)
    for j in range(HV):
        compare(log, "fold_state", state[0, 1, j], ref[j], STATE_RTOL, STATE_ATOL * rms(ref[j], (-2, -1)),
                head=j, count=count, sum_g=cum[j].item())
    extra["per_head"] = [dict(head=e["head"], sum_g=e["sum_g"], tol_ratio=e["tol_ratio"], max_rel=e["max_rel"],
                              rel_rms=e["rel_rms"]) for e in log]
    assert_within_tolerance(log)
