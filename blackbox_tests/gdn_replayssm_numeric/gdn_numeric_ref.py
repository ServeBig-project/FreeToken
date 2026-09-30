"""fp64 oracle and checking harness for the public GDN ReplaySSM operator contract
(docs/replayssm-public-contract.md, sections 5 and 5.1). Written only from that contract;
the operators under test are never used as their own reference.

Tolerances, fixed from the contract's precision rules before any comparison was run:

  out, record u, record k_hat (bf16) vs fp64:  |x - x*| <= 2^-8 |x*| + 2^-12 rms(x*)
  record g (fp32) vs fp64:                     |g - g*| <= 1e-5 |g*| + 1e-6 exp(A_log)
  written state (fp32: fold, export, fold=True write-back) vs fp64:
                                               |S - S*| <= 2^-20 |S*| + 2^-12 rms_head(S*)
  conv out (bf16) vs fp64:                     |o - o*| <= 2^-8 |o*| + 2^-16 sum_j |w_j x_j|
  vs ideal fp64 recurrence (records unrounded): ||x - x*|| / ||x*|| <= 2^-6 per call / state

rms(x*) is taken over the last dim (one head's V or K vector); rms_head over one head's VxK.

Why these numbers:
- 2^-8 is the bf16 unit roundoff: one rounding of an accurate value stays within 2^-8 of it
  relative to the unrounded fp64 reference.
- The contract computes every multiply-add in fp32 and replays records with 3xTF32 ("close to
  fp32"). fp32 over <= 64 replayed records and 128-long dot products costs ~1e-6 of the vector
  RMS; 2^-12 (2.4e-4) keeps >100x margin, while one missing, extra or shifted record, another
  request's state, or a draft record read by verify moves results by >= 1e-2 of the RMS
  (asserted by the negative controls in the tests).
- Records of earlier calls are stored in bf16 (contract). The tight ("conditioned") reference
  therefore replays exactly the stored records, after each was checked against fp64 when it
  was written; the in-call recurrence stays unrounded fp64. Rounding its own fp64 records
  instead would put isolated 2^-7 bf16 flips into the reference.
- The ideal reference never rounds records: bf16 u, k_hat add ~2^-8*sqrt(2) relative error to
  the state; decay <= 1 and the delta-rule update (I - beta k k^T) are non-expansive, so this
  does not grow with steps. 2^-6 leaves ~3x margin.
- g: fp32 exp/softplus are accurate to a few ulp; log(1+e^x) in fp32 loses ~2^-24 absolute
  (times exp(A_log)) when e^x is small, hence the atol.
- conv: bf16 products are exact in fp32; the 4-term fp32 sum and silu cost ~1e-6 sum|w x|.
Non-finite values always fail. Zero padding output and unchanged state / records / windows are
exact (bitwise) checks.
"""
import torch

from freetoken.kernel.triton.gdn_replay import gdn_replay, gdn_replay_advance, gdn_replay_conv, gdn_replay_fold

H, HV, K, V, KW = 16, 32, 128, 128, 4
REP = HV // H
D = 2 * H * K + HV * V
SCALE = K ** -0.5
DEV = "cuda"
F64, F32, BF16, I32 = torch.float64, torch.float32, torch.bfloat16, torch.int32

OUT_RTOL, OUT_ATOL = 2 ** -8, 2 ** -12
G_RTOL, G_ATOL = 1e-5, 1e-6
STATE_RTOL, STATE_ATOL = 2 ** -20, 2 ** -12
IDEAL_REL = 2 ** -6
CONV_RTOL, CONV_ATOL = 2 ** -8, 2 ** -16


# ---------------------------------------------------------------- fp64 reference math

def rms(x, dims):
    return x.pow(2).mean(dim=dims, keepdim=True).sqrt()


def l2norm(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def to_value_heads(x):
    """[..., H, K] -> [..., HV, K]: value head j uses key head j // (HV/H)."""
    return x.repeat_interleave(REP, dim=-2)


def gates(a, b, A_log, dt_bias):
    x = a.to(F64) + dt_bias.to(F64)
    softplus = torch.where(x > 20, x, torch.log1p(torch.exp(x.clamp(max=20))))
    return -torch.exp(A_log.to(F64)) * softplus, torch.sigmoid(b.to(F64))


def recur(S, qkv, a, b, A_log, dt_bias):
    """Token-by-token recurrence from state S [HV,V,K]; returns final S and per-token y, u, k_hat, g,
    and sum_k |S_vk q_k|, the size of the terms summed into y (fp32 rounding scales with it)."""
    x = qkv.to(F64)
    T = x.shape[0]
    q = l2norm(x[:, : H * K].reshape(T, H, K)) * SCALE
    k = l2norm(x[:, H * K: 2 * H * K].reshape(T, H, K))
    v = x[:, 2 * H * K:].reshape(T, HV, V)
    g, beta = gates(a, b, A_log, dt_bias)
    ys, us, mags = [], [], []
    for t in range(T):
        kh, qh = to_value_heads(k[t]), to_value_heads(q[t])
        S = torch.exp(g[t])[:, None, None] * S
        u = beta[t][:, None] * (v[t] - torch.einsum("jvk,jk->jv", S, kh))
        S = S + u[:, :, None] * kh[:, None, :]
        ys.append(torch.einsum("jvk,jk->jv", S, qh))
        mags.append(torch.einsum("jvk,jk->jv", S.abs(), qh.abs()))
        us.append(u)
    return S, torch.stack(ys), torch.stack(us), k, g, torch.stack(mags)


def replay_records(S, u, k, g):
    """Apply records S <- exp(g) S + u k^T in order; u [n,HV,V], k [n,H,K], g [n,HV]."""
    if u.shape[0] == 0:
        return S
    cs = g.cumsum(0)
    later_decay = torch.exp(cs[-1] - cs)
    return (torch.exp(cs[-1])[:, None, None] * S
            + torch.einsum("ij,ijv,ijk->jvk", later_decay, u, to_value_heads(k)))


def conv_ref(weight, window_row, x, p):
    """out_t = silu(sum_j w[:,j] x(p+t-KW+1+j)); positions < p come from window[q mod W]."""
    W = window_row.shape[0]
    hist = window_row[[q % W for q in range(p - KW + 1, p)]]
    xs = torch.cat([hist, x]).to(F64).unfold(0, KW, 1)  # [T, D, KW]
    terms = xs * weight.to(F64)
    z = terms.sum(-1)
    return z * torch.sigmoid(z), terms.abs().sum(-1)


# ---------------------------------------------------------------- checks

def tol_ratio(got, ref, rtol, atol):
    d = (got.to(F64) - ref).abs()
    return torch.where(d == 0, 0.0, d / (rtol * ref.abs() + atol)).max().item()


# Tolerance checks only log (tol_ratio > 1 is a violation) so one violation does not hide the
# rest of a test; every test ends with assert_within_tolerance. Non-finite values fail at once.

def compare(log, kind, got, ref, rtol, atol, **info):
    """Elementwise |got-ref| <= rtol|ref| + atol, plus finiteness; logs the error figures."""
    assert torch.isfinite(got).all(), f"{kind}: non-finite values {info}"
    ref = ref.to(F64)
    d = (got.to(F64) - ref).abs()
    r = ref.abs()
    big = r >= rms(ref, tuple(range(ref.dim())))
    log.append(dict(kind=kind, **info, tol_ratio=tol_ratio(got, ref, rtol, atol), max_abs=d.max().item(),
                    max_rel=(d[big] / r[big]).max().item() if big.any() else 0.0,
                    rel_rms=(d.norm() / r.norm()).item() if r.norm() > 0 else float(d.norm() > 0)))


def compare_ideal(log, kind, got, ideal, **info):
    assert torch.isfinite(got).all(), f"{kind}: non-finite values {info}"
    d = got.to(F64) - ideal
    log.append(dict(kind=kind, **info, tol_ratio=(d.norm() / ideal.norm()).item() / IDEAL_REL,
                    rel_rms=(d.norm() / ideal.norm()).item(), max_abs=d.abs().max().item()))


def assert_within_tolerance(log):
    bad = {}
    for e in log:
        if e["kind"] != "negative_control" and e.get("tol_ratio", 0.0) > 1.0:
            worst = bad.setdefault(e["kind"], [0, e])
            worst[0] += 1
            if e["tol_ratio"] > worst[1]["tol_ratio"]:
                worst[1] = e
    assert not bad, "outside tolerance: " + "; ".join(
        f"{kind}: {n} checks, worst {e}" for kind, (n, e) in bad.items())


# ---------------------------------------------------------------- recurrence harness

def i32(values):
    return torch.tensor(values, dtype=I32, device=DEV)


def positions_of(specs, Ts):
    return i32([s["p"] + t for s, T in zip(specs, Ts) for t in range(T)])


class Seq:
    """One request: record row m, full-state slot, next input position p (its b lives in World.b[m])."""

    def __init__(self, m, slot, pos, name):
        self.m, self.slot, self.p, self.p0, self.name = m, slot, pos, pos, name

    def spec(self, x, p=None, phase=""):
        p = self.p if p is None else p
        return dict(m=self.m, slot=self.slot, p=p, x=x, name=self.name, step=p - self.p0, phase=phase)


def pad_spec(x, slot=0):
    return dict(m=-1, slot=slot, p=0, x=x, name="pad", step=0, phase="pad")


class World:
    """Operator tensors plus two fp64 mirrors: `ref` (replays the stored, already verified
    records) and `ideal` (re-runs the recurrence from the raw inputs, never rounding).
    `b[m]` is the expected content of the operator's `start` array."""

    def __init__(self, log, L=1, slots=4, rows=4, R=16, seed=0):
        self.log, self.L, self.R = log, L, R
        self.gen = torch.Generator(device=DEV).manual_seed(seed)
        self.A_log = self.uniform((L, HV), -4.0, 4.3)
        self.dt_bias = self.uniform((L, HV), -5.3, 4.0)
        # the checkpoint has both extremes: a head reset almost every token, a head with decay ~1
        self.dt_bias[:, 0] = 15.5
        self.A_log[:, 1], self.dt_bias[:, 1] = -4.0, -5.0
        self.state = torch.zeros(L, slots, HV, V, K, device=DEV)
        # records start as garbage that must never be read
        self.u = self.randn(L, rows, HV, R, V).to(BF16)
        self.k = (self.randn(L, rows, H, R, K) * K ** -0.5).to(BF16)
        self.g = -2 * torch.rand(L, rows, HV, R, generator=self.gen, device=DEV)
        self.start = torch.zeros(rows, dtype=I32, device=DEV)
        self.stats = torch.zeros(2, dtype=torch.int64, device=DEV)
        self.b, self.expect_stats = [0] * rows, [0, 0]
        self.ref = self.state.to(F64)
        self.ideal = self.ref.clone()
        self.snap = [self.u.clone(), self.k.clone(), self.g.clone()]
        self.hist = {}  # (layer, row, position) -> latest raw input written there

    def randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=DEV)

    def uniform(self, shape, lo, hi):
        return lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=DEV)

    def x(self, T):
        return (self.randn(T, D).to(BF16), (1.5 * self.randn(T, HV)).to(BF16), self.randn(T, HV).to(BF16))

    def inputs(self, T):
        return [self.x(T) for _ in range(self.L)]

    def warm_slot(self, slot, n=24):
        """A realistic non-zero start: fp64 recurrence over n random tokens, stored as fp32."""
        for l in range(self.L):
            S = recur(torch.zeros(HV, V, K, dtype=F64, device=DEV), *self.x(n), self.A_log[l], self.dt_bias[l])[0]
            self.state[l, slot] = S.to(F32)
            self.ref[l, slot] = self.ideal[l, slot] = self.state[l, slot].to(F64)

    def copy_slot(self, src, dst):
        for t in (self.state, self.ref, self.ideal):
            t[:, dst] = t[:, src]

    def assign(self, m, pos):
        """Caller-side setup: a request takes row m while its slot holds the state at `pos`."""
        self.start[m] = pos
        self.b[m] = pos

    def save(self):
        return [t.clone() for t in (self.u, self.k, self.g, self.state)], dict(self.hist), self.ref.clone(), \
            self.ideal.clone(), list(self.b), list(self.expect_stats), self.start.clone(), self.stats.clone()

    def restore(self, saved):
        recs, hist, ref, ideal, b, st, start, stats = saved
        for live, s in zip((self.u, self.k, self.g, self.state), recs):
            live.copy_(s)
        for snap, s in zip(self.snap, recs):
            snap.copy_(s)
        self.hist, self.ref, self.ideal, self.b, self.expect_stats = dict(hist), ref.clone(), ideal.clone(), list(b), list(st)
        self.start.copy_(start)
        self.stats.copy_(stats)

    def start_ref(self, l, m, slot, b, p, snap=None):
        su, sk, sg = snap or self.snap
        idx = torch.arange(b, p, device=DEV) & (self.R - 1)
        return replay_records(self.ref[l, slot], su[l, m][:, idx].transpose(0, 1).to(F64),
                              sk[l, m][:, idx].transpose(0, 1).to(F64), sg[l, m][:, idx].T.to(F64))

    def start_ideal(self, l, m, slot, b, p):
        S = self.ideal[l, slot]
        if p > b:
            xs = [self.hist[(l, m, q)] for q in range(b, p)]
            S = recur(S, *(torch.cat(c) for c in zip(*xs)), self.A_log[l], self.dt_bias[l])[0]
        return S

    def check_state(self, kind, l, slot, ref, ideal, **info):
        compare(self.log, kind, self.state[l, slot], ref, STATE_RTOL, STATE_ATOL * rms(ref, (-2, -1)), **info)
        compare_ideal(self.log, kind + "_vs_ideal", self.state[l, slot], ideal, **info)

    def call(self, specs, fold=False, qkv_pad=0, tag=""):
        """One gdn_replay per layer over `specs` (then gdn_replay_advance when fold=True, as the
        model does after its last layer); checks outputs, records, state writes, untouched tensors.
        Returns per layer the list of per-spec output slices."""
        Ts = [s["x"][0][0].shape[0] for s in specs]
        cu = [0]
        for T in Ts:
            cu.append(cu[-1] + T)
        real = [i for i, s in enumerate(specs) if s["m"] >= 0]
        folds = {i for i in real if fold and specs[i]["p"] + Ts[i] - self.b[specs[i]["m"]] > self.R}
        args = (self.start, i32(cu), i32([s["slot"] for s in specs]), i32([s["m"] for s in specs]),
                positions_of(specs, Ts), SCALE)
        result = []
        for l in range(self.L):
            qkv, a, b = (torch.cat(c) for c in zip(*(s["x"][l] for s in specs)))
            if qkv_pad:  # production splits these out of wider projection rows
                buf = torch.zeros(cu[-1], D + qkv_pad, dtype=BF16, device=DEV)
                buf[:, :D] = qkv
                ab = torch.zeros(cu[-1], 2 * HV + qkv_pad, dtype=BF16, device=DEV)
                ab[:, :HV], ab[:, HV: 2 * HV] = a, b
                qkv, a, b = buf[:, :D], ab[:, :HV], ab[:, HV: 2 * HV]
            state0, start0 = self.state[l].clone(), self.start.clone()
            out = gdn_replay(qkv, a, b, self.A_log[l], self.dt_bias[l], self.state[l], self.u[l], self.k[l],
                             self.g[l], *args, fold=fold)
            assert out.shape == (cu[-1], HV, V) and out.dtype == BF16, (out.shape, out.dtype)
            assert torch.equal(self.start, start0), "gdn_replay modified start"
            expect = [t[l].clone() for t in self.snap]
            new_states = {}
            for i, s in enumerate(specs):
                o = out[cu[i]: cu[i + 1]]
                info = dict(tag=tag, layer=l, name=s["name"], step=s["step"], phase=s["phase"], T=Ts[i],
                            hist=s["p"] - self.b[s["m"]] if s["m"] >= 0 else 0, fold=fold)
                if s["m"] < 0:
                    assert torch.equal(o, torch.zeros_like(o)), f"padding row produced output {info}"
                    continue
                m, T, p, slot = s["m"], Ts[i], s["p"], s["slot"]
                S0 = self.start_ref(l, m, slot, self.b[m], p)
                if i in folds:  # the prefix state of length p is written back to the slot
                    Si = self.start_ideal(l, m, slot, self.b[m], p)
                    self.check_state("replay_fold_state", l, slot, S0, Si, **info)
                    new_states[slot] = (S0, Si)
                _, y, u, kh, g, mag = recur(S0, *s["x"][l], self.A_log[l], self.dt_bias[l])
                compare(self.log, "out", o, y, OUT_RTOL, OUT_ATOL * rms(y, -1), **info)
                # diagnostics only: error beyond bf16 rounding relative to the size of the terms
                # summed into y, and for a violation how small that head's y is against those terms
                d = (o.to(F64) - y).abs()
                e = self.log[-1]
                e["excess_over_terms"] = ((d - OUT_RTOL * y.abs()).clamp(min=0) / mag).max().item()
                if e["tol_ratio"] > 1:
                    t, j = divmod(int((d / (OUT_RTOL * y.abs() + OUT_ATOL * rms(y, -1))).amax(-1).argmax()), HV)
                    e["worst_head_y_rms_over_terms_rms"] = (rms(y[t, j], 0) / rms(mag[t, j], 0)).item()
                yi = recur(self.start_ideal(l, m, slot, self.b[m], p), *s["x"][l], self.A_log[l],
                           self.dt_bias[l])[1]
                compare_ideal(self.log, "out_vs_ideal", o, yi, **info)
                idx = torch.arange(p, p + T, device=DEV) & (self.R - 1)
                got_u = self.u[l, m][:, idx].transpose(0, 1)
                got_k = self.k[l, m][:, idx].transpose(0, 1)
                got_g = self.g[l, m][:, idx].T
                compare(self.log, "rec_u", got_u, u, OUT_RTOL, OUT_ATOL * rms(u, -1), **info)
                compare(self.log, "rec_k", got_k, kh, OUT_RTOL, OUT_ATOL * rms(kh, -1), **info)
                compare(self.log, "rec_g", got_g, g, G_RTOL, G_ATOL * torch.exp(self.A_log[l].to(F64)), **info)
                for ex, live in zip(expect, (self.u, self.k, self.g)):
                    ex[m][:, idx] = live[l, m][:, idx]
            for ex, live, name in zip(expect, (self.u, self.k, self.g), "ukg"):
                assert torch.equal(live[l], ex), f"gdn_replay changed record {name} outside the written windows"
            for slot in range(self.state.shape[1]):
                if slot not in new_states:
                    assert torch.equal(self.state[l, slot], state0[slot]), f"gdn_replay wrote state slot {slot}"
            for slot, (S0, Si) in new_states.items():
                self.ref[l, slot], self.ideal[l, slot] = S0, Si
            for snap, live in zip(self.snap, (self.u, self.k, self.g)):
                snap[l].copy_(live[l])
            result.append([out[cu[i]: cu[i + 1]] for i in range(len(specs))])
        for i in real:
            for t in range(Ts[i]):
                for l in range(self.L):
                    self.hist[(l, specs[i]["m"], specs[i]["p"] + t)] = tuple(c[t: t + 1] for c in specs[i]["x"][l])
        if fold:
            widths = 1 if all(T == 1 for T in Ts) else [Ts[i] for i in real]
            self.advance([specs[i]["m"] for i in real], [specs[i]["p"] for i in real], widths)
        return result

    def advance(self, rows, ends, widths):
        """gdn_replay_advance; checks start and stats against the contract's rule."""
        warg = widths if isinstance(widths, int) else i32(widths)
        gdn_replay_advance(self.start, self.stats, i32(rows), i32(ends), warg, self.R)
        for m, end, width in zip(rows, ends, [widths] * len(rows) if isinstance(widths, int) else widths):
            if m >= 0 and end + width - self.b[m] > self.R:
                self.expect_stats[0] += 1
                self.expect_stats[1] += end - self.b[m]
                for l in range(self.L):  # inputs folded into the row's slot are no longer read
                    for q in range(self.b[m], end):
                        self.hist.pop((l, m, q), None)
                self.b[m] = end
        assert self.start.tolist() == self.b, f"start {self.start.tolist()} != expected {self.b}"
        assert self.stats.tolist() == self.expect_stats, f"stats {self.stats.tolist()} != {self.expect_stats}"

    def fold(self, plan, tag=""):
        """gdn_replay_fold over all layers; plan rows (src, dst, m, end, width). Returns which ran."""
        state0, start0 = self.state.clone(), self.start.clone()
        gdn_replay_fold(self.state, self.u, self.k, self.g, self.start, i32(plan))
        for live, snap, name in zip((self.u, self.k, self.g), self.snap, "ukg"):
            assert torch.equal(live, snap), f"gdn_replay_fold modified record {name}"
        assert torch.equal(self.start, start0), "gdn_replay_fold modified start"
        new, ran = {}, []
        for src, dst, m, end, width in plan:
            b = self.b[m]
            ran.append(width == 0 or end + width - b > self.R)
            if not ran[-1]:
                continue
            for l in range(self.L):
                info = dict(tag=tag, layer=l, src=src, dst=dst, b=b, end=end, width=width)
                ref, ideal = self.start_ref(l, m, src, b, end), self.start_ideal(l, m, src, b, end)
                self.check_state("fold_state", l, dst, ref, ideal, **info)
                if end == b:
                    self.log.append(dict(kind="fold_count0_bitwise_copy", **info,
                                         value=bool(torch.equal(self.state[l, dst], state0[l, src]))))
                new[(l, dst)] = (ref, ideal)
        dsts = {row[1] for row, r in zip(plan, ran) if r}
        for slot in range(self.state.shape[1]):
            if slot not in dsts:
                assert torch.equal(self.state[:, slot], state0[:, slot]), f"fold touched slot {slot}"
        for (l, dst), (ref, ideal) in new.items():
            self.ref[l, dst], self.ideal[l, dst] = ref, ideal
        self.log.append(dict(kind="fold_ran", tag=tag, value=ran))
        return ran

    def prefold(self, reqs, tag="prefold"):
        """What SD does before a round: conditional in-place fold per request, then advance."""
        plan = [(s.slot, s.slot, s.m, s.p, width) for s, width in reqs]
        ran = self.fold(plan, tag=tag)
        self.advance([s.m for s, _ in reqs], [s.p for s, _ in reqs], [width for _, width in reqs])
        return ran


def assert_rejects(w, got, alt_start, x, l, name):
    """The tight output tolerance must not accept the output of a plausible wrong start state."""
    y = recur(alt_start, *x, w.A_log[l], w.dt_bias[l])[1]
    ratio = tol_ratio(got, y, OUT_RTOL, OUT_ATOL * rms(y, -1))
    w.log.append(dict(kind="negative_control", name=name, tol_ratio=ratio))
    assert ratio > 1.0, f"tolerance cannot tell '{name}' from the correct result (ratio {ratio})"


# ---------------------------------------------------------------- conv harness

class ConvWorld:
    def __init__(self, log, rows, W, wdtype, seed):
        self.log, self.W = log, W
        self.gen = torch.Generator(device=DEV).manual_seed(seed)
        self.weight = (0.3 * torch.randn(D, KW, generator=self.gen, device=DEV)).to(wdtype)
        # random everywhere: stands for whatever the caller wrote for earlier positions
        self.window = torch.randn(rows, W, D, generator=self.gen, device=DEV).to(BF16)
        self.snap = self.window.clone()

    def x(self, T):
        return torch.randn(T, D, generator=self.gen, device=DEV).to(BF16)

    def call(self, specs, tag=""):
        """specs: dict(m, p, x[T,D], name). Checks outputs, window writes and untouched slots."""
        Ts = [s["x"].shape[0] for s in specs]
        cu = [0]
        for T in Ts:
            cu.append(cu[-1] + T)
        x = torch.cat([s["x"] for s in specs])
        out = gdn_replay_conv(x, self.weight, self.window, i32(cu), i32([s["m"] for s in specs]),
                              positions_of(specs, Ts))
        assert out.shape == x.shape and out.dtype == x.dtype, (out.shape, out.dtype)
        expect = self.snap.clone()
        for i, s in enumerate(specs):
            o = out[cu[i]: cu[i + 1]]
            info = dict(tag=tag, name=s["name"], p=s["p"], T=Ts[i])
            if s["m"] < 0:
                assert torch.equal(o, torch.zeros_like(o)), f"padding row produced conv output {info}"
                continue
            ref, mag = conv_ref(self.weight, self.snap[s["m"]], s["x"], s["p"])
            compare(self.log, "conv_out", o, ref, CONV_RTOL, CONV_ATOL * mag, **info)
            expect[s["m"], [(s["p"] + t) % self.W for t in range(Ts[i])]] = s["x"]
        assert torch.equal(self.window, expect), "conv window differs from writes at (p+t) % W"
        self.snap = self.window.clone()
        return [out[cu[i]: cu[i + 1]] for i in range(len(specs))]
