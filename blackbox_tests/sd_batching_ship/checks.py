"""Scenario checks shared by the per-configuration sessions. Section numbers refer to the contract."""
import time

from . import env, view
from .client import common_prefix, count_prompt, record


def exact_len(r, n):
    assert r["usage"].get("completion_tokens") == n, f"completion_tokens {r['usage']} != max_tokens {n}"
    assert r["finish"] == "length", r["finish"]


def effective(se, batching, sd, phase=None, drafter=None):
    """Section 2/4: ready stats report the actual batching and SD state separately."""
    s = se.c.stats()
    assert (s.get("execution") or {}).get("effective") is not None, f"no execution.effective: {s}"
    for k in ("sd_inwave", "sd_outwave"):  # section 4: per-phase counters exist (zero) from ready, SD on or off
        assert isinstance(view.get(s, k), (int, float)), f"{k} not a number at ready: {view.get(s, k)}"
    b = str(view.get(s, "batching")).lower()
    assert batching in b, f"effective batching {b!r}, expected {batching!r}"
    assert view.sd_on(s) == sd, f"effective SD on={view.sd_on(s)}, expected {sd}: {view.text(s)[:1500]}"
    if sd:
        assert view.num(s, "steps") == se.steps, f"effective steps {view.get(s, 'steps')} != {se.steps}"
    if phase:
        assert phase in str(view.get(s, "phase")).lower(), view.get(s, "phase")
    if drafter:
        assert drafter in str(view.get(s, "drafter")).lower(), view.get(s, "drafter")
    return s


def sd_zero(se):
    """Section 2: SD off means zero SD execution counters."""
    s = se.c.stats()
    for k in ("drafted", "rounds"):
        assert view.num(s, k) == 0, f"SD off but {k}={view.get(s, k)}"
    assert not view.sd_on(s)


def decode_sd(se, max_tokens=96, expect_sd=True):
    """Pure decode outside any prefill wave: real SD must execute (section 3, 'no AR posing as SD')."""
    b = se.c.stats()
    r = se.c.complete(count_prompt(), max_tokens)
    exact_len(r, max_tokens)
    a = se.c.stats()
    d = {k: view.delta(b, a, k) for k in ("drafted", "accepted", "rounds")}
    record(se.name + ":decode_sd", d)
    if expect_sd:
        assert d["rounds"] > 0 and d["drafted"] > 0, f"no real SD during plain decode: {d}"
        assert 0 <= d["accepted"] <= d["drafted"], d
        # each round commits at least one token; with ignore_eos the request needs max_tokens commits
        assert d["rounds"] <= max_tokens, d
    else:
        assert d["rounds"] == 0 and d["drafted"] == 0, f"SD ran though not expected: {d}"
    return r, d


def draft_hist_bounded(se):
    """Section 4: draft-length distribution exists and no draft exceeds the enabled maximum."""
    h = view.get(se.c.stats(), "draft_len_hist")
    items = h.items() if isinstance(h, dict) else enumerate(h)
    for k, n in items:
        if n:
            assert int(k) <= se.steps, f"draft length {k} > max steps {se.steps}: {h}"


def staggered_wave(se, long_tokens=2500, n_long=3, decode_tokens=256, wave_tokens=48):
    """One request decodes while long prompts arrive 0.5 s apart (prefill waves). All must finish."""
    tok = se.tok
    longs = [tok.filler(long_tokens + 37 * i, seed=1000 + 10 * i + len(se.name)) for i in range(n_long)]
    b = se.c.stats()
    jobs = [(se.c.complete, (count_prompt(5), decode_tokens), {})]
    jobs += [(se.c.complete, (p, wave_tokens), {}) for p in longs]
    res = se.c.parallel(jobs, stagger_s=0.5)
    a = se.c.stats()
    exact_len(res[0], decode_tokens)
    for r in res[1:]:
        exact_len(r, wave_tokens)
    d = {k: view.delta(b, a, k) for k in ("rounds", "sd_inwave", "sd_outwave", "drafted")}
    d["ar_by_reason"] = {k: v - view.reasons(b).get(k, 0) for k, v in view.reasons(a).items()
                         if isinstance(v, (int, float))}
    record(se.name + ":staggered_wave", d)
    return d


def phase_counts(se, phase):
    """Section 2/4: phase policy shows up in real in-wave/out-of-wave SD and phase-rule AR counts."""
    b = se.c.stats()
    d = staggered_wave(se)
    a = se.c.stats()
    phase_ar = view.reason_count(a, "phase") - view.reason_count(b, "phase")
    if phase == "outwave":
        assert d["sd_inwave"] == 0, f"outwave policy ran SD inside a wave: {d}"
        assert d["sd_outwave"] > 0, d
        assert phase_ar > 0, f"in-wave AR not counted under a phase reason: {d}"
    elif phase == "inwave":
        assert d["sd_outwave"] == 0, f"inwave policy ran SD outside waves: {d}"
        assert d["sd_inwave"] > 0, d
        assert phase_ar > 0, f"out-of-wave AR not counted under a phase reason: {d}"
    else:
        assert d["sd_inwave"] > 0 and d["sd_outwave"] > 0, d
    return d


def shapes(se, concurrency=(1, 3, 4, 5)):
    """Section 6 shape/tail: irregular prompts, outputs before/at/after the draft window, C>max-running."""
    n = max(se.steps, 1)
    outs = [1, max(n - 1, 1), n, n + 1, 2 * n, 2 * n + 1, 3 * n + 2]
    lens = [7, 41, 263, 1100, 2049, 37, 515]
    for c in concurrency:
        jobs = []
        for i in range(c):
            p = se.tok.filler(lens[(i + c) % len(lens)], seed=50 * c + i)
            jobs.append((se.c.complete, (p, outs[(i + c) % len(outs)]), {}))
        for (_, (_, m), _), r in zip(jobs, se.c.parallel(jobs)):
            exact_len(r, m)


def context_edge(se):
    """Prompt + output exactly at the context limit completes; the draft window cannot exceed it."""
    p = se.tok.filler(env.CTX - 40, seed=4242)
    pt = se.c.complete(p, 1)["usage"]["prompt_tokens"]
    m = env.CTX - pt
    r = se.c.complete(p, m)
    exact_len(r, m)
    over = se.c.post("/v1/completions", dict(model=se.c.model, prompt=p, max_tokens=m + 8, temperature=0,
                                              ignore_eos=True))
    record(se.name + ":context_over", {"status": over.status_code, "body": over.text[:300]})
    assert over.status_code in (200, 400, 413, 422), over.status_code
    if over.status_code == 200:
        assert over.json()["usage"]["completion_tokens"] <= m


def stop_and_eos(se):
    """Section 3: stop strings cut visible output; usage counts nothing after the stop."""
    r = se.c.complete(count_prompt(), 64, ignore_eos=False, stop=["\n20\n"])
    assert r["finish"] == "stop", r
    assert "20" not in r["text"].split(), r["text"]
    used = r["usage"]["completion_tokens"]
    assert used <= se.tok.n(r["text"]) + se.tok.n("\n20\n") + 1, (used, r["text"])
    r = se.c.chat("Reply with exactly the word: yes", 1200, ignore_eos=False)
    record(se.name + ":eos", {"finish": r["finish"], "usage": r["usage"]})
    assert r["finish"] in ("stop", "length")
    if r["finish"] == "stop":
        assert r["usage"]["completion_tokens"] < 1200


def stream_consistent(se):
    """SSE: visible text equals the non-streamed text of the same greedy request, usage once per token."""
    p = se.tok.filler(300, seed=77)
    fresh = se.c.complete(p, 40)  # both compared requests below then see the same prefix-cache state
    full = se.c.complete(p, 40)
    st = se.c.stream(p, 40)
    assert st["done"] and st["usage"]["completion_tokens"] == 40, st
    record(se.name + ":stream_vs_full", {"fresh_vs_hit_prefix": common_prefix(fresh["text"], full["text"]),
                                         "prefix": common_prefix(st["text"], full["text"]),
                                         "len": len(full["text"]), "equal": st["text"] == full["text"]})


def greedy_batch_drift(se):
    """Quantify greedy text drift between execution plans (section 6: report, do not gate)."""
    p = count_prompt(30)
    alone = se.c.complete(p, 64)["text"]
    others = [(se.c.complete, (se.tok.filler(900 + i, seed=300 + i), 32), {}) for i in range(3)]
    batched = se.c.parallel([(se.c.complete, (p, 64), {})] + others, stagger_s=0.3)[0]["text"]
    record(se.name + ":greedy_drift", {"prefix": common_prefix(alone, batched), "len": len(alone)})
    return alone


def cache_numbers(se):
    st = se.c.stats()
    obj = {"cache": se.c.cache_status(), "kv": st.get("kv"), "mamba": st.get("mamba")}
    return {k: v for k, v in view.flat(obj).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _idle_usage(se, wait_s=30):
    """Pages/slots held by requests (kv.used_pages, mamba.used_slots) once no request is running."""
    end = time.time() + wait_s
    while True:
        st = se.c.stats()  # kv/mamba are null before the first request
        u = {"kv": (st.get("kv") or {}).get("used_pages", 0), "mamba": (st.get("mamba") or {}).get("used_slots", 0)}
        if u == {"kv": 0, "mamba": 0} or time.time() > end:
            return u
        time.sleep(1)


def cancels(se):
    """Section 5: cancel while queued, during prefill and during decode; nothing leaks or blocks."""
    long_p = se.tok.filler(3000, seed=9001)
    for _ in range(3):  # warm: whatever the cache keeps for this prompt is now kept
        se.c.stream(long_p, 200, stop_after_s=0.3)
        se.c.stream(count_prompt(), 300, stop_after_chunks=3)
    exact_len(se.c.complete(count_prompt(), 32), 32)
    base = _idle_usage(se)
    for i in range(8):
        se.c.stream(long_p, 200, stop_after_s=0.3)          # during prefill
        se.c.stream(count_prompt(), 300, stop_after_chunks=3 + i)  # during decode / draft-verify
    # queued: 6 requests on 4 running slots, cancel the last two before they can start
    jobs = [(se.c.complete, (se.tok.filler(500, seed=600 + i), 64), {}) for i in range(4)]
    jobs += [(se.c.stream, (long_p, 64), {"stop_after_s": 0.1}) for _ in range(2)]
    res = se.c.parallel(jobs)
    for r in res[:4]:
        exact_len(r, 64)
    exact_len(se.c.complete(count_prompt(), 32), 32)
    after = _idle_usage(se)
    record(se.name + ":cancel_cache", {"base": base, "after": after})
    assert all(after[k] <= base[k] for k in base), f"usage grew across repeated cancels: {base} -> {after}"


def hot_prefix_and_groups(se):
    """Section 3/5: hot reuse hits within a cache_group, never across groups; SD still runs after a hit."""
    p = se.tok.filler(1500, seed=31337)
    first = se.c.complete(p, 24, cache_group="ga")
    b = se.c.stats()
    again = se.c.complete(p, 24, cache_group="ga")
    a = se.c.stats()
    other = se.c.complete(p, 24, cache_group="gb")
    hits = [x["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) for x in (first, again, other)]
    record(se.name + ":hot_prefix", {"cached": hits, "prefix": common_prefix(first["text"], again["text"]),
                                     "sd_rounds_on_hit": view.delta(b, a, "rounds")})
    assert hits[1] > 0, f"repeat in same cache_group did not hit: {hits}"
    assert hits[2] == 0, f"different cache_group hit another group's prefix: {hits}"
    return view.delta(b, a, "rounds")


def _rebuild(se, body):
    r = se.c.post("/v1/cache/rebuild", body, timeout=900)
    try:
        j = r.json()
    except ValueError:
        j = {"text": r.text}
    return r.status_code, j


def _ok(code, j):
    return code == 200 and "reject" not in view.text(j) and "error" not in str(j.get("status", "")).lower()


def geometry(se):
    st = se.c.cache_status()
    g = {}
    for k in ("num_pages", "moe_cache_size", "num_mamba_slots"):
        hits = {p: v for p, v in view.flat(st).items() if p.split(".")[-1] == k}
        if hits:
            g[k] = next(iter(hits.values()))
    return g


def maintenance(se, has_state, after=None):
    """Section 5: idle rebuild variants, busy rejection, illegal geometry rejection, real SD afterwards."""
    g0 = geometry(se)
    if not has_state:  # no recurrent-state pool: a slot count is not part of this model's geometry
        g0.pop("num_mamba_slots", None)
    assert "num_pages" in g0 and "moe_cache_size" in g0, f"cache status lacks geometry: {se.c.cache_status()}"
    log = {}
    # busy: a long generation is running
    jobs = [(se.c.complete, (count_prompt(), 400), {}), (lambda: (time.sleep(4), _rebuild(se, dict(g0)))[1], (), {})]
    gen, (code, j) = se.c.parallel(jobs)
    log["busy"] = (code, j)
    assert not _ok(code, j), f"rebuild accepted while busy: {code} {j}"
    exact_len(gen, 400)
    # illegal targets rejected before old resources are destroyed
    for bad in ({"num_pages": 0}, {"moe_cache_size": 10 ** 7}, {"num_pages": 10 ** 8}):
        code, j = _rebuild(se, {**g0, **bad})
        log[f"bad{bad}"] = (code, j)
        assert not _ok(code, j), f"illegal rebuild {bad} accepted: {code} {j}"
        assert all(geometry(se).get(k) == v for k, v in g0.items())
        exact_len(se.c.complete(count_prompt(), 16), 16)
    # legal idle rebuilds: same, KV-only, expert-only, state-only
    targets = [dict(g0), {**g0, "num_pages": int(g0["num_pages"] * 0.9)},
               {**g0, "num_pages": int(g0["num_pages"] * 0.9), "moe_cache_size": g0["moe_cache_size"] - 64}]
    if has_state:
        assert "num_mamba_slots" in g0, f"no recurrent-state geometry: {se.c.cache_status()}"
        targets.append({**targets[-1], "num_mamba_slots": g0["num_mamba_slots"] + 1})
    for t in targets:
        code, j = _rebuild(se, t)
        log[f"ok{t}"] = (code, j)
        assert _ok(code, j), f"legal idle rebuild {t} failed: {code} {j}"
        g = geometry(se)
        assert all(g.get(k) == v for k, v in t.items()), f"geometry after rebuild {g} != target {t}"
        exact_len(se.c.complete(count_prompt(), 16), 16)
    record(se.name + ":maintenance", log)
    b = se.c.stats()
    (after or (lambda: decode_sd(se, expect_sd=view.sd_on(b))))()
    a = se.c.stats()
    return view.delta(b, a, "graph_replays")


def greedy_probe(se):
    """Fixed greedy probes whose text other sessions compare against (reported, not gated)."""
    return {"count": se.c.complete(count_prompt(30), 64)["text"],
            "prose": se.c.complete(se.tok.filler(600, seed=5), 64)["text"]}


def kv_pressure(se, plen=3000, out=600):
    """Section 5: concurrent long contexts near KV capacity finish within budget, no deadlock."""
    b = se.c.stats()
    jobs = [(se.c.complete, (se.tok.filler(plen + 11 * i, seed=8100 + i), out), {}) for i in range(4)]
    for r in se.c.parallel(jobs, stagger_s=0.5):
        exact_len(r, out)
    a = se.c.stats()
    record(se.name + ":kv_pressure", {"rounds": view.delta(b, a, "rounds"),
                                      "ar_by_reason": {k: v - view.reasons(b).get(k, 0)
                                                       for k, v in view.reasons(a).items()
                                                       if isinstance(v, (int, float))}})


def _cached(r):
    return (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def _restores(se):
    return {k: v for k, v in cache_numbers(se).items()
            if any(w in k.lower() for w in ("restor", "host_reused", "h2d"))}


def _evict(se, salt, n=6, plen=2000):
    for i in range(n):
        se.c.complete(se.tok.filler(plen, seed=salt + i), 2)


def cold_restore(se):
    """Section 5: data pushed to the host tier is restored on reuse, counted only after restore, and
    real SD continues afterwards; concurrent waiters do not multiply the restore traffic."""
    p = se.tok.filler(2000, seed=7001)
    fresh = se.c.complete(p, 48)
    _evict(se, 7100)
    r0 = _restores(se)
    b = se.c.stats()
    again = se.c.complete(p, 48)
    a = se.c.stats()
    r1 = _restores(se)
    one = {k: r1[k] - r0.get(k, 0) for k in r1}
    assert _cached(again) > 0, f"no reuse after eviction to host: {again['usage']}"
    assert any(v > 0 for v in one.values()), f"host-tier counters unchanged, not a cold restore: {one}"
    if view.sd_on(a):
        assert view.delta(b, a, "rounds") > 0, "no real SD after a cold restore"
    other = se.c.complete(p, 8, cache_group="cold-other")
    assert _cached(other) == 0, f"another cache_group reused this prefix: {other['usage']}"
    _evict(se, 7200)
    r2 = _restores(se)
    res = se.c.parallel([(se.c.complete, (p + f"\nVariant {i}: tell me more.", 24), {}) for i in range(3)])
    r3 = _restores(se)
    three = {k: r3[k] - r2.get(k, 0) for k in r3}
    for r in res:
        exact_len(r, 24)
    record(se.name + ":cold_restore", {"cached": [_cached(again)] + [_cached(r) for r in res],
                                       "restore_one": one, "restore_three_waiters": three,
                                       "fresh_vs_restored_prefix": common_prefix(fresh["text"], again["text"]),
                                       "len": len(fresh["text"])})
    assert all(_cached(r) > 0 for r in res), [r["usage"] for r in res]
    for k, v in one.items():
        if v > 0 and k in three and "h2d" in k:  # bytes/batches moved, not per-user reused tokens
            assert three[k] < 2 * v, f"{k}: 3 waiters moved {three[k]} vs single restore {v}"


def multi_turn(se):
    """Section 6 prefix: multi-round continuation and a shared fork both reuse the earlier history."""
    t1 = se.tok.filler(1200, seed=7301) + "\nQuestion: summarize the notes."
    r1 = se.c.complete(t1, 48)
    pt1 = r1["usage"]["prompt_tokens"]
    hist = t1 + r1["text"]
    forks = se.c.parallel([(se.c.complete, (hist + q, 32), {}) for q in
                           ("\nQuestion: and then?", "\nQuestion: list three items.")])
    r3 = se.c.complete(hist + "\nQuestion: and then?" + forks[0]["text"] + "\nQuestion: finally?", 32)
    got = [_cached(x) for x in (*forks, r3)]
    record(se.name + ":multi_turn", {"pt1": pt1, "cached": got})
    assert all(g >= 0.8 * pt1 for g in got), (pt1, got)


def cancel_cold(se):
    """Section 5: cancel while waiting for a cold restore; service and later reuse continue."""
    p = se.tok.filler(2000, seed=7401)
    se.c.complete(p, 4)
    for rnd in range(4):
        _evict(se, 7500 + 10 * rnd)
        se.c.stream(p, 64, stop_after_s=0.05)
        se.c.stream(p, 64, stop_after_s=0.05)
        exact_len(se.c.complete(count_prompt(), 16), 16)
        time.sleep(2)
        nums = cache_numbers(se)
        used = {k: v for k, v in nums.items() if "host" in k.lower() and "used" in k.lower()}
        record(se.name + ":cancel_cold", {"round": rnd, "host_used": used})
    r = se.c.complete(p, 16)
    exact_len(r, 16)


def resources(se, sd):
    """Section 5 / round-2 contract: execution.resources carries both figures with SD on or off."""
    r = (se.c.stats().get("execution") or {}).get("resources") or {}
    for k in ("speculative_graph_reserved_bytes", "cpu_executor_pinned_io_bytes"):
        assert isinstance(r.get(k), int) and r[k] >= 0, f"execution.resources.{k} missing/invalid: {r}"
    if not sd:
        assert r["speculative_graph_reserved_bytes"] == 0, f"SD off but SD graph bytes reserved: {r}"
    return r


def _shapes(stats):
    return {(x["phase"], x["batch_size"], x["query_tokens"]): x["replays"]
            for x in (stats.get("cuda_graph") or {}).get("replay_shapes") or []}


def graph_ladder(se, max_bs):
    """Target-decode ladder = 1..min(max_bs, max running 4); each size up to it really replays a Graph,
    larger concurrency stays within the captured sizes (eager outside coverage)."""
    s = se.c.stats()
    got = view.get(s, "graph")["batch_sizes"] if max_bs else (view.get(s, "graph") or {}).get("batch_sizes")
    cover = min(max_bs, 4)
    want = list(range(1, cover + 1))
    assert (got or []) == want, f"target-decode Graph ladder {got}, expected {want}"
    seen = {}
    # layered admits one prompt per wave, so the c-th request joins decode only after c-1 waves:
    # outputs must outlast those waves for all c requests to decode together
    for c in range(1, 5):
        b = se.c.stats()
        res = se.c.parallel([(se.c.complete, (count_prompt(10 * i + 1), 240), {}) for i in range(c)])
        for r in res:
            exact_len(r, 240)
        a = se.c.stats()
        sb, sa = _shapes(b), _shapes(a)
        grew = {k: v - sb.get(k, 0) for k, v in sa.items() if v > sb.get(k, 0)}
        seen[c] = sorted({k[1] for k in grew})
        assert all(k[1] <= cover for k in grew), f"Graph replayed batch {grew} beyond coverage {cover}"
        if c <= cover:
            assert c in seen[c], f"{c} concurrent decodes never replayed a batch-{c} Graph: {grew}"
    record(se.name + ":graph_ladder", {"ladder": got, "replayed_batch_sizes": seen,
                                       "eager": (a.get("cuda_graph") or {}).get("speculative_eager")})
