"""Load scenarios shared by the per-configuration service modules (contract sections 3-5).

Each scenario drives the public generation API, asserts what every configuration must satisfy
(completion, no repeated or missing output, usage, held <= budget) and returns the observations a
caller gates differently per configuration (restore versus recompute, concurrency bounds).
"""
import time

import pytest

from service_common import (GIB, Watch, assert_enum, assert_length, cached_tokens, components, counter_delta, dump,
                            enum_prompt, graph_replays, overlap, record, run_streams, sd_num,
                            start_streams, tok, wait_streams)

PREAMBLE = ("The following is one long list of consecutive item identifiers; every identifier is "
            "the word item followed by a number, and each number is one more than the previous one. "
            ) * 6
KV_BYTES_PER_TOKEN = 10 * 2 * 2 * 256 * 2  # 10 full-attention layers, 2 KV heads of 256, K and V, bf16


def concurrency(svc, cap=6):
    return min(cap, svc.rt()["max_running_requests"])


def _check_enum_streams(streams, starts, out):
    for s, start in zip(streams, starts):
        assert s.error is None and s.done and s.finish == "length", s.summary()
        assert s.usage["completion_tokens"] == out, s.usage
        assert_enum(s.text, start, out)


def pause_round(svc, salt, out=1200, items=40, preamble="", probe_busy=True):
    """`concurrency` enumeration requests of `out` tokens at once, then nothing else until they
    end: every request finishes with exactly `out` tokens, its sequence has no gap or repeat, usage
    counts only the real prompt and output, and maintenance during the round answers busy."""
    k = concurrency(svc)
    starts = [salt + 1000 * i for i in range(k)]
    prompts = [enum_prompt(s, items, preamble) for s in starts]
    streams = [svc.c.stream(p, out, ignore_eos=True) for p in prompts]
    rt0 = svc.rt()
    busy = None
    with Watch(svc.c) as w:
        ex, futures = start_streams(streams)
        end = time.monotonic() + 1800
        while not all(f.done() for f in futures) and time.monotonic() < end:
            if probe_busy and busy is None and svc.rt()["paused"] > rt0["paused"]:
                busy = svc.c.rebuild({"runtime_cache_gib": rt0["budget_bytes"] / GIB}, timeout=1)
            time.sleep(0.5)
        wait_streams(ex, futures, 60)
    delta = counter_delta(rt0, svc.rt())
    record(f"{svc.name}:pause_round", k=k, out=out, delta=delta, busy=busy, watch=w.report(),
           streams=[s.summary() for s in streams])
    dump(f"{svc.name}_pause_texts", [s.text for s in streams])
    assert not w.violations, w.report()
    _check_enum_streams(streams, [s + items for s in starts], out)
    prompt_tokens = {s.usage["prompt_tokens"] for s in streams}
    assert len(prompt_tokens) == 1, prompt_tokens  # same shape of prompt, same count, paused or not
    assert abs(next(iter(prompt_tokens)) - tok().n(prompts[0])) <= 1, (prompt_tokens, tok().n(prompts[0]))
    if not preamble:
        assert all(cached_tokens(s.usage) == 0 for s in streams), [s.usage for s in streams]
    if busy is not None:
        assert busy[1].get("status") == "busy", busy  # HTTP 503 or 409 by the published configuration
    svc.c.wait_idle()
    return dict(delta=delta, busy=busy, streams=streams, watch=w)


def cancel_round(svc, salt, out=1200, items=40, preamble=PREAMBLE):
    """Same load with a shared prefix; once a pause is observed the newest requests (two, or
    one when only two run) are cancelled. The others finish intact, the cancelled ones are
    reclaimed, service continues."""
    k = concurrency(svc)
    if k < 2:
        pytest.skip(f"effective concurrency {k} leaves no request to cancel beside a survivor")
    n = min(2, k - 1)
    starts = [salt + 1000 * i for i in range(k)]
    streams = [svc.c.stream(enum_prompt(s, items, preamble), out, ignore_eos=True) for s in starts]
    rt0 = svc.rt()
    ex, futures = start_streams(streams, stagger_s=0.3)
    cancelled_at = None
    end = time.monotonic() + 1800
    while cancelled_at is None and time.monotonic() < end and not all(f.done() for f in futures):
        if svc.rt()["paused"] > rt0["paused"]:
            for s in streams[-n:]:
                s.cancel()
            cancelled_at = time.monotonic()
        time.sleep(0.2)
    wait_streams(ex, futures, 1800)
    delta = counter_delta(rt0, svc.rt())
    record(f"{svc.name}:cancel_round", k=k, delta=delta, cancelled=cancelled_at is not None,
           streams=[s.summary() for s in streams])
    if cancelled_at is None:
        pytest.skip("no pause was observed in this round, so cancel-during-pause was not exercised")
    _check_enum_streams(streams[:-n], [s + items for s in starts[:-n]], out)
    for s in streams[-n:]:
        assert s.cancelled and not s.done, s.summary()
    svc.c.wait_idle(120)  # the cancelled requests are released as well
    assert_length(svc.c.complete(enum_prompt(salt + 777, 12), 32), 32)
    return delta


def early_stop_round(svc, salt, min_overlap, max_tokens=8000, items=12, run=30):
    """`concurrency` requests with a large max_tokens that stop after ~`run` items: admission
    must not reserve the full output limit, so they run together and nothing is paused."""
    k = concurrency(svc)
    starts = [salt + 1000 * i for i in range(k)]
    streams = [svc.c.stream(enum_prompt(s, items), max_tokens, stop=[f"item{s + items + run}"])
               for s in starts]
    rt0 = svc.rt()
    run_streams(streams, 900)
    delta = counter_delta(rt0, svc.rt())
    seen = overlap(streams)
    record(f"{svc.name}:early_stop", k=k, overlap=seen, delta=delta, streams=[s.summary() for s in streams])
    for s, start in zip(streams, starts):
        assert s.error is None and s.done and s.finish == "stop", s.summary()
        assert s.usage["completion_tokens"] < 20 * run, s.usage
        assert_enum(s.text, start + items, 0)
    assert seen >= min(k, min_overlap), f"only {seen} of {k} early-stop requests overlapped"
    assert delta["paused"] == 0, delta
    svc.c.wait_idle()
    return seen


def over_context(svc, salt):
    """A prompt beyond the published single-request cap ends with the public error, streaming or
    not, and the service keeps serving; a prompt that fits alone but whose output limit does not
    follows the existing length semantics and is never a hang or an empty success."""
    ctx = svc.rt()["context_tokens"]
    t = tok()
    big = t.filler(ctx + 300, seed=salt)
    assert t.n(big) > ctx
    code, j = svc.c.generate(big, 16, ignore_eos=True, timeout=300)
    assert 400 <= code < 500 and (j.get("error") or {}).get("code") == "context_length_exceeded", (code, j)
    assert j["error"]["message"].startswith("prompt is too long"), j  # the existing length error, before admission
    s = svc.c.stream(big, 16, ignore_eos=True)
    run_streams([s], 300)
    assert s.error and s.error.get("code") == "context_length_exceeded" and not s.text, s.summary()
    assert svc.c.status()["state"] == "serving"
    svc.c.wait_idle(60)
    near = t.filler(ctx - 100, seed=salt + 1)
    code, j = svc.c.generate(near, 400, ignore_eos=True, timeout=900)
    if code == 200:
        u = j["usage"]
        assert 0 < u["completion_tokens"] <= ctx - u["prompt_tokens"], u
        assert j["choices"][0]["finish_reason"] == "length", j["choices"][0]
    else:
        assert 400 <= code < 500 and (j.get("error") or {}).get("code") == "context_length_exceeded", (code, j)
    record(f"{svc.name}:over_context", ctx=ctx, big_tokens=t.n(big), near=(code, j.get("usage") or j))
    svc.c.wait_idle(60)
    assert_length(svc.c.complete(enum_prompt(salt, 12), 16), 16)


def _short_phase(svc, k, salt, out=160):
    starts = [salt + 1000 * i for i in range(k)]
    streams = [svc.c.stream(enum_prompt(s, 12), out, ignore_eos=True) for s in starts]
    with Watch(svc.c) as w:
        run_streams(streams, 900)
    _check_enum_streams(streams, [s + 12 for s in starts], out)
    svc.c.wait_idle()
    return w


def _long_phase(svc, n, plen, out=48):
    t = tok()
    streams = [svc.c.stream(t.filler(plen, seed=5000 + i) + "\nSummary:", out, ignore_eos=True)
               for i in range(n)]
    with Watch(svc.c) as w:
        run_streams(streams, 1800)
    for s in streams:
        assert s.error is None and s.done and s.finish == "length", s.summary()
        assert s.usage["completion_tokens"] == out, s.usage
    svc.c.wait_idle()
    return w


def short_long_short(svc, salt):
    """Many short requests, then a few long prompts whose KV alone needs more than the budget
    minus the short phase's state holdings, then short again: the component holdings move in
    opposite directions, both peaks could not coexist under any fixed split, nothing exceeds the
    budget, the expert capacity is untouched and no rebuild happens (sections 1, 4, 5)."""
    k = concurrency(svc)
    rt0, g0 = svc.rt(), svc.c.geometry()
    rebuild0 = svc.c.status().get("last_rebuild")
    plen = min(7000, rt0["context_tokens"] - 200)
    n = min(6, max(3, -(-21000 // plen)))
    a = _short_phase(svc, k, salt)
    b = _long_phase(svc, n, plen)
    c = _short_phase(svc, k, salt + 50000)
    budget = rt0["budget_bytes"]
    names = set(a.max_component) | set(b.max_component) | set(c.max_component)
    assert names, "runtime.components is empty"

    def peak(w, name):
        return max((smp.get(name, 0) for smp in w.series), default=0)

    # identified by behaviour (published names: kv grows, gdn_state gives way); compared at one instant
    x = max(names, key=lambda n: peak(b, n) - peak(a, n))
    y = max(names - {x}, key=lambda n: peak(a, n))
    at = max(b.series, key=lambda smp: smp.get(x, 0))  # the long-phase sample where x holds most
    record(f"{svc.name}:short_long_short", k=k, n=n, plen=plen, budget=budget, grew=x, gave_way=y,
           short_peak={n_: peak(a, n_) for n_ in names}, at_long_peak=at,
           short_again_peak={n_: peak(c, n_) for n_ in names},
           phases={p: w.report() for p, w in (("short", a), ("long", b), ("short_again", c))},
           geometry_after=svc.c.geometry())
    for w in (a, b, c):
        assert not w.violations, w.report()
    assert svc.c.geometry()["moe_cache_size"] == g0["moe_cache_size"] == 2048
    assert svc.c.status().get("last_rebuild") == rebuild0
    assert at.get(x, 0) + peak(a, y) > budget, (
        f"{x} at its long-phase peak {at.get(x, 0)} plus {y} at its short-phase peak {peak(a, y)} fit "
        f"the {budget} budget together, so this load never needed capacity to move")
    assert at.get(y, 0) < peak(a, y), f"{y} kept {at.get(y, 0)} while {x} peaked: {at}"
    assert peak(c, y) > at.get(y, 0), f"{y} did not regain capacity after the long phase"
    grow = {n_: at.get(n_, 0) - peak(a, n_) for n_ in names}
    return grow


def combo_round(svc, sd, graph, salt, out=300):
    """One function-combination check (contract section 6): concurrent enumeration requests
    complete intact under the budget, and the SD and Graph counters move only when that path is
    configured, so neither is replaced by plain AR or eager execution."""
    k = concurrency(svc, cap=4)
    starts = [salt + 1000 * i for i in range(k)]
    streams = [svc.c.stream(enum_prompt(s, 40), out, ignore_eos=True) for s in starts]
    before = svc.c.stats()
    with Watch(svc.c) as w:
        run_streams(streams, 1200)
    after = svc.c.stats()
    rounds = sd_num(after, "rounds") - sd_num(before, "rounds") if sd else None
    replays = graph_replays(after) - graph_replays(before)
    record(f"{svc.name}:combo", k=k, sd_rounds=rounds, graph_replays=replays, watch=w.report(),
           streams=[s.summary() for s in streams])
    assert not w.violations, w.report()
    _check_enum_streams(streams, [s + 40 for s in starts], out)
    if sd:
        assert rounds > 0, "SD configured but no verify round ran"
    assert (replays > 0) == graph, f"Graph {'on' if graph else 'off'} but replay counter moved by {replays}"
    svc.c.wait_idle()


def ready_combo(svc, batching, sd, graph, draft=None):
    """The requested combination is the effective one (section 6): batching policy, SD, Graph,
    and the drafter history component (`draft_kv` for full storage, `draft_*` when compact)."""
    stats, rt = svc.c.stats(), svc.rt()
    eff = (stats.get("execution") or {}).get("effective") or {}
    names = set(components(rt))
    record(f"{svc.name}:ready", effective=eff, cuda_graph=stats.get("cuda_graph"), components=sorted(names),
           runtime={k: v for k, v in rt.items() if k != "components"})
    assert eff.get("batching_policy") == batching, eff
    assert bool((stats.get("speculative") or {}).get("enabled")) == sd, stats.get("speculative")
    assert bool(stats["cuda_graph"]["enabled"]) == graph, stats["cuda_graph"]
    assert {"kv", "gdn_state", "gdn_conv"} <= names, names
    if draft:
        assert draft in names, names
    assert rt["held_bytes"] <= rt["budget_bytes"], rt


def burst(svc, n, prompt_tokens, out, seed, **kw):
    """`n` requests of about `prompt_tokens` input and exactly `out` output tokens arrive at
    once (ignore_eos); all must finish with their full output under the budget."""
    t = tok()
    streams = [svc.c.stream(t.filler(prompt_tokens, seed=seed + i) + "\nSummary:", out, ignore_eos=True, **kw)
               for i in range(n)]
    with Watch(svc.c) as w:
        run_streams(streams, 1800)
    record(f"{svc.name}:burst", n=n, prompt_tokens=prompt_tokens, out=out, kw=kw, overlap=overlap(streams),
           watch=w.report(), streams=[s.summary() for s in streams])
    assert not w.violations, w.report()
    for s in streams:
        assert s.error is None and s.done and s.finish == "length", s.summary()
        assert s.usage["completion_tokens"] == out, s.usage
    svc.c.wait_idle()
    assert svc.c.get("/health").get("status") == "ok"
    return streams


def shared_stats_contract(svc):
    """Shared mode reports runtime memory in prefix_cache.runtime, not as a page total in
    /v1/stats (as for GDN slots and windows)."""
    kv = svc.c.stats().get("kv") or {}
    record(f"{svc.name}:stats_kv", kv=kv)
    assert not kv.get("total_pages"), f"/v1/stats still reports a KV page total in shared mode: {kv}"
