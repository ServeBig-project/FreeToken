"""P4: fixed / adaptive / observe-only control, ragged batches, EOS/stop/max_tokens, AR fallback and
recovery, sampled-distribution check, Graph/eager, C16, self-SD regression.
Contract: sections 1 (control flags), 2 (sampling, outputs), 4 (stats), 5 (execution/batch/termination rows)."""

import time

from harness import dflash, qwen36, text
from p1 import REAL_ACCEPT_MIN, quality
from p2 import cancel_cycles, fresh, idle_window_clean, pollution_check
from workloads import (COIN_ALPHA, COIN_PROMPT, COPY_MIN_RATIO, acceptance, chi_square_two_sample, coin_category,
                       copy_prompt, copy_ratio, counter_consistency)

COIN_SAMPLES = 400
CLIP_ALLOWANCE = 2   # per request: rounds whose length is not a configured block may only come from tail clipping
INIT_ALLOWANCE = 4   # observe-only: one real initialisation per action (AR/2/4/8)

CONFIGS = {
    "nvfp4_n2": qwen36() + dflash(2),
    "nvfp4_n4": qwen36() + dflash(4),
    "nvfp4_n8_eager": qwen36(graph=0) + dflash(8),
    "nvfp4_adaptive": qwen36() + dflash(8, "--speculative-adaptive-cost"),
    "nvfp4_observe": qwen36() + dflash(8, "--speculative-adaptive-cost", "--dflash-adaptive-observe-only"),
    "nvfp4_c16": qwen36(tokens=131072, running=16, graph=16, gdn=4600000000, policy="layered-pipeline") + dflash(8),
    "nvfp4_self_sd": qwen36() + ["--speculative-num-steps", "4", "--speculative-draft-experts", "3"],
    "nvfp4_lp_inwave": qwen36(policy="layered-pipeline") + dflash(8, "--speculative-phase", "inwave"),
    "nvfp4_lp_all": qwen36(policy="layered-pipeline") + dflash(8, "--speculative-phase", "all"),
}


def steps(server):
    return int(server.flag("--speculative-num-steps", 0))


def shapes_delta(after, before):
    def index(stats):
        return {(s["phase"], s["batch_size"], s["query_tokens"], s["physical_query_tokens"]): s["replays"]
                for s in stats["cuda_graph"].get("replay_shapes", [])}
    a, b = index(after), index(before)
    return [{"phase": k[0], "batch_size": k[1], "query_tokens": k[2], "physical_query_tokens": k[3],
             "replays": n - b.get(k, 0)} for k, n in a.items() if n - b.get(k, 0) > 0]


def graph_ragged(server, c):
    n = steps(server)
    work = [(copy_prompt(110 + k, s), m) for k, (s, m) in enumerate(((30, 1), (60, 2), (90, 150), (120, 400)))]
    before = server.stats()
    resps = server.parallel([lambda w=w: server.complete(w[0][0], w[1], group=fresh()) for w in work])
    after = server.wait_idle()
    from workloads import spec_delta
    delta = spec_delta(after, before)
    shapes = shapes_delta(after, before)
    hist = delta["histogram"]
    c.check("ragged_outputs", [r["body"]["usage"]["completion_tokens"] for r in resps[:2]] == [1, 2]
            and min(copy_ratio(src, text(r)) for ((_, src), _), r in zip(work[2:], resps[2:])) >= COPY_MIN_RATIO)
    bad = counter_consistency(delta, server.outwave)
    c.check("ragged_counter_consistency", not bad, violations=bad)
    c.check("ragged_draft_lengths_within_config", all(x == 0 for x in hist[n + 1:]) and hist[n] > 0, histogram=hist)
    c.check("short_request_does_not_zero_others", hist[0] <= 2 * CLIP_ALLOWANCE, histogram=hist)
    graph = after["cuda_graph"]
    if int(server.flag("--cuda-graph-max-bs")) > 0:
        replays = {p: sum(s["replays"] for s in shapes if s["phase"] == p) for p in ("draft", "verify")}
        c.check("graph_used_for_draft_and_verify", replays["draft"] > 0 and replays["verify"] > 0, replays=replays)
        c.check("graph_tail_batches", len({s["batch_size"] for s in shapes if s["phase"] == "verify"}) >= 2,
                batch_sizes=sorted({s["batch_size"] for s in shapes}))
        c.check("graph_physical_ge_real", all(s["physical_query_tokens"] >= s["query_tokens"] for s in shapes))
        c.note("graph_padded_shapes", padded=[s for s in shapes if s["physical_query_tokens"] > s["query_tokens"]][:6])
    else:
        c.check("eager_when_graph_disabled", not graph.get("enabled") and not shapes, cuda_graph=graph)
    return {"delta": delta, "shapes": shapes}


def stop_eos_limits(server, c):
    prompt, source = copy_prompt(3, 40)
    exact = server.complete(prompt, 37)
    c.check("max_tokens_exact", exact["body"]["usage"]["completion_tokens"] == 37
            and exact["body"]["choices"][0]["finish_reason"] == "length", usage=exact["body"]["usage"])
    streamed = server.stream(prompt, 37)
    c.check("stream_matches_nonstream", streamed["done"] and streamed["text"] == text(exact))
    stop = "Note 3-12:"
    resp, delta = server.measured(lambda: server.complete(prompt, 1000, stop=[stop]))
    out = text(resp)
    cut = source.index(stop)
    c.check("stop_string_honoured", resp["body"]["choices"][0]["finish_reason"] == "stop" and stop not in out
            and copy_ratio(source[:cut], out) >= COPY_MIN_RATIO, tail=out[-80:])
    if steps(server):
        bad = counter_consistency(delta, server.outwave)
        c.check("stop_counter_consistency", not bad, violations=bad,
                delta=delta)
    eos = server.measured(lambda: server.post_chat("Reply with exactly the single word: yes", 1500))[0]
    body = eos["body"]
    c.check("eos_stops_generation", eos["status"] == 200 and body["choices"][0]["finish_reason"] == "stop"
            and body["usage"]["completion_tokens"] < 1500, usage=body.get("usage"),
            finish=body.get("choices", [{}])[0].get("finish_reason"))
    pollution_check(server, c, "after_stop_eos")
    return {"stop_delta": delta, "stop_text": out[-200:], "eos": body.get("choices", [{}])[0].get("message")}


def sampling_distribution(server, c):
    before = server.stats()
    counts = {}
    body = {"temperature": 1.0, "top_p": 1.0}
    for start in range(0, COIN_SAMPLES, 4):
        resps = server.parallel([lambda: server.complete(COIN_PROMPT, 14, group=fresh(), **body) for _ in range(4)])
        for r in resps:
            key = coin_category(text(r))
            counts[key] = counts.get(key, 0) + 1
    after = server.wait_idle()
    from workloads import spec_delta
    delta = spec_delta(after, before)
    if steps(server):
        c.check("sampled_sd_active", delta["draft_tokens"] > 0 and 0 < delta["accepted_draft_tokens"]
                < delta["draft_tokens"], delta=delta)
        bad = counter_consistency(delta, server.outwave)
        c.check("sampled_counter_consistency", not bad, violations=bad)
    return {"counts": counts, "delta": delta}


STATS_CONCEPTS = {  # contract section 4 items -> any key path containing one of these fragments
    "round_time": ["round_ms", "round_gpu_ms", "round_time"],
    "proposal_time": ["proposal"],
    "verify_time": ["dflash_verify", "verify_ms", "verify_gpu_ms"],
    "timing_scope": ["timing_scope"],
    "fixed_rounds": ["fixed_rounds", "dflash_fixed"],
    "resource_clipped_requests": ["clip"],
    "controller_cpu_time": ["cost_control_ms", "decision"],
    "cost_samples": ["cost_samples", "dflash_samples"],
    "dropped_samples": ["drop"],
    "initialisation": ["init"],
    "probes": ["probe"],
    "ar_fallback": ["cost_ar_requests", "fallback"],
}
TIMED = ("round_time", "proposal_time", "verify_time")  # must also be > 0 after drafting


def key_paths(value, prefix=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield prefix + k, v
            yield from key_paths(v, prefix + k + ".")


def positive(value):
    if isinstance(value, dict):
        return any(positive(v) for v in value.values())
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def stats_fields(server, c, concepts, label="stats_field"):
    paths = list(key_paths(server.stats()["speculative"]))
    for concept, fragments in concepts.items():
        found = [(p, v) for p, v in paths if any(f in p for f in fragments)]
        ok = bool(found) and (concept not in TIMED or any(positive(v) for _, v in found))
        c.check(f"{label}:{concept}", ok, found=[(p, v if not isinstance(v, dict) else "{...}") for p, v in found][:6])
    return [p for p, _ in paths]


def fixed_stats_and_api(server, c):
    """Section 4 in fixed mode: per-round timing is adaptive-only; clipped_requests counts requests whose
    draft was shortened (tail = own remaining output, capacity = resources). Token-ID prompts unsupported."""
    from workloads import spec_delta
    prompt, _ = copy_prompt(140, 40)
    before = server.stats()
    server.parallel([lambda: server.complete(prompt, 13, group=fresh()),  # tail shorter than one block
                     lambda: server.complete(prompt, 200, group=fresh())])
    after = server.wait_idle()
    spec0, spec1 = before["speculative"], after["speculative"]
    clipped = spec1.get("clipped_requests")
    c.check("clipped_requests_reported", isinstance(clipped, dict) and {"tail", "capacity"} <= set(clipped),
            clipped=clipped)
    if isinstance(clipped, dict):
        tail = clipped.get("tail", 0) - (spec0.get("clipped_requests") or {}).get("tail", 0)
        c.check("tail_clip_counted", tail > 0, tail_delta=tail, delta=spec_delta(after, before))
    timing = [p for p, _ in key_paths(spec1) if any(f in p for f in ("timing_scope", "proposal", "round_ms"))]
    c.note("fixed_mode_timing_fields", present=timing)
    ids = server.complete([1, 2, 3], 4)
    c.check("token_id_prompt_rejected", 400 <= ids["status"] < 500, status=ids["status"], body=str(ids["body"])[:200])


def adaptive_policy(server, c):
    from workloads import spec_delta
    low = {"temperature": 1.0, "top_p": 1.0}
    prompt = "Invent forty unusual fantasy words that do not exist in any language, separated by commas:"
    before = server.stats()
    for _ in range(3):
        server.parallel([lambda: server.complete(prompt, 200, group=fresh(), **low) for _ in range(4)])
    mid = server.wait_idle()
    work = [copy_prompt(120 + k, 60) for k in range(4)]
    resps = server.parallel([lambda w=w: server.complete(w[0], 600, group=fresh()) for w in work])
    after = server.wait_idle()
    a, b = spec_delta(mid, before), spec_delta(after, mid)
    blocks = {0, 2, 4, 8}
    for label, d in (("low", a), ("high", b)):
        odd = sum(n for i, n in enumerate(d["histogram"]) if i not in blocks)
        c.check(f"adaptive_whole_blocks:{label}", odd <= CLIP_ALLOWANCE * d["requests"], histogram=d["histogram"])
    c.check("adaptive_high_acceptance_drafts", acceptance(b) >= REAL_ACCEPT_MIN and sum(b["histogram"][1:]) > 0
            and min(copy_ratio(s, text(r)) for (_, s), r in zip(work, resps)) >= COPY_MIN_RATIO,
            acceptance=acceptance(b), histogram=b["histogram"])
    fell_back = a["histogram"][0] > 0 or a.get("cost_ar_requests", 0) > 0
    if fell_back:
        c.check("adaptive_recovers_after_ar", sum(b["histogram"][2:]) > 0, low=a, high=b)
    else:
        c.note("adaptive_ar_fallback_not_reached", low=a)
    paths = stats_fields(server, c, STATS_CONCEPTS)
    return {"low": a, "high": b, "stats_paths": paths}


def observe_only(server, c):
    from workloads import spec_delta
    before = server.stats()
    work = [copy_prompt(130 + k, 60) for k in range(4)]
    server.parallel([lambda w=w: server.complete(w[0], 400, group=fresh()) for w in work])
    server.parallel([lambda: server.complete("Invent forty unusual fantasy words:", 200, group=fresh(),
                                             temperature=1.0, top_p=1.0) for _ in range(4)])
    after = server.wait_idle()
    d = spec_delta(after, before)
    other = sum(n for i, n in enumerate(d["histogram"]) if i != 8)
    c.check("observe_executes_configured_length", other <= CLIP_ALLOWANCE * d["requests"] + INIT_ALLOWANCE,
            histogram=d["histogram"])
    paths = stats_fields(server, c, {**STATS_CONCEPTS, "suggested_vs_executed": ["suggest", "observe", "recommend"]})
    return {"delta": d, "stats_paths": paths}


def wave_overlap(server, c):
    """Long prompts arrive while other requests decode, so prefill waves overlap decoding.
    outwave: no SD beside a wave; inwave: SD only beside a wave; all: both (layered-pipeline only)."""
    from workloads import spec_delta
    phase = server.flag("--speculative-phase", "outwave")
    early = [copy_prompt(150 + k, 30) for k in range(2)]
    late = [copy_prompt(160 + k, 200) for k in range(2)]
    before = server.stats()

    def delayed(work):
        import time
        time.sleep(1.5)
        return server.complete(work[0], 200, group=fresh())
    calls = [lambda w=w: server.complete(w[0], 700, group=fresh()) for w in early]
    calls += [lambda w=w: delayed(w) for w in late]
    resps = server.parallel(calls)
    after = server.wait_idle()
    d = spec_delta(after, before)
    ratios = [copy_ratio(src, text(r)) for (_, src), r in zip(early + late, resps)]
    c.check("wave_overlap_outputs", min(ratios) >= COPY_MIN_RATIO, ratios=ratios)
    rounds = d.get("verify_rounds")
    c.check("wave_rounds_reported", isinstance(rounds, dict) and {"inwave", "outwave"} <= set(rounds), delta=d)
    if isinstance(rounds, dict) and server.flag("--batching-policy") == "layered-pipeline":
        expect = {"outwave": rounds.get("inwave") == 0 and rounds.get("outwave", 0) > 0,
                  "inwave": rounds.get("outwave") == 0 and rounds.get("inwave", 0) > 0,
                  "all": rounds.get("inwave", 0) > 0 and rounds.get("outwave", 0) > 0}[phase]
        c.check(f"wave_phase_respected:{phase}", expect, verify_rounds=rounds,
                fallback=d.get("fallback_requests"), histogram=d["histogram"])
    return d


def c16_matrix(server, c):
    out = []
    for wave in range(2):
        sizes = [20 + 12 * k for k in range(16)] if wave == 0 else [210, 210, 210] + [25 + 9 * k for k in range(13)]
        work = [(copy_prompt(200 + 16 * wave + k, s), 32 + 25 * k) for k, s in enumerate(sizes)]
        before = server.stats()
        resps = server.parallel([lambda w=w: server.complete(w[0][0], w[1], group=fresh()) for w in work])
        after = server.wait_idle()
        from workloads import spec_delta
        delta, shapes = spec_delta(after, before), shapes_delta(after, before)
        ratios = [copy_ratio(src, text(r)) for ((_, src), _), r in zip(work, resps)]
        c.check(f"c16_complete:{wave}", all(r["status"] == 200 for r in resps) and min(ratios) >= COPY_MIN_RATIO,
                ratios=[round(x, 3) for x in ratios])
        c.check(f"c16_counter_consistency:{wave}", not counter_consistency(delta, server.outwave), violations=counter_consistency(delta, server.outwave))
        sizes_seen = sorted({s["batch_size"] for s in shapes if s["phase"] == "verify"})
        c.check(f"c16_graph_batches:{wave}", max(sizes_seen, default=0) >= 9 and len(sizes_seen) >= 3
                and all(s["physical_query_tokens"] >= s["query_tokens"] for s in shapes), batch_sizes=sizes_seen)
        c.check(f"c16_n8_drafts:{wave}", delta["histogram"][8] > 0, histogram=delta["histogram"])
        out.append({"delta": delta, "shapes": shapes, "ratios": ratios})
    idle_window_clean(server, c, "c16")
    return out


def self_sd_regression(server, c):
    spec = server.stats()["speculative"]
    c.check("self_sd_not_dflash", spec.get("drafter") != "dflash"
            and not (server.status()["geometry"].get("dflash") or {}).get("active"), drafter=spec.get("drafter"))
    return graph_ragged(server, c)


PLAN = {
    "nvfp4_ar": [stop_eos_limits, sampling_distribution],
    "nvfp4_n8": [graph_ragged, stop_eos_limits, sampling_distribution, wave_overlap, fixed_stats_and_api],
    "nvfp4_n2": [quality, graph_ragged, fixed_stats_and_api],
    "nvfp4_n4": [quality, graph_ragged, fixed_stats_and_api],
    "nvfp4_n8_eager": [quality, graph_ragged],
    "nvfp4_n8_noreplay": [graph_ragged],
    "nvfp4_adaptive": [quality, adaptive_policy],
    "nvfp4_observe": [quality, observe_only],
    "nvfp4_c16": [c16_matrix],
    "nvfp4_lp_inwave": [wave_overlap],  # smoke
    "nvfp4_lp_all": [wave_overlap],     # smoke
    "nvfp4_self_sd": [quality],  # smoke


}


def coin_distribution(sessions, c):
    ref = sessions.get("nvfp4_ar", {}).get("artifacts", {}).get("sampling_distribution")
    sd = sessions.get("nvfp4_n8", {}).get("artifacts", {}).get("sampling_distribution")
    if ref and sd:
        stat, dof, p = chi_square_two_sample(ref["counts"], sd["counts"])
        c.check("sampled_distribution_matches_ar", p >= COIN_ALPHA, chi2=stat, dof=dof, p=p,
                ar=ref["counts"], dflash=sd["counts"])


def graph_eager_report(sessions, c):
    from harness import common_prefix
    g = sessions.get("nvfp4_n8", {}).get("artifacts", {}).get("quality")
    e = sessions.get("nvfp4_n8_eager", {}).get("artifacts", {}).get("quality")
    if g and e:
        c.note("graph_vs_eager_greedy", code=[common_prefix(x["text"], y["text"]) for x, y in zip(g["code"], e["code"])],
               copy=common_prefix(g["copy"], e["copy"]), copy_len=len(g["copy"]),
               acceptance=(acceptance(g["copy_delta"]), acceptance(e["copy_delta"])))


COMPARE = [coin_distribution, graph_eager_report]
