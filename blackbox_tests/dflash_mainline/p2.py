"""P2: exact window slot allocation and reuse — wrap-around, rejection/cancel, shared prefixes,
rebuild conservation. Contract: sections 2 (history purity, sharing), 3 (status, rebuild), 5 (shared/lifecycle)."""

import threading
import time
import uuid

from harness import cached, document, text
from p1 import REAL_ACCEPT_MIN, TINY
from workloads import COPY_MIN_RATIO, NEEDLE, acceptance, copy_prompt, copy_ratio, needle_prompt

REUSE_ACCEPT_FRACTION = 0.6  # reused-prefix acceptance must keep this share of the cold-prefix acceptance
REUSE_MIN_SHARE = 0.9        # cached tokens / prompt tokens for a prefix that is certainly cached


def fresh():
    return uuid.uuid4().hex


def note_question(seed, i):
    value = (i * 37 + seed * 11) % 997
    return f"\n\nQuestion: What value is recorded in Note {seed}-{i}?\nAnswer: Note {seed}-{i} records value", str(value)


def idle_window_clean(server, c, label):
    server.wait_idle()
    w = server.status()["prefix_cache"]["window_slots"]
    c.check(f"idle_window_slots_released:{label}", w["request_owned"] == 0 and w["copy_inflight"] == 0
            and w["tree_locked"] == 0 and w["free"] + w["tree_evictable"] == w["total"], window_slots=w)
    return w


def probe(server):
    prompt, source = copy_prompt(21, 60)
    resp, delta = server.measured(lambda: server.complete(prompt, 300, group=fresh()))
    return {"text": text(resp), "accepted": delta["accepted_draft_tokens"], "ratio": copy_ratio(source, text(resp))}


def determinism_baseline(server, c):
    first, second = probe(server), probe(server)
    c.note("determinism_baseline", same_text=first["text"] == second["text"],
           accepted=(first["accepted"], second["accepted"]))
    server.baseline = first if first["text"] == second["text"] else None
    return {"first": first, "second": second}


def pollution_check(server, c, label):
    if getattr(server, "baseline", None) is None:
        c.note(f"pollution_probe_inconclusive:{label}", reason="baseline not deterministic or not run")
        return None
    now = probe(server)
    c.check(f"no_pollution:{label}", now["text"] == server.baseline["text"],
            accepted=(server.baseline["accepted"], now["accepted"]))
    return now


def window_wrap_cycles(server, c):
    tiny = server.flag("--speculative-draft-model-path") == str(TINY)
    out = []
    for cycle in range(3):
        group = fresh()
        work = [(copy_prompt(30 + 4 * cycle + k, s), m) for k, (s, m) in
                enumerate(((165, 300), (190, 120), (103, 500), (230, 64)))]
        resps = server.parallel([lambda w=w: server.complete(w[0][0], w[1], group=group) for w in work])
        ratios = [copy_ratio(src, text(r)) for ((_, src), _), r in zip(work, resps)]
        c.check(f"wrap_cycle_fidelity:{cycle}", min(ratios) >= COPY_MIN_RATIO, ratios=ratios)
        out.append({"ratios": ratios, "window_slots": idle_window_clean(server, c, f"wrap{cycle}")})
    if not tiny:
        out.append(pollution_check(server, c, "after_wrap"))
    return out


def reject_no_pollution(server, c):
    """Continue a sampled-free copy from the reused prefix; drafting must not lose or mix history."""
    group = fresh()
    prompt, source = copy_prompt(40, 120)
    first = server.complete(prompt, 400, group=group)
    follow = prompt + text(first)
    reused, d_reused = server.measured(lambda: server.complete(follow, 400, group=group))
    cold, d_cold = server.measured(lambda: server.complete(follow, 400, group=fresh()))
    share = cached(reused) / first["body"]["usage"]["prompt_tokens"]
    c.check("continuation_reuses_prefix", share >= REUSE_MIN_SHARE, cached=cached(reused),
            first_prompt_tokens=first["body"]["usage"]["prompt_tokens"])
    c.check("reused_prefix_keeps_draft_history",
            acceptance(d_reused) >= REUSE_ACCEPT_FRACTION * acceptance(d_cold),
            reused=acceptance(d_reused), cold=acceptance(d_cold))
    c.note("reused_vs_cold_text_equal", equal=text(reused) == text(cold))
    return {"reused": text(reused), "cold": text(cold), "d_reused": d_reused, "d_cold": d_cold}


def cancel_cycles(server, c):
    tiny = server.flag("--speculative-draft-model-path") == str(TINY)
    out = []
    for cycle in range(2):
        group = fresh()
        work = [copy_prompt(50 + 4 * cycle + k, 80 + 30 * k) for k in range(4)]
        calls = [lambda w=work[k], n=n: server.stream(w[0], 600, group=group, cancel_after=n)
                 for k, n in enumerate((3, 15, 40))]
        calls.append(lambda: server.complete(work[3][0], 300, group=group))
        results = server.parallel(calls)
        ratio = copy_ratio(work[3][1], text(results[3]))
        c.check(f"survivor_after_cancels:{cycle}", ratio >= COPY_MIN_RATIO, ratio=ratio)
        time.sleep(2)
        out.append(idle_window_clean(server, c, f"cancel{cycle}"))
    if not tiny:
        out.append(pollution_check(server, c, "after_cancel"))
    return out


def shared_prefix_forks(server, c):
    seed, base = 9, document(9, 170)
    group, out = fresh(), {}
    server.complete(base, 1, group=group)  # prompt end at the fork boundary: a reuse point exists there
    forks = [(note_question(seed, i), m) for i, m in ((17, 8), (60, 64), (101, 200), (150, 400))]
    resps = server.parallel([lambda f=f: server.complete(base + f[0][0], f[1], group=group) for f in forks])
    answers = [text(r) for r in resps]
    shares = [cached(r) / r["body"]["usage"]["prompt_tokens"] for r in resps]
    c.check("forks_correct", all(f[0][1] in a[:24] for f, a in zip(forks, answers)), answers=[a[:40] for a in answers],
            expected=[f[0][1] for f in forks])
    c.check("forks_reuse_shared_prefix", min(shares) >= REUSE_MIN_SHARE, shares=shares)
    other = server.complete(base + forks[0][0][0], 8, group=fresh())
    c.check("cache_group_isolation", cached(other) == 0 and forks[0][0][1] in text(other)[:24],
            cached=cached(other), text=text(other))
    details = (other["body"].get("usage") or {}).get("prompt_tokens_details") or {}
    c.check("zero_hit_omits_cached_tokens", "cached_tokens" not in details, usage=other["body"].get("usage"))
    q, want = note_question(seed, 77)
    revisit_prompt = base + forks[3][0][0] + answers[3] + q
    revisit = server.complete(revisit_prompt, 8, group=group)
    c.check("old_branch_revisit", cached(revisit) >= REUSE_MIN_SHARE * resps[3]["body"]["usage"]["prompt_tokens"]
            and want in text(revisit)[:24], cached=cached(revisit), text=text(revisit),
            old_prompt_tokens=resps[3]["body"]["usage"]["prompt_tokens"])
    out.update(answers=answers, shares=shares, revisit=text(revisit))
    if server.flag("--speculative-draft-model-path"):
        out["window_slots"] = idle_window_clean(server, c, "forks")
    return out


def _rebuilt(server, c, label, body, check_geometry):
    before = server.status()["geometry"]
    resp = server.rebuild(body, timeout=900)
    after = server.status()["geometry"]
    fixed = ("weight_bytes", "workspace_bytes", "window_slots", "window_context_bytes")  # metadata may follow KV size
    c.check(f"rebuild_ok:{label}", resp["status"] == 200 and check_geometry(after), response=resp["body"],
            geometry={k: after[k] for k in ("num_pages", "moe_cache_size", "num_mamba_slots")})
    d0, d1 = before["dflash"], after["dflash"]
    c.check(f"rebuild_keeps_dflash_budget:{label}", d1["active"] and all(d0[k] == d1[k] for k in fixed)
            and d1["reserved_bytes"] == d1["weight_bytes"] + d1["context_bytes"] + d1["metadata_bytes"]
            + d1["workspace_bytes"], before={k: d0[k] for k in fixed}, after={k: d1[k] for k in fixed})
    tokens = after["num_pages"] * after["page_size"]
    if d1["attention_window"] == 0:
        c.check(f"rebuild_full_history_tracks_kv:{label}", tokens * d1["full_token_bytes"] <= d1["full_context_bytes"]
                <= (tokens + 64) * d1["full_token_bytes"], full=d1["full_context_bytes"], tokens=tokens)
    w = server.status()["prefix_cache"]["window_slots"]
    # A KV/state rebuild clears cached prefixes; an expert-only rebuild may keep them as evictable.
    c.check(f"rebuild_window_free:{label}", w["free"] + w["tree_evictable"] == w["total"] == d1["window_slots"]
            and w["tree_locked"] == w["request_owned"] == 0, window_slots=w)
    group = fresh()
    prompt, source = copy_prompt(60, 50)
    server.complete(prompt, 16, group=group)
    resp2, delta = server.measured(lambda: server.complete(prompt, 300, group=group))
    c.check(f"rebuild_then_reuse_and_draft:{label}", cached(resp2) > 0 and copy_ratio(source, text(resp2))
            >= COPY_MIN_RATIO and acceptance(delta) >= REAL_ACCEPT_MIN,
            cached=cached(resp2), acceptance=acceptance(delta))
    return {"before": before, "after": after, "response": resp["body"]}


def rebuild_conservation(server, c):
    g0 = server.status()["geometry"]
    pages, slots, moe = g0["num_pages"], g0["num_mamba_slots"], g0["moe_cache_size"]
    out = {"kv": _rebuilt(server, c, "kv", {"num_pages": pages - 4096}, lambda g: g["num_pages"] == pages - 4096)}
    out["kv_back"] = _rebuilt(server, c, "kv_back", {"num_pages": pages}, lambda g: g["num_pages"] == pages)
    if slots:
        out["state"] = _rebuilt(server, c, "state", {"num_mamba_slots": slots - 4},
                                lambda g: g["num_mamba_slots"] == slots - 4)
        out["state_back"] = _rebuilt(server, c, "state_back", {"num_mamba_slots": slots},
                                     lambda g: g["num_mamba_slots"] == slots)
    out["moe"] = _rebuilt(server, c, "moe", {"moe_cache_size": moe - 256}, lambda g: g["moe_cache_size"] == moe - 256)
    out["moe_back"] = _rebuilt(server, c, "moe_back", {"moe_cache_size": moe}, lambda g: g["moe_cache_size"] == moe)
    prompt, source = copy_prompt(61, 60)
    holder = {}
    worker = threading.Thread(target=lambda: holder.update(r=server.complete(prompt, 500, group=fresh())))
    worker.start()
    time.sleep(3)
    busy = server.rebuild({"num_pages": pages - 4096})
    worker.join()
    c.check("busy_rebuild_rejected", busy["status"] == 503 and busy["body"].get("status") == "busy", response=busy)
    c.check("busy_rebuild_keeps_resources_and_request", server.status()["geometry"]["num_pages"] == pages
            and copy_ratio(source, text(holder["r"])) >= COPY_MIN_RATIO)
    answer = text(server.complete(needle_prompt(), 12))
    c.check("serves_after_busy_rebuild", NEEDLE in answer, text=answer)
    out["busy"] = busy["body"]
    return out


PLAN = {
    "nvfp4_ar": [shared_prefix_forks],
    "nvfp4_n8": [determinism_baseline, window_wrap_cycles, reject_no_pollution, cancel_cycles,
                 shared_prefix_forks, rebuild_conservation],
    "nvfp4_n8_noreplay": [rebuild_conservation],

}
