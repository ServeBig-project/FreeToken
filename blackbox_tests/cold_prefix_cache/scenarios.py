"""Workloads built only from public requests; each returns observations and appends failures."""

import http.client
import json
import time

from harness import cached, common_prefix_len, delta, document, get, post, prompt_tokens, record, text

COUNTERS = ("h2d_bytes", "d2h_bytes", "h2d_batches", "d2h_batches", "checkpoint_created",
            "checkpoint_deduplicated", "checkpoint_pruned", "checkpoint_evicted", "gpu_reused_tokens",
            "host_reused_tokens", "recomputed_tokens", "gpu_checkpoint_peak", "host_checkpoint_peak")


class Checks:
    """Collects soft failures so one launch can report every broken condition."""

    def __init__(self, label, strict_text=True):
        # strict_text=False for modes documented as not run-to-run deterministic (layered, DSV4 concurrency)
        self.label, self.failures, self.notes, self.strict_text = label, [], [], strict_text

    def expect(self, ok, name, **detail):
        record(f"{self.label}/{name}", ok=bool(ok), **detail)
        if not ok:
            self.failures.append({"check": name, **detail})
        return ok

    def note(self, name, **detail):
        record(f"{self.label}/{name}", ok=None, **detail)
        self.notes.append({"check": name, **detail})


def invariants(c, pc, where, prev=None, gdn=True):
    """Section 3 relations that must hold in every snapshot."""
    c.expect(pc["host_allocated_bytes"] <= pc["host_budget_bytes"], "allocated<=budget", where=where,
             allocated=pc["host_allocated_bytes"], budget=pc["host_budget_bytes"])
    c.expect(pc["host_used_bytes"] <= pc["host_allocated_bytes"], "used<=allocated", where=where,
             used=pc["host_used_bytes"], allocated=pc["host_allocated_bytes"])
    c.expect(pc["host_inflight_bytes"] <= pc["host_used_bytes"], "inflight<=used", where=where,
             inflight=pc["host_inflight_bytes"], used=pc["host_used_bytes"])
    comps = pc["components"]
    kinds = {x["storage_kind"] for x in comps}
    c.expect(kinds <= {"paged", "window", "boundary_state", "composite"}, "component kinds", where=where, kinds=sorted(kinds))
    for key in ("host_used_bytes", "h2d_bytes", "d2h_bytes"):
        total = sum(x[key] for x in comps)
        c.expect(total == pc[key], f"components sum {key}", where=where, components=total, top=pc[key])
    c.expect(pc["gpu_checkpoint_peak"] >= pc["gpu_checkpoint_count"], "gpu peak>=count", where=where)
    c.expect(pc["host_checkpoint_peak"] >= pc["host_checkpoint_count"], "host peak>=count", where=where)
    if not gdn:
        zeros = {k: pc[k] for k in ("gpu_checkpoint_count", "gpu_checkpoint_peak", "host_checkpoint_count",
                                    "host_checkpoint_peak", "active_state_count", "checkpoint_created",
                                    "checkpoint_deduplicated", "checkpoint_pruned", "checkpoint_evicted")}
        c.expect(not any(zeros.values()), "non-GDN checkpoint fields zero", where=where, values=zeros)
    if prev is not None:
        dropped = {k: (prev[k], pc[k]) for k in COUNTERS if pc[k] < prev[k]}
        c.expect(not dropped, "cumulative counters monotonic", where=where, dropped=dropped)


def snap(s, c, where, prev=None, gdn=True):
    pc = s.pc()
    invariants(c, pc, where, prev, gdn)
    return pc


def drained(s, timeout=3):
    """Idle snapshot after in-flight copies finish (or the last snapshot if they never do)."""
    deadline = time.monotonic() + timeout
    while True:
        pc = s.pc()
        if pc["host_inflight_bytes"] == 0 or time.monotonic() > deadline:
            return pc
        time.sleep(0.5)


def tokens(s, prompt):
    """Public prompt token count, measured in a throwaway cache group."""
    return prompt_tokens(s.complete(prompt, 1, group=f"count-{time.monotonic_ns()}"))


def pressure(s, seeds, sentences, group="pressure"):
    for seed in seeds:
        r = s.complete(document(seed, sentences), 4, group=group)
        assert r["status"] == 200, r


def abort_stream(s, prompt, after_s, group, max_tokens=64):
    """Client disconnect of a streaming request after a fixed delay."""
    conn = http.client.HTTPConnection("127.0.0.1", s.port, timeout=60)
    body = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "cache_group": group}
    conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    time.sleep(after_s)
    conn.sock.close()
    conn.close()


def hot_pair(s, prompt, n, group):
    first = s.complete(prompt, n, group=group)
    hot = s.complete(prompt, n, group=group)
    return first, hot


def compare_text(c, name, got, want, solo=True, **detail):
    same = got == want
    info = dict(same=same, common_chars=common_prefix_len(got, want), got=got[:120], want=want[:120], **detail)
    if solo and c.strict_text:
        c.expect(same, name, **info)
    else:
        c.note(name, **info)
    return same


def cold_restore(s, c, prompt, n, group, pressure_seeds, sentences, gdn, expect_h2d=True):
    """Hot reference, GPU pressure from other requests, then reaccess."""
    first = s.complete(prompt, n, group=group)
    pre_hot = drained(s)
    hot = s.complete(prompt, n, group=group)
    hot_delta = delta(drained(s), pre_hot)
    c.expect(hot_delta["h2d_bytes"] == 0 and hot_delta["host_reused_tokens"] == 0,
             "hot hit without CPU round trip", delta=hot_delta)
    c.expect(hot_delta["gpu_reused_tokens"] == (cached(hot) or 0), "hot hit counted as GPU reuse",
             delta=hot_delta, cached=cached(hot))
    pt = prompt_tokens(first)
    c.expect(cached(first) in (None, 0) or cached(first) < pt, "cold first cached<prompt", cached=cached(first), prompt=pt)
    c.expect((cached(hot) or 0) > 0, "hot hit cached>0", cached=cached(hot), prompt=pt)
    pressure(s, pressure_seeds, sentences)
    before = drained(s)
    again = s.complete(prompt, n, group=group)
    after = snap(s, c, "after cold restore", before, gdn)
    d = delta(after, before)
    hit = cached(again) or 0
    obs = dict(prompt_tokens=pt, cached_hot=cached(hot), cached_restore=hit, delta=d)
    if expect_h2d:
        c.expect(d["h2d_bytes"] > 0 and d["host_reused_tokens"] > 0, "cold restore uses H2D and host reuse", **obs)
    c.expect(d["host_reused_tokens"] + d["gpu_reused_tokens"] == hit, "reused segments == cached_tokens", **obs)
    c.expect(d["recomputed_tokens"] == pt - hit, "recomputed == prompt - cached", **obs)
    c.expect(hit <= pt, "cached<=prompt", **obs)
    compare_text(c, "restored output == GPU-hot output", text(again), text(hot),
                 solo=cached(hot) == hit, cached_hot=cached(hot), cached_restore=hit)
    record(f"{c.label}/cold_restore_obs", **obs)
    return dict(obs, h2d=d["h2d_bytes"], hot_text=text(hot), first_text=text(first))


def shared_restore(s, c, prompt, n, group, pressure_seeds, sentences, single, waiters=3, gdn=True):
    _, hot = hot_pair(s, prompt, n, group)
    pressure(s, pressure_seeds, sentences)
    before = drained(s)
    outs = s.parallel([lambda: s.complete(prompt, n, group=group)] * waiters)
    after = snap(s, c, "after shared restore", before, gdn)
    d = delta(after, before)
    hits = [cached(r) or 0 for r in outs]
    texts = [text(r) for r in outs]
    c.expect(all(r["status"] == 200 for r in outs), "shared restore all complete")
    compare_text(c, "shared restore identical outputs", texts[0], texts[-1], solo=True,
                 texts=[t[:80] for t in texts])
    compare_text(c, "shared restore output vs GPU-hot", texts[0], text(hot), solo=False, cached=hits)
    c.expect(d["host_reused_tokens"] + d["gpu_reused_tokens"] == sum(hits), "shared reused == sum cached",
             hits=hits, delta=d)
    if single["h2d"]:  # needs a measured single-restore size to compare against
        per_token = single["h2d"] / max(single["cached_restore"], 1)
        limit = 1.5 * per_token * max(hits)
        c.expect(d["h2d_bytes"] <= limit, "shared restore not copied per waiter", h2d=d["h2d_bytes"],
                 single_restore_bytes_per_token=per_token, hits=hits, limit=limit)
    c.expect(d["h2d_bytes"] > 0, "shared restore did use H2D", delta=d)
    return dict(hits=hits, delta=d)


def isolation(s, c, prompt, n):
    a1 = s.complete(prompt, n, group="iso-a")
    a2 = s.complete(prompt, n, group="iso-a")
    b1 = s.complete(prompt, n, group="iso-b")
    d1 = s.complete(prompt, n)
    c.expect((cached(a2) or 0) > 0, "isolation same group reuses", cached=cached(a2))
    c.expect((cached(b1) or 0) == 0, "isolation other group no reuse", cached=cached(b1))
    c.expect((cached(d1) or 0) == 0, "isolation default group no reuse", cached=cached(d1))
    c.expect(text(b1) == text(a1) and text(d1) == text(a1), "isolation cold outputs equal",
             a=text(a1)[:80], b=text(b1)[:80], d=text(d1)[:80])


def linear_chain(s, c, seed, rounds, n, group, sentences, gdn):
    """Strict continuation: each prompt = previous prompt + previous output + new user text."""
    cur, rows = document(seed, sentences), []
    prompts, outs = [], []
    for i in range(rounds):
        r = s.complete(cur, n, group=group)
        pc = drained(s)
        invariants(c, pc, f"linear round {i}", gdn=gdn)
        rows.append(dict(round=i, prompt=prompt_tokens(r), cached=cached(r),
                         gpu_ckpt=pc["gpu_checkpoint_count"], host_ckpt=pc["host_checkpoint_count"],
                         created=pc["checkpoint_created"], pruned=pc["checkpoint_pruned"],
                         host_used=pc["host_used_bytes"], inflight=pc["host_inflight_bytes"]))
        prompts.append(cur)
        outs.append(text(r))
        if i:
            prev_total = rows[i - 1]["prompt"] + n
            # text re-tokenization at the turn boundary can shorten the strict token prefix, so record only
            c.note("linear round reuse vs previous history", round=i, cached=cached(r), previous_total=prev_total)
        cur = cur + text(r) + f" User turn {i}: please continue the notes."
    record(f"{c.label}/linear_rows", rows=rows)
    return rows, prompts, outs


def forks(s, c, seed, sentences, n, group, gdn):
    base = document(seed, sentences) + "\nQuestion:"
    tails = [" what is the first note about?", " which value appears most often?", " summarize in one line."]
    common = tokens(s, base)
    results = []
    for i, tail in enumerate(tails):
        r = s.complete(base + tail, n, group=group)
        results.append(dict(fork=i, prompt=prompt_tokens(r), cached=cached(r), text=text(r)))
        if i:
            c.expect((cached(r) or 0) <= common, "fork cached<=common prefix tokens", fork=i,
                     cached=cached(r), common=common)
    again = s.complete(base + tails[0], n, group=group)
    hot = s.complete(base + tails[0], n, group=group)
    full = (cached(again) or 0) >= prompt_tokens(again) - 2
    if gdn:  # a recurrent state exists only at saved positions; shorter honest reuse is allowed
        c.note("old fork reuse length", full=full, cached=cached(again), prompt=prompt_tokens(again))
    else:
        c.expect(full, "old fork full reuse", cached=cached(again), prompt=prompt_tokens(again))
    compare_text(c, "old fork output stable", text(again), text(hot), solo=True)
    rows = [{k: v for k, v in r.items() if k != "text"} for r in results]
    record(f"{c.label}/fork_rows", common=common, rows=rows, again=cached(again))
    return dict(common=common, rows=rows, texts=[r["text"] for r in results] + [text(again)])


def history_edit(s, c, prompts, outs, n, group):
    """Modify an earlier assistant output and request again."""
    if len(prompts) < 3:
        return None
    original = prompts[2]
    out1 = outs[1]
    cut = len(prompts[1]) + max(1, len(out1) // 2)
    edited = original[:cut] + " [edited] " + original[cut:]
    common = tokens(s, original[:cut])
    r = s.complete(edited, n, group=group)
    c.expect((cached(r) or 0) <= common, "edited history cached<=true common prefix", cached=cached(r), common=common)
    hot = s.complete(edited, n, group=group)
    compare_text(c, "edited history output vs its hot repeat", text(r), text(hot), solo=False,
                 cached_first=cached(r), cached_hot=cached(hot))
    return dict(common=common, cached=cached(r), text=text(r))


def revisit_pruned(s, c, prompts, outs, n, group):
    """Return to round 1 of a continued chain; its state may have been pruned."""
    r = s.complete(prompts[1], n, group=group)
    compare_text(c, "revisit pruned round output", text(r), outs[1], solo=False, cached=cached(r),
                 prompt=prompt_tokens(r))
    return dict(cached=cached(r), prompt=prompt_tokens(r), same=text(r) == outs[1])


def cancellations(s, c, prompt, n, group, pressure_seeds, sentences, gdn):
    """Cancel while generating, while waiting, and right after submit (restore window)."""
    ref = s.complete(prompt, n, group=group)
    out = {}
    # generating: a co-running request on the same prefix must not be disturbed
    results = s.parallel([lambda: s.stream(prompt, 64, group=group, cancel_after=3),
                          lambda: s.complete(prompt, n, group=group)])
    compare_text(c, "co-runner unaffected by cancel during generation", text(results[1]), text(ref), solo=False)
    out["generating_partner_same"] = text(results[1]) == text(ref)
    # waiting: occupy all slots, then abort a queued request
    slots = [lambda i=i: s.complete(document(80 + i, sentences // 2), 48, group="cancel-fill") for i in range(4)]
    waiting = lambda: (time.sleep(0.3), abort_stream(s, prompt + " queued", 0.4, group))
    filled = s.parallel(slots + [waiting])
    c.expect(all(r["status"] == 200 for r in filled[:4]), "slots complete after waiting cancel")
    # restore window: cold prefix, abort shortly after submit; repeat to look for accumulation
    used = []
    for cycle in range(3):
        pressure(s, [p + cycle for p in pressure_seeds], sentences)
        abort_stream(s, prompt, 0.02, group)
        time.sleep(1.0)
        pc = drained(s)
        invariants(c, pc, f"after restore-cancel cycle {cycle}", gdn=gdn)
        used.append((pc["host_used_bytes"], pc["host_inflight_bytes"]))
    after = s.complete(prompt, n, group=group)
    hot = s.complete(prompt, n, group=group)
    compare_text(c, "request after restore cancels == its hot repeat", text(after), text(hot),
                 solo=cached(after) == cached(hot), cached=cached(after))
    compare_text(c, "request after restore cancels vs pre-cancel reference", text(after), text(ref), solo=False)
    c.note("idle in-flight bytes after cancel cycles", used=used)
    c.expect(used[-1][0] <= max(u for u, _ in used[:1]) * 1.5 + 1, "host use not accumulating over cancel cycles",
             used=used)
    out["used"] = used
    return out


def rebuild(s, c, prompt, n, group, gdn):
    """Rejected rebuild keeps cache; accepted idle rebuild leaves a working server."""
    hot = s.complete(prompt, n, group=group)
    before = drained(s)
    geometry = s.status()["geometry"]
    bad = post(s.url, "/v1/cache/rebuild", {"num_pages": 10 ** 9})
    c.expect(bad["status"] != 200 and bad["body"].get("status") == "rejected", "huge rebuild rejected",
             status=bad["status"], body=str(bad["body"])[:300])
    c.expect(s.status()["state"] == "serving", "serving after rejected rebuild")
    keep = s.complete(prompt, n, group=group)
    c.expect((cached(keep) or 0) >= (cached(hot) or 0), "rejected rebuild keeps cache", hot=cached(hot), after=cached(keep))
    compare_text(c, "output after rejected rebuild", text(keep), text(hot), solo=cached(keep) == cached(hot))
    ok = post(s.url, "/v1/cache/rebuild", {"num_pages": geometry["num_pages"]})
    c.expect(ok["status"] == 200, "idle rebuild accepted", status=ok["status"], body=str(ok["body"])[:300])
    deadline = time.monotonic() + 300
    while s.status()["state"] != "serving" and time.monotonic() < deadline:
        time.sleep(1)
    after_pc = snap(s, c, "after rebuild", before, gdn)
    for key in ("gpu_checkpoint_peak", "host_checkpoint_peak", "d2h_bytes", "gpu_reused_tokens"):
        c.expect(after_pc[key] >= before[key], f"rebuild keeps cumulative {key}", before=before[key], after=after_pc[key])
    post_resp = s.complete(prompt, n, group=group)
    hot2 = s.complete(prompt, n, group=group)
    c.expect(post_resp["status"] == 200 and s.alive(), "request after rebuild completes")
    compare_text(c, "post-rebuild output vs post-rebuild hot repeat", text(post_resp), text(hot2),
                 solo=cached(post_resp) == cached(hot2))
    compare_text(c, "post-rebuild output vs pre-rebuild", text(post_resp), text(hot), solo=False,
                 cached=cached(post_resp))
    return dict(bad=bad["status"], ok=ok["status"], cached_after=cached(post_resp),
                h2d_after=s.pc()["h2d_bytes"] - after_pc["h2d_bytes"])


def stats(s):
    return get(s.url, "/v1/stats")["body"]


def run_matrix(s, c, sentences, pressure_count, gdn, policy, n=12, parts="all"):
    """Common section-5 rows on one launch, starting from a fresh server for clean counts."""
    obs = {}
    graph_before = stats(s).get("cuda_graph") or {}
    start = snap(s, c, "startup", gdn=gdn)
    c.expect(start["enabled"] and start["policy"] == policy, "status enabled/policy", enabled=start["enabled"],
             policy=start["policy"])
    c.expect(start["transfer_device_bytes"] > 0, "transfer buffer reported when enabled",
             transfer=start["transfer_device_bytes"])
    if parts == "all":
        rows, prompts, outs = linear_chain(s, c, 3, 5, n, "linear", sentences, gdn)
        obs["linear"] = rows
        if gdn and policy == "continuation":
            tail = rows[2:]
            c.expect(max(r["gpu_ckpt"] for r in tail) <= 2 and max(r["host_ckpt"] for r in tail) <= 2,
                     "continuation keeps <=2 GDN snapshots per linear chain after drain", rows=rows)
            c.expect(rows[-1]["pruned"] > rows[1]["pruned"], "continuation prunes replaced snapshots", rows=rows)
            c.expect(rows[-1]["host_used"] <= rows[2]["host_used"] * 1.2, "host use not growing per round", rows=rows)
        if not gdn:
            c.expect(all(r["gpu_ckpt"] == 0 and r["host_ckpt"] == 0 for r in rows), "no GDN snapshots without GDN")
        obs["revisit"] = revisit_pruned(s, c, prompts, outs, n, "linear")
        obs["edit"] = history_edit(s, c, prompts, outs, n, "linear")
    seeds = list(range(40, 40 + pressure_count))
    single = cold_restore(s, c, document(1, sentences), n, "cold", seeds, sentences, gdn)
    obs["cold"] = single
    obs["shared"] = shared_restore(s, c, document(2, sentences), n, "shared", seeds, sentences, single, gdn=gdn)
    obs["batch_tails"] = batch_tails(s, c, sentences, [x + 150 for x in seeds], gdn)
    if parts == "all":
        isolation(s, c, document(5, sentences // 2), n)
        obs["forks"] = forks(s, c, 6, sentences, n, "fork", gdn)
    obs["cancel"] = cancellations(s, c, document(7, sentences), n, "cancel", [60 + i for i in range(pressure_count)],
                                  sentences, gdn)
    # checked before the rebuild step, which legitimately re-captures graphs
    graph_after = stats(s).get("cuda_graph") or {}
    if graph_before.get("enabled"):
        c.expect(graph_after.get("capture_seconds") == graph_before.get("capture_seconds"),
                 "CUDA graphs not re-captured by restores", before=graph_before.get("capture_seconds"),
                 after=graph_after.get("capture_seconds"))
    obs["graph"] = {k: graph_after.get(k) for k in ("enabled", "target_decode", "draft", "verify", "capture_seconds")}
    if parts == "all":
        obs["rebuild"] = rebuild(s, c, document(1, sentences), n, "cold", gdn)
    final = drained(s, timeout=30)
    invariants(c, final, "final", gdn=gdn)
    c.note("idle in-flight bytes 30s after last request", inflight=final["host_inflight_bytes"],
           used=final["host_used_bytes"])
    c.expect(final["active_state_count"] == 0, "idle: no request working states left",
             active=final["active_state_count"])
    obs["final"] = {k: v for k, v in final.items() if k != "components"}
    obs["components"] = final["components"]
    record(f"{c.label}/matrix_summary", obs={k: v for k, v in obs.items() if k not in ("cold",)},
           cold={k: v for k, v in single.items() if not k.endswith("text")})
    return obs


def exact_prompt(s, length, salt=""):
    """Prompt whose public prompt_tokens equals length, built from common one-token words."""
    words = [" the", " cat", " sat", " on", " a", " mat", " and", " then", " it", " ran", " to", " see",
             " his", " dog", " in", " big", " red", " house", " by", " sea"]
    head = f"Story {salt}:"
    k = length
    for _ in range(8):
        prompt = head + "".join(words[(i * 7 + len(salt)) % len(words)] for i in range(max(k, 0)))
        got = tokens(s, prompt)
        if got == length:
            return prompt
        k += length - got
    raise AssertionError(f"could not build a {length}-token prompt")


def small_capacity(s, c, sentences, pressure_count, gdn, rounds=3):
    """Budget never exceeded, no hang, honest reuse when CPU space is short."""
    rows = []
    for i in range(rounds):
        prompt = document(90 + i, sentences)
        first = s.complete(prompt, 12, group="small")
        hot = s.complete(prompt, 12, group="small")
        pressure(s, range(100 + i * pressure_count, 100 + (i + 1) * pressure_count), sentences)
        before = drained(s)
        again = s.complete(prompt, 12, group="small")
        after = snap(s, c, f"small round {i}", before, gdn)
        d = delta(after, before)
        hit = cached(again) or 0
        c.expect(d["host_reused_tokens"] + d["gpu_reused_tokens"] == hit, "small: reused segments == cached",
                 round=i, cached=hit, delta=d)
        c.expect(d["host_reused_tokens"] == 0 or d["h2d_bytes"] > 0, "small: host reuse implies H2D", round=i, delta=d)
        compare_text(c, "small: output after pressure vs hot", text(again), text(hot), solo=hit == cached(hot),
                     cached=hit, cached_hot=cached(hot))
        rows.append(dict(round=i, prompt=prompt_tokens(first), cached_hot=cached(hot), cached_after=hit,
                         host_used=after["host_used_bytes"], allocated=after["host_allocated_bytes"],
                         host_ckpt=after["host_checkpoint_count"], evicted=after["checkpoint_evicted"], delta=d))
    record(f"{c.label}/small_rows", rows=rows)
    return rows


def sd_window(s):
    sp = stats(s)["speculative"]
    return {k: sp[k] for k in ("draft_tokens", "accepted_draft_tokens", "verify_steps")}, sp


def sd_checks(s, c, sentences, steps, gdn=True):
    """Drafting still happens on restored prefixes; rejected tails never become reusable history."""
    out = {}
    _, sp = sd_window(s)
    c.expect(sp["enabled"] and sp["max_draft_steps"] == steps, "SD enabled with configured steps",
             enabled=sp["enabled"], max_draft_steps=sp["max_draft_steps"])
    prompt = document(11, sentences)
    first = s.complete(prompt, 48, group="sd")
    hot = s.complete(prompt, 48, group="sd")
    pressure(s, range(120, 126), sentences)
    before_pc = drained(s)
    before, _ = sd_window(s)
    again = s.complete(prompt, 48, group="sd")
    after, _ = sd_window(s)
    d = {k: after[k] - before[k] for k in after}
    pd = delta(snap(s, c, "after SD restore", before_pc, gdn), before_pc)
    out["restore"] = dict(sd=d, cache=pd, cached=cached(again))
    c.expect(pd["host_reused_tokens"] > 0, "SD: prefix restored from CPU", delta=pd)
    c.expect(d["draft_tokens"] > 0 and d["accepted_draft_tokens"] > 0 and d["verify_steps"] > 0,
             "SD: restored request drafts and accepts", sd=d)
    compare_text(c, "SD: restored output vs hot", text(again), text(hot), solo=cached(again) == cached(hot))
    compare_text(c, "SD: hot output vs first", text(hot), text(first), solo=False)
    # low-acceptance continuation, then extend the accepted history
    noisy = "Random tokens: " + " ".join(f"{(i * 7919) % 104729:x}" for i in range(120))
    b, _ = sd_window(s)
    r1 = s.complete(noisy, 40, group="sd-noisy")
    a, _ = sd_window(s)
    out["noisy"] = {k: a[k] - b[k] for k in a}
    follow = noisy + text(r1) + " Next:"
    hist = tokens(s, noisy + text(r1))
    r2 = s.complete(follow, 24, group="sd-noisy")
    c.expect((cached(r2) or 0) <= hist, "SD: reuse never exceeds accepted history", cached=cached(r2), history=hist)
    r3 = s.complete(follow, 24, group="sd-noisy")
    compare_text(c, "SD: continuation output vs hot repeat", text(r2), text(r3), solo=cached(r2) == cached(r3))
    pressure(s, range(126, 132), sentences)
    r4 = s.complete(follow, 24, group="sd-noisy")
    compare_text(c, "SD: continuation restored vs hot", text(r4), text(r3), solo=cached(r4) == cached(r3),
                 cached=cached(r4))
    # early stop: the model ends before max_tokens
    body = {"model": "m", "messages": [{"role": "user", "content": "Reply with only the word OK."}],
            "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    e = post(s.url, "/v1/chat/completions", body)
    out["early_stop"] = dict(status=e["status"], finish=e["body"].get("choices", [{}])[0].get("finish_reason"),
                             tokens=(e["body"].get("usage") or {}).get("completion_tokens"))
    c.expect(e["status"] == 200 and out["early_stop"]["finish"] == "stop", "SD: early stop request completes",
             **out["early_stop"])
    _, sp = sd_window(s)
    out["histogram"] = sp["draft_length_histogram"]
    c.expect(len(sp["draft_length_histogram"]) == steps + 1, "SD: histogram length matches steps",
             histogram=sp["draft_length_histogram"])
    record(f"{c.label}/sd_obs", **out)
    return out


def lengths_restore(s, c, lengths, n, group, pressure_seeds, sentences, gdn):
    """Exact-length prompts around a public window/page/compression size: hot, pressure, restore."""
    prompts = {L: exact_prompt(s, L, salt=f"{group}{L}") for L in lengths}
    hot = {}
    for L, p in prompts.items():
        s.complete(p, n, group=group)
        hot[L] = s.complete(p, n, group=group)
    pressure(s, pressure_seeds, sentences)
    rows = []
    for L, p in prompts.items():
        before = drained(s)
        r = s.complete(p, n, group=group)
        after = snap(s, c, f"length {L} restore", before, gdn)
        d = delta(after, before)
        hit = cached(r) or 0
        c.expect(hit <= L and d["host_reused_tokens"] + d["gpu_reused_tokens"] == hit,
                 "length restore honest reuse", length=L, cached=hit, delta=d)
        compare_text(c, "length restore output vs hot", text(r), text(hot[L]), solo=hit == cached(hot[L]),
                     length=L, cached=hit, cached_hot=cached(hot[L]))
        comps = {x["name"]: (x["h2d_bytes"] - y["h2d_bytes"]) for x, y in zip(after["components"], before["components"])}
        rows.append(dict(length=L, cached_hot=cached(hot[L]), cached_restore=hit, host_reused=d["host_reused_tokens"],
                         component_h2d=comps, same=text(r) == text(hot[L])))
    record(f"{c.label}/length_rows", group=group, rows=rows)
    return rows


def continuation_and_fork(s, c, prompt_len, n, group, pressure_seeds, sentences, gdn):
    """Prefill, decode, pressure, then continue the history and fork from an older point."""
    p = exact_prompt(s, prompt_len, salt=group)
    r0 = s.complete(p, n, group=group)
    cont = p + text(r0) + " Then"
    hist = tokens(s, p + text(r0))
    r1 = s.complete(cont, n, group=group)
    c.expect((cached(r1) or 0) <= hist, "continuation reuse <= real history", cached=cached(r1), history=hist)
    h1 = s.complete(cont, n, group=group)
    pressure(s, pressure_seeds, sentences)
    r2 = s.complete(cont, n, group=group)
    compare_text(c, "continuation restored vs hot", text(r2), text(h1), solo=cached(r2) == cached(h1),
                 cached=cached(r2), cached_hot=cached(h1))
    fork = p + " Instead, describe the sea."
    f1 = s.complete(fork, n, group=group)
    f2 = s.complete(fork, n, group=group)
    c.expect((cached(f1) or 0) <= prompt_len, "old fork reuse <= shared prefix", cached=cached(f1), shared=prompt_len)
    compare_text(c, "old fork output vs hot", text(f1), text(f2), solo=cached(f1) == cached(f2),
                 cached=cached(f1), cached_hot=cached(f2))
    back = s.complete(p, n, group=group)
    compare_text(c, "original prompt after continuation/fork", text(back), text(r0), solo=False, cached=cached(back))
    row = dict(prompt=prompt_len, history=hist, cached_cont=cached(r1), cached_restore=cached(r2),
               cached_fork=cached(f1), cached_back=cached(back))
    record(f"{c.label}/continuation_fork", **row)
    return row


def overlapping_windows(s, c, shared_len, n, pressure_seeds, sentences, gdn):
    """Two forks whose recent windows overlap; cancel one under pressure, the other stays correct."""
    d = exact_prompt(s, shared_len, salt="overlap")
    a, b = d + " Tell me about the cat.", d + " Tell me about the dog."
    s.complete(b, n, group="overlap")
    hot_b = s.complete(b, n, group="overlap")
    s.complete(a, n, group="overlap")
    pressure(s, pressure_seeds, sentences)
    res = s.parallel([lambda: s.stream(a, 64, group="overlap", cancel_after=4),
                      lambda: s.complete(b, n, group="overlap")])
    compare_text(c, "overlap: surviving fork vs its hot output", text(res[1]), text(hot_b),
                 solo=False, cached=cached(res[1]))
    pressure(s, [x + 50 for x in pressure_seeds], sentences)
    again = s.complete(b, n, group="overlap")
    hot_again = s.complete(b, n, group="overlap")
    compare_text(c, "overlap: surviving fork after cancel+pressure vs hot", text(again), text(hot_again),
                 solo=cached(again) == cached(hot_again), cached=cached(again))
    snap(s, c, "after overlap", gdn=gdn)
    a2 = s.complete(a, n, group="overlap")
    c.expect(a2["status"] == 200, "overlap: cancelled fork usable again")


def batch_tails(s, c, sentences, pressure_seeds, gdn):
    """Non-full concurrent batch mixing restored and new prompts with uneven output lengths."""
    jobs = [(document(1, sentences), 5, "cold"), (document(2, sentences), 23, "shared"),
            (document(13, sentences // 3), 17, "tail")]
    solo = [s.complete(p, n, group=g) for p, n, g in jobs]
    pressure(s, pressure_seeds, sentences)
    outs = s.parallel([lambda p=p, n=n, g=g: s.complete(p, n, group=g) for p, n, g in jobs])
    rows = []
    for (p, n, g), a, b in zip(jobs, solo, outs):
        usage = b["body"]["usage"]
        c.expect(usage["completion_tokens"] == n and b["body"]["choices"][0]["finish_reason"] == "length",
                 "batch tail: each request gets its own length", group=g, want=n, got=usage["completion_tokens"])
        compare_text(c, "batch tail: concurrent restored output vs solo", text(b), text(a), solo=False, group=g,
                     cached=cached(b))
        rows.append(dict(group=g, max_tokens=n, cached=cached(b), same_as_solo=text(a) == text(b)))
    snap(s, c, "after batch tails", gdn=gdn)
    record(f"{c.label}/batch_tail_rows", rows=rows)
    return rows
