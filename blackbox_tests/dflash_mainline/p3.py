"""P3: long generation releasing old window protection, forks, copies during splits, cold restore,
pressure admission, physical memory saving. Contract: sections 2 (cold windows), 3, 5 (input/shared/cold rows)."""

import json
import math
import threading
import time
import urllib.request

from transformers import AutoTokenizer

from harness import cached, document, dflash, post, qwen36, text
from p1 import REAL_ACCEPT_MIN, TINY
from p2 import REUSE_ACCEPT_FRACTION, REUSE_MIN_SHARE, fresh, idle_window_clean, note_question
from workloads import COPY_MIN_RATIO, acceptance, copy_prompt, copy_ratio

WINDOW_MARGIN = 64       # report-only: request-owned slots above the native window (release is chunked)
PROGRESS_MARGIN = 256    # output tokens past the window before old-window protection must be gone
OVERLAP_COPY_FACTOR = 3.0  # restore bytes for 3 forks of P + 1 other cold prompt, in single-restore units
PHYSICAL_SAVING_SHARE = 0.5

CONFIGS = {
    "nvfp4_n8_cold": qwen36(policy="layered-pipeline") + dflash(8) + ["--prefix-cache-host-gib", "4"],
    "nvfp4_n8_coldsmall": qwen36() + dflash(8) + ["--prefix-cache-host-gib", "0.25"],
    "nvfp4_lp_tool": qwen36(policy="layered-pipeline") + dflash(8) + ["--enable-special-token-ckpt"],
}


def long_generation(server, c):
    window = server.status()["geometry"]["dflash"]["window_tokens"]
    tokenizer = AutoTokenizer.from_pretrained(server.flag("--model-path"))
    sentences = max(20, math.ceil(2.2 * window / 29))
    prompt, source = copy_prompt(80, sentences)
    budget = len(tokenizer(source)["input_ids"]) + 200
    progress, samples, done = [], [], threading.Event()
    body = {"model": "m", "prompt": prompt, "max_tokens": budget, "temperature": 0, "stream": True,
            "cache_group": fresh()}

    def run():
        req = urllib.request.Request(server.url + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        out = ""
        with urllib.request.urlopen(req, timeout=1800) as response:
            for raw in response:
                line = raw.decode().strip()
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    chunk = json.loads(line[5:])
                    if chunk.get("choices"):
                        out += chunk["choices"][0].get("text", "")
                        progress.append((time.monotonic(), out))
        done.set()

    before = server.stats()
    worker = threading.Thread(target=run)
    worker.start()
    while not done.is_set():
        samples.append((time.monotonic(), server.status()["prefix_cache"]["window_slots"]))
        time.sleep(0.5)
    worker.join()
    after = server.wait_idle()
    from workloads import spec_delta
    delta = spec_delta(after, before)
    output = progress[-1][1] if progress else ""
    rows = []
    for t, w in samples:
        seen = [p for p in progress if p[0] <= t]
        produced = len(tokenizer(seen[-1][1])["input_ids"]) if seen else 0
        rows.append({"output_tokens": produced, **w})
    late = [r for r in rows if r["output_tokens"] > window + PROGRESS_MARGIN]
    c.check("long_generation_fidelity", copy_ratio(source, output) >= COPY_MIN_RATIO,
            ratio=copy_ratio(source, output))
    c.check("long_generation_covered_window", bool(late), max_output=max((r["output_tokens"] for r in rows), default=0))
    c.check("old_window_protection_released", all(r["tree_locked"] == 0 for r in late),
            worst=max(late, key=lambda r: r["tree_locked"]) if late else None)
    worst = max((r["request_owned"] for r in rows), default=0)
    # Contract: no permanent "old window + new window" per request, and no growth with output length.
    third = len(late) // 2
    c.check("request_window_bounded", worst < 2 * window and (not late or max(r["request_owned"] for r in late[third:])
            <= max(r["request_owned"] for r in late[:third or 1]) + WINDOW_MARGIN), worst=worst, window=window)
    c.note("request_window_over_native", worst_excess=worst - window, margin=WINDOW_MARGIN)
    if server.flag("--speculative-draft-model-path") != str(TINY):
        c.check("long_generation_acceptance", acceptance(delta) >= REAL_ACCEPT_MIN, acceptance=acceptance(delta))
    revisit = server.complete(prompt + output + "\n\nNow write only the first note again:\n", 24,
                              group=body["cache_group"])
    first_prompt_tokens = len(tokenizer(prompt)["input_ids"])
    c.check("revisit_after_window_slides", cached(revisit) >= REUSE_MIN_SHARE * first_prompt_tokens
            and "80-0" in text(revisit), cached=cached(revisit), text=text(revisit)[:80])
    idle_window_clean(server, c, "long_generation")
    return {"samples": rows, "delta": delta, "output_chars": len(output)}


def pressure_admission(server, c):
    out = []
    for wave, shape in enumerate((((500, 256),) * 4, ((520, 200), (330, 200), (120, 200)))):
        group = fresh()
        work = [(copy_prompt(90 + 5 * wave + k, s), m) for k, (s, m) in enumerate(shape)]
        resps = server.parallel([lambda w=w: server.complete(w[0][0], w[1], group=group) for w in work])
        ratios = [copy_ratio(src, text(r)) for ((_, src), _), r in zip(work, resps)]
        c.check(f"pressure_wave_complete:{wave}", all(r["status"] == 200 for r in resps)
                and min(ratios) >= COPY_MIN_RATIO, ratios=ratios,
                prompt_tokens=[r["body"].get("usage", {}).get("prompt_tokens") for r in resps])
        out.append(ratios)
    idle_window_clean(server, c, "pressure")
    return out


def pressure_mixed(server, c):
    """Short requests keep decoding while long prompts are admitted under window-pool pressure."""
    short = [copy_prompt(170 + k, 30) for k in range(2)]
    long = [copy_prompt(175 + k, 500) for k in range(2)]

    def delayed(work):
        time.sleep(2)
        return server.complete(work[0], 128, group=fresh())
    calls = [lambda w=w: server.complete(w[0], 800, group=fresh()) for w in short]
    calls += [lambda w=w: delayed(w) for w in long]
    resps = server.parallel(calls)
    ratios = [copy_ratio(src, text(r)) for (_, src), r in zip(short + long, resps)]
    c.check("pressure_mixed_complete", all(r["status"] == 200 for r in resps) and min(ratios) >= COPY_MIN_RATIO,
            ratios=ratios)
    idle_window_clean(server, c, "pressure_mixed")
    return ratios


def cancel_long_prompt(server, c):
    """Disconnect while a long prompt is still being prefilled, with another request decoding."""
    long_prompt, _ = copy_prompt(180, 500)
    short, short_src = copy_prompt(181, 30)
    body = json.dumps({"model": "m", "prompt": long_prompt, "max_tokens": 64, "temperature": 0, "stream": True,
                       "cache_group": fresh()}).encode()

    def cancelled(wait):
        time.sleep(1.0)
        try:
            req = urllib.request.Request(server.url + "/v1/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=wait) as response:
                response.readline()
        except Exception:
            pass
    results = server.parallel([lambda: server.complete(short, 600, group=fresh()),
                               lambda: cancelled(1.5), lambda: cancelled(4.0)])
    c.check("decoding_request_unaffected", copy_ratio(short_src, text(results[0])) >= COPY_MIN_RATIO)
    time.sleep(5)
    c.check("server_alive_after_prefill_cancel", server.alive())
    idle_window_clean(server, c, "cancel_long_prompt")
    again = server.complete(short, 200, group=fresh())
    c.check("serves_after_prefill_cancel", copy_ratio(short_src, text(again)) >= COPY_MIN_RATIO)


SHORT_ADMIT_SECONDS = 10  # a short request beside nearly finished long decodes must finish within this


def short_admit_near_end(server, c):
    """While long-prompt requests decode their last tokens and the window pool has room, a short new
    request is admitted promptly instead of waiting for them to finish."""
    longs = [copy_prompt(185 + k, 500) for k in range(3)]
    first, finished = {}, {}

    def run_long(k):
        body = {"model": "m", "prompt": longs[k][0], "max_tokens": 600, "temperature": 0, "stream": True,
                "cache_group": fresh()}
        req = urllib.request.Request(server.url + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        out = ""
        with urllib.request.urlopen(req, timeout=900) as response:
            for raw in response:
                line = raw.decode().strip()
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    chunk = json.loads(line[5:])
                    if chunk.get("choices"):
                        first.setdefault(k, time.monotonic())
                        out += chunk["choices"][0].get("text", "")
        finished[k] = time.monotonic()
        return {"text": out}

    def run_short():
        while len(first) < 3:
            time.sleep(0.2)
        time.sleep(6)  # all three decode, well into their 600-token outputs
        t0 = time.monotonic()
        resp = server.complete("The capital of France is", 16, group=fresh())
        return resp, time.monotonic() - t0, t0
    results = server.parallel([lambda k=k: run_long(k) for k in range(3)] + [run_short])
    resp, seconds, t0 = results[3]
    still_running = sum(1 for t in finished.values() if t > t0 + seconds)
    if still_running:
        c.check("short_request_admitted_promptly", resp["status"] == 200 and seconds <= SHORT_ADMIT_SECONDS,
                seconds=seconds, longs_still_running=still_running)
    else:
        c.note("short_admit_not_exercised", seconds=seconds, reason="long requests finished first")
    c.check("longs_complete_beside_short", all(copy_ratio(src, r["text"]) >= COPY_MIN_RATIO
                                               for (_, src), r in zip(longs, results[:3])))
    idle_window_clean(server, c, "short_admit")
    return {"seconds": seconds, "long_finish_after_short_start": [t - t0 for t in finished.values()]}


TOOLS = [{"type": "function", "function": {
    "name": "lookup_note", "description": "Return the recorded value of one note.",
    "parameters": {"type": "object", "properties": {"note_id": {"type": "string"}}, "required": ["note_id"]}}}]


def tool_ckpt_pressure(server, c):
    """--enable-special-token-ckpt: concurrent long prompts whose replies call a tool, while a new long
    prompt arrives; the server stays up and idle accounting is consistent."""
    def tool_chat(seed):
        body = {"model": "m", "max_tokens": 200, "temperature": 0, "tools": TOOLS, "cache_group": fresh(),
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": document(seed, 350) +
                              f"\n\nUse the lookup_note tool to fetch note {seed}-7. Call the tool now."}]}
        return post(server.url, "/v1/chat/completions", body, 420)

    def late():
        time.sleep(3)
        prompt, src = copy_prompt(199, 450)
        return server.complete(prompt, 128, group=fresh()), src
    results = server.parallel([lambda s=s: tool_chat(190 + s) for s in range(3)] + [late])
    chats, (last, src) = results[:3], results[3]
    calls = [r["body"]["choices"][0].get("message", {}).get("tool_calls") for r in chats if r["status"] == 200]
    c.check("tool_replies_ok", all(r["status"] == 200 for r in chats) and any(calls),
            statuses=[r["status"] for r in chats], tool_calls=[bool(x) for x in calls])
    c.check("late_long_prompt_ok", copy_ratio(src, text(last)) >= COPY_MIN_RATIO)
    time.sleep(5)  # idle integrity is checked by the service while idle
    c.check("server_alive_after_tool_pressure", server.alive() and server.stats()["requests"]["active"] == 0)
    idle_window_clean(server, c, "tool_ckpt")


def pc(server):
    return server.status()["prefix_cache"]


def pc_delta(after, before):
    keys = ("h2d_bytes", "d2h_bytes", "host_reused_tokens", "gpu_reused_tokens", "recomputed_tokens")
    out = {k: after[k] - before[k] for k in keys}
    comp0 = {x["name"]: x for x in before["components"]}
    out["components_h2d"] = {x["name"]: x["h2d_bytes"] - comp0[x["name"]]["h2d_bytes"] for x in after["components"]}
    return out


def pressure(server, c, label):
    """Evict older prefixes from GPU with distinct real requests exceeding the KV capacity."""
    worst = 0
    for k in range(6):
        server.complete(document(100 + k, 450) + "\nEnd of notes.", 4, group=fresh())
        p = pc(server)
        worst = max(worst, p["host_used_bytes"] - p["host_budget_bytes"])
    c.check(f"host_budget_respected:{label}", worst <= 0, over_budget_bytes=worst)


def cold_restore(server, c):
    host = float(server.flag("--prefix-cache-host-gib", 0))
    group = fresh()
    prompt, source = copy_prompt(70, 120)
    first, d_first = server.measured(lambda: server.complete(prompt, 300, group=group))
    pressure(server, c, "cold_restore")
    before = pc(server)
    again, d_again = server.measured(lambda: server.complete(prompt, 300, group=group))
    delta = pc_delta(pc(server), before)
    tokens = again["body"]["usage"]["prompt_tokens"]
    c.check("cold_access_fidelity", copy_ratio(source, text(again)) >= COPY_MIN_RATIO)
    c.check("cold_access_keeps_drafting", acceptance(d_again) >= REUSE_ACCEPT_FRACTION * acceptance(d_first),
            cold=acceptance(d_first), again=acceptance(d_again))
    if host >= 1:
        drafted = {k: v for k, v in delta["components_h2d"].items() if "draft" in k}
        c.check("cold_restore_hit", delta["host_reused_tokens"] > 0 and cached(again) >= REUSE_MIN_SHARE * tokens,
                delta=delta, cached=cached(again), prompt_tokens=tokens)
        c.check("cold_restore_includes_draft_history", drafted and all(v > 0 for v in drafted.values()),
                components=delta["components_h2d"])
    elif host == 0:
        c.check("host0_no_restore", delta["h2d_bytes"] == 0 and delta["host_reused_tokens"] == 0, delta=delta)
    c.note("cold_text_equal_first", equal=text(first) == text(again))
    server.single_restore_h2d, server.cold_group = delta["h2d_bytes"], group
    return {"delta": delta, "cached": cached(again), "prompt_tokens": tokens}


def fork_copy_overlap(server, c):
    """Three forks of one cold prefix plus another cold prompt restore at the same time."""
    seed = 71
    base, g = document(seed, 120), fresh()
    server.complete(base, 1, group=g)
    other, other_src = copy_prompt(72, 120)
    _, d_fresh = server.measured(lambda: server.complete(other, 300, group=g))
    pressure(server, c, "fork_overlap")
    before = pc(server)
    forks = [(note_question(seed, i), m) for i, m in ((11, 8), (55, 40), (99, 120))]
    calls = [lambda f=f: server.complete(base + f[0][0], f[1], group=g) for f in forks]
    calls.append(lambda: server.complete(other, 300, group=g))
    stats0 = server.stats()
    resps = server.parallel(calls)
    after_stats = server.wait_idle()
    delta = pc_delta(pc(server), before)
    answers = [text(r) for r in resps[:3]]
    c.check("cold_forks_correct", all(f[0][1] in a[:24] for f, a in zip(forks, answers)),
            answers=[a[:40] for a in answers], expected=[f[0][1] for f in forks])
    c.check("cold_other_fidelity", copy_ratio(other_src, text(resps[3])) >= COPY_MIN_RATIO)
    single = getattr(server, "single_restore_h2d", 0)
    c.check("shared_restore_not_per_waiter", single > 0 and delta["h2d_bytes"] <= OVERLAP_COPY_FACTOR * single,
            h2d=delta["h2d_bytes"], single=single)
    from workloads import spec_delta
    c.note("overlap_acceptance", fresh=acceptance(d_fresh), overlap_batch=acceptance(spec_delta(after_stats, stats0)))
    p = pc(server)
    c.check("idle_no_inflight_copies", p["host_inflight_bytes"] == 0 and p["window_slots"]["copy_inflight"] == 0,
            host_inflight=p["host_inflight_bytes"], window=p["window_slots"])
    return {"delta": delta, "answers": answers}


def cancel_waiting_restore(server, c):
    prompt, source = copy_prompt(70, 120)
    pressure(server, c, "cancel_restore")
    body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": 300, "temperature": 0, "stream": True,
                       "cache_group": server.cold_group}).encode()
    for wait in (0.2, 1.0):
        try:
            req = urllib.request.Request(server.url + "/v1/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=wait) as response:
                response.readline()
                response.readline()
        except Exception:
            pass
    time.sleep(5)
    server.wait_idle()
    p = pc(server)
    w = p["window_slots"]
    c.check("cancel_last_waiter_clean", p["host_inflight_bytes"] == 0 and w["copy_inflight"] == 0
            and w["request_owned"] == 0, host_inflight=p["host_inflight_bytes"], window=w)
    again = server.complete(prompt, 300, group=server.cold_group)
    c.check("serves_after_cancel_restore", copy_ratio(source, text(again)) >= COPY_MIN_RATIO)
    return {"window": w}


PLAN = {
    "nvfp4_n8": [long_generation, pressure_admission, pressure_mixed, short_admit_near_end, cold_restore,
                 cancel_long_prompt],
    "nvfp4_lp_tool": [tool_ckpt_pressure],

    "nvfp4_n8_cold": [cold_restore, fork_copy_overlap, cancel_waiting_restore],
    "nvfp4_n8_coldsmall": [cold_restore],  # smoke
}


def physical_saving(sessions, c):
    a, b = sessions.get("nvfp4_n8"), sessions.get("nvfp4_n8_full")
    if not a or not b or "process_gpu_mib" not in a or "process_gpu_mib" not in b:
        return
    saved = (b["process_gpu_mib"] - a["process_gpu_mib"]) * 2 ** 20
    reported = (b["status0"]["geometry"]["dflash"]["context_bytes"]
                - a["status0"]["geometry"]["dflash"]["context_bytes"])
    c.check("compact_saves_physical_memory", reported > 0 and saved >= PHYSICAL_SAVING_SHARE * reported,
            process_mib=(a["process_gpu_mib"], b["process_gpu_mib"]), reported_context_saving=reported)
    c.check("saving_not_refilled_into_gdn", a["status0"]["geometry"]["num_mamba_slots"]
            == b["status0"]["geometry"]["num_mamba_slots"])


COMPARE = [physical_saving]
