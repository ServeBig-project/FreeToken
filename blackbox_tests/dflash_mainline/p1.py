"""P1: capability grouping, independent DFlash budget, defaults, BF16/NVFP4 targets,
native-window compact vs full storage, no-GDN target without a GDN budget, illegal capacity.
Contract: sections 1, 2 (math A/B), 3 (budget/status/illegal rebuild), 5 (format/model rows)."""

import math
import uuid
from pathlib import Path

from safetensors import safe_open

from harness import BF16, DRAFTER, LOG_DIR, QWEN3, Server, StartupError, dflash, gpu_used_mib, qwen36, text
from tiny_drafter import build
from workloads import (CODE_HEADER, CODE_MIN_PASS, CODE_STOP, CODE_TASKS, COPY_MIN_RATIO, NEEDLE, acceptance,
                       code_passes, copy_prompt, copy_ratio, counter_consistency, needle_prompt)

TINY = LOG_DIR / "tiny_drafter"
if not (TINY / "config.json").exists():
    build(TINY)

QWEN3_ARGS = ["--model-path", QWEN3, "--moe-backend", "offload", "--moe-cache-size", "1024", "--batching-policy", "legacy",
              "--num-tokens", "65536", "--max-running-requests", "4", "--attention-backend", "fi",
              "--cuda-graph-max-bs", "4", "--max-seq-len-override", "32768"]
REAL_ACCEPT_MIN = 0.5      # greedy copying with the matched real drafter
WINDOW_AB_TOLERANCE = 0.03  # |acceptance(compact) - acceptance(full)| for identical drafter math


def stored_weight_bytes(directory):
    with safe_open(str(Path(directory) / "model.safetensors"), "pt") as f:
        return 2 * sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())  # BF16 checkpoints


def rejections(checks):
    """Configurations that must fail before ready, with a reason naming the problem."""
    base = qwen36() + dflash(8)
    cases = {
        "triton_attention": (base + ["--attention-backend", "triton"], ["attention", "flashinfer", "fi"]),
        "kv_over_capacity": ([a if a != "65536" else "2000000" for a in base], ["gib", "exceed", "memory", "capacity"]),
        "moe_over_capacity": ([a if a != "2048" else "14000" for a in base], ["gib", "exceed", "memory", "capacity"]),
        "triton_attention_eager": (qwen36(graph=0) + dflash(8) + ["--attention-backend", "triton"],
                                   ["attention", "flashinfer"]),
        "cpu_experts": (qwen36(backend="cpu") + dflash(8), ["cpu", "expert"]),
        "cpu_expert_layers": (base + ["--moe-cpu-layers", "4"], ["cpu", "expert"]),
        "window_without_drafter": (qwen36() + ["--dflash-attention-window", "256"], ["window", "draft"]),
        "adaptive_under_layered": (qwen36(policy="layered-pipeline") + dflash(8, "--speculative-adaptive-cost"),
                                   ["adaptive", "legacy"]),
        "inwave_under_legacy": (base + ["--speculative-phase", "inwave"], ["phase", "legacy", "layered"]),
    }
    out = {}
    for name, (args, words) in cases.items():
        try:
            server = Server(f"reject_{name}", args)
        except StartupError as error:
            reasons = [line for line in error.log_tail.splitlines() if "Error" in line and "Warning" not in line][-2:]
            out[name] = reasons
            raw_oom = any("OutOfMemoryError" in line for line in reasons)
            ok = not raw_oom and any(w in "\n".join(reasons).lower() for w in words)
            if name.endswith("_ar"):
                checks.note(f"rejected_before_ready:{name}", clear_reason=ok, reasons=reasons)
            else:
                checks.check(f"rejected_before_ready:{name}", ok, reasons=reasons)
            continue
        server.close()
        checks.check(f"rejected_before_ready:{name}", False, reason="service reached serving state")
    return out


CONFIGS = {
    "rejections": rejections,
    "nvfp4_ar": qwen36() + ["--speculative-num-steps", "0"],
    # SD off ignores the draft path and DFlash-only settings
    "nvfp4_off_with_drafter": qwen36() + ["--speculative-num-steps", "0", "--speculative-draft-model-path", DRAFTER,
                                          "--dflash-attention-window", "256", "--no-dflash-compact-kv"],
    "nvfp4_lp_n8": qwen36(policy="layered-pipeline") + dflash(8),
    # representative hybrid config: default phase, draft-step count omitted (contract: 4 with a draft path)
    "nvfp4_hybrid": qwen36(policy="layered-pipeline", backend="hybrid")
    + ["--speculative-draft-model-path", DRAFTER],
    "nvfp4_n8": qwen36() + dflash(8),
    "nvfp4_n8_full": qwen36() + dflash(8, "--no-dflash-compact-kv"),
    "nvfp4_n8_noreplay": [a for a in qwen36(tokens=32768) if a != "--enable-gdn-replayssm"] + dflash(8),
    "nvfp4_n8_cap": qwen36() + dflash(8, "--dflash-attention-window", "512"),
    "nvfp4_n8_cap_full": qwen36() + dflash(8, "--dflash-attention-window", "512", "--no-dflash-compact-kv"),
    "bf16_n8": qwen36(BF16, moe=1024) + dflash(8),
    "qwen3_ar": QWEN3_ARGS,
    "qwen3_tiny": QWEN3_ARGS + ["--speculative-num-steps", "8", "--speculative-draft-model-path", str(TINY)],
}


def geometry(server, c):
    status, stats = server.status(), server.stats()
    g, spec = status["geometry"], stats["speculative"]
    drafter = server.flag("--speculative-draft-model-path")
    tokens = int(server.flag("--num-tokens"))
    c.check("explicit_kv_capacity_kept", g["num_pages"] * g["page_size"] == tokens, num_pages=g["num_pages"])
    c.check("explicit_moe_capacity_kept", g["moe_cache_size"] == int(server.flag("--moe-cache-size")),
            moe=g["moe_cache_size"])
    d = g.get("dflash") or {}
    components = {x["name"] for x in status["prefix_cache"].get("components", [])}
    if not drafter:
        c.check("no_dflash_without_drafter", not d.get("active") and spec.get("drafter") != "dflash"
                and not any("draft" in name for name in components), dflash=d, components=sorted(components))
        return {"geometry": g, "speculative": spec, "gpu_used_mib": gpu_used_mib()}
    window_slots = status["prefix_cache"].get("window_slots")  # only compact storage has a window slot pool
    c.check("dflash_active", d.get("active") is True, dflash=d)
    c.check("reserved_is_sum", d["reserved_bytes"] == d["weight_bytes"] + d["context_bytes"] + d["metadata_bytes"]
            + d["workspace_bytes"], dflash=d)
    c.check("context_is_full_plus_window", d["context_bytes"] == d["full_context_bytes"] + d["window_context_bytes"],
            dflash=d)
    c.check("weight_bytes_match_checkpoint", d["weight_bytes"] == stored_weight_bytes(drafter),
            reported=d["weight_bytes"], stored=stored_weight_bytes(drafter))
    compact = "--no-dflash-compact-kv" not in server.cmd
    cap = int(server.flag("--dflash-attention-window", 0))
    c.check("compact_and_cap_reported", d["compact_kv"] == compact and d["attention_window"] == cap,
            compact_kv=d["compact_kv"], attention_window=d["attention_window"])
    if cap == 0:
        c.check("full_layers_pay_full_history", tokens * d["full_token_bytes"] <= d["full_context_bytes"]
                <= (tokens + 64) * d["full_token_bytes"], full=d["full_context_bytes"], tokens=tokens)
    if compact:
        c.check("compact_window_bounded", d["window_context_bytes"] < tokens * d["window_token_bytes"],
                window=d["window_context_bytes"])
    else:
        c.check("full_storage_window_covers_capacity", d["window_context_bytes"] >= tokens * d["window_token_bytes"],
                window=d["window_context_bytes"])
    if compact:
        c.check("window_slots_consistent", d["window_slots"] == window_slots["total"]
                and d["window_free_slots"] == window_slots["free"], geometry=d, prefix_cache=window_slots)
    return {"geometry": g, "speculative": spec, "window_slots": window_slots, "gpu_used_mib": gpu_used_mib()}


def default_steps(server, c):
    spec = server.stats()["speculative"]
    c.check("omitted_steps_default_four", spec.get("enabled") and spec.get("max_draft_steps") == 4, speculative=spec)


def sd_off_ignores_drafter(server, c):
    status, spec = server.status(), server.stats()["speculative"]
    d = status["geometry"].get("dflash") or {}
    c.check("sd_off_ignores_drafter", not spec.get("enabled") and not d.get("active")
            and not any("draft" in x["name"] for x in status["prefix_cache"].get("components", [])),
            speculative_enabled=spec.get("enabled"), dflash=d)
    answer = text(server.complete(needle_prompt(), 12))
    c.check("sd_off_serves", NEEDLE in answer, text=answer)


def quality(server, c):
    """Self-checking tasks: code with unit tests, needle beyond the drafter window, copying."""
    drafter = server.flag("--speculative-draft-model-path")
    out, passes = {"code": [], "deltas": []}, 0
    for signature, name, cases in CODE_TASKS:
        resp, delta = server.measured(lambda: server.complete(CODE_HEADER + signature, 160, stop=CODE_STOP))
        ok = code_passes(signature, name, cases, text(resp))
        passes += ok
        out["code"].append({"name": name, "text": text(resp), "pass": ok})
        out["deltas"].append(delta)
    c.check("code_tasks_pass", passes >= CODE_MIN_PASS, passes=passes)
    resp, delta = server.measured(lambda: server.complete(needle_prompt(), 12))
    out["needle"], out["needle_prompt_tokens"] = text(resp), resp["body"]["usage"]["prompt_tokens"]
    out["deltas"].append(delta)
    c.check("needle_retrieved", NEEDLE in text(resp), text=text(resp))
    prompt, source = copy_prompt(3, 40)
    resp, delta = server.measured(lambda: server.complete(prompt, 1300))
    out["copy"], out["copy_delta"] = text(resp), delta
    ratio = copy_ratio(source, text(resp))
    c.check("copy_fidelity", ratio >= COPY_MIN_RATIO, ratio=ratio)
    if drafter:
        bad = [b for d in out["deltas"] + [delta] for b in counter_consistency(d, server.outwave)]
        c.check("counter_consistency", not bad, violations=bad)
        c.check("dflash_drafts_used", delta["draft_tokens"] > 0
                and server.stats()["speculative"].get("drafter") == "dflash", delta=delta)
        if Path(drafter) != TINY:
            c.check("copy_acceptance", acceptance(delta) >= REAL_ACCEPT_MIN, acceptance=acceptance(delta))
    return out


def long_window_ab(server, c):
    """Copying sources longer than the 4095-token native window, alone and as a concurrent pair."""
    group, out = uuid.uuid4().hex, {}
    for seed, sentences in ((11, 180), (12, 200)):
        prompt, source = copy_prompt(seed, sentences)
        resp, delta = server.measured(lambda: server.complete(prompt, 1000, group=group))
        out[f"single_{seed}"] = {"text": text(resp), "delta": delta, "prompt_tokens": resp["body"]["usage"]["prompt_tokens"],
                                 "ratio": copy_ratio(source, text(resp))}
    pair = [copy_prompt(13, 170), copy_prompt(14, 190)]
    resps, delta = server.measured(lambda: server.parallel(
        [lambda p=p: server.complete(p[0], 800, group=group) for p in pair]))
    out["pair"] = {"texts": [text(r) for r in resps], "delta": delta,
                   "ratios": [copy_ratio(s, text(r)) for (_, s), r in zip(pair, resps)]}
    ratios = [v["ratio"] for k, v in out.items() if k.startswith("single")] + out["pair"]["ratios"]
    c.check("long_copy_fidelity", min(ratios) >= COPY_MIN_RATIO, ratios=ratios)
    out["acceptance"] = {k: acceptance(v["delta"]) for k, v in out.items()}
    c.note("long_window_acceptance", **out["acceptance"])
    return out


def invalid_rebuild(server, c):
    before = server.status()["geometry"]
    resp = server.rebuild({"num_pages": -1})
    body = resp["body"]
    c.check("invalid_rebuild_rejected", resp["status"] == 503 and body.get("status") == "rejected"
            and "num_pages" in str(body.get("error")), response=resp)
    after = server.status()["geometry"]
    keys = ("num_pages", "moe_cache_size", "num_mamba_slots")
    budget = ("reserved_bytes", "context_bytes", "window_slots", "active")
    c.check("invalid_rebuild_keeps_resources", all(before[k] == after[k] for k in keys)
            and all(before["dflash"][k] == after["dflash"][k] for k in budget),
            before={k: before[k] for k in keys}, after={k: after[k] for k in keys},
            dflash=[{k: g["dflash"][k] for k in budget} for g in (before, after)])
    answer = text(server.complete(needle_prompt(), 12))
    c.check("serves_after_invalid_rebuild", NEEDLE in answer, text=answer)
    return body


def no_gdn(server, c):
    status = server.status()
    g = status["geometry"]
    c.check("no_gdn_pool", g["num_mamba_slots"] == 0 and not (g.get("gdn_replayssm") or {}).get("active"),
            num_mamba_slots=g["num_mamba_slots"])
    c.check("no_gdn_budget_flag_needed", "--gdn-state-budget-bytes" not in server.cmd and g["dflash"]["active"])
    c.check("tiny_native_window", g["dflash"]["window_tokens"] == 63, window_tokens=g["dflash"]["window_tokens"])
    resp = server.rebuild({"num_pages": 61440})
    after = server.status()["geometry"]
    c.check("no_gdn_kv_rebuild", resp["status"] == 200 and after["num_pages"] == 61440, response=resp)
    answer = text(server.complete(needle_prompt(), 12))
    c.check("no_gdn_serves_after_rebuild", NEEDLE in answer, text=answer)
    server.rebuild({"num_pages": 65536})
    return {"rebuild": resp["body"], "geometry": after}


# Sessions that repeat another session's whole scenario list under a different execution choice.
ALIASES = {"nvfp4_lp_n8": "nvfp4_n8"}

PLAN = {
    "nvfp4_off_with_drafter": [sd_off_ignores_drafter],
    "nvfp4_hybrid": [default_steps, quality],  # smoke
    "nvfp4_ar": [geometry, quality, long_window_ab],
    "nvfp4_n8": [geometry, quality, long_window_ab, invalid_rebuild],
    "nvfp4_n8_full": [geometry, quality, long_window_ab],
    "nvfp4_n8_noreplay": [geometry, quality],
    "nvfp4_n8_cap": [geometry, quality, long_window_ab],
    "nvfp4_n8_cap_full": [geometry, quality, long_window_ab],
    "bf16_n8": [quality],  # smoke
    "qwen3_ar": [geometry, quality],
    "qwen3_tiny": [no_gdn, quality],  # smoke
}


def _art(sessions, name, scenario):
    return sessions.get(name, {}).get("artifacts", {}).get(scenario)


def budget_independent(sessions, c):
    """Same explicit expert/KV capacities and GDN byte budget with and without DFlash; DFlash bytes are
    reported outside the GDN pool. GDN slots may differ only through the GDN pool's own SD workspace."""
    ar = _art(sessions, "nvfp4_ar", "geometry")
    for other in ("nvfp4_n8", "nvfp4_n8_full", "nvfp4_self_sd"):
        g = _art(sessions, other, "geometry")
        if not (ar and g):
            continue
        a, b = ar["geometry"], g["geometry"]
        ga, gb = a["gdn_replayssm"], b["gdn_replayssm"]
        c.check(f"explicit_capacities_equal:nvfp4_ar~{other}", a["num_pages"] == b["num_pages"]
                and a["moe_cache_size"] == b["moe_cache_size"] and ga["state_budget_bytes"] == gb["state_budget_bytes"]
                and gb["reserved_bytes"] <= gb["state_budget_bytes"], ar=ga, other=gb)
        c.note(f"gdn_slots:nvfp4_ar~{other}", slots=(a["num_mamba_slots"], b["num_mamba_slots"]),
               conv_workspace=(ga["conv_workspace_bytes"], gb["conv_workspace_bytes"]))


def window_independent_of_target_capacity(sessions, c):
    a, b = _art(sessions, "nvfp4_n8", "geometry"), _art(sessions, "nvfp4_n8_noreplay", "geometry")
    if a and b:
        da, db = a["geometry"]["dflash"], b["geometry"]["dflash"]
        c.check("window_bytes_independent_of_kv_capacity", da["window_context_bytes"] == db["window_context_bytes"],
                kv=(a["geometry"]["num_pages"], b["geometry"]["num_pages"]),
                window=(da["window_context_bytes"], db["window_context_bytes"]),
                full=(da["full_context_bytes"], db["full_context_bytes"]))


def compact_math_ab(sessions, c):
    """Native-window compact and full storage share math: drafter acceptance must agree."""
    a, b = _art(sessions, "nvfp4_n8", "long_window_ab"), _art(sessions, "nvfp4_n8_full", "long_window_ab")
    if a and b:
        diffs = {k: abs(a["acceptance"][k] - b["acceptance"][k]) for k in a["acceptance"]}
        c.check("compact_vs_full_acceptance", max(diffs.values()) <= WINDOW_AB_TOLERANCE,
                compact=a["acceptance"], full=b["acceptance"])


def cap_kept_without_compact(sessions, c):
    """The cap is drafter-only math: compact on/off with the same cap must agree, and the cap
    must be visible against the uncapped run (otherwise this comparison cannot show it was kept)."""
    a, b = _art(sessions, "nvfp4_n8_cap", "long_window_ab"), _art(sessions, "nvfp4_n8_cap_full", "long_window_ab")
    n = _art(sessions, "nvfp4_n8", "long_window_ab")
    if a and b:
        diffs = {k: abs(a["acceptance"][k] - b["acceptance"][k]) for k in a["acceptance"]}
        c.check("cap_compact_vs_full_acceptance", max(diffs.values()) <= WINDOW_AB_TOLERANCE,
                cap_compact=a["acceptance"], cap_full=b["acceptance"])
    if a and n:
        c.note("cap_vs_uncapped_acceptance", cap=a["acceptance"], uncapped=n["acceptance"])


def greedy_report(sessions, c):
    """Greedy text agreement with pure AR (report only; quality is enforced in-session)."""
    from harness import common_prefix
    ref = _art(sessions, "nvfp4_ar", "quality")
    for name, session in sessions.items():
        q = session.get("artifacts", {}).get("quality")
        if not ref or not q or name == "nvfp4_ar" or not name.startswith(("nvfp4", "bf16")):
            continue
        c.note(f"greedy_vs_ar:{name}", code=[common_prefix(x["text"], y["text"]) for x, y in zip(q["code"], ref["code"])],
               copy=common_prefix(q["copy"], ref["copy"]), copy_len=len(ref["copy"]))


COMPARE = [budget_independent, window_independent_of_target_capacity, compact_math_ab, cap_kept_without_compact,
           greedy_report]
