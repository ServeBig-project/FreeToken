"""Candidate-only checks: reported configuration, cold prefix restore, executed paths, lifecycle.

Field paths are the ones /v1/stats and /v1/cache/status actually serve (see view.at).
"""
import math
import time

import pytest

from . import limits
from .client import cached
from .scenarios import _single, compare, wait_idle
from .view import at, flat

__all__ = ["test_00_effective_configuration", "test_00_dense_bytes_reported", "test_12_cold_prefix",
           "test_13_paths_executed", "test_14_idle_lifecycle"]


def requested(se, flag):
    a = se.cfg["args"]
    return a[a.index(flag) + 1] if flag in a else "auto"


def test_00_effective_configuration(se):
    """Requested/effective precision, batching, Graph, Replay and prefix-cache budget follow the flags."""
    s, cs, fails = se.rec["stats_ready"], se.rec["cache_ready"], []
    want = {
        "execution.requested.dense_quant": requested(se, "--dense-quant"),
        "execution.requested.kv_dtype": requested(se, "--kv-dtype"),
        "execution.effective.dense_quant": se.cfg["dense"],  # auto -> source BF16, whatever the dir name
        "execution.effective.kv_dtype": se.cfg["kv"],
        "cuda_graph.enabled": se.cfg["graph"],
        "gdn_replayssm.active": se.cfg["replay"],
    }
    for path, v in want.items():
        got = at(s, path)
        if got != v:
            fails.append(f"{path} = {got!r}, expected {v!r}")
    eff_batch = at(s, "execution.effective.batching_policy")
    if se.cfg["batching"] not in eff_batch:
        fails.append(f"effective batching {eff_batch!r}, expected {se.cfg['batching']}")
    sizes = at(s, "execution.effective.cuda_graph.batch_sizes")
    if bool(sizes) != se.cfg["graph"]:
        fails.append(f"Graph batch sizes {sizes} with Graph {'on' if se.cfg['graph'] else 'off'}")
    pc = at(cs, "prefix_cache")
    if pc["enabled"] != se.cfg["radix"]:
        fails.append(f"prefix_cache.enabled {pc['enabled']} with radix={se.cfg['radix']}")
    host = se.cfg["host"] * 2 ** 30
    if pc["host_budget_bytes"] != host or pc["host_allocated_bytes"] > host:
        fails.append(f"host budget/allocated {pc['host_budget_bytes']}/{pc['host_allocated_bytes']} != {host}")
    geo = at(cs, "geometry")
    se.rec["resources"] = dict(kv_per_token=geo["unit_bytes"]["kv_per_token"], moe_cache_size=geo["moe_cache_size"],
                               kv_tokens=geo["num_pages"] * geo["page_size"], mamba_slots=geo["num_mamba_slots"])
    assert not fails, fails


def test_00_dense_bytes_reported(se):
    """Contract 7: actual dense bytes are observable (stats / cache status)."""
    keys = {k: v for src in ("stats_ready", "cache_ready") for k, v in flat(se.rec[src]).items()
            if "dense" in k.lower() and "byte" in k.lower()}
    se.rec["dense_bytes"] = keys
    assert keys, "no /v1/stats or /v1/cache/status field reports actual dense bytes"


def prefix(se):
    return se.c.get("/v1/cache/status")["prefix_cache"]


def drained(se, timeout=30):
    end = time.time() + timeout
    while True:
        p = prefix(se)
        if p["host_inflight_bytes"] == 0 or time.time() > end:
            return p


DELTA = ("h2d_bytes", "d2h_bytes", "gpu_reused_tokens", "host_reused_tokens", "recomputed_tokens")


def test_12_cold_prefix(se):
    """Hot hit, GPU pressure beyond the KV capacity, re-access: restore from host (H>0) or recompute (H=0)."""
    if not se.cfg["radix"]:
        pytest.skip("naive cache keeps no prefixes")
    p, m, chk = _single(se)["needle_3k"]
    first = se.c.complete(p, m, group="cold")
    hot = se.c.complete(p, m, group="cold")
    geo = se.rec["cache_ready"]["geometry"]
    cap = geo["num_pages"] * geo["page_size"]
    n = math.ceil(limits.PRESSURE_FACTOR * cap / 3000)
    for i in range(n):
        se.c.complete(se.tok.chat([{"role": "user", "content": se.tok.prose(3000, 500 + i)}]), 1, group="cold")
    before = drained(se)
    again = se.c.complete(p, m, group="cold")
    after = drained(se)
    d = {k: after[k] - before[k] for k in DELTA}
    pt = again["usage"]["prompt_tokens"]
    se.rec["cold"] = dict(first=first, hot=hot, again=again, pressure_prompts=n, delta=d,
                          cached=[cached(first), cached(hot), cached(again)], after=after)
    fails = []
    if not chk(again["text"]):
        fails.append(f"re-access after pressure wrong: {again['text']!r}")
    if cached(hot) <= 0:
        fails.append("immediate repeat did not reuse the prefix")
    if cached(again) == cached(hot) and again["text"] != hot["text"]:
        fails.append(f"same reuse boundary, different output: {again['text']!r} vs {hot['text']!r}")
    compare(se, "cold:again_vs_hot", again["text"], hot["text"], chk, fails)
    if d["gpu_reused_tokens"] + d["host_reused_tokens"] != cached(again):
        fails.append(f"reused tokens {d} != cached_tokens {cached(again)}")
    if cached(again) > pt:
        fails.append(f"cached {cached(again)} > prompt {pt}")
    if se.cfg["host"] > 0:
        if d["h2d_bytes"] <= 0 or d["host_reused_tokens"] <= 0:
            fails.append(f"host budget {se.cfg['host']} GiB but no restore from host after pressure: {d}")
    elif d["h2d_bytes"] or d["host_reused_tokens"] or after["host_used_bytes"]:
        fails.append(f"host budget 0 but host traffic/usage: {d}, host_used={after['host_used_bytes']}")
    assert not fails, fails


def test_13_paths_executed(se):
    """Graph sessions replay decode graphs (incl. batch > 1); eager sessions none; Replay runs when on."""
    s, fails = se.c.get("/v1/stats"), []
    g, r = at(s, "cuda_graph"), at(s, "gdn_replayssm")
    se.rec["paths"] = dict(cuda_graph=g, gdn_replayssm=r)
    shapes = [x for x in g.get("replay_shapes", []) if x.get("phase") == "target_decode" and x.get("replays")]
    if se.cfg["graph"]:
        if g["target_decode"] <= 0 or not any(x["batch_size"] > 1 for x in shapes):
            fails.append(f"Graph on but decode graph replays missing: {g}")
    elif g["target_decode"] or shapes:
        fails.append(f"Graph off (--cuda-graph-max-bs 0) but graph replays reported: {g}")
    if se.cfg["replay"]:
        if r["ar_tokens"] <= 0:
            fails.append(f"Replay on but no AR tokens went through it: {r}")
    elif r.get("active") or r.get("ar_tokens"):
        fails.append(f"Replay off but active: {r}")
    assert not fails, fails


def test_14_idle_lifecycle(se):
    """After all work and cancellations: no active request, copies drained, host use within budget."""
    ok, last = wait_idle(se)
    s, pc = se.c.get("/v1/stats"), drained(se)
    se.rec["idle"] = dict(requests=last, kv=s.get("kv"), mamba=s.get("mamba"), prefix_cache=pc)
    assert ok, f"not idle: {last}"
    assert pc["host_inflight_bytes"] == 0, pc
    assert pc["host_used_bytes"] <= pc["host_allocated_bytes"] <= max(pc["host_budget_bytes"], 0), pc
    single = _single(se)
    p, m, chk = single["capital"]
    r = se.c.complete(p, m, group="after-idle")
    assert chk(r["text"]), r
