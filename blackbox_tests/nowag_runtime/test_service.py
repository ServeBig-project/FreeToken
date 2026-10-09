"""Real-service acceptance (contract §3, §5, §6, §7): execution modes, cache capacity, batching
policies, concurrency/tail batches/cancel, HTTP semantics, SD, TP2, DSV4, non-NoWAG
regression and paired performance. Needs GPU approval; TP2 and baseline runs need more.

Output comparisons follow the protocol frozen in harness.py.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from cases import (QWEN36_BASE, QWEN36_BF16, QWEN36_SIDE, DSV4_BASE, DSV4_SIDE,  # noqa: E402
                   DFLASH_DRAFT, need_gpu, need_path, need_tp2)
from harness import (BASELINE_SOURCE, LOG_DIR, PROMPTS, Server, StartupFailed, cross_path,  # noqa: E402
                     experts, loops, run_prompts, same_execution, task_ok, text)
from test_cache_status import sizes  # noqa: E402

QWEN_CACHE = os.environ.get("NOWAG_QWEN36_CACHE", "1536")
DSV4_CACHE = os.environ.get("NOWAG_DSV4_CACHE", "640")
CACHE_SIZES = os.environ.get("NOWAG_QWEN36_CACHE_SIZES", "1024,1536,4096").split(",")
DFLASH_ARGS = os.environ.get(
    "NOWAG_DFLASH_ARGS", "--speculative-draft-experts 3 --enable-gdn-replayssm "
                         "--gdn-state-budget-bytes 3000000000").split()


def qwen(*extra, backend="offload", cache=QWEN_CACHE):
    args = ["--model", need_path(QWEN36_BASE, "Qwen3.6 base"),
            "--nowag-expert-path", need_path(QWEN36_SIDE, "Qwen3.6 sidecar"),
            "--moe-backend", backend]
    if backend in ("offload", "hybrid"):
        args += ["--moe-cache-size", cache]
    return args + list(extra)


_REF = {}


def offload_reference():
    """Greedy outputs of the default NoWAG configuration (offload, eager defaults)."""
    if "qwen" not in _REF:
        with Server("svc_ref_offload", qwen(), need_gpu()) as s:
            _REF["qwen"] = run_prompts(s)
            assert task_ok(_REF["qwen"]), _REF["qwen"]
    return _REF["qwen"]


@pytest.mark.parametrize("backend", ["fused", "cpu", "hybrid"])
def test_execution_modes(backend):
    gpu = need_gpu()
    ref = offload_reference()
    side = sizes(QWEN36_SIDE)
    with Server(f"svc_mode_{backend}", qwen(backend=backend), gpu) as s:
        out, block = run_prompts(s), experts(s.status())
    cross_path(ref, out, f"offload vs {backend}")
    r = block["ranks"][0]
    if backend == "fused":
        assert r["expert_device_bytes"] >= 0.9 * side["compressed"]
        assert r["expert_device_bytes"] < 0.5 * side["bf16"]
    else:   # cpu / hybrid: no full BF16 expert copy anywhere
        assert r["expert_host_bytes"] < 0.5 * side["bf16"]


@pytest.mark.parametrize("cpu_layers", ["8", "0.5"])
def test_offload_with_cpu_layers(cpu_layers):
    """CPU expert compute for part of the layers (§9: accepted via the service only)."""
    gpu = need_gpu()
    with Server(f"svc_cpu_layers_{cpu_layers}", qwen("--moe-cpu-layers", cpu_layers), gpu) as s:
        cross_path(offload_reference(), run_prompts(s), f"--moe-cpu-layers {cpu_layers}")
        mixed_load(s)


@pytest.mark.parametrize("cache", CACHE_SIZES)
def test_cache_capacity(cache):
    gpu = need_gpu()
    with Server(f"svc_cache_{cache}", qwen(cache=cache), gpu) as s:
        cross_path(offload_reference(), run_prompts(s), f"cache {cache}")


@pytest.mark.parametrize("policy", ["legacy", "mixed", "layered", "joint", "layered-pipeline"])
def test_batching_policy(policy):
    """Each policy either works with the same weights or is refused before ready."""
    gpu = need_gpu()
    try:
        server = Server(f"svc_policy_{policy}", qwen("--batching-policy", policy), gpu)
    except StartupFailed as refused:
        pytest.skip(f"{policy} refused before ready (exit {refused.code}); record as rejection")
    with server as s:
        cross_path(offload_reference(), run_prompts(s), f"policy {policy}")
        mixed_load(s)


def mixed_load(s):
    """Concurrent long/short requests, natural tail batches (2,3,5) and a cancelled stream.
    Batched rows may break near-ties differently from solo runs, so at least half of the
    full-length rows must equal their solo text and none may loop; short rows must stop at
    their own max_tokens."""
    prompts = [p for p, _ in PROMPTS]
    solo = s.greedy(prompts, 24)
    for width in (2, 3, 5):
        lengths = [24 if i < 4 else 7 for i in range(width)]
        outs = s.parallel([lambda i=i, n=n: s.complete(prompts[i % 4], n)
                           for i, n in enumerate(lengths)])
        full = [(text(r), solo[i % 4]) for i, r in enumerate(outs) if lengths[i] == 24]
        assert sum(a == b for a, b in full) * 2 >= len(full), (width, full)
        assert not any(loops(a) for a, _ in full), full
        for r, n in zip(outs, lengths):
            if n == 7:
                assert r["body"]["usage"]["completion_tokens"] <= 7
    cut = s.stream("Write a long story about a lighthouse keeper.", 200, cancel_after=3)
    assert len(cut["chunks"]) == 3
    assert s.alive()
    cross_path(solo, s.greedy(prompts, 24), "after cancel")


def test_concurrency_tail_batches_and_cancel():
    gpu = need_gpu()
    with Server("svc_concurrency", qwen(), gpu) as s:
        mixed_load(s)


def test_http_semantics():
    gpu = need_gpu()
    with Server("svc_http", qwen(), gpu) as s:
        r = s.complete("The capital of France is", 5)
        body = r["body"]
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["usage"]["completion_tokens"] == 5
        r = s.complete("The capital of France is", 64, stop=["."])
        assert r["body"]["choices"][0]["finish_reason"] == "stop" and "." not in text(r)
        streamed = s.stream("The capital of France is", 16)        # same (warm) cache state
        assert streamed["done"] and streamed["text"] == text(s.complete("The capital of France is", 16))
        sampled = s.complete("Once upon a time", 16, temperature=0.8)
        assert sampled["status"] == 200 and text(sampled)


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_self_speculative(steps):
    gpu = need_gpu()
    with Server(f"svc_selfsd_{steps}", qwen("--speculative-num-steps", steps), gpu) as s:
        cross_path(offload_reference(), run_prompts(s), f"self-SD N={steps}")
        mixed_load(s)


@pytest.mark.parametrize("steps", [2, 8])
def test_dflash_speculative(steps):
    gpu = need_gpu()
    draft = need_path(DFLASH_DRAFT, "DFlash draft for Qwen3.6 (NOWAG_DFLASH_DRAFT)")
    args = qwen("--speculative-num-steps", steps, "--speculative-draft-model-path", draft,
                *DFLASH_ARGS)
    with Server(f"svc_dflash_{steps}", args, gpu) as s:
        cross_path(offload_reference(), run_prompts(s), f"DFlash N={steps}")
        mixed_load(s)


def test_tp2_native():
    gpus = need_tp2()
    with Server("svc_tp2", qwen("--tensor-parallel-size", 2), gpus) as s:
        out, ranks = run_prompts(s), experts(s.status())["ranks"]
        mixed_load(s)
    assert sorted(r["rank"] for r in ranks) == [0, 1]
    assert len({str(r["device"]) for r in ranks}) == 2
    cross_path(offload_reference(), out, "TP1 vs TP2")


@pytest.mark.parametrize("backend", ["offload", "cpu", "hybrid"])
def test_dsv4_service(backend):
    gpu = need_gpu()
    args = ["--model", need_path(DSV4_BASE, "DSV4 base"),
            "--nowag-expert-path", need_path(DSV4_SIDE, "DSV4 sidecar"), "--moe-backend", backend]
    if backend != "cpu":
        args += ["--moe-cache-size", DSV4_CACHE]
    try:
        server = Server(f"svc_dsv4_{backend}", args, gpu)
    except StartupFailed as refused:
        if backend == "offload":
            raise
        pytest.skip(f"DSV4 {backend} refused before ready (exit {refused.code})")
    with server as s:
        out = run_prompts(s)
        assert task_ok(out), out
        mixed_load(s)


# ------------------------------------------------------------------ non-NoWAG regression

REGRESSION = {"qwen36_nvfp4": lambda: [QWEN36_BASE, "offload", QWEN_CACHE],
              "qwen36_bf16": lambda: [QWEN36_BF16, "offload", QWEN_CACHE]}


@pytest.mark.parametrize("config", sorted(REGRESSION))
def test_non_nowag_matches_baseline(config):
    gpu = need_gpu()
    if not BASELINE_SOURCE:
        pytest.skip("NOWAG_BASELINE_SOURCE (python/ of the pre-feature baseline) unset")
    model, backend, cache = REGRESSION[config]()
    args = ["--model", need_path(model, config), "--moe-backend", backend, "--moe-cache-size", cache]
    with Server(f"reg_base_{config}", args, gpu, source=BASELINE_SOURCE) as s:
        base_out, base_geo = run_prompts(s), s.status().get("geometry", {})
    with Server(f"reg_cand_{config}", args, gpu) as s:
        cand_out, cand_geo = run_prompts(s), s.status().get("geometry", {})
    same_execution(base_out, cand_out, f"{config} baseline vs candidate")
    cand_geo = {k: v for k, v in cand_geo.items() if k != "experts"}
    assert cand_geo == {k: v for k, v in base_geo.items() if k != "experts"}


def test_default_backend_unchanged_without_nowag():
    gpu = need_gpu()
    if not BASELINE_SOURCE:
        pytest.skip("NOWAG_BASELINE_SOURCE unset")
    args = ["--model", need_path(QWEN36_BASE, "Qwen3.6 base")]       # no backend, no cache flags
    with Server("reg_default_base", args, gpu, source=BASELINE_SOURCE) as s:
        base = (run_prompts(s), s.status().get("geometry", {}))
    with Server("reg_default_cand", args, gpu) as s:
        cand = (run_prompts(s), s.status().get("geometry", {}))
    same_execution(base[0], cand[0], "default backend")
    assert {k: v for k, v in cand[1].items() if k != "experts"} == \
        {k: v for k, v in base[1].items() if k != "experts"}


# ------------------------------------------------------------------ paired performance
# Three interleaved baseline/candidate blocks, same weights/device/work; gate: candidate
# median TTFT and end-to-end within +5% of baseline, decode throughput within -5%.

PERF = {"dsv4_nowag_offload": lambda: ["--model", need_path(DSV4_BASE, "DSV4 base"),
                                       "--nowag-expert-path", need_path(DSV4_SIDE, "DSV4 sidecar"),
                                       "--moe-backend", "offload", "--moe-cache-size", DSV4_CACHE],
        "qwen36_nvfp4_offload": lambda: ["--model", need_path(QWEN36_BASE, "Qwen3.6 base"),
                                         "--moe-backend", "offload", "--moe-cache-size", QWEN_CACHE]}


def measure(s, tokens=128):
    s.greedy(["warm up the server please"], 8)
    rows = []
    for p, _ in PROMPTS:
        r = s.stream(p, tokens)
        n = max(len(r["chunks"]) - 1, 1)
        rows.append({"ttft": r["ttft"], "e2e": r["seconds"],
                     "decode_tps": n / max(r["seconds"] - r["ttft"], 1e-9)})
    return {k: statistics.median(row[k] for row in rows) for k in rows[0]}


@pytest.mark.parametrize("config", sorted(PERF))
def test_paired_performance(config):
    gpu = need_gpu()
    if not BASELINE_SOURCE:
        pytest.skip("NOWAG_BASELINE_SOURCE unset")
    args = PERF[config]()
    blocks = []
    for i in range(3):
        block = {}
        for side, source in (("baseline", BASELINE_SOURCE), ("candidate", None)):
            try:
                with Server(f"perf_{config}_{side}_{i}", args, gpu, source=source) as s:
                    block[side] = measure(s)
            except StartupFailed:
                if side == "baseline":
                    pytest.skip(f"{config} not supported before the feature; report absolute values")
                raise
        blocks.append(block)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"perf_{config}_{time.strftime('%H%M%S')}.json").write_text(json.dumps(blocks, indent=1))
    med = {side: {k: statistics.median(b[side][k] for b in blocks) for k in blocks[0][side]}
           for side in ("baseline", "candidate")}
    b, c = med["baseline"], med["candidate"]
    assert c["ttft"] <= 1.05 * b["ttft"], med
    assert c["e2e"] <= 1.05 * b["e2e"], med
    assert c["decode_tps"] >= 0.95 * b["decode_tps"], med
