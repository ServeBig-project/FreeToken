"""Public cache-capacity, transfer, maintenance and existing SD-control paths.

Capacity formulas were supplied by the coordinator on 2026-10-09. Lifecycle behavior
also follows docs/runtime-pool-public-contract.md and docs/dflash-public-contract.md.
No candidate execution is allowed without the existing GPU approval gate.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sidecar as S
from cases import DFLASH_DRAFT, QWEN36_SIDE, need_gpu, need_path
from harness import LOG_DIR, Server, cross_path, expect_rejected, experts, get, post, run_prompts, text
from test_service import check_sd_observed, mixed_load, offload_reference, qwen


CAPACITIES = [
    ("legacy", True, 2), ("legacy", False, 1),
    ("mixed", True, 2), ("mixed", False, 1),
    ("layered", True, 3), ("layered-pipeline", True, 2),
]


def expert_count():
    return S.manifest(need_path(QWEN36_SIDE, "Qwen3.6 sidecar"))["num_experts"]


@pytest.mark.parametrize("policy,overlap,layers", CAPACITIES)
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_minimum_cache_capacity(policy, overlap, layers, delta):
    gpu = need_gpu()
    count = layers * expert_count() + delta
    options = ["--batching-policy", policy, "--prefill-layer-group-size", 1]
    if not overlap:
        options += ["--disable-moe-prefill-overlap"]
    if policy == "layered-pipeline":
        options += ["--attention-backend", "triton,fi"]
    args = qwen(*options, cache=str(count))
    label = f"minimum_{policy}_{overlap}_{count}"
    if delta < 0:
        expect_rejected(label, args, gpu)
    else:
        reference = offload_reference()
        with Server(label, args, gpu) as server:
            cross_path(reference, run_prompts(server), label)
            mixed_load(server)


@pytest.mark.parametrize("path", ["streaming-overlap", "streaming-serial", "hit-d2d", "layered-concurrent"])
def test_prefill_transfer_paths(path):
    gpu = need_gpu()
    reference = offload_reference()
    options = ["--batching-policy", "layered" if path == "layered-concurrent" else "legacy"]
    if path == "streaming-serial":
        options += ["--disable-moe-prefill-overlap"]
    if path == "hit-d2d":
        import torch
        if tuple(map(int, torch.version.cuda.split(".")[:2])) < (13, 0):
            pytest.skip("public hit-D2D path requires CUDA >= 13")
        options += ["--moe-prefill-hit-d2d"]
    if path == "layered-concurrent":   # concurrent layered prefill needs triton attention
        options += ["--prefill-execution", "concurrent", "--attention-backend", "triton"]
    with Server(f"transfer_{path}", qwen(*options), gpu) as server:
        cross_path(reference, run_prompts(server), path)
        mixed_load(server)
        mixed_load(server)  # same workload with populated expert/prefix caches


@pytest.mark.parametrize("shared", [False, True])
def test_idle_rebuild_and_rejected_rebuild_keep_serving(shared):
    gpu = need_gpu()
    options = ["--batching-policy", "legacy"]
    if shared:
        options += ["--runtime-cache-gib", 4]
    with Server(f"rebuild_{shared}", qwen(*options), gpu) as server:
        baseline = run_prompts(server)
        initial = experts(server.status())["ranks"][0]
        for slots in (2 * expert_count(), 3 * expert_count()):
            result = post(server.url, "/v1/cache/rebuild", {"mode": "if_idle", "moe_cache_size": slots})
            assert result["status"] == 200, result
            current = experts(server.status())["ranks"][0]
            assert current["shared_device_bytes"] == initial["shared_device_bytes"]
            assert current["workspace_device_bytes"] > 0
            cross_path(baseline, run_prompts(server), f"rebuild slots={slots}")
        before = server.status()["geometry"]
        result = post(server.url, "/v1/cache/rebuild", {"mode": "if_idle", "moe_cache_size": 0})
        assert result["status"] == 503 and result["body"]["status"] == "rejected", result
        assert server.status()["geometry"] == before
        cross_path(baseline, run_prompts(server), "after invalid rebuild")


def test_cache_group_isolation_and_conversation_continuation():
    gpu = need_gpu()
    prompt = ("A library keeps books on labelled shelves. " * 64) + "\nThe capital of France is"
    with Server("cache_groups", qwen("--batching-policy", "legacy"), gpu) as server:
        cold = server.complete(prompt, 16, cache_group="nowag-a")
        warm = server.complete(prompt, 16, cache_group="nowag-a")
        isolated = server.complete(prompt, 16, cache_group="nowag-b")
        cached = lambda r: r["body"]["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        assert cached(cold) == 0 and cached(warm) > 0 and cached(isolated) == 0
        cross_path([text(cold)], [text(isolated)], "group isolation", task=False)
        continuation = prompt + text(warm) + "\nThe chemical symbol for gold is"
        results = [server.complete(continuation, 16, cache_group=group)
                   for group in ("nowag-a", "nowag-b")]
        cross_path([text(r) for r in results[:1]], [text(r) for r in results[1:]],
                   "conversation continuation", task=False)


@pytest.mark.parametrize("control", ["resident", "load-missing", "verify-prefetch", "adaptive"])
def test_existing_self_sd_controls(control):
    gpu = need_gpu()
    reference = offload_reference()
    options = ["--batching-policy", "legacy", "--speculative-num-steps", 4,
               "--speculative-draft-experts", 3]
    if control in ("resident", "load-missing"):
        options += ["--speculative-draft-residency", "router"]
    if control == "load-missing":
        options += ["--speculative-draft-load-missing"]
    if control == "verify-prefetch":
        options += ["--speculative-verify-prefetch"]
    if control == "adaptive":
        options += ["--speculative-adaptive-cost"]
    with Server(f"sd_control_{control}", qwen(*options), gpu) as server:
        before = get(server.url, "/v1/stats")["body"]
        cross_path(reference, run_prompts(server), control)
        mixed_load(server)
        check_sd_observed(server, before)


@pytest.mark.parametrize("phase,backend", [("outwave", "offload"), ("inwave", "hybrid")])
def test_layered_sd_phases(phase, backend):
    gpu = need_gpu()
    reference = offload_reference()
    # SD CUDA Graph needs FlashInfer attention; with triton,fi the public rule is eager SD
    options = ["--batching-policy", "layered-pipeline", "--attention-backend", "triton,fi",
               "--speculative-num-steps", 4, "--speculative-phase", phase, "--cuda-graph-max-bs", 0]
    with Server(f"layered_sd_self_{phase}_{backend}", qwen(*options, backend=backend), gpu) as server:
        before = get(server.url, "/v1/stats")["body"]
        cross_path(reference, run_prompts(server), phase)
        mixed_load(server)
        check_sd_observed(server, before)
        assert get(server.url, "/v1/stats")["body"]["execution"]["effective"]["batching_policy"] == "layered-pipeline"


@pytest.mark.parametrize("policy,attention", [("legacy", "fi"), ("layered-pipeline", "triton,fi")])
def test_dflash_phase_all_rejected(policy, attention):
    """No public combination exists: phase "all" needs layered-pipeline batching, which needs
    Triton prefill, while DFlash accepts only fi attention. Each combination must be refused
    before ready (DFlash serving is covered by test_dflash_speculative)."""
    gpu = need_gpu()
    options = ["--batching-policy", policy, "--attention-backend", attention, "--page-size", 1,
               "--speculative-num-steps", 4, "--speculative-phase", "all",
               "--speculative-draft-model-path", need_path(DFLASH_DRAFT, "Qwen3.6 DFlash"),
               "--enable-gdn-replayssm", "--gdn-state-budget-bytes", 3000000000]
    expect_rejected(f"dflash_all_{policy}", qwen(*options, backend="hybrid"), gpu)


def test_layered_sd_with_graph_and_triton_attention_rejected():
    """Existing SD restriction (contract §3): SD CUDA Graph requires FlashInfer attention, so
    triton,fi with graphs left on is refused before ready."""
    gpu = need_gpu()
    expect_rejected("layered_sd_graph_triton", qwen(
        "--batching-policy", "layered-pipeline", "--attention-backend", "triton,fi",
        "--speculative-num-steps", 4, "--speculative-phase", "outwave"), gpu)


@pytest.mark.parametrize("graph", [0, 16])
def test_service_graph_configuration_and_tails(graph):
    gpu = need_gpu()
    reference = offload_reference()
    with Server(f"service_graph_{graph}", qwen("--batching-policy", "legacy",
                                              "--cuda-graph-max-bs", graph), gpu) as server:
        before = get(server.url, "/v1/stats")["body"]
        cross_path(reference, run_prompts(server), f"graph {graph}")
        mixed_load(server)
        stats = get(server.url, "/v1/stats")["body"]
        assert stats["cuda_graph"]["enabled"] == bool(graph)
        replays = stats["cuda_graph"]["target_decode"] - before["cuda_graph"]["target_decode"]
        assert (replays > 0) if graph else (replays == 0)
        assert stats["execution"]["effective"]["batching_policy"] == "legacy"
        (LOG_DIR / f"{server.label}-stats.json").write_text(json.dumps(stats, indent=2))


def test_busy_rebuild_is_refused_and_cancel_releases_requests():
    gpu = need_gpu()
    with Server("rebuild_busy", qwen("--batching-policy", "legacy"), gpu) as server:
        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(server.stream, "List the integers from one to one thousand in order.",
                                  1024, cancel_after=16)
            deadline = time.monotonic() + 30
            while get(server.url, "/v1/stats")["body"]["requests"]["active"] == 0:
                assert not request.done() and time.monotonic() < deadline, "never observed an active request"
                time.sleep(0.05)
            # a normal timeout: timeout 0 returns 504 before the scheduler's busy reply arrives
            result = post(server.url, "/v1/cache/rebuild", {"mode": "if_idle", "timeout": 10,
                                                            "moe_cache_size": 3 * expert_count()})
            assert result["body"]["status"] == "busy", result
            cancelled = request.result(timeout=900)
            assert len(cancelled["chunks"]) == 16 and not cancelled["done"]
        deadline = time.monotonic() + 30
        while get(server.url, "/v1/stats")["body"]["requests"]["active"]:
            assert time.monotonic() < deadline, "cancelled request remained active"
            time.sleep(0.1)
        assert text(server.complete("The capital of France is", 16))
