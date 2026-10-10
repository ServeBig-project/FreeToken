"""One public maintenance transaction transfers a feasible budget from runtime to experts.

The public fields and nvidia-smi observation were supplied by the coordinator. Capacities
are selected once from those observations, without probing allocations until they fail.
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from cases import need_gpu
from harness import LOG_DIR, Server, cross_path, experts, get, post, run_prompts, text
from test_service import qwen
from test_service_paths import expert_count

GIB = 1 << 30
OPTIONS = ["--batching-policy", "legacy", "--cuda-graph-max-bs", 0,
           "--max-running-requests", 1, "--max-seq-len-override", 4096,
           "--max-prefill-length", 256, "--memory-ratio", 0.9]


def device_memory(gpu):
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.total,memory.free",
         "--format=csv,noheader,nounits"], text=True)
    total, free = (int(value.strip()) * (1 << 20) for value in output.strip().split(","))
    return {"total_bytes": total, "free_bytes": free}


def wait_idle(server):
    deadline = time.monotonic() + 30
    while get(server.url, "/v1/stats")["body"]["requests"]["active"]:
        assert time.monotonic() < deadline, "completed requests remained active"
        time.sleep(0.1)


def test_shrink_runtime_and_grow_experts_in_one_rebuild():
    gpu = need_gpu()
    slots = 2 * expert_count()
    evidence = {}
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    def record(name, value):
        evidence[name] = value
        (LOG_DIR / "rebuild-shared-exchange.json").write_text(json.dumps(evidence, indent=2))

    with Server("exchange_calibration", qwen(*OPTIONS, "--runtime-cache-gib", 4,
                                             cache=str(slots)), gpu) as server:
        assert text(server.complete("The capital of France is", 8))
        wait_idle(server)
        calibration, memory = server.status(), device_memory(gpu)
        record("calibration", {"status": calibration, "device": memory})
    geometry, runtime = calibration["geometry"], calibration["prefix_cache"]["runtime"]
    if not 23 * GIB <= memory["total_bytes"] <= 25 * GIB:
        pytest.skip("this pressure scenario requires the assigned 24 GiB GPU")
    expert_bytes = experts(calibration)["ranks"][0]["expert_device_bytes"]
    granularity = runtime["granularity_bytes"]
    limit = min(geometry["cache_budget_bytes"] - expert_bytes - GIB // 2,
                geometry["runtime_cache_bytes"] + memory["free_bytes"] - 3 * GIB)
    old_runtime = limit // granularity * granularity
    if old_runtime <= 4 * GIB:
        pytest.skip("insufficient available capacity for the planned runtime-to-expert exchange")

    with Server("exchange_shared_pools", qwen(*OPTIONS, "--runtime-cache-gib", old_runtime / GIB,
                                              cache=str(slots)), gpu) as server:
        baseline = run_prompts(server)
        wait_idle(server)
        before, memory = server.status(), device_memory(gpu)
        geometry, runtime = before["geometry"], before["prefix_cache"]["runtime"]
        granularity = runtime["granularity_bytes"]
        rank = experts(before)["ranks"][0]
        old_runtime = geometry["runtime_cache_bytes"]
        old_slots, layer_experts = geometry["moe_cache_size"], geometry["num_experts"]
        unit = geometry["unit_bytes"]["moe_per_expert"]
        extra_slots = math.ceil((memory["free_bytes"] + 2 * GIB) / (unit * layer_experts)) * layer_experts
        new_slots = old_slots + extra_slots
        growth = extra_slots * unit
        new_runtime = (old_runtime - growth - GIB // 2) // granularity * granularity
        record("before", {"status": before, "device": memory})
        target = {"mode": "if_idle", "runtime_cache_gib": new_runtime / GIB, "moe_cache_size": new_slots}
        record("target", {"request": target, "expert_growth_bytes": growth,
                          "predicted_pool_bytes": new_runtime + rank["expert_device_bytes"] + growth})
        if (new_slots > layer_experts * geometry["num_moe_layers"] or new_runtime < 4 * GIB
                or old_runtime + rank["expert_device_bytes"] < 0.9 * geometry["cache_budget_bytes"]):
            pytest.skip("the observed resources cannot provide this safe, near-budget pressure case; see evidence")
        assert runtime["budget_bytes"] == old_runtime
        assert growth > memory["free_bytes"]
        assert new_runtime + rank["expert_device_bytes"] + growth < geometry["cache_budget_bytes"]

        result = post(server.url, "/v1/cache/rebuild", target)
        record("rebuild", result)
        assert result["status"] == 200, result
        after_output = run_prompts(server)
        cross_path(baseline, after_output, "single runtime-shrink/expert-grow rebuild")
        wait_idle(server)
        after = server.status()
        record("after", after)
        assert after["geometry"]["moe_cache_size"] == new_slots
        assert after["geometry"]["runtime_cache_bytes"] == new_runtime
        assert after["prefix_cache"]["runtime"]["budget_bytes"] == new_runtime
        assert experts(after)["ranks"][0]["expert_device_bytes"] == rank["expert_device_bytes"] + growth

        invalid = {"mode": "if_idle", "runtime_cache_gib": (after["geometry"]["cache_budget_bytes"] + GIB) / GIB,
                   "moe_cache_size": new_slots}
        rejected = post(server.url, "/v1/cache/rebuild", invalid)
        record("invalid_rebuild", {"request": invalid, "response": rejected})
        assert rejected["status"] == 503 and rejected["body"]["status"] == "rejected", rejected
        preserved = server.status()["geometry"]
        for key in ("moe_cache_size", "runtime_cache_bytes", "experts"):
            assert preserved[key] == after["geometry"][key], key
        cross_path(after_output, run_prompts(server), "service preserved after infeasible target")
