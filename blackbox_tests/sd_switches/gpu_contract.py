"""Public GPU checks for the SD switch contract; starts one server per case."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import httpx

TESTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TESTS / "sd_graph"), str(TESTS / "sd_concurrency")]
from benchmark_http import idle
from inputs import PROMPTS

PACKAGE = TESTS.parent / "python"
MODEL = "/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B"
REMOVED = ("adaptive_enabled", "adaptive_stops", "reuse_enabled", "reuse_changed_routes",
           "draft_expert_replacements")
PHASES = ("target_decode", "draft", "verify")
BUDGET = ["--batching-policy", "legacy", "--max-running-requests", "32", "--num-tokens", "4096",
          "--max-seq-len-override", "1024", "--max-prefill-length", "512", "--cache-type", "radix"]
OFFLOAD = ["--moe-backend", "offload", "--moe-cache-size", "1706"]
SD8_GRAPH = [*OFFLOAD, "--speculative-num-steps", "8", "--cuda-graph-max-bs", "32"]
ROUTER_COST = ["--speculative-draft-residency", "router", "--speculative-draft-load-missing",
               "--speculative-adaptive-cost"]
FUSED = ["--moe-backend", "fused", "--speculative-num-steps", "4"]
STEPS9 = [*OFFLOAD, "--speculative-num-steps", "9"]
CASES = {  # name: (expected outcome, options); fused leaves --cuda-graph-max-bs at its default
    "fused-graph": ("reject", FUSED),
    "fused-eager": ("eager", [*FUSED, "--cuda-graph-max-bs", "0"]),
    "steps9-graph": ("reject", [*STEPS9, "--cuda-graph-max-bs", "32"]),
    "steps9-eager": ("eager", [*STEPS9, "--cuda-graph-max-bs", "0"]),
    **{f"k{k}": ("cost-graph", [*SD8_GRAPH, *ROUTER_COST, "--speculative-draft-experts", str(k)])
       for k in (1, 5, 8)},
    "all-controls": ("cost-graph", [*SD8_GRAPH, *ROUTER_COST, "--speculative-verify-prefetch"]),
    "plain-sd": ("graph", [*SD8_GRAPH, "--speculative-draft-residency", "off"]),
}
# All BF16 experts (~57 GB) cannot be resident on one 24 GB RTX 4090, so a fused
# startup failure without the Graph rejection message is inconclusive, not a failure.
MAY_NOT_FIT = {"fused-graph", "fused-eager"}
WAVES = ((1, 16), (6, 24))  # (concurrent requests, max_tokens)


def wait_ready(process, client, timeout=900):
    """None once healthy, otherwise why startup did not succeed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return f"exit code {process.returncode}"
        with suppress(httpx.TransportError):
            health = client.get("/health", timeout=2).json()
            if health["status"] == "ok":
                return None
            if health["status"] == "error":
                return f"health error: {health.get('message')}"
        time.sleep(1)
    return "timeout"


def stop(process):
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def complete(client, index, limit):
    body = {"model": "sd-switches", "prompt": PROMPTS[index][1], "temperature": 0,
            "max_tokens": limit, "ignore_eos": True}
    response = client.post("/v1/completions", json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    return {"request": body, "text": data["choices"][0]["text"],
            "finish_reason": data["choices"][0]["finish_reason"], "usage": data["usage"]}


def workload(client):
    snapshots, responses, offset = [idle(client)], [], 0
    for count, limit in WAVES:
        with ThreadPoolExecutor(max_workers=count) as pool:
            responses += pool.map(lambda index: complete(client, index, limit), range(offset, offset + count))
        offset += count
        snapshots.append(idle(client))
    return snapshots, responses


def served_checks(kind, steps, snapshots, responses):
    first, last = snapshots[0]["speculative"], snapshots[-1]["speculative"]
    checks = {
        "removed_fields_absent": all("moe_residency" not in stats
                                     and not any(key in stats["speculative"] for key in REMOVED)
                                     for stats in snapshots),
        "outputs_complete": all(row["text"] and row["finish_reason"] == "length"
                                and row["usage"]["completion_tokens"] == row["request"]["max_tokens"]
                                for row in responses),
        "drafted_and_verified": last["enabled"] is True and last["draft_tokens"] > first["draft_tokens"]
                                and last["verify_steps"] > first["verify_steps"],
        "draft_ceiling": len(last.get("draft_length_histogram") or []) == steps + 1,
    }
    graphs = [stats["cuda_graph"] for stats in snapshots]
    if kind == "eager":
        checks["graphs_disabled"] = all(not graph["enabled"] and all(graph[p] == 0 for p in PHASES)
                                        for graph in graphs)
        return checks
    shapes = {(row["phase"], row["batch_size"]) for row in graphs[-1]["replay_shapes"] if row["replays"]}
    checks["graphs_enabled"] = all(graph["enabled"] for graph in graphs)
    # One verification round is one batched verification forward.
    checks["verify_rounds_replayed"] = all(stats["speculative"]["verify_steps"] == stats["cuda_graph"]["verify"]
                                           for stats in snapshots)
    checks["draft_verify_b1_and_over4"] = all((phase, 1) in shapes and any(p == phase and b > 4 for p, b in shapes)
                                              for phase in ("draft", "verify"))
    if kind == "cost-graph":
        checks["target_decode_replayed"] = graphs[-1]["target_decode"] > 0
        checks["cost_samples_equal_replays"] = all(
            stats["speculative"]["cost_samples"][sample] == stats["cuda_graph"][phase]
            for stats in snapshots for sample, phase in (("ar", "target_decode"), ("draft", "draft"), ("verify", "verify")))
    return checks


def run_case(name, gpu, output):
    kind, options = CASES[name]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [sys.executable, "-c", "from freetoken.cli import main; main()", "serve", "--model-path", MODEL,
               "--gpu", gpu, "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "sd-switches",
               *BUDGET, *options]
    log_path = output / f"{name}.log"
    record = {"name": name, "kind": kind, "command": command, "log": str(log_path)}
    failure = None
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env={**os.environ, "PYTHONPATH": str(PACKAGE)}, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=600) as client:
                failure = record["startup_failure"] = wait_ready(process, client)
                if failure is None and kind != "reject":
                    record["stats"], record["responses"] = workload(client)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            stop(process)
    text = log_path.read_text(errors="replace")
    record["log_tail"] = text.splitlines()[-30:]
    eager_hint = bool(re.search(r"--cuda-graph-max-bs[ =]0\b", text + str(failure)))
    if kind == "reject":
        record["checks"] = {"startup_rejected": failure not in (None, "timeout"), "names_eager_switch": eager_hint}
    elif failure is not None:
        record["checks"] = {"started": False, "not_graph_rejection": not eager_hint}
    elif "error" in record:
        record["checks"] = {"workload_completed": False}
    else:
        steps = int(options[options.index("--speculative-num-steps") + 1])
        record["checks"] = served_checks(kind, steps, record["stats"], record["responses"])
    passed = all(record["checks"].values())
    unfit = name in MAY_NOT_FIT and failure not in (None, "timeout") and not eager_hint
    record["status"] = "pass" if passed else "inconclusive" if unfit else "fail"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="GPU UUID or nvidia-smi index")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"gpu": args.gpu, "model": MODEL, "cases": {}}
    for name in args.cases:
        record = report["cases"][name] = run_case(name, args.gpu, args.output)
        (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        failed = [check for check, passed in record["checks"].items() if not passed]
        print(f"{record['status'].upper():<12} {name} {' '.join(failed)}", flush=True)
    failed = [name for name, record in report["cases"].items() if record["status"] == "fail"]
    report["passed"] = not failed
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("OVERALL", f"FAIL: {' '.join(failed)}" if failed else "PASS", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
