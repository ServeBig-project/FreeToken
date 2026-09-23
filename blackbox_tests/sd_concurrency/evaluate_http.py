"""Frozen high-concurrency public HTTP workload; never starts a model server."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sd_graph"))
from benchmark_http import FROZEN, TASKS, coding_prompt, idle, stats_delta, stream
from collect_http import shape_counts
from inputs import PROMPTS

CONCURRENCIES = (4, 8, 16, 32)
TAIL_LENGTHS = (1, 2, 3, 4, 5, 7, 17)


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("ar", "off", "router"), required=True)
    parser.add_argument("--execution", choices=("eager", "graph"), required=True)
    parser.add_argument("--part", choices=("performance", "profile", "acceptance"), default="performance")
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    if args.quality and (args.part != "performance" or args.execution != "graph"):
        parser.error("--quality follows performance and is restricted to Graph modes")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"part": args.part, "mode": args.mode, "execution": args.execution,
              "base_url": args.base_url, "prompts": PROMPTS, "batches": [], "checks": {}}
    filename = "acceptance.json" if args.part == "acceptance" else "http.json"

    def save():
        (args.output / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2))

    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok", health.text
        models = client.get("/v1/models")
        assert models.status_code == 200, models.text
        report["models"] = models.json()
        model = report["models"]["data"][0]["id"]

        def measure(label, prompts, limits, ignore_eos=True):
            before = idle(client)
            (args.output / "phase.txt").write_text(label)
            report["current_phase"] = {"label": label, "prompts": prompts, "limits": limits}
            save()
            started_at, start = time.time(), time.monotonic()
            with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                rows = list(pool.map(lambda item: stream(client, model, item[0][1], item[1], ignore_eos), zip(prompts, limits)))
            elapsed = time.monotonic() - start
            exit_response = client.get("/v1/stats")
            assert exit_response.status_code == 200, exit_response.text
            at_exit = exit_response.json()
            after = idle(client)
            old, new = shape_counts(before), shape_counts(after)
            replays = [{"phase": key[0], "batch_size": key[1], "query_tokens": key[2],
                        "replays": value - old.get(key, 0)}
                       for key, value in sorted(new.items()) if value > old.get(key, 0)]
            for row in replays:
                if row["phase"] == "verify":
                    row["batch_mean_candidates"] = row["query_tokens"] / row["batch_size"] - 1
            delta = stats_delta(before, after)
            spec = delta.get("speculative", {})
            drafted = spec.get("draft_tokens", 0)
            tokens = sum(row["usage"]["completion_tokens"] if row["usage"] else 0 for row in rows)
            seconds = [row["seconds"] for row in rows]
            ttft = [row["ttft_seconds"] for row in rows if row["ttft_seconds"] is not None]
            graph = after["cuda_graph"]
            checks = {
                "resources": (after["model"]["ctx"] == 1024 and after["kv"]["total_pages"] == 4096
                              and after["moe_residency"]["cache_slots"] == 1706
                              and after["moe_residency"]["resident_experts"] == 0),
                "features": (after["speculative"]["enabled"] == (args.mode != "ar")
                             and after["speculative"]["draft_residency"] == ("router" if args.mode == "router" else "off")
                             and not after["speculative"]["adaptive_enabled"]
                             and not after["speculative"]["reuse_enabled"]),
                "execution": graph["enabled"] == (args.execution == "graph"),
                "idle": after["requests"]["active"] == 0,
                "responses": all(row["done"] and row["usage"] is not None
                                 and row["usage"]["total_tokens"] == row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"]
                                 and ((row["finish_reason"] == "length" and row["usage"]["completion_tokens"] == limit)
                                      if ignore_eos else (row["finish_reason"] in ("stop", "length") and row["usage"]["completion_tokens"] <= limit))
                                 for row, limit in zip(rows, limits)),
            }
            if args.execution == "eager":
                checks["no_replays"] = all(graph[phase] == 0 for phase in ("target_decode", "draft", "verify")) and not replays
            else:
                checks["replays_observed"] = bool(replays)
            record = {"label": label, "scored": label.startswith("performance-"),
                      "concurrency": len(prompts), "prompt_names": [p[0] for p in prompts], "limits": limits,
                      "started_at_unix": started_at, "seconds": elapsed,
                      "completion_tokens": tokens, "completion_tps": tokens / elapsed,
                      "request_seconds_p50": percentile(seconds, 0.5), "request_seconds_p95": percentile(seconds, 0.95),
                      "ttft_seconds_p50": percentile(ttft, 0.5), "ttft_seconds_p95": percentile(ttft, 0.95),
                      "responses": rows, "stats_before": before, "stats_at_response_exit": at_exit,
                      "stats_after": after, "stats_delta": delta, "replay_delta": replays,
                      "active_at_response_exit": at_exit["requests"]["active"], "active_after_idle": after["requests"]["active"],
                      "draft_acceptance_rate": spec.get("accepted_draft_tokens", 0) / drafted if drafted else None,
                      "checks": checks}
            print(json.dumps({"label": label, "seconds": elapsed, "tokens": tokens,
                              "actual_graph_B": sorted({r["batch_size"] for r in replays})}), flush=True)
            return record

        def record(label, count, limits):
            batch = measure(label, PROMPTS[:count], limits)
            report["batches"].append(batch)
            save()
            assert all(batch["checks"].values()), {"label": label, "checks": batch["checks"]}

        if args.part == "acceptance":
            for count in CONCURRENCIES[1:]:
                record(f"acceptance-tails-c{count}", count, [TAIL_LENGTHS[i % len(TAIL_LENGTHS)] for i in range(count)])
        else:
            repeats = 1 if args.part == "profile" else 3
            for count in CONCURRENCIES:
                record(f"warmup-c{count}", count, [16] * count)
                for repeat in range(repeats):
                    record(f"performance-{repeat}-c{count}", count, [64] * count)

        report["points"] = []
        for count in CONCURRENCIES:
            batches = [b for b in report["batches"] if b["concurrency"] == count and not b["label"].startswith("warmup-")]
            if not batches:
                continue
            histogram = Counter()
            for batch in batches:
                histogram.update({(r["phase"], r["batch_size"], r["query_tokens"]): r["replays"] for r in batch["replay_delta"]})
            phase_batches = {phase: sorted({b for p, b, q in histogram if p == phase}) for phase in ("target_decode", "draft", "verify")}
            rows = [r for batch in batches for r in batch["responses"]]
            total_seconds = sum(b["seconds"] for b in batches)
            total_tokens = sum(b["completion_tokens"] for b in batches)
            speculative = {key: sum(b["stats_delta"]["speculative"][key] for b in batches)
                           for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps", "residency_stops")}
            point = {"concurrency": count, "batch_seconds": [b["seconds"] for b in batches],
                     "completion_tokens": total_tokens, "seconds": total_seconds, "completion_tps": total_tokens / total_seconds,
                     "request_seconds_p50": percentile([r["seconds"] for r in rows], 0.5),
                     "request_seconds_p95": percentile([r["seconds"] for r in rows], 0.95),
                     "ttft_seconds_p50": percentile([r["ttft_seconds"] for r in rows if r["ttft_seconds"] is not None], 0.5),
                     "ttft_seconds_p95": percentile([r["ttft_seconds"] for r in rows if r["ttft_seconds"] is not None], 0.95),
                     "actual_graph_batches": phase_batches,
                     "speculative_delta": speculative,
                     "router_all_fallback": args.mode == "router" and speculative["draft_tokens"] == 0,
                     "full_concurrency_graph_observed": {phase: count in values for phase, values in phase_batches.items()},
                     "verify_batch_mean_candidates": [{"batch_size": b, "query_tokens": q, "batch_mean_candidates": q / b - 1, "replays": n}
                                                        for (p, b, q), n in sorted(histogram.items()) if p == "verify"]}
            report["points"].append(point)
            if args.execution == "graph" and count > 4:
                report["checks"][f"c{count}:actual_graph_B_over_4"] = any(b > 4 for values in phase_batches.values() for b in values)
                if args.mode == "off" and args.part != "acceptance":
                    for phase in ("draft", "verify"):
                        report["checks"][f"c{count}:{phase}_B_over_4"] = any(b > 4 for b in phase_batches[phase])
        report["completed"] = True
        report["passed"] = all(report["checks"].values())
        save()
        assert report["passed"], report["checks"]

        if args.quality:
            spec = importlib.util.spec_from_file_location("frozen_s2_judge", FROZEN / "evaluate_http.py")
            frozen = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(frozen)
            quality_dir = args.output / "quality"
            quality_dir.mkdir(exist_ok=True)
            quality = {"mode": args.mode, "execution": args.execution, "quality_total": len(TASKS), "quality": []}
            batch = measure("quality-c8", [(task["name"], coding_prompt(task)) for task in TASKS], [256] * len(TASKS), False)
            quality["batch"] = batch
            quality_path = args.output / "quality.json"
            quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2))
            assert all(batch["checks"].values()), batch["checks"]
            for task, row in zip(TASKS, batch["responses"]):
                quality["quality"].append({"task": task["name"], **frozen.judge(task, row, quality_dir)})
                quality["quality_passed"] = sum(r["passed"] for r in quality["quality"])
                quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2))
        print(f"Saved public evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()
