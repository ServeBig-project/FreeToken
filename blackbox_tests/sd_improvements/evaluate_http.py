"""Fixed public workload for the SD improvements; coordinator owns every server."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import time

import httpx

TESTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TESTS), str(TESTS / "sd_concurrency"), str(TESTS / "sd_graph")]
from sd_concurrency.inputs import PROMPTS
from sd_concurrency.evaluate_http import percentile, shape_counts
from benchmark_http import FROZEN, TASKS, coding_prompt, idle, stats_delta, stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--execution", choices=("eager", "graph"), required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    if not all(1 <= c <= len(PROMPTS) for c in args.concurrencies):
        parser.error("concurrencies must be between1 and32")
    if min(args.warmup_tokens, args.max_tokens, args.repetitions) < 1:
        parser.error("token limits and repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"execution": args.execution, "base_url": args.base_url,
              "concurrencies": args.concurrencies, "warmup_tokens": args.warmup_tokens,
              "max_tokens": args.max_tokens, "repetitions": args.repetitions, "batches": []}

    def save():
        (args.output / "http.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok", health.text
        response = client.get("/v1/models")
        assert response.status_code == 200, response.text
        report["models"] = response.json()
        model = report["models"]["data"][0]["id"]

        def measure(label, prompts, limit, ignore_eos=True):
            before = idle(client)
            report["current_phase"] = label
            (args.output / "phase.txt").write_text(label)
            save()
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                rows = list(pool.map(lambda prompt: stream(client, model, prompt[1], limit, ignore_eos), prompts))
            elapsed = time.monotonic() - start
            response = client.get("/v1/stats")
            assert response.status_code == 200, response.text
            at_exit, after = response.json(), idle(client)
            old, new = shape_counts(before), shape_counts(after)
            replays = [{"phase": key[0], "batch_size": key[1], "query_tokens": key[2],
                        "physical_query_tokens": key[3], "replays": value - old.get(key, 0)}
                       for key, value in sorted(new.items()) if value > old.get(key, 0)]
            graph = after["cuda_graph"]
            checks = {
                "resources": (after["model"]["ctx"] == 1024 and after["kv"]["total_pages"] == 4096
                              and after["moe_residency"]["cache_slots"] == 1706
                              and after["moe_residency"]["resident_experts"] == 0),
                "execution": graph["enabled"] == (args.execution == "graph"),
                "replays": (bool(replays) if args.execution == "graph" else
                            not replays and all(graph[p] == 0 for p in ("target_decode", "draft", "verify"))),
                "idle": after["requests"]["active"] == 0,
                "responses": all(r["done"] and r["usage"] is not None
                                 and r["usage"]["total_tokens"] == r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]
                                 and ((r["finish_reason"] == "length" and r["usage"]["completion_tokens"] == limit)
                                      if ignore_eos else (r["finish_reason"] in ("stop", "length") and r["usage"]["completion_tokens"] <= limit))
                                 for r in rows),
            }
            tokens = sum(r["usage"]["completion_tokens"] if r["usage"] else 0 for r in rows)
            return {"label": label, "scored": label.startswith("performance-"), "concurrency": len(prompts),
                    "prompt_names": [p[0] for p in prompts], "responses": rows, "seconds": elapsed,
                    "completion_tokens": tokens, "completion_tps": tokens / elapsed,
                    "request_seconds_p50": percentile([r["seconds"] for r in rows], 0.5),
                    "request_seconds_p95": percentile([r["seconds"] for r in rows], 0.95),
                    "ttft_seconds_p50": percentile([r["ttft_seconds"] for r in rows if r["ttft_seconds"] is not None], 0.5),
                    "ttft_seconds_p95": percentile([r["ttft_seconds"] for r in rows if r["ttft_seconds"] is not None], 0.95),
                    "stats_before": before, "stats_at_response_exit": at_exit, "stats_after": after,
                    "stats_delta": stats_delta(before, after), "replay_delta": replays, "checks": checks}

        for count in args.concurrencies:
            phases = [(f"warmup-c{count}", args.warmup_tokens)]
            phases += [(f"performance-{repeat}-c{count}", args.max_tokens) for repeat in range(args.repetitions)]
            for label, limit in phases:
                batch = measure(label, PROMPTS[:count], limit)
                report["batches"].append(batch)
                save()
                print(json.dumps({"label": label, "seconds": batch["seconds"], "tokens": batch["completion_tokens"]}), flush=True)
                assert all(batch["checks"].values()), {"label": label, "checks": batch["checks"]}
        report["completed"] = True
        report["passed"] = True
        save()

        if args.quality:
            spec = importlib.util.spec_from_file_location("frozen_s2_judge", FROZEN / "evaluate_http.py")
            frozen = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(frozen)
            quality_dir = args.output / "quality"
            quality_dir.mkdir(exist_ok=True)
            quality = {"execution": args.execution, "quality_total": len(TASKS), "quality": []}
            batch = measure("quality-c8", [(t["name"], coding_prompt(t)) for t in TASKS], 256, False)
            quality["batch"] = batch
            path = args.output / "quality.json"
            path.write_text(json.dumps(quality, ensure_ascii=False, indent=2))
            assert all(batch["checks"].values()), batch["checks"]
            for task, row in zip(TASKS, batch["responses"]):
                quality["quality"].append({"task": task["name"], **frozen.judge(task, row, quality_dir)})
                quality["quality_passed"] = sum(r["passed"] for r in quality["quality"])
                path.write_text(json.dumps(quality, ensure_ascii=False, indent=2))
        print(f"Public HTTP evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()
