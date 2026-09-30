#!/usr/bin/env python3
"""Replay frozen public requests for throughput only; no quality comparison."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from http_client import check, idle, request


CAPACITY = ("num_pages", "moe_cache_size", "num_mamba_slots", "cache_budget_bytes")


def replay(run, body):
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    raw = request(run, path, body)
    packets = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
    check(run, "SSE terminates with DONE", bool(packets) and packets[-1] == "[DONE]")
    text, usages, reasons = "", [], []
    for packet in packets[:-1]:
        chunk = json.loads(packet)
        if chunk.get("usage") is not None:
            usages.append(chunk["usage"])
        for choice in chunk.get("choices", []):
            text += (choice.get("delta", {}).get("content") or "") if "messages" in body else (
                choice.get("text") or "")
            if choice.get("finish_reason") is not None:
                reasons.append(choice["finish_reason"])
    check(run, "final streaming usage present", bool(usages))
    usage = usages[-1]
    check(run, "frozen output token count", usage["completion_tokens"] == body["max_tokens"], usage)
    check(run, "requested output completed", reasons == ["length"], reasons)
    return {"request": body, "usage": usage, "text": text, "finish_reason": reasons[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen = json.loads(Path(args.reference).read_text())
    run = {"url": args.url.rstrip("/"), "base_url": args.url.rstrip("/"), "timeout": 180,
           "reference": args.reference, "checks": [], "http": [], "batches": [],
           "completed": False, "passed": False}
    try:
        run["models"] = request(run, "/v1/models")
        model = run["models"]["data"][0]["id"]
        initial = request(run, "/v1/cache/status")
        run["cache_initial"] = initial
        geometry = initial["geometry"]
        for source in frozen["batches"]:
            bodies = [dict(response["request"], model=model) for response in source["responses"]]
            batch = {"label": source["label"], "scored": source["scored"],
                     "concurrency": source["concurrency"], "responses": [None] * len(bodies)}
            run["batches"].append(batch)
            check(run, "one frozen wave per batch", len(bodies) == batch["concurrency"])
            batch["stats_before"] = idle(run)
            batch["cache_before"] = request(run, "/v1/cache/status")
            barrier = threading.Barrier(batch["concurrency"])

            def worker(index, body):
                barrier.wait()
                batch["responses"][index] = replay(run, body)

            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=batch["concurrency"]) as pool:
                jobs = [pool.submit(worker, index, body) for index, body in enumerate(bodies)]
                for job in jobs:
                    job.result()
            batch["seconds"] = time.monotonic() - started
            batch["completion_tokens"] = sum(response["usage"]["completion_tokens"]
                                             for response in batch["responses"])
            batch["completion_tps"] = batch["completion_tokens"] / batch["seconds"]
            batch["stats_after"] = idle(run)
            batch["cache_after"] = request(run, "/v1/cache/status")
            batch["checks"] = {
                "responses": all(response["usage"]["completion_tokens"] == body["max_tokens"]
                                 for response, body in zip(batch["responses"], bodies)),
                "idle": batch["stats_after"]["requests"]["active"] == 0,
                "resources": all(snapshot["geometry"][key] == geometry[key]
                                 for snapshot in (batch["cache_before"], batch["cache_after"])
                                 for key in CAPACITY),
            }
            batch["stats_delta"] = {
                section: {key: batch["stats_after"][section].get(key, 0) -
                          batch["stats_before"][section].get(key, 0) for key in keys}
                for section, keys in (("speculative", ("draft_tokens", "accepted_draft_tokens",
                                                       "verify_steps", "state_slot_stops")),
                                      ("cuda_graph", ("target_decode", "draft", "verify")))}
            check(run, "batch keeps starting capacities and completes", all(batch["checks"].values()),
                  batch["checks"])
            print(json.dumps({key: batch[key] for key in ("label", "scored", "seconds",
                                                          "completion_tokens", "completion_tps")}), flush=True)
        run["completed"] = True
        run["passed"] = True
    except Exception as error:
        run["error"] = f"{type(error).__name__}: {error}"
    scored = [batch for batch in run["batches"] if batch["scored"] and "seconds" in batch]
    run["scored_seconds"] = sum(batch["seconds"] for batch in scored)
    run["scored_completion_tokens"] = sum(batch["completion_tokens"] for batch in scored)
    if run["scored_seconds"]:
        run["scored_completion_tps"] = run["scored_completion_tokens"] / run["scored_seconds"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output), "completed": run["completed"], "passed": run["passed"]}))
    return 0 if run["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
