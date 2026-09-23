"""Fixed C1/mixed-C4 HTTP measurements; all GPU services belong to the coordinator."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import time

import httpx

FROZEN = Path(__file__).resolve().parents[1] / "s2_methods"
sys.path.insert(0, str(FROZEN))
from corpus import PERFORMANCE, TASKS, coding_prompt


def idle(client):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        response = client.get("/v1/stats")
        assert response.status_code == 200, response.text
        stats = response.json()
        if stats["requests"]["active"] == 0:
            return stats
        time.sleep(0.2)
    raise AssertionError(f"Requests did not return to idle: {stats}")


def stats_delta(before, after):
    result = {}
    for key, value in after.items():
        previous = before.get(key)
        if isinstance(value, dict) and isinstance(previous, dict):
            result[key] = stats_delta(previous, value)
        elif type(value) in (int, float) and type(previous) in (int, float):
            result[key] = value - previous
    return result


def stream(client, model, prompt, limit=64, ignore_eos=True):
    body = {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": limit,
            "ignore_eos": ignore_eos, "stream": True, "stream_options": {"include_usage": True}}
    started_at, start = time.time(), time.monotonic()
    first_at = first = None
    parts, usage, finish, done = [], None, None, False
    with client.stream("POST", "/v1/completions", json=body) as response:
        assert response.status_code == 200, response.read().decode()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk["choices"]:
                text = choice.get("text") or ""
                if text and first is None:
                    first_at, first = time.time(), time.monotonic() - start
                parts.append(text)
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
    finished_at, elapsed = time.time(), time.monotonic() - start
    return {"request": body, "text": "".join(parts), "usage": usage, "finish_reason": finish,
            "truncated": finish == "length", "done": done, "ttft_seconds": first, "seconds": elapsed,
            "started_at_unix": started_at, "first_token_at_unix": first_at, "finished_at_unix": finished_at}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--quality", action="store_true", help="Run the frozen eight tasks after performance, serially")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"part": "graph-performance", "base_url": args.base_url, "batches": [], "quality": []}
    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok", health.text
        models = client.get("/v1/models")
        assert models.status_code == 200, models.text
        model = report["model"] = models.json()["data"][0]["id"]

        def measure(label, prompts, limit=64, ignore_eos=True):
            if isinstance(prompts, str):
                prompts = [prompts]
            before = idle(client)
            (args.output / "phase.txt").write_text(label)
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                rows = list(pool.map(lambda prompt: stream(client, model, prompt, limit, ignore_eos), prompts))
            elapsed = time.monotonic() - start
            at_exit = client.get("/v1/stats")
            assert at_exit.status_code == 200, at_exit.text
            exit_stats = at_exit.json()
            after = idle(client)
            tokens = sum(row["usage"]["completion_tokens"] if row["usage"] else 0 for row in rows)
            delta = stats_delta(before, after)
            spec = delta.get("speculative", {})
            drafted = spec.get("draft_tokens", 0)
            record = {"label": label, "scored": label.startswith("performance-"),
                      "timed": label != "warmup", "concurrency": len(prompts), "seconds": elapsed,
                      "completion_tokens": tokens, "completion_tps": tokens / elapsed, "responses": rows,
                      "stats_before": before, "stats_at_response_exit": exit_stats, "stats_after": after,
                      "active_at_response_exit": exit_stats["requests"]["active"],
                      "active_after_idle": after["requests"]["active"], "stats_delta": delta,
                      "draft_acceptance_rate": spec["accepted_draft_tokens"] / drafted if drafted else None}
            print(json.dumps({"label": label, "seconds": elapsed, "completion_tokens": tokens}), flush=True)
            return record

        mixed = [PERFORMANCE[0][1], PERFORMANCE[1][1]] * 2
        phases = [("warmup", mixed)]
        for repeat in range(3):
            phases += [(f"performance-{repeat}-{name}-c1", [prompt]) for name, prompt in PERFORMANCE]
            phases.append((f"performance-{repeat}-mixed-c4", mixed))
        for label, prompt in phases:
            record = measure(label, prompt)
            report["batches"].append(record)
            (args.output / "http.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            assert all(row["done"] and row["usage"] is not None
                       and row["usage"]["completion_tokens"] == 64 and row["finish_reason"] == "length"
                       for row in record["responses"]), record

        if args.quality:
            spec = importlib.util.spec_from_file_location("frozen_s2_judge", FROZEN / "evaluate_http.py")
            frozen = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(frozen)
            quality_dir = args.output / "quality"
            quality_dir.mkdir(exist_ok=True)
            quality = {"base_url": args.base_url, "model": model, "batches": [], "quality": [],
                       "quality_total": len(TASKS), "quality_passed": 0}
            for task in TASKS:
                record = measure(f"quality-{task['name']}", coding_prompt(task), 256, False)
                quality["batches"].append(record)
                (args.output / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2))
                row = record["responses"][0]
                assert row["done"] and row["usage"] is not None and row["finish_reason"] in {"stop", "length"}, record
                assert row["usage"]["completion_tokens"] <= 256, record
                quality["quality"].append({"task": task["name"], **frozen.judge(task, row, quality_dir)})
                quality["quality_passed"] = sum(item["passed"] for item in quality["quality"])
                (args.output / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2))
    scored = [row for row in report["batches"] if row["scored"]]
    print(json.dumps({"output": str(args.output), "scored_requests": sum(row["concurrency"] for row in scored),
                      "scored_tokens": sum(row["completion_tokens"] for row in scored)}), flush=True)


if __name__ == "__main__":
    main()
