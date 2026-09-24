"""Frozen HTTP performance/quality inputs; the coordinator owns all GPU servers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import httpx

from corpus import PERFORMANCE, TASKS, coding_prompt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "self_speculative"))
from test_serving import Server


def request(prompt, limit=256, ignore_eos=True):
    return {"prompt": prompt, "temperature": 0, "max_tokens": limit, "ignore_eos": ignore_eos}


def stream(client, model, body):
    body = {"model": model, **body, "stream": True, "stream_options": {"include_usage": True}}
    start, first = time.monotonic(), None
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
                    first = time.monotonic() - start
                parts.append(text)
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
    assert done and usage is not None and finish is not None
    return {"request": body, "text": "".join(parts), "usage": usage, "finish_reason": finish,
            "truncated": finish == "length", "ttft_seconds": first, "seconds": time.monotonic() - start}


def stats_delta(before, after):
    result = {}
    for key, value in after.items():
        previous = before.get(key)
        if isinstance(value, dict) and isinstance(previous, dict):
            result[key] = stats_delta(previous, value)
        elif type(value) in (int, float) and type(previous) in (int, float):
            result[key] = value - previous
    return result


def judge(task, response, output):
    text = response["text"].strip()
    fenced = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", text, re.S)
    code = fenced.group(1) if fenced else text
    (output / f"{task['name']}.py").write_text(code)
    if response["truncated"]:
        return {"passed": False, "reason": "truncated", "passed_cases": 0, "total_cases": len(task["cases"])}
    try:
        with tempfile.TemporaryDirectory(prefix="s2-code-") as work:
            run = subprocess.run([sys.executable, "-I", "-S", str(Path(__file__).with_name("judge_code.py"))],
                                 input=json.dumps({**task, "code": code}), text=True, capture_output=True,
                                 cwd=work, timeout=3)
        if run.returncode != 0:
            return {"passed": False, "reason": "worker_error", "stderr": run.stderr}
        return json.loads(run.stdout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "reason": "timeout"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--part", choices=("evaluate", "draft-ablation"), default="evaluate")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"part": args.part, "base_url": args.base_url, "batches": [], "quality": []}
    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok", health.text
        model = client.get("/v1/models").json()["data"][0]["id"]
        report["model"] = model
        observer = Server(args.output.name, client)

        def batch(label, bodies):
            before = observer.idle()
            if args.part == "draft-ablation":
                (args.output / "phase.txt").write_text(label)
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
                rows = list(pool.map(lambda body: stream(client, model, body), bodies))
            elapsed = time.monotonic() - start
            after = observer.idle()
            tokens = sum(row["usage"]["completion_tokens"] for row in rows)
            record = {"label": label, "seconds": elapsed, "completion_tokens": tokens,
                      "completion_tps": tokens / elapsed, "responses": rows,
                      "stats_before": before, "stats_after": after, "stats_delta": stats_delta(before, after)}
            if args.part == "draft-ablation":
                speculative = record["stats_delta"].get("speculative", {})
                drafted = speculative.get("draft_tokens", 0)
                record["scored"] = label != "warmup"
                record["draft_acceptance_rate"] = (speculative["accepted_draft_tokens"] / drafted
                                                     if drafted else None)
            report["batches"].append(record)
            (args.output / "http.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            if args.part == "draft-ablation":
                assert all(row["usage"]["completion_tokens"] == 64 and row["finish_reason"] == "length"
                           for row in rows), "Incomplete fixed-length workload; see saved HTTP evidence"
            print(json.dumps({"label": label, "seconds": elapsed, "completion_tokens": tokens}), flush=True)
            return rows

        batch("warmup", [request(PERFORMANCE[0][1], 64)] * 4)
        limit = 64 if args.part == "draft-ablation" else 256
        for repeat in range(2):
            for name, prompt in PERFORMANCE:
                for concurrency in (1, 4):
                    batch(f"performance-{repeat}-{name}-c{concurrency}", [request(prompt, limit)] * concurrency)
        if args.part == "evaluate":
            for offset in range(0, len(TASKS), 4):
                tasks = TASKS[offset:offset + 4]
                responses = batch(f"quality-{offset // 4}", [request(coding_prompt(task), ignore_eos=False) for task in tasks])
                for task, response in zip(tasks, responses):
                    report["quality"].append({"task": task["name"], **judge(task, response, args.output)})
            report["quality_passed"] = sum(row["passed"] for row in report["quality"])
            report["quality_total"] = len(TASKS)
    (args.output / "http.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(args.output), "quality_passed": report.get("quality_passed")}), flush=True)


if __name__ == "__main__":
    main()
