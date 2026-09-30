#!/usr/bin/env python3
"""Eight fixed public input/output tasks, scored identically for AR and SD."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from http_client import check, compare, idle, request


TASKS = [
    {"id": "arithmetic_integer", "kind": "json", "expected": 869,
     "prompt": "Compute 37 * 24 - 19. Return only the answer as a JSON integer."},
    {"id": "arithmetic_fraction", "kind": "json", "expected": {"numerator": 19, "denominator": 12},
     "prompt": "Compute 3/4 + 5/6 as a reduced fraction. Return only a JSON object with keys "
               "numerator and denominator containing integers."},
    {"id": "structured_extract", "kind": "json",
     "expected": {"name": "Mira", "count": 3, "active": True},
     "prompt": "Mira has three notebooks and her membership is active. Extract only this JSON "
               "object: name (string), count (integer number of notebooks), active (boolean)."},
    {"id": "structured_sort", "kind": "json", "expected": ["a", "b", "c"],
     "prompt": 'Records: [{"id":"c","score":2},{"id":"a","score":3},{"id":"b","score":2}]. '
               "Sort by descending score, then ascending id for ties. Return only the ordered "
               "ids as a JSON array."},
    {"id": "coding_comprehension", "kind": "json", "expected": [1, 9, 25],
     "prompt": "What does this Python expression evaluate to? [x*x for x in range(6) if x % 2]. "
               "Return only the result as a JSON array."},
    {"id": "coding_alias", "kind": "json", "expected": [[1, 2, 3], [9, 2, 3]],
     "prompt": "Execute this Python mentally:\nx = [1, 2]\ny = x\ny.append(3)\nz = x.copy()\n"
               "z[0] = 9\nprint([x, z])\nReturn only the printed value as a JSON array."},
    {"id": "coding_unique", "kind": "python", "function": "unique_values",
     "cases": [([[]], []), ([[3, 1, 3, 2, 1]], [3, 1, 2]),
               ([["a", "b", "a"]], ["a", "b"]), ([[0, 0]], [0])],
     "prompt": "Write only a Python function unique_values(values) that removes duplicate "
               "hashable items, keeps first occurrence order, and returns a list. "
               "No imports, type annotations, examples, or prose."},
    {"id": "coding_clamp", "kind": "python", "function": "clamp",
     "cases": [([-2, 0, 5], 0), ([3, 0, 5], 3), ([9, 0, 5], 5),
               ([2, 2, 2], 2), ([-4, -10, -1], -4)],
     "prompt": "Write only a Python function clamp(x, low, high) returning low if x is below "
               "low, high if x is above high, otherwise x. Inputs are integers and low <= high. "
               "No imports, type annotations, examples, or prose."},
]


def grade(task, text):
    try:
        if task["kind"] == "json":
            actual = json.loads(text)
            return {"correct": actual == task["expected"], "actual": actual,
                    "expected": task["expected"]}
        code = text.strip()
        if code.startswith("```") and code.endswith("```"):
            code = "\n".join(code.splitlines()[1:-1])
        program = code + "\n"
        for args, expected in task["cases"]:
            program += f"assert {task['function']}(*{args!r}) == {expected!r}\n"
        result = subprocess.run([sys.executable, "-I", "-c", program], text=True,
                                capture_output=True, timeout=2)
        return {"correct": result.returncode == 0, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr, "cases": task["cases"]}
    except (ValueError, subprocess.TimeoutExpired) as error:
        return {"correct": False, "error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("graph", "eager"), required=True)
    parser.add_argument("--expected-steps", type=int, choices=range(9), required=True)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--reference", help="AR or earlier report with identical task requests")
    parser.add_argument("--only", nargs="+", choices=[task["id"] for task in TASKS],
                        help="Run only selected tasks without changing their inputs or grading")
    args = parser.parse_args()
    tasks = [task for task in TASKS if args.only is None or task["id"] in args.only]
    run = {"url": args.url.rstrip("/"), "timeout": args.timeout, "arguments": vars(args),
           "checks": [], "http": [], "tasks": {}, "reference_only": args.expected_steps == 0}
    started = time.monotonic()
    try:
        model = args.model or request(run, "/v1/models")["data"][0]["id"]
        geometry = request(run, "/v1/cache/status")["geometry"]
        run["geometry"] = geometry
        off = geometry["reasoning"]["kwargs"]["off"]
        before = idle(run)
        run["before"] = before
        check(run, "effective maximum draft length", before["speculative"]["max_draft_steps"]
              == args.expected_steps)
        check(run, "effective SD setting", before["speculative"]["enabled"] ==
              (args.expected_steps > 0))
        check(run, "effective Graph setting", before["cuda_graph"]["enabled"] ==
              (args.mode == "graph"))
        for task in tasks:
            body = {"model": model, "messages": [{"role": "user", "content": task["prompt"]}],
                    "chat_template_kwargs": off, "temperature": 0, "top_k": 1, "top_p": 1,
                    "max_tokens": 256, "ignore_eos": False, "stream": False,
                    "cache_group": "sd-quality-" + task["id"]}
            response = request(run, "/v1/chat/completions", body)
            choice = response["choices"][0]
            text = choice["message"]["content"]
            run["tasks"][task["id"]] = {"input": body, "text": text,
                                        "usage": response["usage"],
                                        "finish_reason": choice["finish_reason"],
                                        "grade": grade(task, text)}
        after = idle(run)
        run["after"] = after
        if args.expected_steps:
            for field in ("draft_tokens", "verify_steps"):
                check(run, f"quality actually exercises SD: {field}",
                      after["speculative"][field] > before["speculative"][field],
                      status="uncovered")
            if args.mode == "graph":
                for field in ("draft", "verify"):
                    check(run, f"quality actually exercises Graph: {field}",
                          after["cuda_graph"][field] > before["cuda_graph"][field],
                          status="uncovered")
        if args.reference:
            old = json.loads(Path(args.reference).read_text())
            for name, result in run["tasks"].items():
                previous = old["tasks"][name]
                check(run, f"{name}: identical reference input", result["input"] == previous["input"])
                result["reference_correct"] = previous["grade"]["correct"]
                check(run, f"{name}: no task regression", not previous["grade"]["correct"] or
                      result["grade"]["correct"], result["grade"], status="regression")
                compare(run, f"{name}: full text comparison", previous["text"], result["text"])
    except Exception as error:
        run["checks"].append({"name": "quality run completed", "status": "failed",
                              "detail": f"{type(error).__name__}: {error}"})
    score = sum(task["grade"]["correct"] for task in run["tasks"].values())
    counts = {status: sum(item["status"] == status for item in run["checks"])
              for status in ("passed", "failed", "investigate", "uncovered", "regression")}
    run["summary"] = {**counts, "task_correct": score, "task_total": len(tasks)}
    run["seconds"] = time.monotonic() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output), "seconds": run["seconds"], **run["summary"]}))
    return (1 if counts["failed"] or counts["regression"] else 4 if score < len(tasks) else
            2 if counts["investigate"] else 3 if counts["uncovered"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
