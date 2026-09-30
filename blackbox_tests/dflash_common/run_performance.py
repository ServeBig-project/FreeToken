"""Frozen HTTP performance workload; no service startup or lifecycle mutations."""

import argparse
import copy
import json
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from http_client import ServiceError, request, stream


def require(condition, message):
    if not condition:
        raise ServiceError(message)


def numeric_difference(before, after):
    if isinstance(after, bool):
        return None
    if isinstance(after, (int, float)):
        return after - before
    if isinstance(after, list):
        prior = before if isinstance(before, list) else []
        return [numeric_difference(prior[index] if index < len(prior) else 0, value)
                for index, value in enumerate(after)]
    if isinstance(after, dict):
        prior = before if isinstance(before, dict) else {}
        result = {key: numeric_difference(prior.get(key, 0), value) for key, value in after.items()}
        return {key: value for key, value in result.items() if value is not None}
    return None


def add_counters(left, right):
    if isinstance(right, dict):
        result = copy.deepcopy(left)
        for key, value in right.items():
            result[key] = add_counters(result.get(key, {} if isinstance(value, dict) else
                                                  [0] * len(value) if isinstance(value, list) else 0), value)
        return result
    if isinstance(right, list):
        return [add_counters(a, b) for a, b in zip(left, right)]
    return left + right


def stats_delta(before, after):
    result = {}
    for section in ("speculative", "gdn_replayssm"):
        if section in before and section in after:
            result[section] = numeric_difference(before[section], after[section])
    result.get("speculative", {}).pop("max_draft_steps", None)
    result["requests"] = {key: after["requests"][key] - before["requests"][key]
                          for key in ("completed", "prompt_tokens_total", "completion_tokens_total")}
    result["cuda_graph"] = {key: after["cuda_graph"][key] - before["cuda_graph"][key]
                            for key in ("target_decode", "draft", "verify")}
    prior = {(row["phase"], row["batch_size"], row["query_tokens"], row["physical_query_tokens"]): row["replays"]
             for row in before["cuda_graph"]["replay_shapes"]}
    shapes = []
    for row in after["cuda_graph"]["replay_shapes"]:
        key = (row["phase"], row["batch_size"], row["query_tokens"], row["physical_query_tokens"])
        count = row["replays"] - prior.get(key, 0)
        require(count >= 0, "Graph replay counters reset during a measured wave")
        if count:
            shapes.append({**row, "replays": count})
    return result, shapes


def summarize(waves):
    outputs = [response for wave in waves for response in wave["responses"]]
    seconds = sum(wave["seconds"] for wave in waves)
    tokens = sum(response["usage"]["completion_tokens"] for response in outputs)
    totals, rows = {}, {}
    for wave in waves:
        totals = add_counters(totals, wave["stats_delta"])
        for row in wave["graph_replay_shapes_delta"]:
            key = (row["phase"], row["batch_size"], row["query_tokens"], row["physical_query_tokens"])
            rows[key] = {**row, "replays": rows.get(key, {}).get("replays", 0) + row["replays"]}
    shapes = list(rows.values())
    complete_loads = all(wave["expert_loads_delta"]["complete_target_counts"] for wave in waves)
    loads = {"draft": sum(wave["expert_loads_delta"]["draft"] for wave in waves),
             "complete_target_counts": complete_loads}
    for phase in ("ar", "verify"):
        loads[phase] = sum(wave["expert_loads_delta"][phase] for wave in waves) if complete_loads else None
    return {"waves": len(waves), "requests": len(outputs), "seconds": seconds,
            "completion_tokens": tokens, "completion_tps": tokens / seconds,
            "ttft_mean_ms": statistics.mean(item["ttft_ms"] for item in outputs),
            "mean_ms_per_output_token": statistics.mean(item["mean_ms_per_output_token"] for item in outputs),
            "post_first_chunk_ms_per_remaining_token": statistics.mean(
                item["post_first_chunk_ms_per_remaining_token"] for item in outputs),
            "stats_delta": totals, "expert_loads_delta": loads, "graph_replay_shapes_delta": shapes,
            "graph_query_tokens": sum(row["query_tokens"] * row["replays"] for row in shapes),
            "graph_physical_query_tokens": sum(row["physical_query_tokens"] * row["replays"] for row in shapes)}


def compare(before, after):
    result = {"label": before["label"], "same_frozen_inputs": before["frozen_requests"] == after["frozen_requests"],
              "input_differences": [], "greedy_text_differences": [], "greedy_equal": 0}
    old = {(wave["name"], item["template_index"]): item for wave in before["waves"] if wave["scored"]
           for item in wave["responses"]}
    new = {(wave["name"], item["template_index"]): item for wave in after["waves"] if wave["scored"]
           for item in wave["responses"]}
    for key in sorted(old.keys() | new.keys()):
        if key not in old or key not in new:
            result["input_differences"].append({"wave": key[0], "template_index": key[1], "reason": "missing request"})
            continue
        left, right = old[key], new[key]
        normalized = [{k: v for k, v in item["request"].items() if k not in ("model", "cache_group")}
                      for item in (left, right)]
        if normalized[0] != normalized[1]:
            result["input_differences"].append({"wave": key[0], "template_index": key[1],
                                                "before": normalized[0], "after": normalized[1]})
        elif left["text"] != right["text"]:
            result["greedy_text_differences"].append({"wave": key[0], "template_index": key[1],
                                                     "before": left["text"], "after": right["text"]})
        else:
            result["greedy_equal"] += 1
    result["same_inputs"] = result["same_frozen_inputs"] and not result["input_differences"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    frozen = json.loads(Path(__file__).with_name("performance_requests.json").read_text())
    report = {"label": args.label, "url": args.url, "source": frozen["source"],
              "frozen_requests": frozen["requests"], "waves": [], "failures": [],
              "stats_observation": "After HTTP completion and requests.active=0; GPU cost observations may lag generation replies."}
    run_id = uuid.uuid4().hex

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    def get(path):
        response = request(args.url, "GET", path, timeout=args.timeout)
        require(response["status"] == 200, f"GET {path}: HTTP {response['status']}")
        return response["body"]

    def idle(min_completed=0):
        deadline = time.monotonic() + args.timeout
        while True:
            current = get("/v1/stats")
            if current["requests"]["active"] == 0 and current["requests"]["completed"] >= min_completed:
                return current
            require(time.monotonic() < deadline, "Requests did not become idle before the timeout")
            time.sleep(0.05)

    def wave(name, indices, limit, scored):
        entry = {"name": name, "concurrency": len(indices), "scored": scored,
                 "stats_before": idle(), "cache_before": get("/v1/cache/status")}
        report["waves"].append(entry)
        started = {}
        barrier = Barrier(len(indices) + 1, action=lambda: started.update(time=time.perf_counter()))

        def send(index):
            payload = {**copy.deepcopy(frozen["requests"][index]), "model": report["model"],
                       "max_tokens": limit, "cache_group": f"dflash-perf-{run_id}-{name}"}
            text, finish, usage, content_times = [], [], None, []

            def chunk_received(chunk):
                nonlocal usage
                for choice in chunk.get("choices", []):
                    if choice.get("text"):
                        content_times.append(time.perf_counter())
                        text.append(choice["text"])
                    if choice.get("finish_reason") is not None:
                        finish.append(choice["finish_reason"])
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]

            barrier.wait(timeout=args.timeout)
            requested = time.perf_counter()
            stream(args.url, "/v1/completions", payload, args.timeout, on_chunk=chunk_received)
            finished = time.perf_counter()
            require(usage is not None and usage["completion_tokens"] == limit, f"Template {index}: output count differs: {usage}")
            require(usage["total_tokens"] == usage["prompt_tokens"] + limit, f"Template {index}: inconsistent usage")
            require(finish == ["length"], f"Template {index}: completion did not reach the requested length: {finish}")
            require(bool(content_times), f"Template {index}: no nonempty content chunk for TTFT")
            duration = finished - requested
            return {"template_index": index, "request": payload, "text": "".join(text),
                    "finish_reason": finish[0], "usage": usage, "seconds": duration,
                    "ttft_ms": 1000 * (content_times[0] - requested),
                    "mean_ms_per_output_token": 1000 * duration / limit,
                    "post_first_chunk_ms_per_remaining_token": 1000 * (content_times[-1] - content_times[0]) / (limit - 1),
                    "content_chunks": len(content_times), "_finished": finished}

        with ThreadPoolExecutor(max_workers=len(indices)) as pool:
            futures = [pool.submit(send, index) for index in indices]
            barrier.wait(timeout=args.timeout)
            entry["responses"] = [future.result() for future in futures]
        entry["seconds"] = max(item.pop("_finished") for item in entry["responses"]) - started["time"]
        entry["stats_after"] = idle(entry["stats_before"]["requests"]["completed"] + len(indices))
        entry["cache_after"] = get("/v1/cache/status")
        require(entry["cache_before"]["geometry"] == entry["cache_after"]["geometry"], "Cache geometry changed during a performance wave")
        entry["stats_delta"], entry["graph_replay_shapes_delta"] = stats_delta(entry["stats_before"], entry["stats_after"])
        speculative = entry["stats_delta"]["speculative"]
        complete_loads = entry["stats_after"]["speculative"]["adaptive_cost_enabled"]
        entry["expert_loads_delta"] = {"draft": speculative["draft_expert_loads"], "complete_target_counts": complete_loads}
        for phase in ("ar", "verify"):
            entry["expert_loads_delta"][phase] = speculative["cost_transfer_predictions"][phase]["actual_experts"] if complete_loads else None
        entry["completion_tokens"] = sum(item["usage"]["completion_tokens"] for item in entry["responses"])
        entry["completion_tps"] = entry["completion_tokens"] / entry["seconds"]
        save()
        print(f"{name}: {entry['completion_tokens']} tokens, {entry['completion_tps']:.3f} tokens/s", flush=True)

    try:
        require(get("/health").get("status") == "ok", "Service is not healthy")
        report["models"] = get("/v1/models")
        report["model"] = report["models"]["data"][0]["id"]
        report["cache_initial"] = get("/v1/cache/status")
        if args.reference:
            baseline = json.loads(args.reference.read_text())
            require(baseline["frozen_requests"] == report["frozen_requests"], "Reference uses different frozen request templates")
        wave("warmup-c16", list(range(16)), 16, False)
        for index in range(2):
            wave(f"score-c16-{index}", list(range(16)), 64, True)
        for index in range(4):
            wave(f"warmup-c1-{index}", [index], 16, False)
        for index in range(4):
            wave(f"score-c1-{index}", [index], 64, True)
        scored = [item for item in report["waves"] if item["scored"]]
        report["scored"] = {"all": summarize(scored), "c16": summarize([item for item in scored if item["concurrency"] == 16]),
                            "c1": summarize([item for item in scored if item["concurrency"] == 1])}
        report["cache_final"] = get("/v1/cache/status")
        if args.reference:
            report["comparison"] = compare(baseline, report)
            require(report["comparison"]["same_inputs"], "Scored request inputs differ from the reference")
        report["completed"] = True
    except Exception as error:
        report["failures"].append(f"{type(error).__name__}: {error}")
        report["completed"] = False
    save()
    print(json.dumps({"completed": report["completed"], "failures": report["failures"],
                      "scored": {key: {field: value for field, value in item.items() if field not in ("stats_delta", "graph_replay_shapes_delta")}
                                 for key, item in report.get("scored", {}).items()}}, ensure_ascii=False))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
