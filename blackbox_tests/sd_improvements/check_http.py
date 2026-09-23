"""Public SD controls, candidate accounting, boundaries and resource recovery."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import httpx

TESTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TESTS), str(TESTS / "self_speculative"), str(TESTS / "sd_concurrency")]
from sd_concurrency.inputs import PROMPTS
from sd_concurrency.evaluate_http import shape_counts
from test_serving import SAMPLED, Server

LENGTHS = (1, 2, 3, 7, 8, 9, 17)
COUNTERS = ("cost_ar_requests", "cost_stopped_requests", "prefetch_predicted_experts",
            "prefetch_loaded_experts", "prefetch_used_experts", "prefetch_evicted_unused_experts",
            "prefetch_loaded_bytes", "prefetch_used_bytes", "prefetch_evicted_unused_bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--execution", choices=("eager", "graph"), required=True)
    parser.add_argument("--part", choices=("smoke", "boundaries", "lifecycle", "small-cache"), default="smoke")
    parser.add_argument("--speculative-num-steps", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--speculative-draft-residency", choices=("off", "router"), default="router")
    for flag in ("adaptive-cost", "draft-load-missing", "verify-prefetch"):
        parser.add_argument(f"--speculative-{flag}", action="store_true")
    args = parser.parse_args()
    if args.part == "small-cache" and (not args.speculative_draft_load_missing or args.speculative_adaptive_cost):
        parser.error("small-cache isolates load-missing with adaptive cost disabled")
    args.output.mkdir(parents=True, exist_ok=True)
    expected = {"adaptive_cost_enabled": args.speculative_adaptive_cost,
                "draft_load_missing_enabled": args.speculative_draft_load_missing,
                "verify_prefetch_enabled": args.speculative_verify_prefetch}
    report = {"part": args.part, "execution": args.execution, "expected_flags": expected, "stages": [], "rebuilds": []}

    def save():
        (args.output / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    def request(index=0, limit=17, **extra):
        return {"prompt": PROMPTS[index][1], "temperature": 0, "max_tokens": limit,
                "ignore_eos": True, "stream": True, **extra}

    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        assert client.get("/health").json()["status"] == "ok"
        report["models"] = client.get("/v1/models").json()
        model = report["models"]["data"][0]["id"]
        observer = Server(args.part, client)

        def inspect(stats, capacity):
            spec = stats["speculative"]
            assert all(spec[key] is value for key, value in expected.items()), spec
            assert spec["enabled"] and not spec["adaptive_enabled"] and not spec["reuse_enabled"], spec
            assert spec["max_draft_steps"] == args.speculative_num_steps, spec
            assert spec["draft_residency"] == args.speculative_draft_residency, spec
            histogram = spec["draft_length_histogram"]
            assert len(histogram) == args.speculative_num_steps + 1 and all(type(x) is int and x >= 0 for x in histogram), histogram
            assert all(type(spec[key]) is int and spec[key] >= 0 for key in COUNTERS), spec
            for suffix in ("experts", "bytes"):
                assert spec[f"prefetch_used_{suffix}"] + spec[f"prefetch_evicted_unused_{suffix}"] <= spec[f"prefetch_loaded_{suffix}"], spec
            assert stats["moe_residency"]["cache_slots"] == capacity and stats["moe_residency"]["resident_experts"] == 0
            assert stats["kv"]["total_pages"] == 4096 and stats["model"]["ctx"] == 1024
            assert stats["requests"]["active"] == 0
            assert stats["cuda_graph"]["enabled"] is (args.execution == "graph")

        def call(body):
            return observer.call({"model": model, **body}, stream=body.get("stream", True))

        def cancel(body):
            parts = []
            with client.stream("POST", "/v1/completions", json={"model": model, **body}) as response:
                assert response.status_code == 200, response.read().decode()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break
                    chunk = json.loads(line[6:])
                    parts.extend(c.get("text") or "" for c in chunk["choices"])
                    if sum(bool(p) for p in parts) >= 2:
                        snapshot = observer.stats()
                        if snapshot["requests"]["active"] >= 2:
                            return {"cancelled": True, "text": "".join(parts), "snapshot": snapshot}
            return {"cancelled": False, "text": "".join(parts)}

        def run(label, bodies, capacity=1706, cancel_first=False):
            before = observer.idle()
            stage = {"label": label, "inputs": bodies, "stats_before": before}
            report["stages"].append(stage)
            (args.output / "phase.txt").write_text(f"acceptance-{label}")
            save()
            assert isinstance(before["speculative"]["draft_expert_loads"], int), "Enable public --moe-collect-stats for demand-load evidence"
            with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
                rows = list(pool.map(lambda item: cancel(item[1]) if cancel_first and item[0] == 0 else call(item[1]), enumerate(bodies)))
            stage["responses"] = rows
            after = stage["stats_after"] = observer.idle()
            old, new = shape_counts(before), shape_counts(after)
            replay = stage["replay_delta"] = [{"phase": k[0], "batch_size": k[1], "query_tokens": k[2],
                                                "physical_query_tokens": k[3], "replays": n - old.get(k, 0)}
                                               for k, n in sorted(new.items()) if n > old.get(k, 0)]
            previous, current = before["speculative"], after["speculative"]
            delta = stage["counter_delta"] = {key: current[key] - previous[key]
                                               for key in (*COUNTERS, "draft_tokens", "accepted_draft_tokens", "verify_steps", "draft_expert_loads", "residency_stops")}
            histogram = stage["draft_length_delta"] = [b - a for a, b in zip(previous["draft_length_histogram"], current["draft_length_histogram"])]
            stage["coverage"] = {"ordinary_round": histogram[0] > 0, "drafting": delta["draft_tokens"] > 0,
                                 "cost_ar": delta["cost_ar_requests"] > 0, "cost_stop": delta["cost_stopped_requests"] > 0,
                                 "demand_load": delta["draft_expert_loads"] > 0,
                                 "prefetch_load": delta["prefetch_loaded_experts"] > 0,
                                 "prefetch_target_use": delta["prefetch_used_experts"] > 0}
            save()
            inspect(after, capacity)
            assert all(n >= 0 for n in delta.values()) and all(n >= 0 for n in histogram), stage
            if not args.speculative_adaptive_cost:
                assert delta["cost_ar_requests"] == delta["cost_stopped_requests"] == 0, delta
            if not args.speculative_verify_prefetch:
                assert all(delta[key] == 0 for key in COUNTERS if key.startswith(("prefetch_loaded_", "prefetch_used_", "prefetch_evicted_unused_"))), delta
            if args.execution == "eager":
                assert not replay and all(after["cuda_graph"][key] == 0 for key in ("target_decode", "draft", "verify"))
            elif max(body["max_tokens"] for body in bodies) > 1:
                assert replay, "Graph enabled but no actual replay"
                if delta["verify_steps"]:
                    assert sum(r["replays"] for r in replay if r["phase"] == "verify") == delta["verify_steps"], "Verification silently fell back from Graph"
            for index, (body, row) in enumerate(zip(bodies, rows)):
                if cancel_first and index == 0:
                    assert row["cancelled"], row
                elif "stop" in body:
                    assert row["finish_reason"] == "stop" and body["stop"] not in row["text"], row
                else:
                    assert row["finish_reason"] == "length" and row["usage"]["completion_tokens"] == body["max_tokens"], row
            if not cancel_first and all("stop" not in body for body in bodies):
                assert sum(n * count for n, count in enumerate(histogram)) == delta["draft_tokens"], "Completed candidates were not retained in final round lengths"
                if args.execution == "graph":
                    assert sum((r["query_tokens"] - r["batch_size"]) * r["replays"] for r in replay if r["phase"] == "verify") == delta["draft_tokens"], "Completed candidates were not all verified"
            stage["passed"] = True
            save()
            print(json.dumps({"label": label, "histogram": histogram, "coverage": stage["coverage"]}), flush=True)
            return rows

        def rebuild(capacity):
            observer.idle()
            response = client.post("/v1/cache/rebuild", json={"moe_cache_size": capacity, "mode": "if_idle", "timeout": 300.0})
            result = {"capacity": capacity, "http_status": response.status_code, "body": response.json()}
            report["rebuilds"].append(result)
            save()
            assert response.status_code == 200 and result["body"]["status"] == "ok", result

        shrunk = False
        try:
            if args.part == "smoke":
                run("smoke", [request(0), request(1, stream=False)])
            elif args.part == "boundaries":
                for limit in LENGTHS:
                    run(f"limit-{limit}", [request(limit=limit)])
                limits = (*LENGTHS, 17)
                for count in (8, 32):
                    run(f"mixed-c{count}", [request(i, limits[i % len(limits)]) for i in range(count)])
                run("sampled-c4", [{**SAMPLED, "stream": bool(i % 2)} for i in range(4)])
            elif args.part == "lifecycle":
                reference = run("stop-reference", [request()])[0]["text"]
                middle = len(reference) // 2
                stop = reference[middle:middle + 8]
                assert stop
                run("stop-stream", [request(stop=stop)])
                run("stop-plain", [request(stop=stop, stream=False)])
                run("cancel-with-survivors", [request(0, 128)] + [request(i) for i in range(1, 4)], cancel_first=True)
                run("after-cancel", [request(0), request(1)])
            else:
                rebuild(256)
                shrunk = True
                run("load-small-cold", [request(0), request(1)], capacity=256)
                run("load-small-repeat", [request(0), request(1)], capacity=256)
                delta = report["stages"][-2]["counter_delta"]
                assert delta["draft_tokens"] > 0 and delta["verify_steps"] > 0 and delta["draft_expert_loads"] > 0, "Load-missing path not exercised"
                assert delta["residency_stops"] == 0, "Strict residency still stopped load-enabled drafting"
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            save()
            raise
        finally:
            if shrunk:
                rebuild(1706)
                run("restored1706", [request()])
        report["completed"] = True
        report["passed"] = True
        save()


if __name__ == "__main__":
    main()
