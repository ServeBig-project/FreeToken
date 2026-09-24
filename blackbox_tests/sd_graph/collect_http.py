"""Initial public Graph shape/boundary probes; this client never starts a server."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import httpx

from benchmark_http import PERFORMANCE, cache_slots, idle, stream
from corpus import CALIBRATION

PROMPTS = [PERFORMANCE[0][1], PERFORMANCE[1][1], CALIBRATION[0], CALIBRATION[2]]


def shape_counts(stats):
    counts = {}
    for row in stats["cuda_graph"]["replay_shapes"]:
        key = (row["phase"], row["batch_size"], row["query_tokens"])
        counts[key] = counts.get(key, 0) + row["replays"]
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("ar", "off", "router"), default="off")
    parser.add_argument("--execution", choices=("eager", "graph"), default="graph")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "initial shape/boundary probes; lifecycle and paired semantic comparison follow",
              "base_url": args.base_url, "mode": args.mode, "execution": args.execution, "stages": []}

    def save():
        (args.output / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok", health.text
        models = client.get("/v1/models")
        assert models.status_code == 200, models.text
        report["models"] = models.json()
        model = report["models"]["data"][0]["id"]
        initial = report["initial_stats"] = idle(client)
        report["cache_slots"] = cache_slots(client)
        assert report["cache_slots"] == 1706
        assert initial["kv"]["total_pages"] == 4096
        assert initial["speculative"]["enabled"] == (args.mode != "ar")
        assert initial["speculative"]["draft_residency"] == ("router" if args.mode == "router" else "off")
        assert initial["cuda_graph"]["enabled"] == (args.execution == "graph")
        stages = [(f"limit-{limit}", [(PROMPTS[0], limit)]) for limit in (1, 2, 3, 4, 5, 7, 17, 64)]
        stages += [(f"concurrency-{n}", [(prompt, 64) for prompt in PROMPTS[:n]]) for n in (2, 3, 4)]
        stages += [("mixed-short-tails", list(zip(PROMPTS, (1, 2, 3, 4)))),
                   ("mixed-long-tails", list(zip(PROMPTS, (5, 7, 17, 64))))]
        for label, requests in stages:
            stage = {"label": label, "inputs": [{"prompt": p, "max_tokens": n} for p, n in requests],
                     "stats_before": idle(client)}
            report["stages"].append(stage)
            save()
            try:
                (args.output / "phase.txt").write_text(label)
                with ThreadPoolExecutor(max_workers=len(requests)) as pool:
                    rows = list(pool.map(lambda request: stream(client, model, *request), requests))
                stage["responses"] = rows
                stage["stats_after"] = idle(client)
                before = shape_counts(stage["stats_before"])
                after = shape_counts(stage["stats_after"])
                stage["replay_delta"] = [{"phase": key[0], "batch_size": key[1], "query_tokens": key[2],
                                           "replays": value - before.get(key, 0)}
                                          for key, value in after.items() if value > before.get(key, 0)]
                save()
                assert all(row["done"] and row["finish_reason"] == "length"
                           and row["usage"]["completion_tokens"] == request[1]
                           for row, request in zip(rows, requests)), label
                print(json.dumps({"label": label, "replays": stage["replay_delta"]}), flush=True)
            except Exception as error:
                stage["error"] = f"{type(error).__name__}: {error}"
                save()
                raise
        observed = {(row["phase"], row["batch_size"], row["query_tokens"])
                    for stage in report["stages"] for row in stage["replay_delta"]}
        required_phases = ("target_decode",) if args.mode == "ar" else ("draft", "verify")
        missing = [f"{phase}:B{n}" for phase in required_phases for n in (1, 2, 3, 4)
                   if not any(p == phase and b == n for p, b, q in observed)]
        if args.mode != "ar":
            missing += [f"verify:B1:Q{q}" for q in (2, 3, 4, 5) if ("verify", 1, q) not in observed]
        report["missing_graph_coverage"] = missing if args.execution == "graph" else []
        report["probe_completed"] = True
        save()
        assert not observed if args.execution == "eager" else not missing, report["missing_graph_coverage"]
        print(f"Public probe evidence: {args.output / 'acceptance.json'}", flush=True)


if __name__ == "__main__":
    main()
