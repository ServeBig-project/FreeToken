#!/usr/bin/env python3
"""Four long requests in a bounded state pool, using only public HTTP behavior."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from http_client import check, complete, idle, request
from lifecycle import shapes_delta


GEOMETRY = ("num_mamba_slots", "num_pages", "moe_cache_size", "cache_budget_bytes")


def same_geometry(run, name, expected):
    current = request(run, "/v1/cache/status")["geometry"]
    check(run, name, all(current[key] == expected[key] for key in GEOMETRY),
          {"expected": {key: expected[key] for key in GEOMETRY},
           "actual": {key: current[key] for key in GEOMETRY}})
    return current


def wave(run, args, geometry):
    before = idle(run)
    barrier = threading.Barrier(4)
    group = "sd-resource-pressure-" + str(time.time_ns())

    def generate(index):
        barrier.wait()
        return complete(run, f"pressure_{index}",
                        f"Describe the stages of a plant's growth in detail. Example {index}:\n",
                        group + "-" + str(index), count=256)

    peak, observed_geometry = 0, False
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(generate, index) for index in range(4)]
        while not all(job.done() for job in jobs):
            peak = max(peak, request(run, "/v1/stats")["requests"]["active"])
            if peak >= 4 and not observed_geometry:
                run["geometry_during"] = same_geometry(run, "live pool and budget remain fixed", geometry)
                observed_geometry = True
            time.sleep(0.25)
        for job in jobs:
            job.result()
    after = idle(run)
    run["geometry_after"] = same_geometry(run, "completed pool and budget remain fixed", geometry)
    sd_before, sd_after = before["speculative"], after["speculative"]
    delta = {key: sd_after[key] - sd_before.get(key, 0)
             for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps", "state_slot_stops")}
    histogram = [right - left for left, right in zip(sd_before["draft_length_histogram"],
                                                    sd_after["draft_length_histogram"])]
    shapes = shapes_delta(before, after)
    run["pressure"] = {"before": before, "after": after, "active_peak": peak,
                       "counter_delta": delta, "draft_length_delta": histogram,
                       "graph_shape_delta": shapes, "natural_tail_round_bound": 32}
    check(run, "four active requests observed", peak >= 4, peak, status="uncovered")
    check(run, "Graph actually executes batch four",
          any(shape["batch_size"] == 4 and shape["delta"] > 0 for shape in shapes),
          shapes, status="uncovered")
    if args.naive_default:
        check(run, "default naive completes through AR", delta["draft_tokens"] == 0 and
              delta["verify_steps"] == 0 and histogram[0] > 0, delta)
        check(run, "default naive fallback is attributed to state capacity",
              delta["state_slot_stops"] > 0, delta, status="uncovered")
    else:
        short_rounds = sum(histogram[1:8])
        run["pressure"]["short_rounds_outside_tail_bound"] = short_rounds > 32
        check(run, "capacity stops or shortening beyond output tails observed",
              delta["state_slot_stops"] > 0 or short_rounds > 32,
              {"state_slot_stops": delta["state_slot_stops"], "short_rounds": short_rounds,
               "tail_bound": 32}, status="uncovered")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--state-slots", type=int, help="Smaller capacity within public limits")
    mode.add_argument("--naive-default", action="store_true", help="Observe default AR without rebuilding")
    args = parser.parse_args()
    run = {"url": args.url.rstrip("/"), "timeout": args.timeout, "arguments": vars(args),
           "checks": [], "http": [], "samples": {}}
    started, changed, original = time.monotonic(), False, None
    try:
        run["model"] = request(run, "/v1/models")["data"][0]["id"]
        stats = idle(run)
        sd = stats["speculative"]
        check(run, "resource probe uses N8 Graph", sd["enabled"] and sd["max_draft_steps"] == 8
              and stats["cuda_graph"]["enabled"])
        check(run, "other policies cannot cause draft shortening", sd["draft_residency"] == "off"
              and not any(sd[key] for key in ("adaptive_cost_enabled", "draft_load_missing_enabled",
                                              "verify_prefetch_enabled")), sd)
        original = request(run, "/v1/cache/status")["geometry"]
        run["geometry_original"] = original
        geometry = original
        if args.state_slots is not None:
            limits = original["limits"]["mamba_slots"]
            check(run, "smaller capacity lies inside public bounds",
                  limits["min"] <= args.state_slots < original["num_mamba_slots"] and
                  (limits["max"] == 0 or args.state_slots <= limits["max"]), limits)
            changed = True
            result = request(run, "/v1/cache/rebuild", {"mode": "if_idle", "timeout": args.timeout,
                                                       "num_mamba_slots": args.state_slots})
            check(run, "smaller state pool rebuilt", result["status"] == "ok", result)
            geometry = request(run, "/v1/cache/status")["geometry"]
            check(run, "only requested pool capacity changes", geometry["num_mamba_slots"] ==
                  args.state_slots and all(geometry[key] == original[key] for key in
                                          ("num_pages", "moe_cache_size")))
            check(run, "rebuild stays within existing budget",
                  geometry["cache_budget_bytes"] <= original["cache_budget_bytes"])
        run["geometry_selected"] = geometry
        wave(run, args, geometry)
    except Exception as error:
        run["checks"].append({"name": "resource pressure completed", "status": "failed",
                              "detail": f"{type(error).__name__}: {error}"})
    finally:
        if changed:
            try:
                idle(run)
                result = request(run, "/v1/cache/rebuild", {"mode": "if_idle", "timeout": args.timeout,
                                                           "num_mamba_slots": original["num_mamba_slots"]})
                check(run, "original state pool rebuilt", result["status"] == "ok", result)
                same_geometry(run, "original pool and budget restored", original)
            except Exception as error:
                run["checks"].append({"name": "restore original state pool", "status": "failed",
                                      "detail": f"{type(error).__name__}: {error}"})
    run["seconds"] = time.monotonic() - started
    counts = {status: sum(item["status"] == status for item in run["checks"])
              for status in ("passed", "failed", "uncovered")}
    run["summary"] = counts
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output), "seconds": run["seconds"], **counts}))
    return 1 if counts["failed"] else 3 if counts["uncovered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
