#!/usr/bin/env python3
"""Public snapshot lifecycle: shared prefixes, cancellation, repeated waves and rebuild."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from http_client import cancel_stream, check, complete, idle, request
from long_context import make_prompt
from resource_pressure import same_geometry


def cached(sample):
    return sample["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)


def reset(run, slots, expected):
    idle(run)
    result = request(run, "/v1/cache/rebuild", {"num_mamba_slots": slots,
                                               "mode": "if_idle", "timeout": 300})
    check(run, "cache rebuild succeeds", result["status"] == "ok" and
          result["mamba_slots"] == slots, result)
    return same_geometry(run, "rebuild preserves requested capacity and budget", expected)


def wave(run, name, prompt, group, geometry, minimum_hit=0):
    before = idle(run)
    barrier = threading.Barrier(4)

    def generate(index):
        barrier.wait()
        label = f"{name}_{index}"
        complete(run, label, prompt, group, count=32)
        if minimum_hit:
            check(run, label + ": shared generated prefix actually hit",
                  cached(run["samples"][label]) > minimum_hit,
                  run["samples"][label]["usage"], status="uncovered")

    peak = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(generate, index) for index in range(4)]
        while not all(job.done() for job in jobs):
            peak = max(peak, request(run, "/v1/stats")["requests"]["active"])
            time.sleep(0.25)
        for job in jobs:
            job.result()
    after = idle(run)
    run["waves"].append({"name": name, "active_peak": peak, "before": before, "after": after})
    check(run, name + ": concurrent requests observed", peak >= 4, peak, status="uncovered")
    same_geometry(run, name + ": fixed capacity and budget", geometry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--public-tokenizer", required=True)
    parser.add_argument("--mode", choices=("sd", "ar"), required=True)
    parser.add_argument("--small-pool", action="store_true", help="Use the public minimum, then restore")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = {"url": args.url.rstrip("/"), "timeout": 300, "arguments": vars(args),
           "checks": [], "http": [], "samples": {}, "waves": []}
    original, reset_started = None, False
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.public_tokenizer, local_files_only=True,
                                                  trust_remote_code=False)
        run["model"] = request(run, "/v1/models")["data"][0]["id"]
        stats = idle(run)
        check(run, "requested execution mode", stats["speculative"]["enabled"] == (args.mode == "sd"))
        check(run, "Graph enabled", stats["cuda_graph"]["enabled"])
        original = request(run, "/v1/cache/status")["geometry"]
        run["geometry_original"] = original
        limits = original["limits"]["mamba_slots"]
        slots = limits["min"] if args.small_pool else original["num_mamba_slots"]
        check(run, "state capacity is publicly supported", slots > 0 and slots >= limits["min"]
              and slots <= original["num_mamba_slots"] and
              (limits["max"] == 0 or slots <= limits["max"]), limits)
        geometry = dict(original, num_mamba_slots=slots)
        reset_started = True
        reset(run, slots, geometry)
        before = idle(run)
        prompt, tokens, _ = make_prompt(tokenizer, target_tokens=912)
        run["chunked_prompt_tokens"] = tokens
        check(run, "prompt crosses the configured 256-token chunks", 768 < tokens < 992, tokens)
        group = "snapshot-lifecycle-" + str(time.time_ns())
        source = complete(run, "chunked_source", prompt, group, count=32)
        check(run, "fresh group starts without prefix hit", cached(run["samples"]["chunked_source"]) == 0)
        followup = prompt + source + "\nContinue:"
        continued = complete(run, "generated_prefix", followup, group, count=32)
        check(run, "generated prefix actually hit", cached(run["samples"]["generated_prefix"]) > tokens,
              run["samples"]["generated_prefix"]["usage"], status="uncovered")
        complete(run, "other_group", followup, group + "-control", count=32)
        check(run, "different group remains isolated", cached(run["samples"]["other_group"]) == 0)
        shared = followup + continued + "\nContinue:"
        wave(run, "shared_wave_one", shared, group, geometry, minimum_hit=tokens)
        event = cancel_stream(run, dict(run["samples"]["chunked_source"]["input"],
                              max_tokens=256, stream=True, cache_group=group + "-cancel",
                              stream_options={"include_usage": True}), min_chars=32)
        check(run, "cancelled a live generation", event["cancelled"] and
              event.get("active_at_close", 0) > 0, status="uncovered")
        idle(run)
        complete(run, "after_cancel", prompt, group + "-cancel", count=16)
        after = idle(run)
        run["generation_before"], run["generation_after"] = before, after
        if args.mode == "sd":
            for field in ("draft_tokens", "verify_steps"):
                check(run, "actual SD " + field, after["speculative"][field] >
                      before["speculative"][field], status="uncovered")
        for field in (("draft", "verify") if args.mode == "sd" else ("target_decode",)):
            check(run, "actual Graph " + field, after["cuda_graph"][field] >
                  before["cuda_graph"][field], status="uncovered")
        reset(run, slots, geometry)
        complete(run, "after_clear", followup, group, count=16)
        check(run, "rebuild removed retained prefix", cached(run["samples"]["after_clear"]) == 0)
        wave(run, "after_clear_wave", prompt, group + "-new", geometry)
    except Exception as error:
        run["checks"].append({"name": "snapshot lifecycle completed", "status": "failed",
                              "detail": f"{type(error).__name__}: {error}"})
    finally:
        if reset_started:
            try:
                reset(run, original["num_mamba_slots"], original)
            except Exception as error:
                run["checks"].append({"name": "restore starting state capacity", "status": "failed",
                                      "detail": f"{type(error).__name__}: {error}"})
    run["summary"] = {status: sum(item["status"] == status for item in run["checks"])
                      for status in ("passed", "failed", "uncovered")}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output), **run["summary"]}))
    return 1 if run["summary"]["failed"] else 3 if run["summary"]["uncovered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
