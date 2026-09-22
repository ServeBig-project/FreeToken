"""Public call-count regression for stopping before an unaffordable draft."""

import json
from pathlib import Path
import re
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "self_speculative"))
from test_serving import CANDIDATE, LONG, SAMPLED, artifacts, checkpoint, serve


def snapshot(log_path, after=0):
    """Wait for the service's public idle snapshot, not a timing benchmark."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        lines = re.findall(r"MoE cache stats snapshot: ([^\r\n]*)", log_path.read_text())
        if len(lines) > after:
            fields = dict(re.findall(r"(\w+)=(\d+)", lines[-1]))
            return {"index": len(lines), "decode_layer_calls": int(fields["decode_layer_calls"]),
                    "public_line": lines[-1]}
        time.sleep(0.2)
    pytest.fail(f"No new public MoE idle snapshot in {log_path}; --moe-collect-stats is required")


def test_stop_before_unaffordable_draft(checkpoint, artifacts):
    """Extra discarded model execution must fail even when HTTP output is correct."""
    layers = checkpoint[0]["num_hidden_layers"]
    assert layers == 48
    cost = {"target_token_ms": 1.0, "draft_step_ms": 2.0, "expert_bandwidth_gib_s": 16.0}
    profile = artifacts / "dominating-draft-cost.json"
    profile.write_text(json.dumps(cost))
    name = "early-draft-stop"
    report = {"cost_profile": cost, "layers": layers, "phases": []}
    phases = [("single", [{"prompt": LONG, "temperature": 0, "max_tokens": 64,
                           "ignore_eos": True}]),
              ("different-length-concurrent", [
                  {"prompt": LONG, "temperature": 0, "max_tokens": limit, "ignore_eos": True}
                  if limit != 17 else {**SAMPLED, "max_tokens": limit}
                  for limit in (8, 17, 33, 64)])]
    with serve(name, CANDIDATE, artifacts, steps=16, experts=3, cache="naive",
               extra_args=["--speculative-adaptive-profile", str(profile), "--moe-collect-stats",
                           "--cuda-graph-max-bs", "0"]) as server:
        for label, requests in phases:
            before = server.idle()
            start = snapshot(artifacts / f"{name}.log")
            if len(requests) == 1:
                results = [server.call(requests[0])]
            else:
                results = server.batch(requests, label)
            after = server.idle()
            end = snapshot(artifacts / f"{name}.log", start["index"])
            keys = ("draft_tokens", "accepted_draft_tokens", "verify_steps", "adaptive_stops")
            delta = {key: after["speculative"][key] - before["speculative"][key] for key in keys}
            calls = end["decode_layer_calls"] - start["decode_layer_calls"]
            # At most one draft plus one verify per round, and one tail fallback per request.
            bound = layers * (2 * delta["verify_steps"] + len(requests))
            checks = {
                "adaptive_without_reuse": after["speculative"]["adaptive_enabled"]
                                          and not after["speculative"]["reuse_enabled"],
                "first_candidates_verified": delta["draft_tokens"] >= delta["verify_steps"] > 0,
                "drafts_retained": delta["accepted_draft_tokens"] > 0,
                "cost_stops_exercised": delta["adaptive_stops"] > 0,
                "single_candidate_per_round": len(requests) != 1
                                              or delta["draft_tokens"] == delta["verify_steps"],
                "necessary_execution_only": 0 < calls <= bound,
                "requests_released": after["requests"]["active"] == 0,
                "all_requests_completed": after["requests"]["completed"]
                                          - before["requests"]["completed"] == len(requests),
                "full_outputs": all(result["text"] and result["finish_reason"] == "length"
                                    and result["usage"]["completion_tokens"] == request["max_tokens"]
                                    for request, result in zip(requests, results)),
            }
            report["phases"].append({"label": label, "requests": requests, "responses": results,
                                     "stats_before": before, "stats_after": after,
                                     "snapshot_before": start, "snapshot_after": end,
                                     "speculative_delta": delta, "decode_layer_calls": calls,
                                     "maximum_decode_layer_calls": bound, "checks": checks})
            (artifacts / "early-draft-stop-checks.json").write_text(json.dumps(report, indent=2))
    failures = [(phase["label"], check) for phase in report["phases"]
                for check, passed in phase["checks"].items() if not passed]
    assert not failures, f"{failures}; public evidence: {artifacts / 'early-draft-stop-checks.json'}"
