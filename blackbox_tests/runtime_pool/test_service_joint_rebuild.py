"""One-shot public acceptance against an already running, idle hybrid DFlash service.

No startup, retry, model loading, or output equality requirement across rebuild plans.
"""

import argparse
import json

from service_common import Client, GIB, graph_replays, sd_num
from test_service_final import Run, body, history


def geometry(status, runtime_bytes, experts):
    g, rt = status["geometry"], status["prefix_cache"]["runtime"]
    assert status["state"] == "serving", status
    assert g["runtime_cache_bytes"] == rt["budget_bytes"] == runtime_bytes, (g, rt)
    assert g["moe_cache_size"] == experts, g
    assert g["num_pages"] == g["num_mamba_slots"] == 0, g
    assert rt["max_running_requests"] == rt["requested_running_requests"] == 8, rt
    assert rt["context_tokens"] == rt["requested_context_tokens"] == 49152, rt
    assert 0 <= rt["used_bytes"] <= rt["held_bytes"] <= runtime_bytes, rt
    assert g["gdn_replayssm"]["active"] and g["dflash"]["active"], g


def execution(stats):
    assert stats["cuda_graph"]["enabled"], stats["cuda_graph"]
    sd = stats["speculative"]
    assert sd["enabled"] and sd["drafter"] == "dflash" and sd["max_draft_steps"] == 4, sd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    run = Run(args)
    client = Client(args.url)
    summary = {"failures": run.failures}
    try:
        initial, stats_before = run.initial, client.stats()
        geometry(initial, int(8.626953125 * GIB), 5000)
        execution(stats_before)
        assert stats_before["requests"]["active"] == 0, stats_before["requests"]
        g = initial["geometry"]
        target_bytes, target_experts = int(4.75 * GIB), 7200
        unit = g["unit_bytes"]["moe_per_expert"]
        limits = g["limits"]["moe_experts"]
        old_total = g["runtime_cache_bytes"] + g["moe_cache_size"] * unit
        new_total = target_bytes + target_experts * unit
        budget = {"old_total_bytes": old_total, "new_total_bytes": new_total,
                  "cache_budget_bytes": g["cache_budget_bytes"],
                  "saved_bytes": old_total - new_total, "moe_per_expert": unit,
                  "expert_limits": limits, "model_experts": g["num_experts"] * g["num_moe_layers"]}
        run.write("budget.json", budget)
        assert limits["min"] <= target_experts <= limits["max"], budget
        assert target_experts <= budget["model_experts"], budget
        assert target_bytes < g["runtime_cache_bytes"] and target_experts > g["moe_cache_size"], budget
        assert new_total <= old_total <= g["cache_budget_bytes"], budget

        reference_body = body(history(900, 4), 64, "joint-rebuild-reference")
        reference = run.phase("reference", [reference_body])[0]
        assert not run.failures, run.failures
        client.wait_idle(args.timeout)
        request = {"runtime_cache_gib": 4.75, "moe_cache_size": target_experts}
        run.write("rebuild-request.json", {"mode": "if_idle", **request})
        code, response = client.rebuild(request, timeout=args.timeout)
        run.write("rebuild-response.json", {"http_status": code, "response": response})
        assert (code, response.get("status")) == (200, "ok"), (code, response)
        rebuilt, stats_rebuilt = client.status(), client.stats()
        run.write("rebuilt.json", rebuilt)
        run.write("stats-rebuilt.json", stats_rebuilt)
        geometry(rebuilt, target_bytes, target_experts)
        execution(stats_rebuilt)
        assert rebuilt["geometry"]["unit_bytes"] == g["unit_bytes"], rebuilt["geometry"]
        assert rebuilt["geometry"]["cache_budget_bytes"] <= g["cache_budget_bytes"], rebuilt["geometry"]
        assert new_total <= rebuilt["geometry"]["cache_budget_bytes"], rebuilt["geometry"]

        repeated = run.phase("same-request", [reference_body])[0]
        run.phase("new-requests", [body(history(910 + i, 4), n, f"joint-rebuild-new-{i}")
                                   for i, n in enumerate((16, 24, 32, 48, 64, 16, 32, 64))])
        assert not run.failures, run.failures
        assert reference["usages"][-1]["prompt_tokens"] == repeated["usages"][-1]["prompt_tokens"]
        summary["same_prompt_text_equal"] = reference["text"] == repeated["text"]
        client.wait_idle(args.timeout)
        final, stats_after = client.status(), client.stats()
        run.write("final.json", final)
        run.write("stats-after.json", stats_after)
        geometry(final, target_bytes, target_experts)
        execution(stats_after)
        summary["graph_replays_after_rebuild"] = graph_replays(stats_after) - graph_replays(stats_rebuilt)
        summary["sd_after_rebuild"] = {name: sd_num(stats_after, name) - sd_num(stats_rebuilt, name)
                                       for name in ("drafted", "accepted", "rounds")}
        assert summary["graph_replays_after_rebuild"] > 0, summary
        assert summary["sd_after_rebuild"]["drafted"] > 0, summary
        assert summary["sd_after_rebuild"]["rounds"] > 0, summary
        assert summary["sd_after_rebuild"]["accepted"] >= 0, summary
        post_results = run.results[1:]
        expected = {"completed": len(post_results),
                    "prompt_tokens_total": sum(r["usages"][-1]["prompt_tokens"] for r in post_results),
                    "completion_tokens_total": sum(r["usages"][-1]["completion_tokens"] for r in post_results)}
        summary["request_counter_delta"] = {
            key: stats_after["requests"][key] - stats_rebuilt["requests"][key] for key in expected}
        assert summary["request_counter_delta"] == expected, (summary, expected)
        summary["requests"] = len(run.results)
        summary["passed"] = True
    except Exception as error:
        run.failures.append(repr(error))
        summary["passed"] = False
        raise
    finally:
        run.write("summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
