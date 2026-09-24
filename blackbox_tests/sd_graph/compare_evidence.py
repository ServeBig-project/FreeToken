"""Compare public eager/Graph probe evidence without numerical or text tolerance."""

import argparse
import json
from pathlib import Path


def semantic(row):
    return {"text": row["text"], "finish_reason": row["finish_reason"],
            "usage": {key: row["usage"][key] for key in
                      ("prompt_tokens", "completion_tokens", "total_tokens")}}


def configuration(report):
    stats = report["initial_stats"]
    return {"model_root": report["models"]["data"][0]["root"], "context": stats["model"]["ctx"],
            "kv_pages": stats["kv"]["total_pages"], "cache_slots": report["cache_slots"],
            "speculative": {key: stats["speculative"][key] for key in ("enabled", "draft_residency")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eager", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    eager, graph = (json.loads(path.read_text()) for path in (args.eager, args.graph))
    report = {"eager": str(args.eager), "graph": str(args.graph), "differences": [], "checks": {}}
    checks = report["checks"]
    checks["same_mode"] = eager["mode"] == graph["mode"]
    checks["same_configuration"] = configuration(eager) == configuration(graph)
    checks["both_completed"] = eager.get("probe_completed", False) and graph.get("probe_completed", False)
    checks["local_checks"] = eager.get("passed", True) and graph.get("passed", True)
    checks["graph_coverage"] = not graph.get("missing_graph_coverage", ["not reported"])
    checks["execution_labels"] = eager["execution"] == "eager" and graph["execution"] == "graph"
    left = {stage["label"]: stage for stage in eager["stages"]}
    right = {stage["label"]: stage for stage in graph["stages"]}
    checks["same_stage_order"] = list(left) == list(right)
    checks["eager_zero_replays"] = all(
        not stats["cuda_graph"]["enabled"]
        and all(stats["cuda_graph"][phase] == 0 for phase in ("target_decode", "draft", "verify"))
        and not any(row["replays"] for row in stats["cuda_graph"]["replay_shapes"])
        for stage in eager["stages"] for stats in (stage["stats_before"], stage.get("stats_after", stage["stats_before"])))
    labels = [label for label in left if label in right] if checks["both_completed"] else []
    for label in labels:
        a, b = left[label], right[label]
        if a["inputs"] != b["inputs"] or len(a.get("responses", [])) != len(b.get("responses", [])):
            report["differences"].append({"stage": label, "reason": "inputs or response count differ"})
            continue
        for index, (expected, actual) in enumerate(zip(a.get("responses", []), b.get("responses", []))):
            if expected.get("cancelled") or actual.get("cancelled"):
                if not (expected.get("cancelled") and actual.get("cancelled")):
                    report["differences"].append({"stage": label, "request_index": index,
                                                   "reason": "only one execution actually cancelled"})
                continue  # Cancellation time changes the delivered partial length; survivors remain strict.
            if semantic(expected) != semantic(actual):
                offset = next((i for i, (x, y) in enumerate(zip(expected["text"], actual["text"])) if x != y),
                              min(len(expected["text"]), len(actual["text"])))
                report["differences"].append({"stage": label, "request_index": index,
                                               "request": actual["request"], "first_text_character": offset,
                                               "eager": semantic(expected), "graph": semantic(actual)})
    report["passed"] = all(checks.values()) and not report["differences"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"passed": report["passed"], "differences": len(report["differences"]),
                      "failed_checks": [key for key, value in checks.items() if not value], "evidence": str(args.output)}))
    assert report["passed"], "Public eager/Graph difference or missing coverage; inspect saved evidence"


if __name__ == "__main__":
    main()
