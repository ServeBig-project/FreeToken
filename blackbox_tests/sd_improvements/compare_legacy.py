"""Strict diagnostic for fixed-four old baseline versus new-controls-off HTTP."""

import argparse
import json
from pathlib import Path


def semantic(row):
    return {"text": row["text"], "finish_reason": row["finish_reason"],
            "usage": {key: row["usage"][key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    old, new = (json.loads(path.read_text()) for path in (args.baseline, args.candidate))
    report = {"baseline": str(args.baseline), "candidate": str(args.candidate), "checks": {}, "differences": []}
    checks = report["checks"]
    checks["completed"] = old.get("completed", False) and new.get("completed", False)
    checks["same_execution"] = old["execution"] == new["execution"]
    checks["same_checkpoint"] = old["models"]["data"][0]["root"] == new["models"]["data"][0]["root"]
    checks["same_workload"] = all(old[key] == new[key] for key in ("concurrencies", "warmup_tokens", "max_tokens", "repetitions"))
    checks["same_phases"] = [b["label"] for b in old["batches"]] == [b["label"] for b in new["batches"]]
    for before, after in zip(old["batches"], new["batches"]):
        label = after["label"]
        spec = after["stats_after"]["speculative"]
        checks[f"{label}:new_controls_off"] = all(spec[key] is False for key in ("adaptive_cost_enabled", "draft_load_missing_enabled", "verify_prefetch_enabled")) and spec["max_draft_steps"] == 4
        a, b = before["stats_after"], after["stats_after"]
        checks[f"{label}:same_resources"] = (a["model"]["ctx"] == b["model"]["ctx"]
            and a["kv"]["total_pages"] == b["kv"]["total_pages"]
            and a["speculative"]["enabled"] is True and b["speculative"]["enabled"] is True
            and a["speculative"]["draft_residency"] == b["speculative"]["draft_residency"])
        checks[f"{label}:response_count"] = len(before["responses"]) == len(after["responses"])
        for index, (expected, actual) in enumerate(zip(before["responses"], after["responses"])):
            requests = [{k: v for k, v in row["request"].items() if k != "model"} for row in (expected, actual)]
            if requests[0] != requests[1] or semantic(expected) != semantic(actual):
                report["differences"].append({"phase": label, "request_index": index,
                                               "baseline_request": requests[0], "candidate_request": requests[1],
                                               "baseline": semantic(expected), "candidate": semantic(actual)})
    report["passed"] = all(checks.values()) and not report["differences"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"passed": report["passed"], "differences": len(report["differences"]),
                      "failed_checks": [key for key, value in checks.items() if not value]}))
    assert report["passed"], "Preserve and review differences; missing new fields in the old baseline are not checked"


if __name__ == "__main__":
    main()
