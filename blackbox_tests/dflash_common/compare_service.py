"""Compare public observations; greedy differences remain a separate review item."""

import argparse
import json
import sys
from pathlib import Path


def requests(report):
    found = {}
    for case in report["cases"]:
        for item in case.get("requests", []):
            key = f"{case['name']}/{item['name']}"
            found[key] = item
        if "assembled_response" in case:
            found[f"{case['name']}/stream"] = {
                "request": case["stream"]["request"],
                "response": {"status": 200, "body": case["assembled_response"]},
            }
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    result = {"before": before["label"], "after": after["label"],
              "before_failures": before["failures"], "after_failures": after["failures"],
              "before_missing_coverage": before["missing_coverage"], "after_missing_coverage": after["missing_coverage"],
              "before_within_run_text_differences": before["within_run_text_differences"],
              "after_within_run_text_differences": after["within_run_text_differences"],
              "input_differences": [], "greedy_text_differences": [], "greedy_behavior_differences": [], "greedy_equal": 0}
    old, new = requests(before), requests(after)
    for key in sorted(old.keys() | new.keys()):
        if key not in old or key not in new:
            result["input_differences"].append({"case": key, "reason": "request missing from one run"})
            continue
        left, right = old[key], new[key]
        if left["request"] != right["request"]:
            result["input_differences"].append({"case": key, "before": left["request"], "after": right["request"]})
            continue
        if left["request"]["temperature"] != 0 or left["response"]["status"] != 200 or right["response"]["status"] != 200:
            continue
        left_body, right_body = left["response"]["body"], right["response"]["body"]
        left_choice, right_choice = left_body["choices"][0], right_body["choices"][0]
        if left_choice["text"] != right_choice["text"]:
            result["greedy_text_differences"].append({"case": key, "before": left_choice["text"], "after": right_choice["text"]})
        else:
            result["greedy_equal"] += 1
        left_behavior = {"finish_reason": left_choice["finish_reason"], "completion_tokens": left_body["usage"]["completion_tokens"]}
        right_behavior = {"finish_reason": right_choice["finish_reason"], "completion_tokens": right_body["usage"]["completion_tokens"]}
        if left_behavior != right_behavior:
            result["greedy_behavior_differences"].append({"case": key, "before": left_behavior, "after": right_behavior})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in result.items()}, ensure_ascii=False))
    return 1 if result["after_failures"] else (2 if result["after_missing_coverage"] or result["input_differences"] else 0)


if __name__ == "__main__":
    sys.exit(main())
