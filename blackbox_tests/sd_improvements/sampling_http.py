"""One frozen AR/all-on sampling comparison; score choices after prefill."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

import httpx

TESTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TESTS / "self_speculative"), str(TESTS / "sd_graph")]
from benchmark_http import cache_slots
from sampling_confirmation import exact_permutation_p
from test_serving import SAMPLED, Server


def bucket(response):
    text = response["choices"][0]["text"]
    if not re.fullmatch(r"\s*[AB](?:\s+[AB]){7}\s*", text):
        return "other"
    return f"A_after_prefill={text.split()[1:].count('A')}"


def collect(args):
    data = {"arm": args.arm, "execution": "graph", "samples_per_arm": 256,
            "alpha": 0.001, "presampling": [], "scored": [], "batches": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        assert client.get("/health").json()["status"] == "ok"
        data["models"] = client.get("/v1/models").json()
        model = data["models"]["data"][0]["id"]
        observer = Server(args.arm, client)
        try:
            for section, count, limit in (("presampling", 8, 224), ("scored", 256, 8)):
                body = {"model": model, **SAMPLED, "max_tokens": limit, "stream": False}
                data[section + "_request"] = body
                data[section + "_before"] = observer.idle()

                def call(_):
                    response = client.post("/v1/completions", json=body)
                    raw = {"http_status": response.status_code, "response": response.json()}
                    return raw

                for batch_index in range(count // 4):
                    before = observer.idle()
                    (args.output.parent / "phase.txt").write_text(f"sampling-{section}-{batch_index}")
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        rows = list(pool.map(call, range(4)))
                    data[section].extend(rows)
                    after = observer.idle()
                    data["batches"].append({"section": section, "index": batch_index,
                                            "stats_before": before, "stats_after": after})
                    save()
                    assert after["cuda_graph"]["enabled"] is True
                    assert after["requests"]["active"] == 0 and after["kv"]["total_pages"] == 4096
                    assert cache_slots(client) == 1706
                    spec = after["speculative"]
                    assert spec["enabled"] is (args.arm == "all-on")
                    if args.arm == "all-on":
                        assert all(spec[key] is True for key in ("adaptive_cost_enabled", "draft_load_missing_enabled", "verify_prefetch_enabled")), spec
                        assert spec["max_draft_steps"] == 8 and spec["draft_residency"] == "router"
                    for row in rows:
                        result = row["response"]
                        assert row["http_status"] == 200 and len(result["choices"]) == 1, row
                        usage = result["usage"]
                        assert result["choices"][0]["finish_reason"] == "length" and usage["completion_tokens"] == limit, row
                        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], row
                data[section + "_after"] = observer.idle()
            data["histogram"] = dict(Counter(bucket(row["response"]) for row in data["scored"]))
            if args.arm == "all-on":
                before, after = data["scored_before"], data["scored_after"]
                delta = {key: after["speculative"][key] - before["speculative"][key]
                         for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps")}
                histogram = [b - a for a, b in zip(before["speculative"]["draft_length_histogram"], after["speculative"]["draft_length_histogram"])]
                participating = sum(b["stats_after"]["speculative"]["draft_tokens"] > b["stats_before"]["speculative"]["draft_tokens"]
                                    and b["stats_after"]["speculative"]["verify_steps"] > b["stats_before"]["speculative"]["verify_steps"]
                                    for b in data["batches"] if b["section"] == "scored")
                data["sampling_coverage"] = {"counter_delta": delta, "draft_length_delta": histogram,
                                             "scored_batches_with_sd": participating, "scored_batches_total": 64,
                                             "graph_verify_delta": after["cuda_graph"]["verify"] - before["cuda_graph"]["verify"]}
                data["sd_sampling_covered"] = all(n > 0 for n in delta.values()) and sum(histogram[1:]) > 0 and data["sampling_coverage"]["graph_verify_delta"] > 0
            data["interface_passed"] = True
        finally:
            save()


def compare(args):
    arms = [json.loads(path.read_text()) for path in (args.ar, args.all_on)]
    assert [arm["arm"] for arm in arms] == ["ar", "all-on"]
    assert all(arm.get("interface_passed") and len(arm["presampling"]) == 8 and len(arm["scored"]) == 256 for arm in arms)
    left, right = (arm["histogram"] for arm in arms)
    p = exact_permutation_p(left, right)
    covered = arms[1]["sd_sampling_covered"]
    varied = len(set(left) - {"other"}) > 1
    report = {"ar_source": str(args.ar), "all_on_source": str(args.all_on), "samples_per_arm": 256,
              "alpha": 0.001, "projection": "A count in choices2–8; malformed eight-choice outputs are other",
              "ar_histogram": left, "all_on_histogram": right, "exact_permutation_p": p,
              "sd_sampling_covered": covered, "ar_variation": varied,
              "passed": covered and varied and p >= 0.001}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))
    assert report["passed"], "Distribution difference or unexercised sampling path; no resampling until passing"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    collection = commands.add_parser("collect")
    collection.add_argument("arm", choices=("ar", "all-on"))
    collection.add_argument("base_url")
    collection.add_argument("output", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("ar", type=Path)
    comparison.add_argument("all_on", type=Path)
    comparison.add_argument("output", type=Path)
    args = parser.parse_args()
    collect(args) if args.action == "collect" else compare(args)


if __name__ == "__main__":
    main()
