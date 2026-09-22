"""Fixed 256-per-arm adaptive confirmation, using public requests and stats."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import httpx

from evaluate_http import Server, stream
from test_serving import SAMPLED
from sampling_confirmation import exact_permutation_p


def collect(args):
    data = {"arm": args.arm, "presampling": [], "scored": []}
    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        assert client.get("/health").json()["status"] == "ok"
        model = client.get("/v1/models").json()["data"][0]["id"]
        observer = Server(args.arm, client)
        try:
            for section, count, body in [("presampling", 8, {**SAMPLED, "max_tokens": 224}),
                                         ("scored", 256, SAMPLED)]:
                before = observer.idle()
                assert before["speculative"]["enabled"] is True
                assert before["speculative"]["adaptive_enabled"] is (args.arm == "adaptive")
                assert before["speculative"]["reuse_enabled"] is False
                assert before["moe_residency"]["resident_experts"] == 1024
                assert before["moe_residency"]["cache_slots"] == 1536
                for _ in range(count // 4):
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        data[section].extend(pool.map(lambda _: stream(client, model, body), range(4)))
                after = observer.idle()
                data[section + "_stats"] = {"before": before, "after": after}
                assert len(data[section]) == count
                assert all(row["usage"]["completion_tokens"] == body["max_tokens"] for row in data[section])
            before = data["scored_stats"]["before"]["speculative"]
            after = data["scored_stats"]["after"]["speculative"]
            for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps"):
                assert after[key] > before[key], f"Speculation coverage missing: {key}"
            if args.arm == "adaptive":
                assert after["adaptive_stops"] > before["adaptive_stops"], "Adaptive stopping not exercised by scored requests"
            data["passed_collection"] = True
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def compare(args):
    from test_serving import sample_bucket
    arms = [json.loads(path.read_text()) for path in (args.fixed, args.adaptive)]
    assert [arm["arm"] for arm in arms] == ["fixed", "adaptive"]
    assert all(arm["passed_collection"] and len(arm["scored"]) == 256 for arm in arms)
    histograms = [dict(Counter(sample_bucket(row) for row in arm["scored"])) for arm in arms]
    assert len(set(histograms[0]) - {"other"}) > 1, "Fixed-control projection has no variation"
    p = exact_permutation_p(*histograms)
    report = {"samples_per_arm": 256, "alpha": 0.001, "fixed_histogram": histograms[0],
              "adaptive_histogram": histograms[1], "exact_permutation_p": p,
              "fixed_source": str(args.fixed), "adaptive_source": str(args.adaptive)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    assert p >= 0.001, "Adaptive distribution differs in this fixed test; do not resample until passing"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="action", required=True)
    sampling = subcommands.add_parser("collect")
    sampling.add_argument("arm", choices=("fixed", "adaptive"))
    sampling.add_argument("base_url")
    sampling.add_argument("output", type=Path)
    comparison = subcommands.add_parser("compare")
    comparison.add_argument("fixed", type=Path)
    comparison.add_argument("adaptive", type=Path)
    comparison.add_argument("output", type=Path)
    args = parser.parse_args()
    collect(args) if args.action == "collect" else compare(args)


if __name__ == "__main__":
    main()
