"""Replay only ordinary candidate serving with CUDA graphs disabled."""

import json
import os
from pathlib import Path

from test_serving import CANDIDATE, GPU, serve


def load_requests():
    evidence = Path("/data2/servebig-envs/self_speculative_20260921/run4")
    ordinary = json.loads((evidence / "disabled.json").read_text())["requests"]
    single = json.loads((evidence / "single.json").read_text())["requests"]
    story = next(row for row in ordinary if row["request"].get("prompt", "").startswith("Continue a detailed story:"))
    # Saved rows finish out of order; restore the original mixed submission order.
    mixed = sorted(single[-6:], key=lambda row: (1, 17, 16, 32, 9, 8).index(row["request"]["max_tokens"]))
    references = [next(row["response"] for row in ordinary if row["request"] == item["request"]) for item in mixed]
    return story, mixed, references


def main():
    if not GPU:
        raise SystemExit("Set FT_SD_GPU to the coordinator-assigned GPU")
    artifacts = Path(os.environ["FT_SD_ARTIFACTS"])
    artifacts.mkdir(parents=True, exist_ok=True)
    story, mixed, references = load_requests()
    with serve("ordinary-eager", CANDIDATE, artifacts,
               cache=os.environ.get("FT_SD_DIAGNOSTIC_CACHE", "radix"),
               extra_args=["--speculative-num-steps", "0", "--cuda-graph-max-bs", "0"]) as server:
        story_result = server.call(story["request"])
        mixed_results = server.batch([row["request"] for row in mixed], "heterogeneous-concurrency")
        stats = server.idle()
    comparisons = []
    for row, ordinary, result in zip(mixed, references, mixed_results):
        deterministic = row["request"]["temperature"] == 0 or row["request"].get("top_k") == 1
        comparisons.append({"request": row["request"], "ordinary_reference": ordinary,
                            "single_reference": row["response"], "ordinary_eager": result,
                            "matches_ordinary": result == ordinary if deterministic else None,
                            "matches_single": result == row["response"] if deterministic else None})
    report = {"story": {"request": story["request"], "ordinary_reference": story["response"],
                        "ordinary_eager": story_result, "matches_ordinary": story_result == story["response"]},
              "mixed": comparisons, "speculative": stats["speculative"]}
    (artifacts / "ordinary-eager-comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
