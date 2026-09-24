"""One fixed scored segment after fixed public presampling requests."""

from collections import Counter, defaultdict
import json
from math import comb

from test_serving import (
    BASELINE, CANDIDATE, SAMPLED, WIDTH, artifacts, checkpoint, sample_bucket, serve,
)


def exact_permutation_p(left, right):
    n = sum(left.values())
    assert n == sum(right.values())
    keys = left.keys() | right.keys()
    totals = sorted(left.get(key, 0) + right.get(key, 0) for key in keys)
    observed = sum(abs(left.get(key, 0) - right.get(key, 0)) for key in keys)
    # Leaving the largest category last fixes its count from the required sample size.
    states = {(0, 0): 1}
    for total in totals[:-1]:
        updated = defaultdict(int)
        for (selected, distance), ways in states.items():
            for count in range(min(total, n - selected) + 1):
                updated[(selected + count, distance + abs(2 * count - total))] += ways * comb(total, count)
        states = updated
    tail = mass = 0
    largest = totals[-1]
    for (selected, distance), ways in states.items():
        count = n - selected
        if 0 <= count <= largest:
            weight = ways * comb(largest, count)
            mass += weight
            if distance + abs(2 * count - largest) >= observed:
                tail += weight
    assert mass == comb(2 * n, n)
    return tail / mass


def test_fixed_sampling_confirmation(checkpoint, artifacts):
    """A fresh distribution difference or absent speculation blocks confirmation."""
    arms = {}
    presampling_request = {**SAMPLED, "max_tokens": 224, "ignore_eos": True}
    for name, package, steps in [("confirmation-baseline", BASELINE, 0),
                                 ("confirmation-reduced", CANDIDATE, 4)]:
        raw = {"presampling": [], "scored": []}
        phase = "presampling"

        def retain_response(response):
            if response.request.method == "POST":
                response.read()
                raw[phase].append({"request": json.loads(response.request.content),
                                   "status_code": response.status_code, "body": response.text})

        try:
            with serve(name, package, artifacts, steps=steps, experts=3) as server:
                server.client.event_hooks["response"].append(retain_response)
                presamples = []
                for _ in range(8 // WIDTH):
                    presamples.extend(server.batch([presampling_request] * WIDTH, "confirmation-presampling"))
                assert len(presamples) == len(raw["presampling"]) == 8
                assert all(row["usage"]["completion_tokens"] == 224 for row in presamples)
                phase = "scored"
                before = server.idle()
                samples = []
                for _ in range(512 // WIDTH):
                    samples.extend(server.batch([SAMPLED] * WIDTH, "confirmation-sampling"))
                after = server.idle()
                assert len(samples) == len(raw["scored"]) == 512
                arms[name] = {"histogram": dict(Counter(sample_bucket(row) for row in samples)),
                              "speculative_before": before.get("speculative"),
                              "speculative_after": after.get("speculative")}
        finally:
            for section, responses in raw.items():
                (artifacts / f"{name}-{section}-raw.json").write_text(json.dumps(responses, ensure_ascii=False, indent=2))
    baseline = arms["confirmation-baseline"]["histogram"]
    reduced = arms["confirmation-reduced"]["histogram"]
    p = exact_permutation_p(baseline, reduced)
    (artifacts / "sampling-confirmation.json").write_text(json.dumps(
        {"presampling": {"requests_per_arm": 8, "request": presampling_request},
         "samples_per_arm": 512, "alpha": 0.001, "request": SAMPLED,
         "concurrency": WIDTH, "arms": arms, "exact_permutation_p": p}, indent=2))
    assert len(set(baseline) - {"other"}) > 1, "Confirmation coverage missing: baseline projection has no variation"
    before = arms["confirmation-reduced"]["speculative_before"]
    after = arms["confirmation-reduced"]["speculative_after"]
    assert after["enabled"] is True
    for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps"):
        assert after[key] > before[key], f"Confirmation coverage missing: no speculative {key}"
    assert p >= 0.001, f"Fresh target distribution differs (exact permutation p={p}); no automatic resampling"
