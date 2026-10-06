"""Qwen3.6 linear chain and revisits, paired across the original version, host=0 and host>0."""

import os

import pytest

from harness import QWEN36, Server, cached, document, record, text
from scenarios import pressure

BASE = os.environ.get("FT_BASE_SOURCE")
SIZE = ["--num-tokens", "4096", "--max-running-requests", "4"]


def chain(s, rounds=5, n=12):
    cur, prompts, rows = document(3, 30), [], []
    for i in range(rounds):
        r = s.complete(cur, n, group="linear")
        prompts.append(cur)
        rows.append(("round", i, cached(r), text(r)))
        cur = cur + text(r) + f" User turn {i}: please continue the notes."
    for i in (1, 3):
        r = s.complete(prompts[i], n, group="linear")
        rows.append(("revisit", i, cached(r), text(r)))
    pressure(s, range(40, 46), 30)
    r = s.complete(cur, n, group="linear")
    rows.append(("after_pressure", rounds, cached(r), text(r)))
    for i in (1, 3):
        r = s.complete(prompts[i], n, group=f"linear-cold-{i}")
        rows.append(("cold_recompute", i, cached(r), text(r)))
    return rows


ARMS = {
    "host0": SIZE + ["--prefix-cache-host-gib", "0"],
    "host4_baseline": SIZE + ["--prefix-cache-host-gib", "4"],
    "host4_continuation": SIZE + ["--prefix-cache-host-gib", "4", "--prefix-cache-policy", "continuation"],
    "replay_host0": SIZE + ["--enable-gdn-replayssm", "--prefix-cache-host-gib", "0"],
    "replay_host4_continuation": SIZE + ["--enable-gdn-replayssm", "--prefix-cache-host-gib", "4",
                                         "--prefix-cache-policy", "continuation"],
}


@pytest.mark.skipif(not BASE, reason="FT_BASE_SOURCE (original version python/ path) not set")
def test_chain_paired():
    results = {}
    for name, args in [("original", SIZE), ("original_replay", SIZE + ["--enable-gdn-replayssm"])] + list(ARMS.items()):
        s = Server(f"q36_paired_{name}", QWEN36 + args, source=BASE if name.startswith("original") else None)
        try:
            results[name] = chain(s)
        finally:
            s.close()
    record("q36_paired/rows", results=results)
    problems = []
    for name in ARMS:
        ref = results["original_replay" if name.startswith("replay") else "original"]
        colds = {row[1]: row[3] for row in results[name] if row[0] == "cold_recompute"}
        for got, want in zip(results[name], ref):
            if got[3] == want[3]:
                continue
            # a revisit that reused a shorter, earlier state may match the cold-recompute reference instead
            if got[0] == "revisit" and got[2] != want[2] and got[3] == colds.get(got[1]):
                continue
            problems.append({"arm": name, "step": got[:3], "original_cached": want[2],
                             "got": got[3][:80], "original": want[3][:80], "cold": colds.get(got[1], "")[:80]})
    record("q36_paired/verdict", ok=not problems, problems=problems)
    assert not problems, problems
