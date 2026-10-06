"""gpt-oss-20b: the same reuse-after-pressure and chain workload on the original version, host=0 and host=2."""

import os

import pytest

from harness import Server, cached, document, gptoss, record, text
from scenarios import exact_prompt, pressure

BASE = os.environ.get("FT_BASE_SOURCE")
SIZE = ["--num-tokens", "4096", "--max-running-requests", "4"]
ARMS = [("original", SIZE, BASE), ("host0", SIZE + ["--prefix-cache-host-gib", "0"], None),
        ("host2", SIZE + ["--prefix-cache-host-gib", "2"], None),
        ("host2_continuation", SIZE + ["--prefix-cache-host-gib", "2", "--prefix-cache-policy", "continuation"], None)]


def workload(s):
    rows = []
    p = document(1, 30)
    for step in ("first", "hot"):
        r = s.complete(p, 12, group="cold")
        rows.append((step, cached(r), text(r)))
    pressure(s, range(40, 46), 30)
    r = s.complete(p, 12, group="cold")
    rows.append(("after_pressure", cached(r), text(r)))
    r = s.complete(p, 12, group="cold-fresh")
    rows.append(("cold_recompute", cached(r), text(r)))
    q = exact_prompt(s, 301, salt="winchain")
    r0 = s.complete(q, 24, group="chain")
    cont = q + text(r0) + " Then"
    for step in ("cont", "cont_hot"):
        r = s.complete(cont, 24, group="chain")
        rows.append((step, cached(r), text(r)))
    pressure(s, range(340, 346), 30)
    r = s.complete(cont, 24, group="chain")
    rows.append(("cont_after_pressure", cached(r), text(r)))
    r = s.complete(cont, 24, group="chain-fresh")
    rows.append(("cont_cold_recompute", cached(r), text(r)))
    return rows


@pytest.mark.skipif(not BASE, reason="FT_BASE_SOURCE (original version python/ path) not set")
def test_paired():
    results = {}
    for name, args, source in ARMS:
        s = Server(f"oss_paired_{name}", gptoss() + args, source=source)
        try:
            results[name] = workload(s)
        finally:
            s.close()
    record("oss_paired/rows", results=results)
    problems = []
    for name in results:
        rows = dict((r[0], r) for r in results[name])
        orig = dict((r[0], r) for r in results["original"])
        for step, row in rows.items():
            if name != "original" and row[2] != orig[step][2]:
                problems.append({"arm": name, "step": step, "cached": row[1], "original_cached": orig[step][1],
                                 "got": row[2][:80], "original": orig[step][2][:80]})
    record("oss_paired/verdict", ok=not problems, problems=problems)
    assert not problems, problems
