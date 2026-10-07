"""host=0 baseline policy versus the fixed original version, identical public workload."""

import os
import time

import pytest

from harness import QWEN3, QWEN36, Server, cached, chat, document, gpu_used_mib, post, record, text
from scenarios import pressure

BASE = os.environ.get("FT_BASE_SOURCE")


def workload(s, sentences, pressure_count):
    out = {"geometry": {k: s.status()["geometry"].get(k) for k in ("num_pages", "moe_cache_size", "num_mamba_slots",
                                                                 "cache_budget_bytes", "limits")}}
    out["gpu_mib_idle"] = gpu_used_mib()
    p = document(1, sentences)
    for name, group in (("cold", None), ("hot", None), ("other_group", "g"), ("other_group_hot", "g")):
        r = s.complete(p, 16, group=group)
        out[name] = (text(r), cached(r))
    rs = s.parallel([lambda i=i: s.complete(document(10 + i, sentences // 2), 16 + 8 * i) for i in range(3)])
    out["concurrent"] = [(text(r), cached(r)) for r in rs]
    out["stream_cancel_chunks"] = len(s.stream(document(4, sentences), 64, cancel_after=3)["chunks"])
    time.sleep(1)
    r = s.complete(document(4, sentences), 16)
    out["after_cancel"] = (text(r), cached(r))
    full = s.stream(document(4, sentences), 16)
    out["stream_full"] = (full["text"], full["done"])
    pressure(s, range(40, 40 + pressure_count), sentences)
    r = s.complete(p, 16)
    out["after_pressure"] = (text(r), cached(r))
    r = chat(s, "Name three colors.", 24)
    out["chat"] = r["body"]["choices"][0]["message"].get("content")
    bad = post(s.url, "/v1/cache/rebuild", {"num_pages": 10 ** 9})
    out["rebuild_bad"] = (bad["status"], bad["body"].get("status"))
    r = s.complete(p, 16)
    out["after_bad_rebuild"] = (text(r), cached(r))
    ok = post(s.url, "/v1/cache/rebuild", {"num_pages": out["geometry"]["num_pages"]})
    out["rebuild_ok"] = (ok["status"], ok["body"].get("status"))
    while s.status()["state"] != "serving":
        time.sleep(1)
    r = s.complete(p, 16)
    out["after_rebuild"] = (text(r), cached(r))
    r = s.complete(p, 16)
    out["after_rebuild_hot"] = (text(r), cached(r))
    return out


CASES = {
    "q3": (QWEN3 + ["--num-tokens", "3072", "--max-running-requests", "4"], 30, 5),
    "q36": (QWEN36 + ["--num-tokens", "4096", "--max-running-requests", "4"], 30, 6),
}


@pytest.mark.skipif(not BASE, reason="FT_BASE_SOURCE (original version python/ path) not set")
@pytest.mark.parametrize("name", sorted(CASES))
def test_host0_matches_original(name):
    args, sentences, count = CASES[name]
    s = Server(f"off_{name}_base", args, source=BASE)
    try:
        base = workload(s, sentences, count)
        base_status = s.status()
    finally:
        s.close()
    s = Server(f"off_{name}_host0", args + ["--prefix-cache-host-gib", "0", "--prefix-cache-policy", "baseline"])
    try:
        new = workload(s, sentences, count)
        pc = s.pc()
    finally:
        s.close()
    record(f"off_{name}", base=base, new=new, base_has_prefix_cache="prefix_cache" in base_status, pc=pc)
    diffs = {k: (base[k], new[k]) for k in base if k != "gpu_mib_idle" and base[k] != new[k]}
    gpu_delta = new["gpu_mib_idle"] - base["gpu_mib_idle"]
    zeros = {k: pc[k] for k in ("host_budget_bytes", "host_allocated_bytes", "host_used_bytes", "transfer_device_bytes",
                                "h2d_bytes", "d2h_bytes", "host_checkpoint_count", "host_reused_tokens")}
    problems = []
    if diffs:
        problems.append(f"public results differ from original: {diffs}")
    if gpu_delta > 32:
        problems.append(f"host=0 uses {gpu_delta} MiB more GPU memory than original at idle")
    if pc["enabled"] or any(zeros.values()):
        problems.append(f"host=0 status not disabled/zero: enabled={pc['enabled']} {zeros}")
    record(f"off_{name}/verdict", ok=not problems, problems=problems, gpu_delta_mib=gpu_delta)
    assert not problems, problems
