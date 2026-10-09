"""DFlash speculative decoding in shared mode (contract sections 2, 5): real SD and Graph
execution, an explicit concurrency equal to the old default still applied, concurrent SD
requests intact under one budget, a long-prompt greedy output recorded for the split-pool
baseline comparison, and SD still real after a runtime budget rebuild.

    --runtime-cache-gib 4 --max-running-requests 4 --speculative-num-steps 4
    --speculative-draft-model-path <DFlash> --enable-cache-report
"""
import os

import pytest

from service_common import (COMMON, DFLASH_ARGS, GIB, Watch, assert_enum, assert_length, components,
                            dump, enum_prompt, graph_replays, record, run_streams, sd_enabled, sd_num,
                            service, tok)

NAME = "e_dflash"
ARGS = COMMON + ["--runtime-cache-gib", "4", "--max-running-requests", "4", "--enable-cache-report"] + DFLASH_ARGS
PARITY_TOKENS = 160


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once GPU1 is free")
    with service(NAME, ARGS) as s:
        yield s


def parity_prompt():
    return tok().filler(3000, seed=424242) + "\n\nSummary of the notes above:"


def sd_counts(stats):
    return {k: sd_num(stats, k) for k in ("drafted", "accepted", "rounds")}


def test_ready_sd_and_explicit_old_default_concurrency(svc):
    st, stats = svc.c.status(), svc.c.stats()
    g, rt = st["geometry"], st["prefix_cache"]["runtime"]
    assert g["runtime_cache_bytes"] == 4 * GIB == rt["budget_bytes"]
    assert rt["requested_running_requests"] == 4, rt
    assert rt["max_running_requests"] == min(4, rt["resource_running_requests"]), rt
    names = set(components(rt))  # drafter history shares the budget under draft_* names
    assert {"kv", "gdn_state"} <= names and any(n.startswith("draft_") for n in names), names
    assert sd_enabled(stats), stats.get("speculative")
    assert stats["speculative"].get("max_draft_steps") == 4, stats["speculative"]
    assert stats["cuda_graph"]["enabled"], stats.get("cuda_graph")
    record(f"{NAME}:ready", runtime=rt, speculative=stats["speculative"], execution=stats.get("execution"))


def test_decode_runs_real_sd_and_graph(svc):
    before, g0 = sd_counts(svc.c.stats()), graph_replays(svc.c.stats())
    r = svc.c.complete(enum_prompt(1000, 12), 96)
    assert_length(r, 96)
    assert_enum(r["text"], 1012, 96)
    after = sd_counts(svc.c.stats())
    d = {k: after[k] - before[k] for k in after}
    record(f"{NAME}:decode_sd", delta=d, graph=graph_replays(svc.c.stats()) - g0)
    assert d["rounds"] > 0 and d["drafted"] > 0, f"no real SD during plain decode: {d}"
    assert 0 <= d["accepted"] <= d["drafted"], d
    assert graph_replays(svc.c.stats()) > g0, svc.c.stats().get("cuda_graph")


def test_long_prompt_greedy_output_recorded_and_repeatable(svc):
    """Two fresh prefills of the same long prompt (different cache groups) agree exactly; the
    hot repeat within one group is reported, since prefix reuse is a different execution plan."""
    p = parity_prompt()
    first = svc.c.complete(p, PARITY_TOKENS, cache_group="parity-a")
    assert_length(first, PARITY_TOKENS)
    fresh = svc.c.complete(p, PARITY_TOKENS, cache_group="parity-b")
    assert_length(fresh, PARITY_TOKENS)
    hot = svc.c.complete(p, PARITY_TOKENS, cache_group="parity-a")
    dump("e_dflash_parity", {"prompt": p, "max_tokens": PARITY_TOKENS, "text": first["text"],
                             "usage": first["usage"], "args": ARGS})
    record(f"{NAME}:parity_repeat", fresh_equal=fresh["text"] == first["text"],
           hot_equal=hot["text"] == first["text"], hot_usage=hot["usage"])
    assert fresh["text"] == first["text"], (first["text"][:200], fresh["text"][:200])


def test_concurrent_sd_requests_intact_under_one_budget(svc):
    starts = [10000 + 1000 * i for i in range(4)]
    streams = [svc.c.stream(enum_prompt(s, 40), 300, ignore_eos=True) for s in starts]
    before = sd_counts(svc.c.stats())
    with Watch(svc.c) as w:
        run_streams(streams, 900)
    d = {k: v - before[k] for k, v in sd_counts(svc.c.stats()).items()}
    record(f"{NAME}:concurrent_sd", delta=d, watch=w.report(), streams=[s.summary() for s in streams])
    assert not w.violations, w.report()
    for s, start in zip(streams, starts):
        assert s.error is None and s.done and s.finish == "length", s.summary()
        assert s.usage["completion_tokens"] == 300, s.usage
        assert_enum(s.text, start + 40, 300)
    assert d["rounds"] > 0 and d["drafted"] > 0, d
    svc.c.wait_idle()


def test_runtime_budget_rebuild_keeps_real_sd(svc):
    code, j = svc.c.rebuild({"runtime_cache_gib": 3})
    assert (code, j.get("status")) == (200, "ok"), (code, j)
    assert svc.c.geometry()["runtime_cache_bytes"] == 3 * GIB == svc.rt()["budget_bytes"]
    assert svc.c.geometry()["moe_cache_size"] == 2048
    before = sd_counts(svc.c.stats())
    r = svc.c.complete(enum_prompt(20000, 12), 64)
    assert_length(r, 64)
    d = {k: v - before[k] for k, v in sd_counts(svc.c.stats()).items()}
    assert d["rounds"] > 0 and d["drafted"] > 0, d
    code, j = svc.c.rebuild({"runtime_cache_gib": 4})
    assert (code, j.get("status")) == (200, "ok"), (code, j)
    assert svc.c.geometry()["runtime_cache_bytes"] == 4 * GIB
