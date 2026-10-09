"""Shared mode at a comfortable budget with the concurrency limit omitted (contract sections 2,
4, 5): published limits, continuous mixed arrivals, users and multi-turn reuse, admission without
output reservation, Graph execution, and maintenance through the public rebuild interface.

    --runtime-cache-gib 4 --enable-cache-report   (no --max-running-requests)
"""
import os
import time

import pytest

from service_common import (COMMON, GIB, MODEL_CONTEXT, Watch, assert_length, cached_tokens,
                            components, enum_prompt, graph_replays, record, sd_enabled, service,
                            start_streams, tok, wait_streams)
from service_scenarios import early_stop_round

NAME = "b_shared"
ARGS = COMMON + ["--runtime-cache-gib", "4", "--enable-cache-report"]


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once GPU1 is free")
    with service(NAME, ARGS) as s:
        yield s


def test_ready_publishes_shared_mode_and_derived_concurrency(svc):
    st = svc.c.status()
    g, rt = st["geometry"], st["prefix_cache"]["runtime"]
    assert g["runtime_cache_bytes"] == 4 * GIB == rt["budget_bytes"]
    assert g["num_pages"] == 0 and g["num_mamba_slots"] == 0, g
    assert g["address_pages"] > 0 and g["address_mamba_slots"] > 0, g
    assert g["moe_cache_size"] == 2048
    assert 0 <= rt["used_bytes"] <= rt["held_bytes"] <= rt["budget_bytes"], rt
    assert {"kv", "gdn_state", "gdn_conv"} <= set(components(rt)), components(rt)
    assert rt["requested_running_requests"] is None, rt  # omitted, not the old default
    assert rt["max_running_requests"] == rt["resource_running_requests"] >= 1, rt
    assert rt["model_context_tokens"] == MODEL_CONTEXT
    assert 1 <= rt["context_tokens"] <= MODEL_CONTEXT, rt
    assert rt.get("requested_context_tokens") in (None, 0), rt
    stats = svc.c.stats()
    assert not sd_enabled(stats), stats.get("speculative")
    assert stats["cuda_graph"]["enabled"], stats.get("cuda_graph")
    record(f"{NAME}:ready", runtime=rt, geometry=g, execution=stats.get("execution"))


def test_continuous_mixed_arrivals_all_complete(svc):
    """Unknown load: prompts of 30-2500 tokens with 8-200 token outputs arriving one per second
    from four users, plus two chat requests allowed to stop at EOS."""
    t = tok()
    shapes = [(30, 96), (2500, 8), (120, 200), (600, 32), (1500, 96), (30, 200),
              (2500, 32), (600, 8), (120, 96), (1500, 200), (30, 32), (600, 96)]
    streams = [svc.c.stream(t.filler(p, seed=100 + i), n, ignore_eos=True, cache_group=f"user{i % 4}")
               for i, (p, n) in enumerate(shapes)]
    with Watch(svc.c) as w:
        ex, futures = start_streams(streams, stagger_s=1.0)
        chats = [svc.c.chat("Reply with exactly the word: yes", 64) for _ in range(2)]
        wait_streams(ex, futures, 1800)
    record(f"{NAME}:mixed_arrivals", watch=w.report(), streams=[s.summary() for s in streams],
           chats=[(c["finish"], c["usage"]) for c in chats])
    assert not w.violations, w.report()
    for s, (_, n) in zip(streams, shapes):
        assert s.error is None and s.done and s.finish == "length", s.summary()
        assert s.usage["completion_tokens"] == n, s.usage
    for c in chats:
        assert c["finish"] in ("stop", "length") and c["usage"]["completion_tokens"] >= 1, c
    svc.c.wait_idle()


def test_users_and_multi_turn_reuse(svc):
    """A second turn of the same user reuses the first turn; another user does not (section 3)."""
    p = tok().filler(1200, seed=31) + "\nQuestion: summarize the notes.\nAnswer:"
    r1 = svc.c.complete(p, 48, cache_group="u1")
    pt1 = r1["usage"]["prompt_tokens"]
    assert cached_tokens(r1["usage"]) == 0, r1["usage"]
    r2 = svc.c.complete(p + r1["text"] + "\nQuestion: and then?\nAnswer:", 32, cache_group="u1")
    assert cached_tokens(r2["usage"]) >= int(0.8 * pt1), (pt1, r2["usage"])
    r3 = svc.c.complete(p, 48, cache_group="u2")
    assert cached_tokens(r3["usage"]) == 0, r3["usage"]
    record(f"{NAME}:multi_turn", pt1=pt1, cached=[cached_tokens(r["usage"]) for r in (r1, r2, r3)],
           same_text=r1["text"] == r3["text"])
    assert r1["text"] == r3["text"]  # same configuration, serial, both a fresh prefill


def test_admission_does_not_reserve_max_tokens(svc):
    early_stop_round(svc, salt=30000, min_overlap=2)


def test_graph_replays_in_shared_mode(svc):
    before = graph_replays(svc.c.stats())
    assert_length(svc.c.complete(enum_prompt(40000, 12), 48), 48)
    after = graph_replays(svc.c.stats())
    assert after > before, svc.c.stats().get("cuda_graph")


def _keys(g):
    return {k: g[k] for k in ("runtime_cache_bytes", "moe_cache_size", "num_pages", "num_mamba_slots")}


def test_maintenance_runtime_budget_and_old_fields(svc):
    """Section 5: busy while a request runs; the old split-pool fields are rejected and the
    service keeps serving; illegal budgets and a budget too small for the derived concurrency are
    rejected; a legal rebuild keeps the expert capacity unless given; Graph survives the rebuild."""
    c = svc.c
    g0 = _keys(c.geometry())
    s = c.stream(enum_prompt(50000, 12), 400, ignore_eos=True)
    ex, futures = start_streams([s])
    time.sleep(3)
    busy = c.rebuild({"runtime_cache_gib": 4}, timeout=1)
    wait_streams(ex, futures, 900)
    assert busy[1].get("status") == "busy", busy  # HTTP 503 or 409 by the published configuration
    assert s.done and s.finish == "length" and s.usage["completion_tokens"] == 400, s.summary()
    c.wait_idle()
    log = {"busy": busy}
    for field, value in (("num_pages", 4096), ("num_mamba_slots", 64), ("num_swa_pages", 1024),
                         ("swa_full_tokens_ratio", 0.5)):
        code, j = c.rebuild({field: value})
        log[field] = (code, j)
        assert (code, j.get("status")) == (503, "rejected") and j.get("error"), (field, code, j)
        assert _keys(c.geometry()) == g0 and c.status()["state"] == "serving"
        assert_length(c.complete(enum_prompt(51000, 8), 8), 8)
    for value in (0, 100):
        code, j = c.rebuild({"runtime_cache_gib": value})
        log[f"runtime_cache_gib={value}"] = (code, j)
        assert (code, j.get("status")) == (503, "rejected") and j.get("error"), (value, code, j)
        assert _keys(c.geometry()) == g0
        assert_length(c.complete(enum_prompt(52000, 8), 8), 8)
    # the derived concurrency stays for the service lifetime; a budget too small for it is refused
    mrr = c.rt()["max_running_requests"]
    code, j = c.rebuild({"runtime_cache_gib": 3})
    log["runtime_cache_gib=3"] = (code, j)
    assert j.get("status") == "rejected" and "requests at their minimum" in (j.get("error") or ""), (code, j)
    assert _keys(c.geometry()) == g0 and c.rt()["max_running_requests"] == mrr
    assert_length(c.complete(enum_prompt(53000, 12), 48), 48)
    code, j = c.rebuild({"runtime_cache_gib": 4, "moe_cache_size": 1536})
    log["runtime_cache_gib=4,moe=1536"] = (code, j)
    assert (code, j.get("status")) == (200, "ok"), (code, j)
    g = c.geometry()
    assert g["runtime_cache_bytes"] == 4 * GIB and g["moe_cache_size"] == 1536, g
    assert_length(c.complete(enum_prompt(54000, 12), 48), 48)
    code, j = c.rebuild({"runtime_cache_gib": 4, "moe_cache_size": 2048})
    assert (code, j.get("status")) == (200, "ok"), (code, j)
    assert _keys(c.geometry()) == g0
    record(f"{NAME}:maintenance", log=log, runtime_after=c.rt())
    before = graph_replays(c.stats())
    assert_length(c.complete(enum_prompt(55000, 12), 48), 48)
    assert graph_replays(c.stats()) > before, c.stats().get("cuda_graph")
