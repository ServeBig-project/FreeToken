"""Shared mode under memory pressure with host copies (contract sections 1, 3, 4, 5): physical
capacity moves between components, paused requests are saved to the host, restored and finish
intact, cancellation during a pause, admission without output reservation, the single-request
cap, and busy maintenance.

    --runtime-cache-gib 0.5 --max-running-requests 6 --prefix-cache-host-gib 4 --enable-cache-report
"""
import os

import pytest

from service_common import (COMMON, GIB, MODEL_CONTEXT, Watch, assert_length, cached_tokens, components,
                            enum_prompt, overlap, record, run_streams, service, tok)
from service_scenarios import cancel_round, early_stop_round, over_context, pause_round, short_long_short

NAME = "c_pressure"
ARGS = COMMON + ["--runtime-cache-gib", "0.5", "--max-running-requests", "6",
                 "--prefix-cache-host-gib", "4", "--enable-cache-report"]


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once GPU1 is free")
    with service(NAME, ARGS) as s:
        yield s


def test_ready_publishes_limits(svc):
    st = svc.c.status()
    g, rt, pc = st["geometry"], st["prefix_cache"]["runtime"], st["prefix_cache"]
    assert g["runtime_cache_bytes"] == GIB // 2 == rt["budget_bytes"]
    assert g["moe_cache_size"] == 2048
    assert rt["requested_running_requests"] == 6, rt
    assert rt["max_running_requests"] == min(6, rt["resource_running_requests"]) >= 1, rt
    assert 1 <= rt["context_tokens"] <= MODEL_CONTEXT, rt
    assert {"kv", "gdn_state", "gdn_conv"} <= set(components(rt)), components(rt)
    assert pc["host_budget_bytes"] == 4 * GIB, pc
    record(f"{NAME}:ready", runtime=rt, geometry=g, host={k: v for k, v in pc.items() if k.startswith("host")})


def test_short_long_short_shares_physical_capacity(svc):
    short_long_short(svc, salt=100000)


def test_paused_requests_restore_from_host_and_complete(svc):
    r = pause_round(svc, salt=1000)
    d = r["delta"]
    assert d["paused"] >= 1, f"the pressure load paused nothing: {d}"
    assert d["restored"] >= 1, f"paused with a 4 GiB host budget, yet nothing was restored: {d}"
    assert r["busy"] is not None, "no maintenance probe ran while a request was paused"


def test_cancel_during_pause_others_continue(svc):
    cancel_round(svc, salt=20000)


def test_admission_does_not_reserve_max_tokens(svc):
    early_stop_round(svc, salt=30000, min_overlap=3)


def test_request_beyond_context_tokens_gets_public_error(svc):
    over_context(svc, salt=40000)


def test_runtime_budget_rebuild_with_explicit_concurrency(svc):
    """Section 5: with an explicit concurrency the runtime budget is resized in place, the
    concurrency and the expert capacity stay, and the service keeps generating."""
    mrr = svc.rt()["max_running_requests"]
    for gib in (1, 0.5):
        code, j = svc.c.rebuild({"runtime_cache_gib": gib})
        assert (code, j.get("status")) == (200, "ok"), (gib, code, j)
        g, rt = svc.c.geometry(), svc.rt()
        assert g["runtime_cache_bytes"] == int(gib * GIB) == rt["budget_bytes"], (g, rt)
        assert g["moe_cache_size"] == 2048 and rt["max_running_requests"] == mrr, (g, rt)
        assert_length(svc.c.complete(enum_prompt(60000, 12), 48), 48)


def test_two_long_prompts_that_fit_run_together(svc):
    """Admission: two long requests whose combined need fits the budget run at the same time
    instead of one after the other (contract section 4)."""
    t = tok()
    streams = [svc.c.stream(t.filler(3000, seed=70000 + i) + "\nSummary:", 64, ignore_eos=True) for i in range(2)]
    with Watch(svc.c) as w:
        run_streams(streams, 900)
    record(f"{NAME}:long_pair", overlap=overlap(streams), streams=[s.summary() for s in streams], watch=w.report())
    assert not w.violations, w.report()
    for s in streams:
        assert s.done and s.finish == "length" and s.usage["completion_tokens"] == 64, s.summary()
    assert overlap(streams) == 2, [s.summary() for s in streams]


def test_pressure_keeps_cold_prefixes(svc):
    """Eviction frees only what a request needs: after a long request squeezes the budget,
    cold prefixes remain evictable and an earlier prompt still hits (sections 4, 5)."""
    t = tok()
    warm = t.filler(1000, seed=71000) + "\nQuestion:"
    svc.c.complete(warm, 1, cache_group="keep")
    svc.c.wait_idle()
    host0 = svc.c.status()["prefix_cache"].get("host_reused_tokens", 0)
    assert_length(svc.c.complete(t.filler(7000, seed=71001) + "\nSummary:", 16, cache_group="squeeze"), 16)
    svc.c.wait_idle()
    rt = svc.rt()
    again = svc.c.complete(warm, 1, cache_group="keep")
    host1 = svc.c.status()["prefix_cache"].get("host_reused_tokens", 0)
    record(f"{NAME}:keep_cold", evictable=rt["evictable_bytes"], held=rt["held_bytes"],
           cached=cached_tokens(again["usage"]), host_reused=host1 - host0)
    assert rt["evictable_bytes"] > 0, rt
    assert cached_tokens(again["usage"]) > 0, again["usage"]
