"""Shared mode under memory pressure with host copies (contract sections 1, 3, 4, 5): physical
capacity moves between components, paused requests are saved to the host, restored and finish
intact, cancellation during a pause, admission without output reservation, the single-request
cap, and busy maintenance.

    --runtime-cache-gib 0.5 --max-running-requests 6 --prefix-cache-host-gib 4 --enable-cache-report
"""
import os

import pytest

from service_common import COMMON, GIB, MODEL_CONTEXT, components, record, service
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
