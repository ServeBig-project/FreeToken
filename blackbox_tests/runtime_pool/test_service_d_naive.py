"""Shared mode without a public prefix cache and without a host budget (contract sections 2-5):
a large explicit concurrency is bounded by the resource-derived value, paused requests keep their
input and committed output and recompute their state, no undeclared host pool appears, prefix
reuse stays off, cancellation during a pause, admission and the single-request cap.

    --runtime-cache-gib 0.5 --max-running-requests 64 --prefix-cache-host-gib 0 --cache-type naive
"""
import os

import pytest

from service_common import COMMON, GIB, MODEL_CONTEXT, assert_length, cached_tokens, enum_prompt, record, service
from service_scenarios import cancel_round, early_stop_round, over_context, pause_round

NAME = "d_naive"
ARGS = COMMON + ["--runtime-cache-gib", "0.5", "--max-running-requests", "64",
                 "--prefix-cache-host-gib", "0", "--cache-type", "naive", "--enable-cache-report"]


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once GPU1 is free")
    with service(NAME, ARGS) as s:
        yield s


def _host(pc):
    return {k: v for k, v in pc.items() if k.startswith("host")}


def test_ready_bounds_a_large_explicit_concurrency(svc):
    st = svc.c.status()
    g, rt, pc = st["geometry"], st["prefix_cache"]["runtime"], st["prefix_cache"]
    assert g["runtime_cache_bytes"] == GIB // 2 == rt["budget_bytes"]
    assert rt["requested_running_requests"] == 64, rt
    assert rt["max_running_requests"] == min(64, rt["resource_running_requests"]) >= 1, rt
    assert 1 <= rt["context_tokens"] <= MODEL_CONTEXT, rt
    assert pc.get("host_budget_bytes", 0) == 0 and pc.get("host_allocated_bytes", 0) == 0, _host(pc)
    record(f"{NAME}:ready", runtime=rt, geometry=g, host=_host(pc), enabled=pc.get("enabled"))


def test_no_prefix_reuse_in_naive_mode(svc):
    p = enum_prompt(500, 40)
    first = svc.c.complete(p, 16)
    again = svc.c.complete(p, 16)
    assert_length(first, 16)
    assert_length(again, 16)
    assert cached_tokens(first["usage"]) == 0 and cached_tokens(again["usage"]) == 0, (first["usage"], again["usage"])
    assert first["text"] == again["text"]


def test_paused_requests_recompute_and_complete(svc):
    r = pause_round(svc, salt=1000)
    d = r["delta"]
    pc = svc.c.status()["prefix_cache"]
    record(f"{NAME}:recompute", delta=d, host=_host(pc))
    assert d["paused"] >= 1, f"the pressure load paused nothing: {d}"
    assert d["recompute"] >= 1 and d["recomputed_tokens"] > 0, f"host budget 0, yet no recompute: {d}"
    assert d["restored"] == 0, f"host budget 0, yet a restore from a host copy was counted: {d}"
    assert pc.get("host_allocated_bytes", 0) == 0 and pc.get("host_used_bytes", 0) == 0, _host(pc)


def test_cancel_during_pause_others_continue(svc):
    cancel_round(svc, salt=20000, preamble="")


def test_admission_does_not_reserve_max_tokens(svc):
    early_stop_round(svc, salt=30000, min_overlap=3)


def test_request_beyond_context_tokens_gets_public_error(svc):
    over_context(svc, salt=40000)
