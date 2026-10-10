"""Execution memory covers the token table and sampling peaks (round J item 3): with the
concurrency omitted at a large runtime, a burst of the full derived concurrency of sampled
(temperature 0.8) short requests completes and the service stays alive; after a runtime
rebuild the same burst completes again.

    --runtime-cache-gib 6   (no --max-running-requests)
"""
import os

import pytest

from service_common import COMMON, GIB, record, service
from service_scenarios import burst, shared_stats_contract

NAME = "j4_auto_burst"
ARGS = COMMON + ["--runtime-cache-gib", "6", "--enable-cache-report"]
SAMPLED = {"temperature": 0.8, "top_p": 0.95, "top_k": 20}


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once a GPU is free")
    with service(NAME, ARGS) as s:
        yield s


def test_ready(svc):
    rt = svc.rt()
    assert rt["requested_running_requests"] is None and rt["max_running_requests"] >= 1, rt
    record(f"{NAME}:ready", max_running_requests=rt["max_running_requests"])
    shared_stats_contract(svc)


def test_sampled_burst_at_full_derived_concurrency(svc):
    burst(svc, svc.rt()["max_running_requests"], 100, 64, seed=84000, **SAMPLED)


def test_same_burst_after_runtime_rebuild(svc):
    mrr = svc.rt()["max_running_requests"]
    code, j = svc.c.rebuild({"runtime_cache_gib": 7})
    assert (code, j.get("status")) == (200, "ok"), (code, j)
    assert svc.c.geometry()["runtime_cache_bytes"] == 7 * GIB and svc.rt()["max_running_requests"] == mrr
    burst(svc, mrr, 100, 64, seed=85000, **SAMPLED)
