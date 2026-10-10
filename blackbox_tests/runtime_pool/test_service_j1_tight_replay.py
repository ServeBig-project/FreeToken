"""Startup capacity at a tight budget and low concurrency, without DFlash, with ReplaySSM
(round J items 1 and 6): the service starts, serves the full concurrency of ~1000-token
prompts with 256-token outputs, publishes runtime memory instead of a page total, and
reports ReplaySSM metadata bytes.

    --runtime-cache-gib 0.5 --max-running-requests 4 --enable-gdn-replayssm
"""
import os

import pytest

from service_common import COMMON, service
from service_scenarios import burst, shared_stats_contract

NAME = "j1_tight_replay"
ARGS = COMMON + ["--runtime-cache-gib", "0.5", "--max-running-requests", "4", "--enable-gdn-replayssm",
                 "--enable-cache-report"]


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once a GPU is free")
    with service(NAME, ARGS) as s:
        yield s


def test_ready_reports_runtime_not_pages_and_replay_metadata(svc):
    assert svc.rt()["max_running_requests"] == 4, svc.rt()
    shared_stats_contract(svc)
    replay = svc.c.geometry().get("gdn_replayssm") or {}
    assert replay.get("active") and replay.get("metadata_bytes", 0) > 0, replay


def test_full_concurrency_of_long_prompts_completes(svc):
    burst(svc, 4, 1000, 256, seed=81000)
