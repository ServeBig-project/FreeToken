"""Startup capacity at a tight budget with DFlash (round J item 1): the service starts and
serves its full explicit concurrency of ~1000-token prompts with 256-token outputs, with real SD.

    --runtime-cache-gib 0.5 --max-running-requests 6 --speculative-num-steps 4 --speculative-draft-model-path <DFlash>
"""
import os

import pytest

from service_common import COMMON, DFLASH_ARGS, sd_num, service
from service_scenarios import burst, shared_stats_contract

NAME = "j2_tight_dflash"
ARGS = COMMON + ["--runtime-cache-gib", "0.5", "--max-running-requests", "6", "--enable-cache-report"] + DFLASH_ARGS


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once a GPU is free")
    with service(NAME, ARGS) as s:
        yield s


def test_ready(svc):
    rt = svc.rt()
    assert 1 <= rt["max_running_requests"] <= 6, rt
    shared_stats_contract(svc)


def test_full_concurrency_of_long_prompts_completes_with_sd(svc):
    before = sd_num(svc.c.stats(), "rounds")
    burst(svc, svc.rt()["max_running_requests"], 1000, 256, seed=82000)
    assert sd_num(svc.c.stats(), "rounds") > before, "no SD round ran"
