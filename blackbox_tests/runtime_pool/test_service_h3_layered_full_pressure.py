"""Function combination (contract section 6): explicit layered-pipeline batching, DFlash with
full (non-compact) history storage, Graph on, under memory pressure with host copies.

    --runtime-cache-gib 0.5 --max-running-requests 6 --prefix-cache-host-gib 4
    --batching-policy layered-pipeline --no-dflash-compact-kv
    --speculative-num-steps 4 --speculative-draft-model-path <DFlash>
"""
import os

import pytest

from service_common import COMMON, DFLASH_ARGS, sd_num, service
from service_scenarios import combo_round, pause_round, ready_combo

NAME = "h3_layered_full_pressure"
ARGS = COMMON + ["--runtime-cache-gib", "0.5", "--max-running-requests", "6", "--prefix-cache-host-gib", "4",
                 "--batching-policy", "layered-pipeline", "--no-dflash-compact-kv",
                 "--enable-cache-report"] + DFLASH_ARGS


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once a GPU is free")
    with service(NAME, ARGS) as s:
        yield s


def test_ready_combination_is_effective(svc):
    ready_combo(svc, "layered-pipeline", sd=True, graph=True, draft="draft_kv")


def test_concurrent_sd_requests_complete_with_graph(svc):
    combo_round(svc, sd=True, graph=True, salt=1000)


def test_paused_sd_requests_restore_and_complete(svc):
    before = sd_num(svc.c.stats(), "rounds")
    d = pause_round(svc, salt=20000)["delta"]
    assert d["paused"] >= 1 and d["restored"] >= 1, d
    assert sd_num(svc.c.stats(), "rounds") > before, "no SD round during the pressure round"
