"""Optional (RP_EXTRA=1): DFlash with ReplaySSM under memory pressure with host copies
(contract sections 3, 5): paused SD requests restore with their draft history and finish intact,
SD keeps running after the pressure round, and cancellation during a pause.

    --runtime-cache-gib 0.5 --max-running-requests 6 --prefix-cache-host-gib 4 --enable-gdn-replayssm
    --speculative-num-steps 4 --speculative-draft-model-path <DFlash> --enable-cache-report
"""
import os

import pytest

from service_common import (COMMON, DFLASH_ARGS, assert_length, components, enum_prompt, record,
                            sd_enabled, sd_num, service)
from service_scenarios import cancel_round, pause_round

NAME = "g_dflash_pressure"
ARGS = COMMON + ["--runtime-cache-gib", "0.5", "--max-running-requests", "6", "--prefix-cache-host-gib", "4",
                 "--enable-gdn-replayssm", "--enable-cache-report"] + DFLASH_ARGS


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once GPU1 is free")
    if not os.environ.get("RP_EXTRA"):
        pytest.skip("optional configuration; set RP_EXTRA=1 to run it")
    with service(NAME, ARGS) as s:
        yield s


def test_ready_sd_with_replay(svc):
    stats, rt = svc.c.stats(), svc.rt()
    assert sd_enabled(stats), stats.get("speculative")
    names = set(components(rt))
    assert any(n.startswith("draft_") for n in names) and any(n.startswith("replay_") for n in names), names
    record(f"{NAME}:ready", runtime=rt, speculative=stats["speculative"], execution=stats.get("execution"))


def test_paused_sd_requests_restore_and_complete(svc):
    before = sd_num(svc.c.stats(), "rounds")
    r = pause_round(svc, salt=1000)
    d = r["delta"]
    rounds = sd_num(svc.c.stats(), "rounds") - before
    record(f"{NAME}:pause_sd", delta=d, sd_rounds=rounds)
    assert d["paused"] >= 1 and d["restored"] >= 1, d
    assert rounds > 0, "no SD round during the pressure round"
    after = sd_num(svc.c.stats(), "rounds")
    assert_length(svc.c.complete(enum_prompt(90000, 12), 64), 64)
    assert sd_num(svc.c.stats(), "rounds") > after, "SD did not resume after the pressure round"


def test_cancel_during_pause_others_continue(svc):
    cancel_round(svc, salt=20000)
