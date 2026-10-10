"""Function combination (contract section 6): legacy batching, AR, Graph on, under memory
pressure with host copies: paused requests restore and finish intact.

    --runtime-cache-gib 0.5 --max-running-requests 6 --prefix-cache-host-gib 4 --batching-policy legacy
"""
import os

import pytest

from service_common import COMMON, service
from service_scenarios import combo_round, pause_round, ready_combo

NAME = "h2_legacy_pressure"
ARGS = COMMON + ["--runtime-cache-gib", "0.5", "--max-running-requests", "6", "--prefix-cache-host-gib", "4",
                 "--batching-policy", "legacy", "--enable-cache-report"]


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once a GPU is free")
    with service(NAME, ARGS) as s:
        yield s


def test_ready_combination_is_effective(svc):
    ready_combo(svc, "legacy", sd=False, graph=True)


def test_concurrent_requests_complete_with_graph(svc):
    combo_round(svc, sd=False, graph=True, salt=1000)


def test_paused_requests_restore_and_complete(svc):
    d = pause_round(svc, salt=20000)["delta"]
    assert d["paused"] >= 1 and d["restored"] >= 1, d
