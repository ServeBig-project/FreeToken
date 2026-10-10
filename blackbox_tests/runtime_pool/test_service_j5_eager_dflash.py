"""Graph off with DFlash at a tight budget and derived concurrency (round J item 4): the
service starts and completes requests with real SD and no Graph replay.

    --runtime-cache-gib 0.5 --cuda-graph-max-bs 0 --speculative-num-steps 4 --speculative-draft-model-path <DFlash>
"""
import os

import pytest

from service_common import COMMON, DFLASH_ARGS, graph_replays, sd_num, service
from service_scenarios import burst, shared_stats_contract

NAME = "j5_eager_dflash"
ARGS = COMMON + ["--runtime-cache-gib", "0.5", "--cuda-graph-max-bs", "0", "--enable-cache-report"] + DFLASH_ARGS


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once a GPU is free")
    with service(NAME, ARGS) as s:
        yield s


def test_ready(svc):
    assert svc.rt()["requested_running_requests"] is None, svc.rt()
    assert not svc.c.stats()["cuda_graph"]["enabled"], svc.c.stats()["cuda_graph"]
    shared_stats_contract(svc)


def test_requests_complete_with_sd_and_no_graph(svc):
    stats = svc.c.stats()
    rounds, replays = sd_num(stats, "rounds"), graph_replays(stats)
    burst(svc, min(4, svc.rt()["max_running_requests"]), 1000, 256, seed=86000)
    stats = svc.c.stats()
    assert sd_num(stats, "rounds") > rounds, "no SD round ran"
    assert graph_replays(stats) == replays, stats.get("cuda_graph")
