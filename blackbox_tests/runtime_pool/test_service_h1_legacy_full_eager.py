"""Function combination (contract section 6): legacy batching, DFlash with full (non-compact)
history storage, Graph off, in shared mode.

    --runtime-cache-gib 4 --max-running-requests 4 --batching-policy legacy --no-dflash-compact-kv
    --cuda-graph-max-bs 0 --speculative-num-steps 4 --speculative-draft-model-path <DFlash>
"""
import os

import pytest

from service_common import COMMON, DFLASH_ARGS, service
from service_scenarios import combo_round, ready_combo

NAME = "h1_legacy_full_eager"
ARGS = COMMON + ["--runtime-cache-gib", "4", "--max-running-requests", "4", "--batching-policy", "legacy",
                 "--no-dflash-compact-kv", "--cuda-graph-max-bs", "0", "--enable-cache-report"] + DFLASH_ARGS


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once a GPU is free")
    with service(NAME, ARGS) as s:
        yield s


def test_ready_combination_is_effective(svc):
    ready_combo(svc, "legacy", sd=True, graph=False, draft="draft_kv")


def test_concurrent_sd_eager_requests_complete(svc):
    combo_round(svc, sd=True, graph=False, salt=1000)
