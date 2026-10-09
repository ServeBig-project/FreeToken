"""The original split-pool implementation with DFlash, as the comparison for the shared-mode
long-prompt greedy output recorded by test_service_e_dflash.py (contract section 7: same
expert pool, same SD, temperature 0).

    PYTHONPATH=<dflash-mainline>  --num-tokens 65536 --max-running-requests 4
    --speculative-num-steps 4 --speculative-draft-model-path <DFlash> --enable-cache-report
"""
import json
import os

import pytest

from service_common import BASELINE, COMMON, DFLASH_ARGS, PARITY, assert_length, record, sd_num, service

NAME = "f_baseline"
ARGS = COMMON + ["--num-tokens", "65536", "--max-running-requests", "4", "--enable-cache-report"] + DFLASH_ARGS


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once GPU1 is free")
    with service(NAME, ARGS, impl=BASELINE) as s:
        yield s


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_long_prompt_matches_shared_mode(svc):
    assert os.path.exists(PARITY), f"{PARITY} missing: run test_service_e_dflash.py first"
    with open(PARITY) as f:
        ref = json.load(f)
    assert svc.c.status()["geometry"]["num_pages"] > 0  # the split-pool mode, not shared
    before = sd_num(svc.c.stats(), "rounds")
    r = svc.c.complete(ref["prompt"], ref["max_tokens"])
    assert_length(r, ref["max_tokens"])
    assert sd_num(svc.c.stats(), "rounds") > before
    record(f"{NAME}:parity", equal=r["text"] == ref["text"], common_prefix=_common_prefix(r["text"], ref["text"]),
           length=len(ref["text"]), baseline_usage=r["usage"], shared_usage=ref["usage"])
    assert r["text"] == ref["text"], (
        f"shared-mode and split-pool greedy outputs differ after {_common_prefix(r['text'], ref['text'])} "
        f"of {len(ref['text'])} characters:\nshared: {ref['text'][:300]!r}\nbaseline: {r['text'][:300]!r}")
