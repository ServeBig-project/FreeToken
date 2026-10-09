"""Startup-time public errors (contract section 2): conflicting fixed-pool parameters, a zero or
negative budget, an explicit context that the budget cannot execute, and a budget the GPU cannot
hold must all fail before ready with understandable text in the process output.

The conflict and budget-sign cases fail before any model is loaded and run with every GPU hidden;
the capability cases need the test GPU (RP_GPU_OK=1).
"""
import os

import pytest

from service_common import COMMON, expect_startup_error

SHARED = COMMON + ["--runtime-cache-gib", "4"]

CONFLICTS = {
    "num_pages": (["--num-pages", "4096"], ("conflict", "num-pages")),
    "num_tokens": (["--num-tokens", "65536"], ("conflict", "num-tokens")),
    "gdn_state_budget": (["--gdn-state-budget-bytes", "3000000000"], ("conflict", "gdn-state-budget")),
    "kv_reserve_tokens": (["--kv-reserve-tokens", "8192"], ("conflict", "kv-reserve")),
    # the acceptance configuration lists this flag; `--help` does not. Either a conflict message or
    # an argument error is a startup failure with text; the report says which one appeared.
    "linear_state_cache_ratio": (["--linear-state-cache-ratio", "0.5"],
                                 ("conflict", "linear-state-cache-ratio", "unrecognized")),
}


@pytest.mark.parametrize("case", list(CONFLICTS))
def test_fixed_pool_parameters_conflict(case):
    extra, words = CONFLICTS[case]
    log = expect_startup_error(f"a_conflict_{case}", SHARED + extra, gpu=False, timeout=240)
    assert "runtime-cache-gib" in log.lower(), log[-2000:]
    assert any(w in log.lower() for w in words), f"text names neither of {words}:\n{log[-2000:]}"


@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_budget_is_not_a_switch(value):
    log = expect_startup_error(f"a_budget_{value}", COMMON + ["--runtime-cache-gib", value],
                               gpu=False, timeout=240)
    assert "runtime-cache-gib" in log.lower() and "> 0" in log, log[-2000:]


gpu = pytest.mark.skipif(not os.environ.get("RP_GPU_OK"), reason="set RP_GPU_OK=1 once GPU1 is free")


@gpu
def test_explicit_context_beyond_capability_fails_at_startup():
    """Section 2: an explicit maximum context the runtime budget cannot execute for one request
    is a startup error, not a silent reduction."""
    log = expect_startup_error("a_context_override", COMMON + [
        "--runtime-cache-gib", "0.5", "--max-running-requests", "6", "--max-seq-len-override", "262144"],
        gpu=True)
    assert any(w in log.lower() for w in ("context", "seq", "fit")), log[-2000:]


@gpu
def test_budget_larger_than_the_gpu_fails_at_startup():
    """Section 2: a runtime budget the GPU cannot hold next to the weights and the fixed expert
    pool is reported before ready; the expert pool is not squeezed to make room."""
    log = expect_startup_error("a_budget_30gib", COMMON + ["--runtime-cache-gib", "30"], gpu=True)
    assert any(w in log.lower() for w in ("memory", "fit", "runtime", "gib")), log[-2000:]
