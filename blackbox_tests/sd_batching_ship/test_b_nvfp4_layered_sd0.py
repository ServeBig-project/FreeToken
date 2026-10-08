"""Qwen3.6 NVFP4: layered-pipeline + SD explicitly off, with a bogus draft path and phase that must be
ignored (section 2: explicit 0 wins, draft path not parsed). Replay on."""
import json
import os

from . import checks, env, view
from .client import common_prefix, record

SESSION = ("B_nvfp4_layered_sd0", "q36_nvfp4",
           env.BUDGET + ["--moe-backend", "offload", "--batching-policy", "layered-pipeline",
                         "--speculative-num-steps", "0", "--speculative-phase", "all",
                         "--speculative-draft-model-path", "/nonexistent/dflash-dir",
                         "--enable-gdn-replayssm", "--enable-cache-report"], 0)


def test_effective_layered_sd_off(srv):
    assert view.get(srv.c.stats(), "req_steps") == 0
    checks.effective(srv, "layered", False)
    checks.sd_zero(srv)


def test_decode_is_ar(srv):
    checks.decode_sd(srv, expect_sd=False)


def test_waves_and_shapes_stay_ar(srv):
    checks.staggered_wave(srv)
    checks.shapes(srv, concurrency=(1, 4, 5))
    checks.sd_zero(srv)


def test_greedy_vs_sd_default(srv):
    mine = checks.greedy_probe(srv)
    path = os.path.join(env.RESULTS, "A_greedy.json")
    if os.path.exists(path):
        ref = json.load(open(path))
        record("B_vs_A_greedy", {k: {"prefix": common_prefix(mine[k], ref[k]), "len": len(ref[k])} for k in ref})


def test_no_sd_allocation_vs_default(srv):
    path = os.path.join(env.RESULTS, "A_nvfp4_default_ready_stats.json")
    mine = view.flat(srv.c.stats())
    record("B_vs_A_resources", {"B": {k: v for k, v in mine.items() if any(w in k.lower() for w in
                                ("byte", "mem", "peak", "draft", "graph"))},
                                "A_present": os.path.exists(path)})
    for k, v in mine.items():
        if "draft" in k.lower() and ("byte" in k.lower() or "mem" in k.lower()) and isinstance(v, (int, float)):
            assert v == 0, f"SD off but drafter memory reported: {k}={v}"


def test_hot_prefix(srv):
    checks.hot_prefix_and_groups(srv)


def test_cancels(srv):
    checks.cancels(srv)


def test_maintenance(srv):
    checks.maintenance(srv, has_state=True)
    checks.sd_zero(srv)
