"""Qwen3.6 NVFP4 + DFlash, offload, layered-pipeline, SD 8, phase all, Replay on, multi-chunk prefill,
CPU cold prefix cache (baseline policy) under KV pressure."""
from . import checks, env

SESSION = ("D_nvfp4_dflash_all", "q36_nvfp4",
           env.BUDGET + ["--moe-backend", "offload", "--batching-policy", "layered-pipeline",
                         "--speculative-num-steps", "8", "--speculative-phase", "all",
                         "--speculative-draft-model-path", env.DFLASH, "--enable-gdn-replayssm",
                         "--max-prefill-length", "2048", "--num-tokens", "12288",
                         "--prefix-cache-host-gib", "4", "--enable-cache-report"], 8)


def test_effective_dflash(srv):
    checks.effective(srv, "layered", True, phase="all", drafter="dflash")


def test_decode_real_sd(srv):
    checks.decode_sd(srv)
    checks.draft_hist_bounded(srv)


def test_phase_all(srv):
    checks.phase_counts(srv, "all")


def test_shapes_and_tails(srv):
    checks.shapes(srv)


def test_cold_restore_keeps_drafter_history(srv):
    checks.cold_restore(srv)


def test_cancels(srv):
    checks.cancels(srv)


def test_stream(srv):
    checks.stream_consistent(srv)
