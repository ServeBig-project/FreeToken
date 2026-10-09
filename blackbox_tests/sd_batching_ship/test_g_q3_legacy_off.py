"""Qwen3 MoE BF16: restore-old-default config `legacy + 0`, plus ignored phase/draft path."""
from . import checks, env

SESSION = ("G_q3_legacy_off", "q3",
           env.BUDGET + ["--moe-backend", "offload", "--batching-policy", "legacy", "--speculative-num-steps", "0",
                         "--speculative-phase", "all", "--speculative-draft-model-path", "/nonexistent/dflash-dir"], 0)


def test_effective_legacy_ar(srv):
    checks.effective(srv, "legacy", False)
    checks.sd_zero(srv)


def test_decode_ar(srv):
    checks.decode_sd(srv, expect_sd=False)


def test_staggered_and_shapes(srv):
    checks.staggered_wave(srv)
    checks.shapes(srv, concurrency=(1, 4, 5))
    checks.sd_zero(srv)


def test_stop_eos(srv):
    checks.stop_and_eos(srv)


def test_stream(srv):
    checks.stream_consistent(srv)


def test_cancels(srv):
    checks.cancels(srv)
