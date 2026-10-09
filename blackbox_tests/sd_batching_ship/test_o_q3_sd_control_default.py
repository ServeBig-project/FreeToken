"""Qwen3 MoE BF16, legacy, only an SD control (`--speculative-adaptive-cost`) given: steps omitted means an
SD 4 request (section 2), self drafting, legacy kept."""
from . import checks, env, view

SESSION = ("O_q3_sd_control_default", "q3",
           env.BUDGET + ["--moe-backend", "offload", "--batching-policy", "legacy", "--speculative-adaptive-cost"], 4)


def test_effective(srv):
    s = checks.effective(srv, "legacy", True, phase="outwave", drafter="self")
    assert view.get(s, "req_steps") is None
    assert "true" in view.text(s.get("speculative", {}).get("adaptive_cost_enabled")), s.get("speculative")


def test_decode_real_sd(srv):
    checks.decode_sd(srv)
    checks.draft_hist_bounded(srv)
