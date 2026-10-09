"""Qwen3 MoE BF16 (no recurrent state) under a misleading directory name; only `--speculative-phase all`
given (steps omitted => self-SD 4 request). The name must not change the capability decision (sections 1-3)."""
from . import checks, env, view

SESSION = ("H_q3_renamed_default", "q3_renamed", env.BUDGET + ["--moe-backend", "offload", "--speculative-phase", "all", "--enable-cache-report"], 4)


def test_effective(srv):
    s = checks.effective(srv, "layered", True, phase="all", drafter="self")
    assert view.get(s, "req_steps") is None


def test_decode_real_sd(srv):
    checks.decode_sd(srv)
    checks.draft_hist_bounded(srv)


def test_phase_all(srv):
    checks.phase_counts(srv, "all")


def test_shapes(srv):
    checks.shapes(srv)


def test_hot_prefix(srv):
    assert checks.hot_prefix_and_groups(srv) > 0


def test_maintenance(srv):
    assert checks.maintenance(srv, has_state=False) > 0
