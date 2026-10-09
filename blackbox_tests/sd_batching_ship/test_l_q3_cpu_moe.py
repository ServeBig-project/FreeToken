"""Existing path without layered/SD support (CPU MoE backend), all defaults: AR on the original batching,
the batching fallback reported (sections 2, 6)."""
from . import checks, env, view

SESSION = ("L_q3_cpu_moe", "q3", env.BUDGET + ["--moe-backend", "cpu", "--moe-cpu-threads", "8", "--enable-cache-report"], 0)


def test_auto_falls_back_with_reason(srv):
    s = checks.effective(srv, "legacy", False)
    t = view.fallback_text(s)
    assert "batch" in t or "layered" in t, f"auto batching fell back without a reported reason: {t}"


def test_generation(srv):
    checks.decode_sd(srv, expect_sd=False)
    checks.shapes(srv, concurrency=(1, 4))
    checks.stream_consistent(srv)


def test_hot_prefix(srv):
    checks.hot_prefix_and_groups(srv)
