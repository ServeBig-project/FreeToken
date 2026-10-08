"""Existing path without SD support (CPU MoE backend) and SD omitted: AR with a reported reason (section 6)."""
from . import checks, env, view

SESSION = ("L_q3_cpu_moe", "q3", env.BUDGET + ["--moe-backend", "cpu", "--moe-cpu-threads", "8", "--enable-cache-report"], 0)


def test_auto_falls_back_with_reason(srv):
    s = srv.c.stats()
    assert not view.sd_on(s)
    t = view.fallback_text(s)
    assert "spec" in t or "sd" in t or "draft" in t, f"no SD fallback reason reported: {t}"


def test_generation(srv):
    checks.decode_sd(srv, expect_sd=False)
    checks.shapes(srv, concurrency=(1, 4))
    checks.stream_consistent(srv)


def test_hot_prefix(srv):
    checks.hot_prefix_and_groups(srv)
