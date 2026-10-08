"""Qwen3 MoE BF16 (no recurrent state) under a misleading directory name, all defaults:
auto must still pick self-SD 4; batching is layered or a reported fallback (sections 1, 3)."""
from . import checks, env, view

SESSION = ("H_q3_renamed_default", "q3_renamed", env.BUDGET + ["--moe-backend", "offload", "--enable-cache-report"], 4)


def test_effective_defaults(srv):
    s = srv.c.stats()
    b = str(view.get(s, "batching")).lower()
    assert "layered" in b or view.fallback_text(s) not in ("null", "[]", "{}"), \
        f"auto batching {b!r} without a reported reason"
    assert view.sd_on(s) and view.num(s, "steps") == 4, view.text(s.get("execution"))
    assert "self" in str(view.get(s, "drafter")).lower()


def test_decode_real_sd(srv):
    checks.decode_sd(srv)
    checks.draft_hist_bounded(srv)


def test_phase_outwave(srv):
    if "layered" in str(view.get(srv.c.stats(), "batching")).lower():
        checks.phase_counts(srv, "outwave")


def test_shapes(srv):
    checks.shapes(srv)


def test_hot_prefix(srv):
    assert checks.hot_prefix_and_groups(srv) > 0


def test_maintenance(srv):
    assert checks.maintenance(srv, has_state=False) > 0
