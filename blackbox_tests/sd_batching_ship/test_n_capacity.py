"""Section 5 persistent budget: the same tight GDN budget gives AR + reason when SD is omitted and a
startup error when SD is explicit. FT_TIGHT_GDN_BYTES (coordinator value, ReplaySSM off, default cache
type) holds the AR state pool but not one SD window."""
import pytest

from . import checks, env, view
from .client import Client, record
from .server import expect_startup_error, start_ready

BASE = env.model_args("q36_nvfp4") + env.BUDGET + ["--moe-backend", "offload"]


@pytest.mark.skipif(not env.TIGHT_GDN_BYTES, reason="FT_TIGHT_GDN_BYTES not set")
def test_budget_pair():
    tight = ["--gdn-state-budget-bytes", env.TIGHT_GDN_BYTES]
    s = start_ready("N_default_tight_gdn", BASE + tight)
    try:
        c = Client(s.url)
        st = c.stats()
        record("N_default_tight_gdn", {"execution": st.get("execution")})
        assert not view.sd_on(st), f"budget {env.TIGHT_GDN_BYTES} cannot hold an SD window, yet SD is on"
        t = view.fallback_text(st)
        assert any(w in t for w in ("budget", "state", "gdn")), f"no state-budget fallback reason: {t}"
        r = c.complete("Continue: 1, 2, 3,", 32)
        checks.exact_len(r, 32)
    finally:
        s.stop()
    log = expect_startup_error("N_explicit_tight_gdn", BASE + tight + ["--speculative-num-steps", "4"])
    assert any(w in log[-8000:].lower() for w in ("budget", "gdn", "state")), log[-2500:]
