"""Section 5 persistent budget: a tight GDN budget serves AR when no SD is requested and fails before
ready when SD is requested (explicit steps, or a DFlash path with steps omitted). FT_TIGHT_GDN_BYTES (coordinator value, ReplaySSM off, default cache
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
        assert not view.sd_on(st), f"no SD requested, yet SD is on: {st.get('execution')}"
        r = c.complete("Continue: 1, 2, 3,", 32)
        checks.exact_len(r, 32)
    finally:
        s.stop()
    for name, extra in (("N_explicit_tight_gdn", ["--speculative-num-steps", "4"]),
                        ("N_dflash_tight_gdn", ["--speculative-draft-model-path", env.DFLASH])):
        log = expect_startup_error(name, BASE + tight + extra)
        assert any(w in log[-8000:].lower() for w in ("budget", "gdn", "state")), log[-2500:]
