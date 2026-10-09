"""Explicit unsupported or invalid requests fail before ready and name the combination (sections 2, 3, 6)."""
import pytest

from . import env, view
from .client import Client, record
from .server import Server, expect_startup_error

Q3 = env.model_args("q3") + env.BUDGET + ["--moe-backend", "offload"]


def _names(log, *alts):
    tail = log[-8000:].lower()
    assert any(a in tail for a in alts), f"error does not name {alts}; log tail:\n{log[-2500:]}"


CASES = {
    "legacy_sd_all": (["--batching-policy", "legacy", "--speculative-num-steps", "4", "--speculative-phase", "all"],
                      ("phase",)),
    "legacy_inwave_steps_omitted": (["--batching-policy", "legacy", "--speculative-phase", "inwave"], ("phase",)),
    "steps_9": (["--speculative-num-steps", "9"], ("speculative-num-steps", "speculative_num_steps", "steps")),
    # F2: the message must say what is wrong, not only echo the path or an unrelated budget
    "draft_path_missing": (["--speculative-draft-model-path", "/nonexistent/dflash-dir"],
                           ("not exist", "not found", "no such", "missing", "invalid draft")),
    "draft_size_mismatch": (["--speculative-draft-model-path", env.DFLASH],
                            ("target", "match", "compatib", "hidden", "size")),
    "cpu_moe_explicit_sd": (["--moe-backend", "cpu", "--moe-cpu-threads", "8", "--speculative-num-steps", "4"],
                            ("cpu", "moe", "backend")),
}


@pytest.mark.parametrize("case", list(CASES))
def test_rejected_before_ready(case):
    extra, words = CASES[case]
    log = expect_startup_error("M_" + case, Q3 + extra)
    _names(log, *words)


VISIBLE = {
    # unsupported extra SD controls under the new joint path must be rejected, or visibly in effect
    "layered_sd_adaptive_cost": (["--batching-policy", "layered-pipeline", "--speculative-num-steps", "4",
                                  "--speculative-adaptive-cost"], "adaptive"),
    "layered_sd_router_residency": (["--batching-policy", "layered-pipeline", "--speculative-num-steps", "4",
                                     "--speculative-draft-residency", "router"], "residen"),
    # an explicit batching policy the backend cannot run must not be swapped silently
    "cpu_moe_explicit_layered": (["--moe-backend", "cpu", "--moe-cpu-threads", "8",
                                  "--batching-policy", "layered-pipeline", "--speculative-num-steps", "0"], "layered"),
}


@pytest.mark.parametrize("case", list(VISIBLE))
def test_rejected_or_visibly_effective(case):
    extra, word = VISIBLE[case]
    s = Server("M_" + case, Q3 + extra).start()
    try:
        state, detail = s.wait_ready()
        if state == "ok":
            st = Client(s.url).stats()
            record("M_" + case, {"state": "ready", "execution": st.get("execution")})
            eff = view.text((st.get("execution") or {}).get("effective"))
            assert word in eff, f"accepted but effective state does not show '{word}': {eff}"
        else:
            record("M_" + case, {"state": state, "tail": s.log_text()[-1500:]})
            _names(s.log_text(), word)
    finally:
        s.stop()
