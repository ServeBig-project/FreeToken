"""Qwen3.6 BF16 + DFlash, hybrid MoE, layered-pipeline, SD 4 in-wave only, Graph off (eager)."""
from . import checks, env, view

SESSION = ("E_bf16_dflash_inwave_eager", "q36_bf16",
           env.BUDGET + ["--moe-backend", "hybrid", "--moe-cpu-threads", "8",
                         "--batching-policy", "layered-pipeline", "--speculative-num-steps", "4",
                         "--speculative-phase", "inwave", "--speculative-draft-model-path", env.DFLASH,
                         "--cuda-graph-max-bs", "0"], 4)


def test_effective(srv):
    checks.effective(srv, "layered", True, phase="inwave", drafter="dflash")


def test_decode_outside_wave_is_ar(srv):
    b = srv.c.stats()
    checks.decode_sd(srv, expect_sd=False)
    assert view.reason_count(srv.c.stats(), "phase") > view.reason_count(b, "phase"), "phase-rule AR not counted"


def test_phase_inwave(srv):
    checks.phase_counts(srv, "inwave")


def test_graph_off_no_replay(srv):
    b = srv.c.stats()
    checks.shapes(srv, concurrency=(1, 4))
    assert view.delta(b, srv.c.stats(), "graph_replays") == 0, "Graph disabled but replays counted"


def test_stream(srv):
    checks.stream_consistent(srv)


def test_maintenance(srv):
    checks.maintenance(srv, has_state=True,
                       after=lambda: checks.phase_counts(srv, "inwave"))
