"""Qwen3 MoE BF16: the original legacy 2-step self-SD path, Graph on."""
from . import checks, env, view

SESSION = ("J_q3_legacy_n2", "q3",
           env.BUDGET + ["--moe-backend", "offload", "--batching-policy", "legacy", "--speculative-num-steps", "2"], 2)


def test_effective(srv):
    checks.effective(srv, "legacy", True, phase="outwave", drafter="self")


def test_decode_real_sd_with_graph(srv):
    b = srv.c.stats()
    checks.decode_sd(srv)
    assert view.delta(b, srv.c.stats(), "graph_replays") > 0
    checks.draft_hist_bounded(srv)


def test_shapes(srv):
    checks.shapes(srv)


def test_staggered(srv):
    assert checks.staggered_wave(srv)["rounds"] > 0


def test_stream(srv):
    checks.stream_consistent(srv)


def test_drift(srv):
    checks.greedy_batch_drift(srv)
