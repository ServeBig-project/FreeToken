"""Qwen3 MoE BF16: layered-pipeline + SD 1, phase all, Graph off, prompts split over several prefill chunks."""
from . import checks, env, view

SESSION = ("I_q3_layered_n1_eager", "q3",
           env.BUDGET + ["--moe-backend", "offload", "--batching-policy", "layered-pipeline",
                         "--speculative-num-steps", "1", "--speculative-phase", "all",
                         "--cuda-graph-max-bs", "0", "--max-prefill-length", "2048",
                         "--prefill-layer-group-size", "2"], 1)


def test_effective(srv):
    checks.effective(srv, "layered", True, phase="all")


def test_decode_real_sd(srv):
    checks.decode_sd(srv)
    checks.draft_hist_bounded(srv)


def test_phase_all_multichunk(srv):
    checks.phase_counts(srv, "all")


def test_shapes_no_graph(srv):
    b = srv.c.stats()
    checks.shapes(srv)
    assert view.delta(b, srv.c.stats(), "graph_replays") == 0


def test_context_edge(srv):
    checks.context_edge(srv)


def test_stop_eos(srv):
    checks.stop_and_eos(srv)
