"""Qwen3.6 NVFP4, hybrid MoE: legacy batching + explicit SD 4 (must not become layered).
Graph capture only up to bs 2, so 4-way decode is a legal shape outside Graph coverage."""
from . import checks, env, view
from .client import record

SESSION = ("C_nvfp4_legacy_sd4", "q36_nvfp4",
           env.BUDGET + ["--moe-backend", "hybrid", "--moe-cpu-threads", "8", "--batching-policy", "legacy",
                         "--speculative-num-steps", "4", "--cuda-graph-max-bs", "2"], 4)


def test_effective_legacy_sd(srv):
    assert view.get(srv.c.stats(), "req_steps") == 4
    checks.effective(srv, "legacy", True, phase="outwave")


def test_decode_real_sd(srv):
    checks.decode_sd(srv)
    checks.draft_hist_bounded(srv)


def test_staggered_prefill_with_decode(srv):
    d = checks.staggered_wave(srv)
    assert d["rounds"] > 0, d


def test_shapes_and_tails(srv):
    checks.shapes(srv)


def test_graph_coverage(srv):
    b = srv.c.stats()
    checks.decode_sd(srv, max_tokens=32)
    m = srv.c.stats()
    assert view.delta(b, m, "graph_replays") > 0, "bs 1 inside coverage but no Graph replay"
    res = srv.c.parallel([(srv.c.complete, (f"Story {i}: once upon a time", 64), {}) for i in range(4)])
    for r in res:
        checks.exact_len(r, 64)
    a = srv.c.stats()
    record("C:graph_outside_coverage", {"replays": view.delta(m, a, "graph_replays"),
                                         "graph": view.flat(a.get("cuda_graph"))})


def test_ladder_and_resources(srv):
    checks.graph_ladder(srv, 2)
    r = checks.resources(srv, sd=True)
    assert r["speculative_graph_reserved_bytes"] > 0, r
    assert r["cpu_executor_pinned_io_bytes"] > 0, f"hybrid CPU executor ran but no pinned bytes: {r}"


def test_stop_eos(srv):
    checks.stop_and_eos(srv)


def test_stream(srv):
    checks.stream_consistent(srv)
