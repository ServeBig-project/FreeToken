"""Qwen3.6 NVFP4, offload, all SD/batching defaults: layered + AR (no draft model => SD off)."""
from . import checks, env, view
from .client import dump

SESSION = ("A_nvfp4_default", "q36_nvfp4", env.BUDGET + ["--moe-backend", "offload", "--enable-cache-report"], 0)


def test_effective_defaults(srv):
    s = checks.effective(srv, "layered", False)
    checks.sd_zero(srv)
    req = view.get(s, "req_steps")
    assert req in (None, "auto"), f"omitted step count reported as requested {req!r} (section 2: omitted != explicit)"


def test_decode_is_ar(srv):
    checks.decode_sd(srv, expect_sd=False)


def test_waves_stay_ar(srv):
    checks.staggered_wave(srv)
    checks.sd_zero(srv)


def test_shapes_and_tails(srv):
    checks.shapes(srv)


def test_stop_eos(srv):
    checks.stop_and_eos(srv)


def test_stream(srv):
    checks.stream_consistent(srv)


def test_greedy_probe_and_drift(srv):
    dump("A_greedy", checks.greedy_probe(srv))
    checks.greedy_batch_drift(srv)


def test_graph_replays(srv):
    b = srv.c.stats()
    checks.decode_sd(srv, max_tokens=32, expect_sd=False)
    assert view.delta(b, srv.c.stats(), "graph_replays") > 0, "Graph on and in coverage but no replay counted"
    checks.graph_ladder(srv, 4)


def test_hot_prefix_and_groups(srv):
    checks.hot_prefix_and_groups(srv)


def test_context_edge(srv):
    checks.context_edge(srv)


def test_kv_pressure(srv):
    checks.kv_pressure(srv)


def test_resource_report(srv):
    checks.resources(srv, sd=False)


def test_cancels(srv):
    checks.cancels(srv)


def test_maintenance(srv):
    assert checks.maintenance(srv, has_state=True) > 0, "no Graph replay after maintenance"
