"""Qwen3.6 NVFP4, offload, all SD/batching defaults: layered + self-SD(4) + outwave."""
from . import checks, env, view
from .client import dump

SESSION = ("A_nvfp4_default", "q36_nvfp4", env.BUDGET + ["--moe-backend", "offload", "--enable-cache-report"], 4)


def test_effective_defaults(srv):
    s = checks.effective(srv, "layered", True, phase="outwave", drafter="self")
    req = view.get(s, "req_steps")
    assert req in (None, "auto"), f"omitted step count reported as requested {req!r} (section 2: omitted != explicit)"


def test_decode_real_sd(srv):
    checks.decode_sd(srv)
    checks.draft_hist_bounded(srv)


def test_phase_outwave(srv):
    checks.phase_counts(srv, "outwave")


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
    checks.decode_sd(srv, max_tokens=32)
    assert view.delta(b, srv.c.stats(), "graph_replays") > 0, "Graph on and in coverage but no replay counted"


def test_hot_prefix_and_groups(srv):
    assert checks.hot_prefix_and_groups(srv) > 0, "no real SD on a hot-prefix request"


def test_context_edge(srv):
    checks.context_edge(srv)


def test_kv_pressure(srv):
    checks.kv_pressure(srv)


def test_resource_report(srv):
    t = view.text(srv.c.stats()) + view.text(srv.c.cache_status())
    missing = [w for w in ("peak", "graph", "draft", "gdn") if w not in t]
    assert not missing, f"section 5 resource report lacks {missing}"


def test_cancels(srv):
    checks.cancels(srv)


def test_maintenance(srv):
    assert checks.maintenance(srv, has_state=True) > 0, "no Graph replay after maintenance"
