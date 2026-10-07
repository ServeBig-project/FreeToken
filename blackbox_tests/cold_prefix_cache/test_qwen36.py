"""Qwen3.6-35B-A3B (GDN hybrid): both policies, ReplaySSM off/on, small CPU tiers, Graph off."""

from conftest import assert_clean, server_fixture
from harness import QWEN36
from scenarios import Checks, linear_chain, run_matrix, small_capacity

SIZE = ["--num-tokens", "4096", "--max-running-requests", "4"]
CONT = ["--prefix-cache-policy", "continuation"]
REPLAY = ["--enable-gdn-replayssm"]

baseline = server_fixture("q36_host4_baseline", QWEN36 + SIZE + ["--prefix-cache-host-gib", "4"])
continuation = server_fixture("q36_host4_continuation", QWEN36 + SIZE + CONT + ["--prefix-cache-host-gib", "4"])
replay = server_fixture("q36_replay_host4_continuation", QWEN36 + SIZE + CONT + REPLAY + ["--prefix-cache-host-gib", "4"])
# one GDN snapshot is ~61 MiB on this model (status: recurrent_state host bytes per snapshot)
below_one = server_fixture("q36_host_below_one_state", QWEN36 + SIZE + CONT + ["--prefix-cache-host-gib", "0.05"])
few = server_fixture("q36_host_few_states", QWEN36 + SIZE + CONT + ["--prefix-cache-host-gib", "0.25"])
graph_off = server_fixture("q36_replay_graph_off", QWEN36 + SIZE + CONT + REPLAY +
                           ["--prefix-cache-host-gib", "4", "--cuda-graph-max-bs", "0"])


def test_matrix_baseline(baseline):
    c = Checks("q36_host4_baseline")
    run_matrix(baseline, c, sentences=30, pressure_count=6, gdn=True, policy="baseline")
    assert baseline.alive()
    assert_clean(c)


def test_matrix_continuation(continuation):
    c = Checks("q36_host4_continuation")
    run_matrix(continuation, c, sentences=30, pressure_count=6, gdn=True, policy="continuation")
    assert continuation.alive()
    assert_clean(c)


def test_matrix_replay_continuation(replay):
    c = Checks("q36_replay_host4_continuation")
    run_matrix(replay, c, sentences=30, pressure_count=6, gdn=True, policy="continuation")
    assert replay.alive()
    assert_clean(c)


def test_capacity_below_one_state(below_one):
    c = Checks("q36_host_below_one_state")
    rows = small_capacity(below_one, c, sentences=30, pressure_count=6, gdn=True)
    c.expect(all(r["host_ckpt"] == 0 for r in rows), "no CPU GDN snapshot fits", rows=rows)
    assert below_one.alive()
    assert_clean(c)


def test_capacity_few_states(few):
    c = Checks("q36_host_few_states")
    linear_chain(few, c, 3, 4, 12, "linear", 30, True)
    small_capacity(few, c, sentences=30, pressure_count=6, gdn=True, rounds=4)
    assert few.alive()
    assert_clean(c)


def test_graph_off(graph_off):
    c = Checks("q36_replay_graph_off")
    run_matrix(graph_off, c, sentences=30, pressure_count=6, gdn=True, policy="continuation", parts="core")
    assert graph_off.alive()
    assert_clean(c)
