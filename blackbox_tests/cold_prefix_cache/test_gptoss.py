"""gpt-oss-20b (sliding window 128 per config.json): window ranges, cross-window continuation, overlap."""

from conftest import assert_clean, server_fixture
from harness import gptoss
from scenarios import Checks, continuation_and_fork, lengths_restore, overlapping_windows, run_matrix

WINDOW = 128
ARGS = gptoss() + ["--num-tokens", "4096", "--max-running-requests", "4", "--prefix-cache-host-gib", "2"]
server = server_fixture("oss_host2_baseline", ARGS)
cont = server_fixture("oss_host2_continuation", ARGS + ["--prefix-cache-policy", "continuation"])
PRESSURE = list(range(40, 46))


def test_matrix(server):
    c = Checks("oss_host2_baseline")
    run_matrix(server, c, sentences=30, pressure_count=6, gdn=False, policy="baseline")
    assert server.alive()
    assert_clean(c)


def test_window_ranges(server):
    c = Checks("oss_window")
    lengths_restore(server, c, [WINDOW - 28, WINDOW - 1, WINDOW, WINDOW + 1, 3 * WINDOW + 37], 12, "win",
                    [x + 200 for x in PRESSURE], 30, gdn=False)
    continuation_and_fork(server, c, 2 * WINDOW + 45, 24, "winchain", [x + 300 for x in PRESSURE], 30, gdn=False)
    overlapping_windows(server, c, 300, 16, [x + 400 for x in PRESSURE], 30, gdn=False)
    assert server.alive()
    assert_clean(c)


def test_continuation_policy(cont):
    c = Checks("oss_host2_continuation")
    run_matrix(cont, c, sentences=30, pressure_count=6, gdn=False, policy="continuation", parts="core")
    continuation_and_fork(cont, c, 2 * WINDOW + 45, 24, "winchain", [x + 300 for x in PRESSURE], 30, gdn=False)
    assert cont.alive()
    assert_clean(c)
