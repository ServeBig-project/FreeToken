"""Qwen3-30B-A3B (plain attention): matrix with an ample CPU tier, continuation policy, tiny CPU tier."""

from conftest import assert_clean, server_fixture
from harness import QWEN3
from scenarios import Checks, run_matrix, small_capacity

SIZE = ["--num-tokens", "3072", "--max-running-requests", "4"]

baseline = server_fixture("q3_host2_baseline", QWEN3 + SIZE + ["--prefix-cache-host-gib", "2"])
continuation = server_fixture("q3_host2_continuation",
                              QWEN3 + SIZE + ["--prefix-cache-host-gib", "2", "--prefix-cache-policy", "continuation"])
tiny = server_fixture("q3_host_tiny", QWEN3 + SIZE + ["--prefix-cache-host-gib", "0.05"])
few = server_fixture("q3_host_few", QWEN3 + SIZE + ["--prefix-cache-host-gib", "0.3"])


def test_matrix_baseline(baseline):
    c = Checks("q3_host2_baseline")
    run_matrix(baseline, c, sentences=30, pressure_count=5, gdn=False, policy="baseline")
    assert baseline.alive()
    assert_clean(c)


def test_matrix_continuation(continuation):
    c = Checks("q3_host2_continuation")
    run_matrix(continuation, c, sentences=30, pressure_count=5, gdn=False, policy="continuation", parts="core")
    assert continuation.alive()
    assert_clean(c)


def test_tiny_capacity(tiny):
    """0.05 GiB is less than one ~800-token prompt of KV (~79 MB)."""
    c = Checks("q3_host_tiny")
    small_capacity(tiny, c, sentences=30, pressure_count=5, gdn=False)
    assert tiny.alive()
    assert_clean(c)


def test_few_prompts_capacity(few):
    """0.3 GiB holds about three ~800-token prompts; more cold prompts than that must evict, not overrun."""
    c = Checks("q3_host_few")
    small_capacity(few, c, sentences=30, pressure_count=5, gdn=False, rounds=4)
    assert few.alive()
    assert_clean(c)
