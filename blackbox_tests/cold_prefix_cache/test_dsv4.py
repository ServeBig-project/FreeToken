"""DeepSeek-V4 (compressed cache; config compress_ratios 4/128, 128-token window pages)."""

import os

import pytest

from conftest import assert_clean, server_fixture
from harness import DSV4
from scenarios import Checks, continuation_and_fork, lengths_restore, run_matrix

ARGS = DSV4 + ["--num-tokens", "4096", "--max-running-requests", "4", "--prefix-cache-host-gib", "4"]
pytestmark = pytest.mark.skipif(not os.environ.get("FT_RUN_DSV4"), reason="DSV4 deferred; set FT_RUN_DSV4=1")
server = server_fixture("dsv4_host4_baseline", ARGS, timeout=1200)
PRESSURE = list(range(40, 46))


def test_matrix(server):
    c = Checks("dsv4_host4_baseline", strict_text=False)
    run_matrix(server, c, sentences=30, pressure_count=6, gdn=False, policy="baseline", parts="core")
    assert server.alive()
    assert_clean(c)


def test_compression_boundaries(server):
    c = Checks("dsv4_boundaries")
    lengths_restore(server, c, [11, 12, 13, 127, 128, 129, 255, 256, 257], 12, "cb",
                    [x + 200 for x in PRESSURE], 30, gdn=False)
    # decode crosses the 128 boundary, then the continuation is restored after pressure
    continuation_and_fork(server, c, 121, 16, "cbdecode", [x + 300 for x in PRESSURE], 30, gdn=False)
    continuation_and_fork(server, c, 600, 24, "cbchain", [x + 400 for x in PRESSURE], 30, gdn=False)
    assert server.alive()
    assert_clean(c)
