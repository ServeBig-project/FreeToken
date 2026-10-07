"""Qwen3.6 speculative decoding with the CPU tier: DFlash and self-SD (legacy batching)."""

from conftest import assert_clean, server_fixture
from harness import DFLASH, QWEN36
from scenarios import Checks, run_matrix, sd_checks

SIZE = ["--num-tokens", "4096", "--max-running-requests", "4"]
HOST = ["--prefix-cache-host-gib", "4", "--prefix-cache-policy", "continuation"]

dflash = server_fixture("q36_dflash", QWEN36 + DFLASH + SIZE + HOST)
self_sd = server_fixture("q36_self_sd", QWEN36 + ["--speculative-num-steps", "4"] + SIZE + HOST)


def test_dflash(dflash):
    c = Checks("q36_dflash")
    sd_checks(dflash, c, sentences=30, steps=8)
    run_matrix(dflash, c, sentences=30, pressure_count=6, gdn=True, policy="continuation", parts="core")
    assert dflash.alive()
    assert_clean(c)


def test_self_sd(self_sd):
    c = Checks("q36_self_sd")
    sd_checks(self_sd, c, sentences=30, steps=4)
    run_matrix(self_sd, c, sentences=30, pressure_count=6, gdn=True, policy="continuation", parts="core")
    assert self_sd.alive()
    assert_clean(c)
