"""Qwen3.6 BF16, offload, only a DFlash path given (steps/phase omitted => DFlash SD 4, outwave),
CPU cold prefix cache with continuation policy."""
from . import checks, env, view

SESSION = ("F_bf16_default_cold", "q36_bf16",
           env.BUDGET + ["--moe-backend", "offload", "--speculative-draft-model-path", env.DFLASH, "--num-tokens", "12288", "--prefix-cache-host-gib", "4",
                         "--prefix-cache-policy", "continuation", "--enable-cache-report"], 4)


def test_effective_defaults(srv):
    s = checks.effective(srv, "layered", True, phase="outwave", drafter="dflash")
    assert view.get(s, "req_steps") is None


def test_cold_restore(srv):
    checks.cold_restore(srv)


def test_multi_turn_and_fork(srv):
    checks.multi_turn(srv)


def test_cancel_during_cold_restore(srv):
    checks.cancel_cold(srv)


def test_kv_pressure(srv):
    checks.kv_pressure(srv, plen=2500, out=300)


def test_decode_real_sd_after_cache_work(srv):
    checks.decode_sd(srv)
