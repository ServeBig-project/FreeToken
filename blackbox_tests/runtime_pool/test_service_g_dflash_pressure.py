"""Optional (RP_EXTRA=1): DFlash with ReplaySSM under memory pressure with host copies
(contract sections 3, 5): paused SD requests restore with their draft history and finish intact,
SD keeps running after the pressure round, and cancellation during a pause.

    --runtime-cache-gib 0.5 --max-running-requests 6 --prefix-cache-host-gib 4 --enable-gdn-replayssm
    --speculative-num-steps 4 --speculative-draft-model-path <DFlash> --enable-cache-report
"""
import os

import pytest

from service_common import (COMMON, DFLASH_ARGS, Watch, assert_enum, assert_length, components, counter_delta,
                            dump, enum_prompt, record, run_streams, sd_enabled, sd_num, service)
from service_scenarios import cancel_round, pause_round


NAME = "g_dflash_pressure"
ARGS = COMMON + ["--runtime-cache-gib", "0.5", "--max-running-requests", "6", "--prefix-cache-host-gib", "4",
                 "--enable-gdn-replayssm", "--enable-cache-report"] + DFLASH_ARGS


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once GPU1 is free")
    if not os.environ.get("RP_EXTRA"):
        pytest.skip("optional configuration; set RP_EXTRA=1 to run it")
    with service(NAME, ARGS) as s:
        yield s


def test_ready_sd_with_replay(svc):
    stats, rt = svc.c.stats(), svc.rt()
    assert sd_enabled(stats), stats.get("speculative")
    names = set(components(rt))
    assert any(n.startswith("draft_") for n in names) and any(n.startswith("replay_") for n in names), names
    record(f"{NAME}:ready", runtime=rt, speculative=stats["speculative"], execution=stats.get("execution"))


def test_paused_sd_requests_restore_and_complete(svc):
    before = sd_num(svc.c.stats(), "rounds")
    r = pause_round(svc, salt=1000, out=3000, items=600)  # ~3k-token private histories: two cannot both stay
    d = r["delta"]
    rounds = sd_num(svc.c.stats(), "rounds") - before
    record(f"{NAME}:pause_sd", delta=d, sd_rounds=rounds)
    # a request yielding during prefill is recomputed from its reusable prefix, a decoding one is
    # saved to the host and restored (public configuration); either way the work is reported
    assert d["paused"] >= 1 and d["restored"] + d["recompute"] >= 1, d
    if d["recompute"]:
        assert d["recomputed_tokens"] > 0, d
    assert rounds > 0, "no SD round during the pressure round"
    after = sd_num(svc.c.stats(), "rounds")
    assert_length(svc.c.complete(enum_prompt(90000, 12), 64), 64)
    assert sd_num(svc.c.stats(), "rounds") > after, "SD did not resume after the pressure round"


def test_cancel_during_pause_others_continue(svc):
    cancel_round(svc, salt=20000, out=3000, items=600, preamble="")


def _first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def test_shared_prefix_outputs_match_solo_runs(svc):
    """Four requests share a ~3000-token prefix with different suffixes; two at a time cannot
    stay resident, so the round pauses, restores or recomputes. Each output must equal the same
    prompt run alone without pressure (both runs reuse the same warmed prefix), and continue its
    suffix's enumeration without gaps or prompt text."""
    prefix = enum_prompt(1000, 600)
    suffixes = [" " + enum_prompt(5000 + 1000 * i, 40) for i in range(4)]
    out = 2500
    solo = []
    svc.c.complete(prefix, 1, cache_group="solo")
    for sfx in suffixes:
        solo.append(svc.c.complete(prefix + sfx, out, cache_group="solo"))
    svc.c.wait_idle()
    svc.c.complete(prefix, 1, cache_group="press")
    streams = [svc.c.stream(prefix + sfx, out, ignore_eos=True, cache_group="press") for sfx in suffixes]
    rt0 = svc.rt()
    with Watch(svc.c) as w:
        run_streams(streams, 1800)
    d = counter_delta(rt0, svc.rt())
    diffs = [_first_diff(s.text, r["text"]) for s, r in zip(streams, solo)]
    dump(f"{NAME}_shared_prefix_texts", {"solo": [r["text"] for r in solo], "pressure": [s.text for s in streams],
                                        "first_diff": diffs})
    record(f"{NAME}:shared_prefix", delta=d, first_diff=diffs, watch=w.report(),
           solo_usage=[r["usage"] for r in solo], streams=[s.summary() for s in streams])
    assert not w.violations, w.report()
    assert d["paused"] >= 1, f"the shared-prefix round paused nothing: {d}"
    if d["recompute"]:
        assert d["recomputed_tokens"] > 0, d
    for i, (s, r) in enumerate(zip(streams, solo)):
        assert s.done and s.finish == "length" and s.usage["completion_tokens"] == out, s.summary()
        assert s.usage["prompt_tokens"] == r["usage"]["prompt_tokens"], (s.usage, r["usage"])
        assert_enum(r["text"], 5040 + 1000 * i, out)
        assert_enum(s.text, 5040 + 1000 * i, out)
        assert s.text == r["text"], f"request {i} diverges from its solo run at character {diffs[i]}"
