"""Black-box numeric acceptance of gdn_replay_conv (public contract section 5.1), D=8192, KW=4.

Tolerance (fixed before any comparison ran; derivation in gdn_numeric_ref.py):
  |o-o*| <= 2^-8|o*| + 2^-16 sum_j|w_j x_j|   (one bf16 rounding; fp32 4-term sum and silu)
Window writes, untouched window slots and zero padding output are exact checks.
The weight dtype is not stated by the contract: bf16 (activation dtype) and fp32 are both run.
"""
import random

import pytest

from gdn_numeric_ref import (BF16, CONV_ATOL, CONV_RTOL, F32, KW, ConvWorld, assert_within_tolerance, conv_ref,
                             tol_ratio)


@pytest.mark.parametrize("wdtype", [BF16, F32], ids=["w_bf16", "w_fp32"])
@pytest.mark.parametrize("W", [KW - 1 + 9, 16])
def test_conv_matches_fp64(W, wdtype, log):
    """Detects: wrong tap order or offset; history read from the wrong window slot (including
    positions before 0, which live at window[q mod W] as pre-written by the caller); x_t written
    to the wrong slot or other slots overwritten; padding rows producing output or writes;
    rows mixed up in a permuted batch with mixed T; draft (T=1 at p+i) then verify at p."""
    rng = random.Random(W)
    cw = ConvWorld(log, rows=6, W=W, wdtype=wdtype, seed=600 + W)
    tmax = min(9, W - KW + 1)
    reqs = [dict(m=m, p=p, name=f"m{m}@{p}", draft=None)
            for m, p in zip([4, 0, 5, 2, 1, 3], [0, 1, 2, 3, 57, 200003])]
    for step in range(30):
        specs = []
        for r in reqs:
            if r["draft"] is None and rng.random() < 0.3:
                r["draft"] = [rng.randint(1, tmax - 1), 0]
            if r["draft"] and r["draft"][1] < r["draft"][0]:
                specs.append(dict(m=r["m"], p=r["p"] + r["draft"][1], x=cw.x(1), name=r["name"], req=r))
            else:
                T = r["draft"][0] + 1 if r["draft"] else rng.randint(1, tmax)
                specs.append(dict(m=r["m"], p=r["p"], x=cw.x(T), name=r["name"], req=r))
        for _ in range(rng.randint(0, 2)):
            specs.insert(rng.randrange(len(specs) + 1),
                         dict(m=-1, p=rng.randint(0, 50), x=cw.x(rng.randint(1, tmax)), name="pad", req=None))
        rng.shuffle(specs)
        before = cw.snap.clone()
        outs = cw.call(specs, tag=f"step{step}")
        if step == 0:  # the tolerance must reject an output one position off and reversed taps
            j = next(j for j, s in enumerate(specs) if s["req"] and s["x"].shape[0] > 1)
            s, o = specs[j], outs[j]
            ref, mag = conv_ref(cw.weight, before[s["m"]], s["x"], s["p"])
            flipped, fmag = conv_ref(cw.weight.flip(-1), before[s["m"]], s["x"], s["p"])
            for name, got, alt, amag in [("output one position off", o[:-1], ref[1:], mag[1:]),
                                         ("taps reversed", o, flipped, fmag)]:
                ratio = tol_ratio(got, alt, CONV_RTOL, CONV_ATOL * amag)
                log.append(dict(kind="negative_control", name=name, tol_ratio=ratio))
                assert ratio > 1.0, f"tolerance cannot tell '{name}' from the correct output"
        for s in specs:
            r = s["req"]
            if r is None:
                continue
            if r["draft"] and r["draft"][1] < r["draft"][0]:
                r["draft"][1] += 1
            else:
                r["p"] += rng.randint(1, s["x"].shape[0])  # accepted drafts + 1
                r["draft"] = None
    assert_within_tolerance(log)
