"""Numeric acceptance protocol, frozen 2026-10-09 before any candidate result was seen.

Reference: reference.moe in fp32 with only the roundings the contract names (BF16 inputs,
DSV4 E4M3 groups). The candidate returns BF16 [T,H].

Basis (blackbox_tests/nowag_runtime/calibrate.py, H/I/top-k of the real models, D4 and D6):
a legal kernel that rounds gate/up outputs and the final output to BF16 deviates from the
fp32 reference by
  SiLU / GELU-tanh / GPT-OSS : rel_fro 2.7e-3..2.9e-3, max/peak 3.0e-3..4.6e-3
  DSV4 (E4M3 groups)         : rel_fro 9.5e-3..1.0e-2, max/peak 9.7e-3..9.8e-3
(E4M3 amplifies BF16 differences when a value crosses a rounding boundary). Wrong-math
variants are far outside: DSV4 without the E4M3 rounding 4.3e-2..4.5e-2, DSV4 with the route
weight moved to the output 3.3e-2..3.6e-2. Bounds below are ~3x the legal deviation and stay
below the nearest wrong variant; the discrimination check additionally requires the
candidate to sit clearly closer to the right reference than to the named wrong one.

"Cumulative" error is the signed mean error over all outputs relative to output RMS; it
catches a systematic scale/bias (e.g. a normalizer applied twice to a slice) that a
Frobenius bound averages away.
"""

import torch

BOUNDS = {
    "bf16": {"rel_fro": 1.0e-2, "max_over_peak": 1.5e-2, "mean_over_rms": 2.0e-3},
    "dsv4": {"rel_fro": 2.0e-2, "max_over_peak": 3.0e-2, "mean_over_rms": 2.0e-3},
}
# candidate must be at least this much closer to the right reference than to a wrong one
DISCRIMINATION = 0.5


def bound_key(math_):
    return "dsv4" if math_.get("dsv4_round") else "bf16"


def metrics(out, ref):
    out, ref = out.float(), ref.float()
    err = out - ref
    peak = ref.abs().max().clamp_min(1e-30)
    rms = ref.pow(2).mean().sqrt().clamp_min(1e-30)
    return {
        "max_abs": float(err.abs().max()) if err.numel() else 0.0,
        "rel_fro": float(err.norm() / ref.norm().clamp_min(1e-30)) if err.numel() else 0.0,
        "max_over_peak": float(err.abs().max() / peak) if err.numel() else 0.0,
        "mean_over_rms": float(err.mean().abs() / rms) if err.numel() else 0.0,
        "finite": bool(torch.isfinite(out).all()),
    }


def assert_close(out, ref, math_, label=""):
    m = metrics(out, ref)
    bounds = BOUNDS[bound_key(math_)]
    bad = {k: (m[k], v) for k, v in bounds.items() if m[k] > v}
    assert m["finite"] and not bad, f"{label} metrics {m} exceed {bad or 'finite'}"
    return m


def assert_discriminates(out, right, wrong, label=""):
    near = float((out.float() - right).norm())
    far = float((out.float() - wrong).norm())
    assert near <= DISCRIMINATION * far, f"{label}: |out-right|={near:.4g} |out-wrong|={far:.4g}"
