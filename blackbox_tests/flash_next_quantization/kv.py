"""I2: exact INT8 encodings using the final stored BF16 scale."""

import torch

from common import bf16, exact, fp32, run_cases, storage_bytes
from freetoken.quant.kv import quantize_kv_int8


def inputs():
    generator = torch.Generator().manual_seed(227)
    vectors = torch.zeros((6, 256), dtype=torch.bfloat16)
    ties = torch.tensor([0.5, 1.5, 2.5, 3.5, 63.5, 64.5, 126.5],
                        dtype=torch.bfloat16) / 64
    lower = torch.nextafter(ties, torch.full_like(ties, -float("inf")))
    upper = torch.nextafter(ties, torch.full_like(ties, float("inf")))
    probes = torch.cat((lower, ties, upper, -lower, -ties, -upper))
    vectors[1, :len(probes)] = probes
    vectors[1, -2:] = torch.tensor([-127 / 64, 127 / 64], dtype=torch.bfloat16)
    vectors[2] = torch.linspace(-1.5, 1.5, 256).to(torch.bfloat16)
    vectors[2, :2] = torch.tensor([0.75, -0.75], dtype=torch.bfloat16)
    for row, scale in [(3, 0.015), (4, 0.4), (5, 1.2)]:
        vectors[row] = (torch.randn(256, generator=generator) * scale).to(torch.bfloat16)
    return vectors


def reference(values):
    codes, scales = [], []
    for row in values.reshape(-1, values.shape[-1]).tolist():
        maximum = max(abs(value) for value in row)
        scale = bf16(fp32(maximum / 127)) if maximum else 1.0
        codes.append([max(-127, min(127, round(fp32(value / scale)))) for value in row])
        scales.append(scale)
    return (torch.tensor(codes, dtype=torch.int8).reshape(values.shape),
            torch.tensor(scales, dtype=torch.bfloat16).reshape(values.shape[:-1]))


def check(values, device):
    expected_codes, expected_scales = reference(values)
    codes, scales = quantize_kv_int8(values.to(device))
    exact(codes, expected_codes, "INT8 encoding")
    exact(scales, expected_scales, "stored BF16 per-vector scale")
    return {"shape": list(values.shape), "encoding_and_scale": "exact",
            "input_bytes": storage_bytes([values]), "stored_bytes": storage_bytes([codes, scales])}


if __name__ == "__main__":
    bank = inputs()
    fixtures = [
        ("zero-vector", bank[0]),
        ("rounding-boundaries-vector", bank[1]),
        ("tokens-heads-final-bf16-scale", bank.reshape(3, 2, 256)),
        ("multiple-prefix-dimensions", bank.repeat(2, 1).reshape(2, 3, 2, 256)),
    ]
    raise SystemExit(run_cases("I2 INT8 KV", [
        (name, lambda device, values=values: check(values, device)) for name, values in fixtures
    ]))
