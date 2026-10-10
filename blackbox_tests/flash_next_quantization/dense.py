"""I1: independent exact FP8 E4M3FN per-output-row quantization."""

import bisect
import math

import torch

from common import exact, fp32, run_cases, storage_bytes
from freetoken.quant.dense import quant_fp8_per_row


def e4m3_values():
    values = []
    for code in range(127):
        exponent, mantissa = code >> 3, code & 7
        value = mantissa * 2**-9 if exponent == 0 else (1 + mantissa / 8) * 2**(exponent - 7)
        values.append(value)
    return values


E4M3_VALUES = e4m3_values()


def reference(weights):
    scales, codes = [], []
    for row in weights.tolist():
        scale = fp32(max(max(abs(value) for value in row) / 448, 1e-12))
        scales.append(scale)
        encoded = []
        for value in row:
            normalized = fp32(value / scale)
            magnitude = abs(normalized)
            upper = min(bisect.bisect_left(E4M3_VALUES, magnitude), 126)
            candidates = [upper, max(upper - 1, 0)]
            code = min(candidates, key=lambda item: (abs(magnitude - E4M3_VALUES[item]), item & 1))
            encoded.append(code | (128 if math.copysign(1, normalized) < 0 else 0))
        codes.append(encoded)
    return torch.tensor(codes, dtype=torch.uint8), torch.tensor(scales, dtype=torch.float32)


def inputs(dtype):
    generator = torch.Generator().manual_seed(191)
    weights = torch.zeros((6, 256), dtype=dtype)
    weights[1] = torch.randn(256, generator=generator).to(dtype) * 0.025
    weights[2] = torch.randn(256, generator=generator).to(dtype) * 0.8
    weights[3] = torch.linspace(-4e-11, 4e-11, 256).to(dtype)
    for row, scale in [(4, 1 / 256), (5, 1 / 512)]:
        midpoints = torch.tensor([2**-10, 1.0625, 1.1875, 1.9375, 2.125, 15.5, 432.0],
                                dtype=dtype) * scale
        lower = torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf")))
        upper = torch.nextafter(midpoints, torch.full_like(midpoints, float("inf")))
        probes = torch.cat((lower, midpoints, upper, -lower, -midpoints, -upper))
        weights[row, :len(probes)] = probes
        weights[row, -2:] = torch.tensor([-448 * scale, 448 * scale], dtype=dtype)
    return weights


def check(dtype, device):
    weights = inputs(dtype)
    expected_codes, expected_scales = reference(weights)
    quantized, scales = quant_fp8_per_row(weights.to(device))
    if quantized.dtype != torch.float8_e4m3fn:
        raise AssertionError(f"expected E4M3FN output, got {quantized.dtype}")
    exact(quantized.view(torch.uint8), expected_codes, "FP8 encoding bits")
    exact(scales, expected_scales, "FP32 output-row scale")
    return {"shape": list(weights.shape), "encoding_and_scale": "exact",
            "input_bytes": storage_bytes([weights]), "stored_bytes": storage_bytes([quantized, scales])}


if __name__ == "__main__":
    raise SystemExit(run_cases("I1 dense FP8", [
        ("fp32/row-scales-zero-floor-rounding", lambda device: check(torch.float32, device)),
        ("bf16/row-scales-zero-floor-rounding", lambda device: check(torch.bfloat16, device)),
    ]))
