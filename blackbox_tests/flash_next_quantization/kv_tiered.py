"""I2 GPU integration with actual QSA geometry, without rerunning the I3 matrix."""

import math
import sys
from pathlib import Path

import torch

from common import exact, run_cases
from kv import inputs, quantize_kv_int8, reference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flash_next_tiered_kernels"))
from run import COUNTER_START, DeviceInputs, Fixture


def check(dtype, device_name):
    if device_name != "cuda":
        raise ValueError("tiered integration requires an authorized CUDA window")
    fixture = Fixture(dtype, "groups", heads=24, mode="mixed", seed=341, layer=1)
    shape = tuple(fixture.encoded[fixture.layer].shape)
    bank = inputs()
    indices = torch.arange(math.prod(shape[:-1])) % len(bank)
    source = bank[indices].reshape(shape)
    if dtype == torch.int8:
        expected_codes, expected_scales = reference(bank)
        codes, scales = quantize_kv_int8(source.cuda())
        exact(codes, expected_codes[indices].reshape(shape), "GPU INT8 encoding before migration")
        exact(scales, expected_scales[indices].reshape(shape[:-1]), "GPU BF16 scales before migration")
        fixture.encoded[fixture.layer].copy_(codes.cpu())
        fixture.scales[fixture.layer].copy_(scales.cpu())
        fixture.host_scales.copy_(fixture.scales)
    else:
        fixture.encoded[fixture.layer].copy_(source)
    fixture.host.copy_(fixture.encoded)
    device = DeviceInputs(fixture)
    device.gather(fixture)
    device.check_gather(fixture)
    device.check_counter(COUNTER_START + fixture.logical_read_bytes())
    device.check_attention(fixture, device.attend(supplied_out=False))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        device.gather(fixture)
        device.attend()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        device.gather(fixture)
        device.attend()
    device.counter.fill_(COUNTER_START)
    device.reset_outputs()
    graph.replay()
    torch.cuda.synchronize()
    device.check_gather(fixture)
    device.check_counter(COUNTER_START + fixture.logical_read_bytes())
    device.check_attention(fixture, device.out)
    return {"heads": 24, "kv_heads": 2, "head_dim": 256, "tokens": fixture.tokens,
            "selection_width": fixture.width, "modes": ["eager", "cuda_graph"],
            "tiered_vs_gpu": "exact", "reference_atol": 1 / 64, "reference_rtol": 1 / 64}


if __name__ == "__main__":
    raise SystemExit(run_cases("I2 quantized KV tiered integration", [
        ("bf16/actual-qsa-geometry", lambda device: check(torch.bfloat16, device)),
        ("int8/quantization-to-tiered-attention", lambda device: check(torch.int8, device)),
    ], default_device="cuda"))
