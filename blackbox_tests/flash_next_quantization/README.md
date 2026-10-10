# Flash-Next quantization numerical blackbox

These tests use only the published numerical contracts. No production source,
internal tests, implementation notes, or implementation diffs were read.

## I1: dense FP8

`dense.py` compares exact FP8 E4M3FN encoding bits and exact per-output-row FP32
scales for FP32 and BF16 inputs. Its independent reference constructs E4M3 values
from exponent/mantissa bits and chooses the nearest code, breaking ties to even.
Scale is `FP32(max(max_abs / 448, 1e-12))`; normalization uses that stored scale.

Inputs include normal weight ranges, different row scales, zero rows, the scale
floor, positive/negative halfway values, and their neighboring input values.
Encoding and scale acceptance is exact, with no adjustable tolerance.

Candidate `8075e8d` passed both CPU cases using Torch `2.11.0+cu130`, with CUDA
hidden. See [the I1 CPU report](results-i1-8075e8d-cpu.json). For the 6×256 BF16
fixture, encoded weights plus FP32 scales occupy 1560 bytes versus 3072 input bytes.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
PYTHONPATH=/absolute/path/to/candidate/python \
python /absolute/path/to/test-worktree/blackbox_tests/flash_next_quantization/dense.py \
  --report /absolute/path/to/dense-results.json
```

Each runner prints JSON case results without production tracebacks and exits
nonzero on failure. `--device cuda` is reserved for an authorized GPU window.

## I2: INT8 KV

`kv.py` compares exact INT8 encodings and stored BF16 scales. Its scalar reference
rounds scale to BF16 by manipulating IEEE floating-point bits, then divides by
that stored scale and rounds to the nearest even integer before clipping.
All-zero vectors use scale 1. Acceptance is exact, with no numerical tolerance.

Fixtures cover zero vectors, positive/negative ties and neighbors, token/head
scale differences, and supported prefix dimensions ending in `head_dim=256`.
The maximum-1.5 vector contains 0.75: its final BF16 scale produces code 63,
distinguishing it from code 64 obtained using the unrounded scale.

Candidate `8075e8d` passed all four CPU cases with CUDA hidden; see
[the I2 CPU report](results-i2-8075e8d-cpu.json). Each 256-element BF16 vector
uses 512 input bytes and 258 encoded bytes including its BF16 scale.

Run CPU acceptance by replacing `dense.py` above with `kv.py`.
`kv_tiered.py --device cuda` is a separately authorized GPU integration run:
BF16 and actual quantized INT8 payloads use `Hq=24, Hkv=2, D=256`, mixed host/GPU
residency, five queries, and eager/CUDA Graph execution. It reuses the existing
independent selected-token attention reference with unchanged `atol=rtol=1/64`
and exact tiered-versus-all-GPU comparison. It does not rerun the old I3 matrix.
Candidate `0abb433` passed both GPU integration cases using test commit `587963a`
on physical GPU 2, including eager and CUDA Graph; the container exited with code 0.
See [the I2 GPU integration report](results-i2-0abb433-gpu.json).
