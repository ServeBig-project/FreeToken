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
