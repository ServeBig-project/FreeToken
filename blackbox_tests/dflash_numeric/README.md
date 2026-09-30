# Independent DFlash numerical acceptance

The oracle is NumPy float64 mathematics from `docs/dflash-public-contract.md`.
The test author has not read production code, implementation diffs, or existing
tests. Production is accessed only through the documented model constructor,
attributes, `project_context`, and `forward` callbacks.

The fixture follows the current public
[checkpoint configuration](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/blob/f181eece646affea2c38b2765f1aaa01a9734ccd/config.json)
and safetensors header. It keeps six layers, eight target feature layers, 4:1
grouped query attention, a query projection twice the hidden width, a 3× MLP,
default full-head RoPE, five causal 4096-token windows and one noncausal full
attention layer. Hidden width is reduced to 32 and head dimension to 8.

## Predeclared acceptance

These thresholds were approved before the first model execution and are not
adjusted after a failure. Both maximum absolute error and NRMSE must pass.
NRMSE is RMSE divided by the reference tensor's RMS, floored at 1e-12.

| Loaded dtype | Reference | Maximum absolute error | NRMSE |
| --- | --- | ---: | ---: |
| FP32 | Independent ideal FP64 equations | 2e-5 | 2e-6 |
| BF16 | Independent equations with explicit BF16 rounding | 0.0625 | 0.01 |

The BF16 oracle uses round-to-nearest-even for inputs and weights, every linear
projection, the RMS cast and weighted product, RoPE trigonometric values and
products/sum, residual sums, SiLU, MLP products, and returned attention values.
The absolute allowance is eight BF16 steps at unit magnitude across six layers;
the NRMSE bound additionally limits overall drift. Fixture normalization weights
are near one and projection weights scale with their input width. Error against
ideal FP64 using the same quantized inputs and weights is reported separately.

## Coverage and execution

- Context K/V and every layer's noise Q/K/V, attention output, and final states.
- Different absolute positions, a real 4096-token window boundary, and full
  noncausal block attention with later noise positions visible.
- C1, mixed request lengths/order, C16, and callback padding masked from output.
- Repeated target-context extensions; temporary noise K/V stays separate.
- Caller-owned target embeddings and interchangeable output heads; checkpoints
  contain neither embedding nor head weights.
- Exact public shapes/dtypes/parameter bytes, input immutability, repeated-call
  stability, and documented invalid-checkpoint errors.

The padded PyTorch callback is separate from the scalar NumPy attention oracle.
Block widths up to 16 exercise the public checkpoint computation interface;
service acceptance separately observes the current eight-draft-token limit.

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python blackbox_tests/dflash_numeric/run_numeric.py --output /tmp/dflash-numeric.json
```

The runner creates its small public-format checkpoint in a temporary directory,
uses CPU only, records component errors, and exits nonzero on a contract failure.
It does not validate the real BF16 checkpoint, production attention kernels,
CUDA Graph execution, service sampling, or cache lifecycle.

## CPU result

All 8 cases passed on 2026-09-30, with 266 enforced component comparisons per
dtype. The approved thresholds were unchanged.

| Comparison | Maximum absolute error | Maximum NRMSE |
| --- | ---: | ---: |
| FP32 vs independent FP64 | 1.2148523e-5 | 7.0357180e-7 |
| BF16 vs explicit-rounding reference | 0.03125 | 0.0013440983 |
| BF16 final output vs ideal FP64, report only | 0.0480172 | 0.00669873 |

The two maxima in a row can occur in different components. Full per-component
observations are written to the runner's requested JSON output.
