# Flash-Next tiered kernel blackbox

This suite was authored from the public `gather_host_kv` and
`qsa_sparse_paged_attention` contracts. The author has not read production
source, existing internal tests, implementation diffs, or implementation notes.

Run from any directory, choosing the Python environment used by the candidate:

```bash
PYTHONPATH=/absolute/path/to/candidate \
python /absolute/path/to/test-worktree/blackbox_tests/flash_next_tiered_kernels/run.py \
  --report /absolute/path/to/results.json
```

`--suite eager` and `--suite graph` select a subset. The process prints one JSON
record per case and exits nonzero on failure. It records exception type and
message, without a traceback or production source inspection.

## Fixed acceptance rules

- Copied payload and scale elements must match their original encodings exactly.
  GPU-resident positions, padding, and guards surrounding output storage must
  retain their initial values.
- Tiered and all-GPU attention outputs must be elementwise identical, for the
  same encoded data, queries, and selected token positions.
- An independent CPU reference computes GQA attention over selected tokens.
  INT8 data is multiplied by its BF16 scale and rounded to BF16 before the
  dot product. Softmax and accumulation use float64, then round to BF16.
  Tolerances were fixed before running the candidate: `atol=1/64, rtol=1/64`.
- Empty selection rows produce exact zeros. Both an allocated return value and
  caller-supplied output storage are exercised.

## Supported input coverage

- BF16 and INT8, per-token/per-head K/V scales, and zero quantized values.
- `D=256`, two KV heads, 2/4/8 query heads, batch 1/3/5, multiple requests,
  nonsequential physical page mapping, and 64-token page boundaries.
- Four-token groups, one/two/three-token tails, partial/all-row padding, and
  maximum selection width 2051 using 33 physical pages.
- All-host, all-GPU, mixed residency, three layers with different payloads,
  and noncompact residency row stride.
- CUDA Graph capture and four replays with changed legal host addresses,
  residency, selection, tables, queries, and cache contents at fixed shapes.
  Host allocations remain alive throughout replay.

These are numerical and output-integrity tests. They do not measure transfer
volume, prove that whole pages were not copied internally, or establish a
latency threshold: those require an observable performance contract or profiling.
