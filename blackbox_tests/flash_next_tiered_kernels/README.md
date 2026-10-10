# Flash-Next tiered kernel blackbox

This suite was authored from the public `gather_host_kv` and
`qsa_sparse_paged_attention` contracts. The author has not read production
source, existing internal tests, implementation diffs, or implementation notes.

Run from any directory, choosing the Python environment used by the candidate:

```bash
PYTHONPATH=/absolute/path/to/candidate/python \
python /absolute/path/to/test-worktree/blackbox_tests/flash_next_tiered_kernels/run.py \
  --report /absolute/path/to/results.json
```

`--suite eager` and `--suite graph` select a subset. The process prints one JSON
record per case and exits nonzero on failure. It records exception type and
message, without a traceback or production source inspection.

Before counter coverage was added, candidate `7139b31` passed all 11 numerical
cases on physical GPU 2 in
`ft-flash-next-i3a-gpu2`, using the unchanged tolerances below.
See [the numerical acceptance report](results-7139b31.json).

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
- The optional CUDA `int64[1]` counter starts at `2**32 - 137` and accumulates exactly
  `host_selected_slots * Hkv * (2 * D * element_bytes + scale_bytes)` per call,
  where `scale_bytes` is 4 for INT8 and 0 for BF16. GPU-resident and padding
  positions do not contribute. Crossing 4 GiB verifies the accumulator's 64-bit
  behavior. Explicit `counter=None` preserves numerical
  behavior; graph replays accumulate using each replay's selection and residency.

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

The counter reports logical host-read bytes. These tests do not measure PCIe bus
transactions, prove that whole pages were not copied internally, or establish a
latency threshold: those require profiling or a performance contract.
