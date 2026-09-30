# Independent DFlash acceptance

These tests use only the public contract in `docs/dflash-public-contract.md`
and the HTTP service. Their author has not read production code, implementation
diffs, or existing tests. The coordinator owns service startup, GPU allocation,
and execution against the baseline and changed revisions.

## Phase 1: existing self SD

Run the same requests on both supported target models, with fixed and adaptive
SD and Graph enabled. Keep model, launch arguments, GPU allocation, cache
capacity, and sampling settings identical when comparing revisions.

The service suite covers:

- Greedy and sampled responses, response structure, usage, and output limits.
- Streaming completion, explicit stop, and client cancellation followed by reuse.
- Concurrent requests at C1, C4, and C16, plus mixed lengths that create tail
  batches; Graph execution must be confirmed by its public counters.
- Cold and repeated prefixes, shared prefixes, separate cache groups, and a
  long input whose length is supplied by the coordinator for the configured
  prefill chunk size.
- Idle cache rebuild, rejected busy rebuild, and rejected invalid rebuild with
  a successful subsequent request.

Protocol, resource-lifecycle, or Graph coverage failures fail acceptance. Greedy
text differences between revisions are recorded separately for numerical and
quality review; equality alone does not establish numerical correctness.

Public statistics are saved before and after each workload so the coordinator
can assess actual draft, verification, acceptance, and Graph activity. Tests do
not infer private counters or use production modules as a reference.

## Run

The coordinator starts one isolated service with cache reporting enabled, waits
for `/health`, and runs:

```sh
python blackbox_tests/dflash_common/run_self_service.py \
  --url http://127.0.0.1:PORT --label qwen3-fixed-before \
  --adaptive no --graph yes --max-draft-steps 8 \
  --prefill-chunk-size PREFILL_CHUNK_SIZE --output /tmp/qwen3-fixed-before.json
```

Repeat for Qwen3 and Qwen3.6, fixed and adaptive, before and after the refactor.
Use the service's configured prefill chunk size. `--long-prompt-file` supplies a
different real long-input fixture when needed for the selected cache capacity.
Add `--require-c16-n8` only when the selected mode and capacity support eight
draft tokens at C16. A 96-slot non-Replay baseline can cap that batch at N4.
The public API omits `usage.prompt_tokens_details` for zero cache hits; the
tests interpret omission as zero and still require an explicit positive
`cached_tokens` value when an existing prefix must be reused. Default fixtures
fit the current 1024-token context at prefill chunk size 512; cancellation uses
512 output tokens.
The default long input is a token-ID array containing `prefill_chunk_size + 1`
copies of valid token ID 0. At chunk size 512 this gives 513 input tokens plus
17 output tokens, independent of tokenizer; actual usage must still exceed 512.
Rebuild requests include `num_mamba_slots` only when a GDN pool exists with a
positive slot count. Idle waiting retries only HTTP 503 with `status="busy"`;
an explicit parameter/resource error fails immediately.

```sh
python blackbox_tests/dflash_common/compare_service.py \
  /tmp/qwen3-fixed-before.json /tmp/qwen3-fixed-after.json \
  --output /tmp/qwen3-fixed-comparison.json
```

Exit 0 means all asserted behavior and requested coverage passed. Exit 1 means
an observable behavior failed. Exit 2 means coverage was not demonstrated (or,
for comparison, the request inputs differed). Greedy differences are saved
separately, including cold-versus-reused comparisons within a run, and do not
change the exit code automatically. Sampling requests test protocol and limits;
they do not claim a distributional equivalence result.

The runner records every request body and public response needed to reproduce
a failure. It never starts or stops a service and never reads model code.

## DFlash numerical phase

The independent CPU mathematical reference and its measured results are in
[`../dflash_numeric/README.md`](../dflash_numeric/README.md).

## Performance-only comparison

`performance_requests.json` freezes the 16 distinct public request templates
from `/data2/servebig-envs/replayssm_ab_20260929b_gpu2/new-sd8-adaptive/http.json`.
The runner uses the runtime model ID and a fresh cache group for each wave.
It preserves the original greedy, ignore-EOS, streaming decoding parameters.

```sh
python blackbox_tests/dflash_common/run_performance.py \
  --url http://127.0.0.1:PORT --label qwen36-ar --output /tmp/ar-performance.json

python blackbox_tests/dflash_common/run_performance.py \
  --url http://127.0.0.1:PORT --label qwen36-dflash \
  --output /tmp/dflash-performance.json --reference /tmp/ar-performance.json
```

The coordinator starts each comparison service with its chosen, documented
resource budget. The runner performs the following waves:

| Workload | Warmup | Scored output |
| --- | --- | --- |
| C16, all 16 frozen prompts | 1 wave × 16 requests × 16 tokens | 2 waves × 16 requests × 64 tokens |
| C1, first 4 frozen prompts | 4 sequential requests × 16 tokens | 4 sequential requests × 64 tokens |

Scored totals contain 2304 output tokens; all 320 warmup tokens and warmup
durations/counters are excluded. Each wave starts from a fresh cache group while
the service's expert cache remains warm. This entrypoint performs no cache
rebuilds and does not invoke the lifecycle suite.

The result JSON contains:

- `frozen_requests`, `source`, `models`: exact comparison inputs and model identity.
- `waves[]`: scored/warmup designation, concurrency, wall time, every request's
  input/text/usage/timing, complete statistics and cache reports before/after,
  observed counter deltas, and Graph replay rows with actual/physical sizes.
- `scored.all`, `scored.c16`, `scored.c1`: completion tokens, wall time and
  throughput, average TTFT and per-output latency, summed scored counter deltas
  including draft/accepted/verify, GPU cost and load observations when exposed,
  and Graph actual/physical token totals.
- `comparison` when `--reference` is supplied: equality of frozen/scored inputs
  after replacing only model/cache group, and separately reported greedy text
  differences. Text differences have no automatic quality threshold.
- `completed`, `failures`: protocol, output-count, or comparison-input failures.

TTFT measures client request start to first nonempty content chunk. The
`mean_ms_per_output_token` metric is end-to-end request time divided by output
tokens. `post_first_chunk_ms_per_remaining_token` divides the time between
first and last content chunks by output tokens minus one. A chunk can contain
multiple tokens; this last metric is a stream-observed average. Wave throughput
uses synchronized dispatch through the last completed response and excludes
statistics reads. Public statistics are sampled after HTTP completion and idle;
asynchronous GPU timing observations may still lag generation replies.

DFlash block observations are preserved in
`stats_delta.speculative.dflash_block_gpu_ms`, `dflash_block_samples` (keys are
`B:max_draft_tokens`) and `dflash_block_choices` (index 0 is ordinary generation).
`expert_loads_delta` reports draft loads and, when adaptive-cost observations are
available, target AR/verification loads. Fixed-mode target load counts are
`null`, with `complete_target_counts=false`; the coordinator can supplement them
from the service's MoE log. Cache snapshots preserve `geometry.dflash`, including
weights, context, metadata, and reserved bytes.

Exit 0 means the frozen workload completed; it establishes no minimum speed or
automatic quality conclusion. CPU checks confirmed the templates and derived
counter differences against the supplied public result before service use.
