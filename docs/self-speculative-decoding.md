# Self-speculative decoding

## Scope

The shared SD runtime currently has migrations for Qwen3 MoE and Qwen3.5/3.6
MoE (including Qwen3-30B-A3B and Qwen3.6-35B-A3B). Draft and target use the same
checkpoint. Drafting activates fewer routed experts; target verification keeps
the checkpoint's original routing. The component refactor's real-model acceptance
is tracked separately; the presence of an implementation does not establish a new
quality or performance result.

Support is determined by the configured routing, attention/state and expert
execution components, not the checkpoint's name. A checkpoint using the migrated
components reuses the same SD loop. Other model-owned routers or unsupported
attention/state components fail with a component-specific startup error; this
migration does not add their SD support.

The interface includes stochastic sampling, concurrent requests, streaming,
request termination and cancellation. The scheduling policy is `legacy`, on a
single GPU. Unsupported policy/execution combinations fail at startup.

Enabled speculation supports GPU-resident experts (`--moe-backend fused`) and
GPU expert execution with CPU weight offload (`--moe-backend offload`). `auto`
resolves to `offload`. CPU/hybrid expert execution and `--moe-cpu-layers` are
unsupported and fail at startup. Scheduling is synchronous while speculation is
enabled; requests are still batched and can run concurrently. CUDA Graph
execution is described in [speculative CUDA graphs](sd-cuda-graphs.md).

## Public interface

`ft serve` adds:

- `--speculative-num-steps N`: number of proposed tokens per round; integer,
  default `0` (ordinary serving), nonnegative.
- `--speculative-draft-experts K`: routed experts used per draft token; integer,
  default `3`. With speculation enabled, `1 <= K <= target experts per token`.
  Equality is supported as an unchanged-drafter control.

All speculative switches, their defaults and legal combinations:

| Option | Default | Off / default behavior | Constraint |
| --- | --- | --- | --- |
| `--speculative-num-steps N` | `0` | ordinary serving | `N` is the draft ceiling; the three controls below and CUDA Graph need `1 <= N <= 8` |
| `--speculative-draft-experts K` | `3` | — | `1 <= K <=` target experts per token; any K works with CUDA Graph |
| `--speculative-draft-residency off\|router` | `off` | draft uses the original top-K and loads misses | `router`: only cached experts; see [adaptive serving](adaptive-loading.md) |
| `--speculative-draft-load-missing` | off | `router` falls back to ordinary generation when a layer has fewer than K cached experts | requires `router` |
| `--speculative-adaptive-cost` | off | every round drafts up to `N` | — |
| `--speculative-verify-prefetch` | off | no prefetch | — |
| `--cuda-graph-max-bs` | automatic | `0` runs speculation eagerly | SD Graph needs BF16 experts, `--moe-backend offload`, FlashInfer and page size 1; otherwise startup fails |

The last three boolean controls require BF16 experts with `--moe-backend offload`
and may be combined freely with each other, any K and either residency mode,
except `--speculative-draft-load-missing` without `router`.

Existing model, GPU, expert-cache, KV-cache, concurrency, and server arguments
retain their meanings. The main model uses an already-supported checkpoint
format; NoWAG is not required.

The existing HTTP APIs retain their request and response formats. In particular,
`/v1/completions` and `/v1/chat/completions` support existing temperature,
top-k/top-p sampling, `max_tokens`, stopping, and stream/non-stream requests.
Only committed tokens count toward generated-token usage and output limits.
Temporary draft tokens must never be emitted to clients.

`GET /v1/stats` exposes a `speculative` object with `enabled`, `draft_tokens`,
`accepted_draft_tokens`, and `verify_steps`. Counters are cumulative since server
startup. Accepted draft tokens count only tokens retained for output; target
correction/bonus tokens are not accepted draft tokens. Counters include work
spent on subsequently cancelled requests. Disabled serving reports `enabled:
false` and zero counters.

## Required behavior

- Greedy decoding follows the original target checkpoint. Stochastic decoding
  applies an exact acceptance/rejection rule relative to the target's requested
  sampling distribution; matching random seeds is not a promise of identical
  sampled text between execution algorithms.
- Concurrent requests remain independent even when their prompts, sampling
  settings, accepted lengths, output limits, and finish times differ.
- EOS, stop strings, and output limits terminate at the same public boundaries
  as ordinary serving. Responses and token usage exclude discarded drafts.
- Client disconnects and cancellation release the request's resources. Rejected
  drafts do not become reusable prompt history or consume cache capacity
  permanently. Subsequent requests remain correct after completion or abort.
- Batch/token limits and available cache capacity bound speculation. A request
  near its output or context limit still completes correctly; the engine may
  propose fewer tokens or perform ordinary decoding for that round.
- Invalid new CLI values and unsupported enabled combinations fail clearly.
  Speculation disabled preserves the existing supported serving behavior.

## Evaluation

Correctness must be checked by an independent agent using this public contract
and executable interfaces only. Test authors must not inspect implementation,
diffs, internal tests, or implementation notes. Implementation and black-box
tests are separate commits.

Use matched checkpoint, prompts, sampling policy, cache budgets, and concurrency
when comparing enabled and disabled serving. Record request latency, committed
output throughput, draft acceptance, and memory usage. Functional completion
does not itself assert a speedup or a reproduction of full S2-MoE results.

Ordinary serving of this checkpoint can produce different greedy continuations
when CUDA Graph or scheduling settings change, even with speculation disabled.
The acceptance evidence retains those ordinary runs as exact references;
it does not use a text-similarity tolerance across execution modes.

## Run on this machine

From this checkout with the `freetoken-dev` environment active:

```bash
PYTHONPATH=python python -c 'from freetoken.cli import main; main()' serve \
  --model-path /data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B \
  --gpu 1 --host 127.0.0.1 --port 30000 \
  --moe-backend offload --moe-cache-size 512 \
  --batching-policy legacy --max-running-requests 4 --cache-type radix \
  --max-seq-len-override 4096 --num-tokens 8192 --max-prefill-length 2048 \
  --speculative-num-steps 4 --speculative-draft-experts 3
```

This uses the real-model acceptance resource limits, not a tuned lab deployment.
Use an available GPU assigned to the service. Set `--speculative-num-steps 0`
for ordinary serving with the same checkpoint.

## Gated DeltaNet models

Qwen3.5/3.6 MoE has linear-attention state that must agree with the retained
computed prefix. Drafting and verification keep temporary states. Only after
EOS, stop strings and output limits determine the retained output is the matching
state committed for continuation or prefix reuse. Cancellation discards the
round's temporary state and keeps the previously committed prefix.

State allocation uses the current round's admitted lengths, including shorter
calibration rounds and request tails. A request drafting N > 0 tokens needs N + 2
temporary slots; a zero-draft tail in a verifying batch needs one. If capacity is
short, free and evictable prefix-state slots are considered first, then the draft
ceiling is shortened. If even one step for the current batch cannot fit, the
whole batch uses ordinary generation. This does not split batches, enlarge the
state pool or take memory from experts or KV.

`state_slot_stops` counts whole-batch fallbacks due to state capacity.
`draft_length_histogram` records actual request-round draft lengths, after all
resource and cost decisions. The default state budget is not a guarantee that
full concurrency can draft: protected prefixes also consume it. The existing idle
cache rebuild can explicitly resize `num_mamba_slots`.

With `--cache-type naive`, fixed live-state slots and the padding sink are
reserved. The default naive pool has no temporary-state capacity, so SD legally
falls back to ordinary generation; an explicitly enlarged pool can run SD.
Rebuild and Graph capture preserve those ownership boundaries.

Draft and verification CUDA Graphs are implemented for the migrated Gated
DeltaNet models under the same BF16/offload/FlashInfer/page-size-1 conditions.
Requesting unsupported Graph components fails at startup; explicit
`--cuda-graph-max-bs 0` remains the way to choose eager execution.
