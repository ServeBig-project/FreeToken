# Self-speculative decoding: first complete serving version

## Scope

This phase adds self-assisted speculative decoding to FreeToken for the
Qwen3 MoE architecture, with Qwen3-30B-A3B as the real-model acceptance target,
on one RTX 4090. Draft and target use the same checkpoint. Drafting activates
fewer routed experts; target verification retains the checkpoint's original
routing.

The first version includes stochastic sampling, concurrent requests, streaming,
request termination, and cancellation. It is not a greedy-only demonstration.
The supported scheduling policy for this phase is `legacy`; unsupported model
architectures, multi-GPU execution, and other policy combinations must fail at
startup with a clear explanation when speculation is enabled.

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
