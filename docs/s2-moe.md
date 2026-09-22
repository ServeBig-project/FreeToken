# S2-MoE controls

These optional controls extend the existing Qwen3 MoE, single-GPU, legacy
scheduling interface. Ordinary serving and fixed-length SD remain the defaults.
The model checkpoint, sampling API, output limits, stopping, cancellation and
streaming interfaces retain their meanings.

## Resident experts and offline profiling

`--moe-expert-profile counts.json` records ordinary target expert use, including
executed prefill and decode tokens. SD must be disabled. Profiling automatically
uses eager execution so graph padding and capture warmup do not enter the counts.
At idle it writes cumulative counts:

```json
{"num_layers": 2, "num_experts": 4, "counts": [[3, 1, 8, 0], [2, 6, 4, 1]]}
```

Select a fixed number of experts offline:

```sh
ft bench experts --profile counts.json --count 4 --output hot.json
```

Selection orders by descending count, breaking ties by ascending layer and expert
ID. Count may be zero through the total number of model experts. The output is
also the input format of `--moe-resident-experts hot.json`:

```json
{"gpu_experts": [[0, 2], [1, 1], [1, 2], [0, 0]]}
```

Pairs must be unique integers within the current model's layer/expert dimensions.
The list is fixed for the server lifetime. Resident weights remain on the GPU
through requests and cache rebuilds. Drafting can still select other experts.

Residency requires `offload`; `auto` resolves to `offload`. `fused` already holds
the complete model's experts and rejects an extra resident list. CPU/hybrid
expert execution, other architectures, multi-GPU and non-legacy policies are
unsupported for these controls and fail clearly.

`--moe-cache-size` is the total expert-slot budget, including resident weights.
In addition to the list, it must hold the prefill temporary area: two layers of
experts when prefill overlap is enabled, one otherwise. A too-small startup or
runtime rebuild budget fails; a rejected rebuild preserves the running service.

`GET /v1/stats` adds `moe_residency` with `resident_experts`, `cache_slots`, and
`temporary_slots`. These describe the offload pool; `temporary_slots` is total
slots minus resident experts and includes the prefill temporary area. No list
means zero permanently reserved experts. These counts do not claim that all
remaining slots are empty.

## Cost-aware draft length

`--speculative-adaptive-profile cost.json` enables adaptive expansion. SD must be
enabled; offload and fused are supported. All three values must be finite and
strictly positive, measured for this machine, model, cache budget and workload:

```json
{"target_token_ms": 45.0, "draft_step_ms": 27.0, "expert_bandwidth_gib_s": 24.419}
```

The numbers above illustrate units, not a recommended calibration. GiB means
2^30 bytes. The existing `--speculative-num-steps` remains the maximum length.
Resource and output boundaries may shorten a round or require ordinary decoding.

The policy uses the existing draft prefix's cumulative confidence, newly observed
non-resident experts and measured draft cost before drawing the next proposal.
It never removes a sampled candidate merely because its cost looks poor. At least
the first candidate is kept when resources permit. With verification reuse off,
the original target's sampling contract remains in force.

The `speculative` stats object adds `adaptive_enabled` and `adaptive_stops`.
The latter counts request-rounds stopped by the cost decision, not rounds ended
by token/context/cache limits.

## Reuse-aware verification routing

`--speculative-reuse-expert-cap M` enables verification routing reuse; zero is
the default and disables it. An enabled cap must be between target experts per
token and total experts per layer, inclusive, and requires SD.

Only target verification uses this policy. Each request forms its own preferred
expert set, independently of other requests sharing a batch. Every token still
selects the target's original number of experts. Preference changes selection
scores; mixture weights come from the original router scores.

This changes the target computation and is approximate. Original-checkpoint
greedy equality and exact original-target sampling are therefore not promised
with this option enabled. Output accounting, request isolation, stopping,
streaming and cancellation remain required. Quality must be measured separately.

With verification reuse enabled, only the original prompt's KV may enter the
shared prefix cache, subject to its usual page alignment and eviction rules.
Generated KV remains private to the running request and is released on completion
or cancellation. A later prompt containing the previous response recomputes that
generated portion with ordinary target prefill; it cannot reuse the previous
request's approximate generated KV. With reuse disabled, existing prefix-cache
behavior is unchanged.

The `speculative` stats object adds `reuse_enabled` and `reuse_changed_routes`.
The counter sums verification (layer, token) positions whose selected expert
set differs from the unmodified router. It is not a count of changed output
tokens. Disabled controls report false and zero counters.
