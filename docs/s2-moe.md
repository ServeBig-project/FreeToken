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

Draft length is chosen online by `--speculative-adaptive-cost`; see
[adaptive speculative serving](adaptive-loading.md).

## Drafting with already-cached experts

`--speculative-draft-residency {off,router,affinity}` defaults to `off`.
The other modes require SD and use the existing `--speculative-draft-experts K`:

- `router` chooses K distinct experts by original router score among experts
  currently present in the GPU cache. Their original router probabilities supply
  the mixture weights, with the checkpoint's usual normalization.
- `affinity` starts with the original router's K experts and preserves cached
  selections. Missing selections, processed in descending original router-score
  order, are replaced by the closest cached expert not already selected or
  reserved for an original cached selection. Distance ties use the smaller expert
  ID. Replacements keep the original selections' mixture weights.

Affinity uses full gate/up/down weight L2 distances, following the affinity idea
in [SpecMoE §III-B](https://arxiv.org/html/2604.10152v1#S3.SS2), with FreeToken's
existing shared cache. The full squared distances are computed on the CPU in
FP32 at startup without sampling dimensions or normalizing expert weights. This
mode requires unquantized floating-point offload experts; packed/quantized banks
fail clearly. Its one-time build duration is logged separately from generation.
For `fused`, all experts are already resident: both modes retain ordinary draft
routing and no affinity table is computed.

Before each draft round, every layer must have at least K valid cached experts.
Otherwise all requests that could still draft in that batch use ordinary target
generation for that round. This is a shared-cache condition, not a per-request
cache shortage. No experts are reserved or loaded to make the check pass.
Drafting performs zero expert loads and cannot evict experts, so its available
set stays valid for the whole round. Ordinary generation, prefill and target
verification keep their previous routing/loading behavior. The modes compose
with cost-based expansion and the separate approximate verification-reuse option.

With approximate verification reuse enabled, drafts depend on cache contents, so
repeated requests can produce different outputs after different cache histories.
With reuse disabled, the original target contract still applies; the approximate
combination is not lossless.

`/v1/stats.speculative` adds `draft_residency` and `residency_stops`, the number of
request-rounds refused drafting because of that shared shortage. With existing
`--moe-collect-stats` enabled, `draft_expert_loads` counts actual expert-row loads
during draft execution, including the `off` mode as a control, and
`draft_expert_replacements` counts affinity substitutions across layer/token
routes. Router/off modes do not substitute experts and report zero replacements.
Both measurement fields are `null` when collection is disabled. Repeated loads
of the same expert count again; duplicate routes sharing one load do not.

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
