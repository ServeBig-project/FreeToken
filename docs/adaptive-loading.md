# Adaptive speculative serving

The three new options are independent and default off:
`--speculative-adaptive-cost`, `--speculative-draft-load-missing`, and
`--speculative-verify-prefetch`. The migrated Qwen3 and Qwen3.5/3.6 MoE
components use single-GPU BF16 offload, legacy scheduling and a draft ceiling of 1–8 tokens.
They retain full target verification.

## Drafting with already-cached experts

`--speculative-draft-residency {off,router}` defaults to `off`, where drafting
selects the top K of the original router scores and loads missing experts on
demand. `router` requires SD and chooses K distinct experts by original router
score among experts currently present in the GPU cache. Their original router
probabilities supply the mixture weights, with the checkpoint's usual
normalization. For `--moe-backend fused`, all experts are resident, so `router`
behaves like ordinary draft routing.

Without `--speculative-draft-load-missing`, every layer must hold at least K
cached experts before a draft round. Otherwise all requests that could still
draft in that batch use ordinary target generation for that round. No experts are
reserved or loaded to make the check pass; drafting performs zero expert loads,
so its available set stays valid for the whole round. Ordinary generation,
prefill and target verification keep their routing and loading behavior.

`/v1/stats.speculative` reports `draft_residency` and `residency_stops`, the
number of request rounds refused drafting because of that shared shortage. With
`--moe-collect-stats`, `draft_expert_loads` counts actual expert-row loads during
draft execution, including the `off` mode; it is `null` when collection is
disabled. Repeated loads of the same expert count again; duplicate routes sharing
one load do not.

## Cache-first drafting with missing experts

`--speculative-draft-load-missing` requires
`--speculative-draft-residency router`; another residency mode is a startup error.
Every layer reads the current shared-cache mapping. Cached experts have priority;
when fewer than k are available, the highest original router scores among missing
experts fill the remaining places. Mixture weights come from the original router
scores for the chosen experts. The existing cache admits and copies each distinct
missing expert once for that layer's whole batch.

This option removes the whole-batch insufficient-residency fallback, without
requiring extra expert slots. The off setting preserves
the previous strict cache-only behavior. Cost decisions and request/KV limits are
separate from `residency_stops`.

`speculative.draft_length_histogram` in `/v1/stats` counts request rounds: index n
is the number of rounds that actually drafted n candidates. Index 0 includes
ordinary-generation decisions and rounds limited by output, KV or state capacity. With a
ceiling of 8, the list has 9 entries. It is not an output-token count.

## Stepwise cost control

`--speculative-adaptive-cost` chooses ordinary generation before drafting or
reconsiders continuation after each completed draft step. Completed candidates
are retained. The continuation decision uses the known prefix, routes and cache
state, not a filter on the newly sampled candidate. It remains compatible with
stochastic acceptance/rejection and with CUDA Graph execution.

The controller learns AR, draft and verification timings from actual serving
forwards, separately by request batch size. CUDA events also measure demanded
expert copies and MoE computation. Copies are subtracted before fitting compute
cost; forecasts add expected transfers once. Non-MoE compute scales with physical
query rows and MoE compute with logical rows, so padded verification does not make
the next token's computation free. New, unobserved batch sizes initially use the
nearest measured batch's rates until their own samples arrive.

Known draft predictions form a deduplicated target-expert union. For unobserved
positions, per-expert selection frequencies from real target queries estimate
union probability as `1 - (1 - p)^m`. All terms are multiplied by the current
missing-expert mask: cached or prefetched experts have no transfer charge.
Additional logical verification positions can therefore increase expected
transfers even when physical graph rows stay fixed. This occupancy approximation
does not model all correlations or guarantee future target routing; prediction
error is exposed against actual distinct expert loads.

Expected effective outputs `g(n)` use observed accepted-prefix counts for each
length, with a small decreasing prior before observations exist. Raw draft-token
probabilities are not substituted for acceptance rates. Admission compares the
estimated achievable SD cost with AR; continuation compares only the next draft
step and incremental verification cost, excluding already-paid drafting work.

For a new batch size, the controller first observes an AR decode, then one
single-candidate calibration round. Every 16 batch admission decisions it probes
one position beyond the largest observed depth, up to the configured ceiling,
so cost rejection cannot permanently prevent new SD observations. Other rounds
use the stepwise policy. Natural output/KV limits and strict-router availability
still apply. Prefetch-only mode collects costs without forcing these AR probes
or enabling cost-based stopping.

A completed step returns at most one batch control packet to the CPU. Per-layer
event timestamps are read after that feedback; no per-layer CPU synchronization
is added. `cost_control_ms` includes feedback waiting and host control work.
Startup/capture warmups do not enter the learning samples.

Additional `/v1/stats.speculative` counters are cumulative:

- `cost_ar_requests`: eligible request rounds sent to AR by this controller,
  including its initial AR calibration; `cost_stopped_requests`: request rounds
  stopped after completed candidates by a continuation cost decision.
- `cost_probe_requests`: eligible request rounds admitted for calibration or
  periodic exploration. These counts are not generated-token counts.
- `cost_samples`: numbers of observed model forwards for `ar`, `draft`, `verify`.
- `cost_gpu_ms`: elapsed forward time and component time (`moe_compute`,
  `demand_copy`, `prefetch_copy`, `prefetch_wait`); overlapping components must
  not be added to parent forward durations.
- `cost_transfer_predictions`: predicted expected expert loads, actual distinct
  loads and absolute error, separately for AR, draft and verification.

These measurements do not depend on `--moe-collect-stats`.

## Predicted verification prefetch

`--speculative-verify-prefetch` predicts target top-k experts from each draft
layer's original router scores, while drafting still computes only its selected
small k. Candidates form the current round's per-layer union; scores are cleared
at the next round. All weights occupy the existing shared expert pool.

A GPU plan deduplicates missing candidates and protects the current computation's
slots, permanent pins, and unfinished copies. It admits only a cost-limited set:
predicted saved target transfer is compared with exposed copy time and the
expected reload cost of the victim. The copy window and per-expert transfer cost
come from observed serving events. Until transfer cost is known, at most one
expert copy per draft forward is permitted for calibration; this does not force
an AR generation round.

Admission and publication run on the main stream. A separate stream copies the
reserved rows concurrently with draft MoE computation and following attention.
The mapping remains unavailable until completion. Before another admission, the
main stream joins that copy and publishes its rows. The final copy is joined
before model forward returns, including inside CUDA Graphs. This is an overlap
attempt, not a guarantee of hidden latency or speedup; copy kernels consume GPU
resources too. Measured copy and exposed-wait times remain visible in the cost
statistics.

With strict `router` residency and load-missing disabled, prefetch protects the
entire round's original allowed expert set. A full pool can therefore yield zero
prefetches. It cannot silently load extra draft experts or invalidate the frozen
allowed set. With residency `off`, or load-missing enabled, the original allowed
set is not pinned by this rule.

Prefetch counters in `/v1/stats.speculative` are independent of
`--moe-collect-stats`:

- `prefetch_predicted_experts`: sum of per-layer candidate-union sizes at each
  draft step. Requests/positions within each set are deduplicated; this is not a
  lifetime count of unique expert IDs.
- `prefetch_loaded_experts`: actual completed expert copies. A later reload after
  eviction is another copy. `draft_expert_loads` continues to count demand copies
  only, so these two counters do not double-count prefetch.
- `prefetch_used_experts`: first subsequent target decode/verify use of each
  completed prefetch while still cached. Draft use and repeated target hits do
  not increment it.
- `prefetch_evicted_unused_experts`: completed prefetches evicted before that
  first target use, including invalidation by cache rebuild.
- `prefetch_loaded_bytes`, `prefetch_used_bytes`, and
  `prefetch_evicted_unused_bytes`: the corresponding actual expert-row bytes.

Rebuild recaptures against the new cache geometry, retains cumulative counters,
and invalidates old prefetched mappings. Capture warmups do not count as serving
prefetches. The switches may be enabled separately or together; prefetch-only
mode uses cost estimation without enabling adaptive stopping.
