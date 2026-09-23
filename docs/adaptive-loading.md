# Adaptive speculative serving

The three new options are independent and default off:
`--speculative-adaptive-cost`, `--speculative-draft-load-missing`, and
`--speculative-verify-prefetch`. The supported target is single-GPU Qwen3 MoE
BF16 with offload, legacy scheduling, and a draft ceiling of 1–8 tokens.
They retain full target verification. Existing S2 adaptive/reuse and fixed
resident lists remain separate baselines, outside these combinations.

## Cache-first drafting with missing experts

`--speculative-draft-load-missing` requires
`--speculative-draft-residency router`; another residency mode is a startup error.
Every layer reads the current shared-cache mapping. Cached experts have priority;
when fewer than k are available, the highest original router scores among missing
experts fill the remaining places. Mixture weights come from the original router
scores for the chosen experts. The existing cache admits and copies each distinct
missing expert once for that layer's whole batch.

This option removes the whole-batch insufficient-residency fallback, without
requiring fixed resident experts or extra expert slots. The off setting preserves
the previous strict cache-only behavior. Cost decisions and request/KV limits are
separate from `residency_stops`.

`speculative.draft_length_histogram` in `/v1/stats` counts request rounds: index n
is the number of rounds that actually drafted n candidates. Index 0 includes
ordinary-generation decisions and rounds limited by output/KV capacity. With a
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

These measurements do not depend on `--moe-collect-stats`. The new controller
cannot be combined with the older `--speculative-adaptive-profile` policy.
