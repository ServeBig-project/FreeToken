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
