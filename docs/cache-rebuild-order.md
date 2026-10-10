# Split-pool cache rebuild: shrink before grow

Status: problem statement only; no implementation yet. Found by the NoWAG (#11) delivery audit. The order it describes already existed in `main@ea3b9df`, and it is not specific to NoWAG.

## Problem

This concerns split cache pools, i.e. a server started without `--runtime-cache-gib`. A single `POST /v1/cache/rebuild` that shrinks one pool and grows another can run out of GPU memory partway through, even when the final layout fits the budget.

`Engine.rebuild_runtime_cache` (`python/freetoken/engine/engine.py`) first validates the target layout against the budget, then:

1. destroys the CUDA graphs and sets `rebuild_teardown_started = True`; past this point a failure needs a rebuild to restore service;
2. rebuilds the MoE expert cache (`moe_offload_cache.rebuild`) and re-reserves the expert scratch (`_alloc_expert_workspace`);
3. resizes the KV pool (`_resize_kv_pool`), including the SWA window;
4. rebuilds the GDN/mamba state pool (`linear_state_pool.rebuild`).

When the expert cache grows and KV or GDN state shrinks in the same request, step 2 allocates the larger expert cache while the old KV pages are still held. The validation only checks the final total, so the intermediate peak is never checked.

The shared-runtime path (`--runtime-cache-gib`) is not affected: it releases the old runtime before growing the expert cache.

## Example

This is a capacity model of the real rebuild statements, taken from the audit. A 24 GiB card with a 90% budget allows 21.6 GiB.

| | Fixed | Experts | KV | Total |
| --- | ---: | ---: | ---: | ---: |
| Before | 3 GiB | 1 GiB | 17 GiB | 21 GiB |
| Target | 3 GiB | 5 GiB | 12 GiB | 20 GiB (passes validation) |
| Peak with the current order | 3 GiB | 5 GiB | 17 GiB | **25 GiB** |

A real GPU OOM has not been reproduced yet.

## Proposed fix

- Keep all existing target validation as it is.
- Within one rebuild, apply every pool that shrinks before any pool that grows. This covers KV pages, SWA window pages, GDN/mamba state slots, and the MoE expert cache with its scratch.
- Leave resource policy unchanged, and add no model- or format-specific branches.

## Acceptance

- An independent black-box test on split pools: one request shrinks `num_pages` and grows `moe_cache_size`. The final total must be under budget, while the "grow first" peak exceeds the device. The rebuild must succeed and keep serving.
- The same check for the state pool: shrink `num_mamba_slots` and grow the expert cache.
- The existing split-pool and shared-runtime rebuild tests stay green.
