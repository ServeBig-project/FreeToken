# Fixed-step speculative CUDA graphs

The existing `--cuda-graph-max-bs` controls this path. `0` keeps eager execution;
`32` captures every real batch size from 1 through 32 when
`--max-running-requests 32` is set, including all intermediate tail batches.
The capture limit is the smaller of these two settings and 32. Supported SD
configurations are single-GPU Qwen3 MoE BF16 with the offload backend, FlashInfer attention,
page size 1, legacy scheduling, K=3, and at most four proposed tokens. Draft
residency may be `off` or `router`. Adaptive expansion, permanent resident lists,
approximate verification reuse, affinity and quantized experts keep their existing
eager behavior in this phase.

Target decode, one-token drafting, and multi-query verification use separate
model graphs. A full four-token draft replays the draft graph four times, then
verifies five input positions. Tail rounds and requests with different remaining
lengths retain their actual batch size and per-request query lengths. Attention planning, sampling, rejection,
KV allocation and request termination remain outside graph capture. Router-mode
availability masks, expert routing, slot mappings and cached contents remain dynamic.

Expert and usable KV budgets retain their configured capacities. Graph buffers
and private graph memory are additional allocations; startup fails if those do
not fit rather than silently reducing expert slots or KV pages. Graph capture
does not publish output or modify reusable request KV.

With 1706 expert slots, N4 and batch limit 32 capture 136 graphs across the three
phases. B1–4 retain exact verification query shapes; higher batches reuse one
full-width verification graph per B. Short tails append dummy queries after all
real queries. Their attention output is initialized to zero, KV writes use the
reserved dummy slot, and negative expert IDs skip expert admission and compute.
Dummy queries do not load experts or change real routing. The exact-shape range
also expands when needed to preserve the cache's LRU/layer-distance choice.
SD warmups retain expert residency between shapes
and reset it before serving. Startup cost and additional reservation are reported
by `capture_seconds` and `extra_reserved_bytes` below.

`GET /v1/stats` adds `cuda_graph`:

- `enabled`: at least one graph is captured for the current engine.
- `target_decode`, `draft`, `verify`: cumulative successful replay submissions.
- `replay_shapes`: entries with `phase`, `batch_size`, `query_tokens`,
  `physical_query_tokens`, and `replays`.
  `batch_size` counts real requests and `query_tokens` counts logical queries;
  ordinary autoregressive padding (such as B3 executing a physical B4 graph) is
  excluded from these logical counts. `physical_query_tokens` includes padding
  and describes the GPU graph's query rows. Verification query lengths within a
  request group may differ while the batch size and total logical query count match.
- `capture_seconds`: elapsed time constructing the current graph set.
- `extra_reserved_bytes`: net additional PyTorch GPU reservation during capture.
  This is a measured reservation increase, not a sum of logical tensor sizes.

Only actual replays increment the counters. Internal startup/capture warmups do
not count; ordinary HTTP warmup requests do. Counters survive cache rebuilds,
while capture time and reserved-byte measurements describe the latest graph set.
Eager execution reports disabled graphs and zero replay counters. Unsupported SD
combinations retain their existing eager behavior without claiming graph execution.

FlashInfer graph/eager partitioning can produce different BF16 greedy outputs.
The user accepts the quantified differences observed in this evaluation, with
quality regression checks retained; cross-plan text equality is not a release
requirement. Original strict-comparison results remain available. New unexplained
differences or clear degradation still require investigation. See the
[numerical investigation](/data2/servebig-envs/sd_graph_20260923/NUMERICS.md).
