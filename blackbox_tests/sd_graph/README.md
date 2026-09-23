# Independent CUDA Graph HTTP checks

These clients read only public HTTP. The coordinator starts every GPU service.
Use the real Qwen3-30B-A3B BF16 checkpoint, legacy scheduling, synchronous eager
or Graph execution, identical context/prefill limits, KV4096, maximum running requests4,
cache1706, and permanent residency/adaptive/verification reuse disabled.
SD uses maximum4 draft steps and3 draft experts, with residency off or router.
The Graph switch is `--cuda-graph-max-bs 0` or `4`. Record the full command.

## Frozen performance and quality inputs

Run from this worktree after `/health` reports `status=ok`:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/sd_graph/benchmark_http.py http://127.0.0.1:PORT \
  /data2/servebig-envs/sd_graph_20260923_gpu2/ar-graph --quality
```

Use fresh services for six configurations: AR, ordinary SD k3 and router SD k3,
each with eager and Graph execution. Use `--quality` only for the three Graph
configurations; the eager quality baseline already exists.

One mixed C4 warmup precedes three fixed repetitions. Each repetition has sky
C1, merge-sort C1, then mixed C4 `[sky, merge_sort, sky, merge_sort]`.
All requests are greedy, ignore EOS and request64 output tokens. Thus each mode
has9 measured batches,18 measured requests and1152 measured output tokens.
There is no selection of repetitions from their results.

`http.json` keeps every response, full usage, finish reason, first visible text
time, request/batch time, event timestamps, response-exit/idle active state and
public stats before/after every batch. Full `cuda_graph` snapshots preserve
actual replay shapes, capture time and extra reserved memory. `phase.txt` names
the current batch. Report timed capture growth if a later shape captures during
a measured request; one fixed warmup does not justify silently dropping it.

Quality follows performance and is saved separately in `quality.json` and
`quality/`. It reuses the original eight tasks, prompts, five cases per task,
256-token budget, truncation-as-failure rule and three-second Python checker.
No input or judgment rule is changed. These finite tasks do not prove general
quality preservation.

CPU preparation checks compilation and the client entry point only. It does
not establish GPU correctness, replay coverage or performance.

## Shape probes and strict paired comparison

On each coordinator-owned service, after its fixed benchmark history:

```bash
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/sd_graph/collect_http.py http://127.0.0.1:PORT OUTPUT \
  --mode off --execution graph
```

Use mode `ar`, `off` or `router`, and execution `eager` or `graph`. The13 frozen
stages exercise output limits1/2/3/4/5/7/17/64, concurrent2/3/4 distinct prompts,
and mixed short/long tails. They retain each input, text, usage, finish reason
and public stats. Actual replay differences must cover B1/2/3/4 for ordinary
target decode or SD draft/verify, and SD B1 verify query lengths2/3/4/5.
`query_tokens` counts total real query positions; padding is not counted.
Each phase returns to idle before the next. Limit1 is a valid prefill-only case.

After both matching official artifacts exist:

```bash
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/sd_graph/compare_evidence.py EAGER/acceptance.json \
  GRAPH/acceptance.json OUTPUT/comparison.json
```

This checks matching public configuration/inputs, disabled replay counters,
actual Graph coverage and exact text/usage/finish equality. Every difference is
saved with its public request and both responses; none receives a numerical or
text tolerance. Different contexts or an incomplete collection are not a valid
pair. The original ctx4096 debugging collection is separate from the official
matrix. Stop, cancellation and cache-rebuild lifecycle probes are a separate
supplement; the initial13 stages do not claim those paths were covered.

## Lifecycle and finite cache supplement

After performance finishes, on each off/router eager/Graph service:

```bash
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/sd_graph/lifecycle_http.py http://127.0.0.1:PORT OUTPUT \
  --mode off --execution graph
```

Enable the public cache-usage report. This checks streaming/nonstreaming stop,
actual prompt-prefix reuse, varied prompts, cancellation while another request
is active and verification has advanced, survivor completion and subsequent
admission. It rebuilds1706 slots to256, repeats a real prompt, then restores1706
and generates again. Each stage must return to idle with unchanged KV4096;
Graph stages must actually replay. A48-layer top8 target forward needs at least
384 distinct expert blocks, so256 slots exercises a finite replacement pool.
Router fallback at256 is valid; the separate1706 shape probes establish actual
draft/verify replay. No undocumented eviction order is assumed.

The same comparison entry accepts paired `lifecycle.json` files. Every complete
response remains subject to exact text/usage/finish comparison. Cancellation
must actually occur in both runs; its timing-dependent partial text is retained
but not compared for identical length. All survivor responses remain strict.
`lifecycle-*` phase labels separate these requests from performance accounting.
