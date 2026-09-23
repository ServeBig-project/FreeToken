# Independent CUDA Graph HTTP checks

These clients read only public HTTP. The coordinator starts every GPU service.
Use the real Qwen3-30B-A3B BF16 checkpoint, legacy scheduling, synchronous eager
or Graph execution, context1024, KV4096, prefill512, maximum running requests4,
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
