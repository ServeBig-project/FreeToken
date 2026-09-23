# Independent high-concurrency HTTP experiment

Only the coordinator starts services. Use Qwen3-30B-A3B BF16, context1024,
prefill512, KV4096,1706 dynamic expert slots, maximum running requests32,
legacy scheduling and the same synchronous execution setting for every mode.
Compare AR, ordinary SD and router SD, each with eager/Graph32. SD uses k3 and
at most4 draft steps; permanent residency, adaptive draft and verify reuse stay
off. Record the complete public command for every fresh service.

## Frozen inputs and commands

`inputs.py` freezes32 distinct coding/research prompts. Each C uses its ordered
prefix. The real checkpoint tokenizer measured17–23 tokens per prompt;
C32 prompt plus64-token output totals2641 tokens, within KV4096. This C4 starts
with binary search, causal inference, graph shortest path and query planning;
it is a fresh workload anchor, distinct from the old sky/merge experiment.
Exact prompts and CPU token counts are also in the experiment's
`frozen-prompts.json`. No prompts are chosen from their runtime results.

Run from this worktree, with the service already ready:

```bash
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/sd_concurrency/evaluate_http.py http://127.0.0.1:PORT \
  /data2/servebig-envs/sd_concurrency_20260923_gpu2/ar-graph \
  --mode ar --execution graph --part performance --quality
```

Use mode `ar|off|router` and execution `eager|graph`. The parts run independently:

- `performance`: C4→8→16→32. At each C, one16-token warmup followed by three
  64-token rounds, greedy and ignoring EOS. Count only the11520 measured tokens;
  warmups add960. Labels are `warmup-cC` and `performance-R-cC`.
- `profile`: identical ordered inputs and warmups, followed by one64-token
  round per C;4800 tokens including warmups. It keeps the same phase labels,
  with repetition0 only. The coordinator handles external profiling.
- `acceptance`: C8/C16/C32, each with one short wave cycling output
  limits1/2/3/4/5/7/17. Labels are `acceptance-tails-cC`. Uniform64-token
  coverage comes from performance's three rounds and is not repeated here.

`--quality` is permitted only after `performance` on Graph modes. All eight
original Python tasks run together at C8 after every performance request has
finished. Prompts,256-token ceiling, five cases, three-second checker and
truncation rule are unchanged. `quality.json` and generated code are separate
from performance. Compare each task with this run's C8 AR Graph baseline;
the previous serial5/8 result is not a substitute for this new baseline.

## Evidence and checks

`http.json` or `acceptance.json` retains all requests, text, usage, request
start/first-text/end timestamps, TTFT and completion times, batch time,
tokens/second, and stats before/response-exit/idle. Percentiles use the nearest
rank of request latencies; point throughput is total tokens divided by summed
batch time. All three measured repetitions remain visible. `phase.txt` changes
before each wave, allowing the coordinator to align external measurements.

Hard checks cover output count/finish/usage, unchanged resources, feature
settings, eager zero replays, actual Graph replay and return to active0.
For every tested C above4, actual Graph B above4 must occur. Each point also
reports target-decode/draft/verify B values and whether each phase reached the
full submitted concurrency. Passing the B>4 check does not establish B32 in
every phase. Router may legitimately fall back when residency is insufficient;
report unexercised draft/verify coverage explicitly instead of calling it tested.
Points retain residency stops and explicitly label router's zero-draft fallback.
Off SD performance/profile additionally requires actual draft and verify B>4
at each higher C. Tail-only acceptance uses performance for this full-wave evidence.

Full actual `(phase,B,Q,replays)` differences are retained. For verify, Q/B−1
is the batch's mean candidate count; it is exact per-request N only at B1.
Mixed-batch shapes do not reveal individual N values, and eager provides no
replay-derived N distribution. Draft/accepted/verify counters are retained;
accepted/verify is never labeled a per-request acceptance length.

The entry's `passed` field covers the listed interface/resource/replay checks,
not text equivalence or task quality. Full responses support strict offline
comparison of matching new workloads. The prior experiment's accepted,
quantified BF16 differences do not automatically excuse new changes, repeated
words or quality regressions. Quality outcomes are always retained separately.

CPU preparation calibrates inputs and checks Python compilation/CLI only;
it cannot establish runtime correctness or speed. Budget roughly10–25 minutes
per full performance group, under2 minutes for short-tail acceptance and1–3 minutes for C8
quality until actual high-concurrency throughput is measured. Six modes use
identical input histories; no timing repetitions are removed after inspection.
