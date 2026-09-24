# Frozen S2 inputs and HTTP runner

The runner uses only public HTTP and the independent serving helper. It never
starts a GPU server. `corpus.py` freezes the performance prompts, the three
calibration prompts reused by `sd_graph`, and the eight coding tasks.
`judge_code.py` and `evaluate_http.judge` are the coding-task checker that
`sd_graph`, `sd_concurrency` and `sd_improvements` load for `--quality`.

## Frozen evaluation

Use the same checkpoint, expert slots, context, KV, maximum running requests,
dtype and prefill limit for every compared server. Record the complete server
command; each configuration has its own ready URL and output directory.
Server flags are supplied by the coordinator, not inferred by this runner.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/s2_methods/evaluate_http.py http://127.0.0.1:PORT OUTPUT
```

Each evaluation has one fixed 4x64-token warmup, then two fixed rounds of the
two 256-token performance prompts at concurrency 1 and 4 (5,120 measured output
tokens if all limits are reached). No warmup/repetition is selected from results.
Then eight Python tasks each receive at most 256 output tokens, at concurrency
4, and execute against five independent cases each in a fresh Python process
with a three-second timeout.

All requests are greedy; this does not pretend that the API exposes independent
or controlled random seeds. Performance requests intentionally use
`ignore_eos=true`; their length-limited outputs are marked truncated and are not
task-quality successes. A quality response truncated at 256 tokens is a failure,
even if a code fragment could run. Do not change tasks or budgets after seeing
results. Report absolute passed-task counts and every per-task gain/loss versus
the ordinary baseline; only zero lost previously passed tasks supports "no
degradation observed in this finite set", not a general quality-preservation claim.

The JSON preserves text, full usage, finish reason, truncation, time to the first
nonempty streamed text chunk, total request/batch time, actual committed-token
throughput, and public stats before/after each batch with numeric differences.
Generated code and all checker outcomes remain in the output directory.

## Draft expert-count ablation

Use the same runner with `--part draft-ablation` on each ready ordinary or SD
server. Each run has one separate 4x64-token warmup, followed by both
performance prompts at concurrency 1 and 4, in two fixed rounds: 20 measured
requests and 1,280 output tokens. All requests use temperature 0 and
`ignore_eos=true`; incomplete 64-token responses fail after their public
evidence is saved. No coding-quality judging is run.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/s2_methods/evaluate_http.py http://127.0.0.1:PORT OUTPUT --part draft-ablation
```

The coordinator starts all services with the same model, dynamic expert slots,
context/KV/prefill budgets, cache policy and input history. SD separately sets
draft experts; ordinary generation uses SD disabled. Save each service command
and output directory.

Before sending each batch, its label is written to the output's `phase.txt` for
the coordinator's separate observations. `http.json` keeps every repetition and
marks warmup `scored=false`. Acceptance rate is the observed accepted/drafted
ratio, or null when no drafts occurred. Compare the two-round mean for each
prompt/concurrency and total tokens/total time, excluding warmup. No expert
count is assumed faster. Execution-route differences between configurations
must be reported separately; this workload cannot isolate their cost from the
effect of expert count.
