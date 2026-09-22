# Frozen S2 public evaluation

The runner uses only public HTTP and the independent serving helper. It never
starts a GPU server. All outputs belong under `/data2/servebig-envs/s2_methods_20260921`.
Use the same checkpoint, 1,536 total expert slots, context 1,024, KV 4,096,
maximum running requests 4, dtype and prefill limit for every mode. Resident
experts count toward that same slot budget. Record the complete server command.

Run the ready original AR graph baseline:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/s2_methods/evaluate_http.py http://127.0.0.1:60387 \
  /data2/servebig-envs/s2_methods_20260921/ar-graph
```

Apply the same command and input history to seven configurations: AR graph,
AR eager/synchronous, basic SD N=4, basic SD N=16, residency with fixed N=16,
residency with adaptive N<=16, and the latter with verification reuse cap 14.
Each has its own ready URL and output directory. Fixed N=16 is the direct
adaptive-drafting control; N=4 retains the existing practical baseline.
New feature CLI settings and measured cost
values are supplied by the coordinator, not inferred by this runner.

Each evaluation has one fixed 4x64-token warmup, then two fixed rounds of the
two 256-token performance prompts at concurrency 1 and 4 (5,120 measured output
tokens if all limits are reached). No warmup/repetition is selected from results.
Then eight Python tasks each receive at most 256 output tokens, at concurrency
4, and execute against five independent cases each in a fresh Python process
with a three-second timeout. The inputs are frozen in `corpus.py`.

All requests are greedy; this does not pretend that the API exposes independent
or controlled random seeds. Performance requests intentionally use
`ignore_eos=true`; their length-limited outputs are marked truncated and are not
task-quality successes. A quality response truncated at 256 tokens is a failure,
even if a code fragment could run. Do not change tasks or budgets after seeing
results. Report absolute passed-task counts and every per-task gain/loss versus
AR and basic SD; only zero lost previously passed tasks supports "no degradation
observed in this finite set", not a general quality-preservation claim.

The JSON preserves text, full usage, finish reason, truncation, time to the first
nonempty streamed text chunk, total request/batch time, actual committed-token
throughput, and public stats before/after each batch with numeric differences.
Generated code and all checker outcomes remain in the output directory. Compare
both ordinary AR modes so improvements over a slower execution mode are clear.

Calibration uses three different prompts and must run on a separate ordinary
server/profile, before creating the fixed hot-expert list. It is not scored:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/s2_methods/evaluate_http.py http://127.0.0.1:60387 \
  /data2/servebig-envs/s2_methods_20260921/calibration --part calibrate
```

Keep calibration and evaluated requests separate when writing the public expert
count profile. Lifecycle and new-feature contract checks are separate from these
performance and coding-quality measurements.
