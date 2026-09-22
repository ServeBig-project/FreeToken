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

Apply the same command and input history to eight configurations: AR graph,
AR graph with the same 1,024 resident experts, AR eager/synchronous,
basic SD N=4, basic SD N=16, residency with fixed N=16,
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

## Public contracts and lifecycle

CPU-only help/offline selection checks (the profile is the published 2x4 example):

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/nengneng/miniconda3/envs/freetoken-dev/bin/python -m pytest -c /dev/null \
  -p no:cacheprovider --confcutdir=blackbox_tests/s2_methods \
  --basetemp=/data2/servebig-envs/s2_methods_20260921/cpu-contract \
  -q blackbox_tests/s2_methods/test_cli_contract.py -k 'not startup'
```

The coordinator runs startup rejection cases separately with `FT_SD_GPU` set,
when the GPU is not doing performance work. These cover malformed resident lists,
the reachable cache minimum, incompatible controls, finite-positive costs, and
reuse-cap bounds; no internal cost formula is used as an oracle.

On a dedicated idle candidate server with the frozen geometry, run:

```bash
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/s2_methods/lifecycle_http.py http://127.0.0.1:60387 \
  /data2/servebig-envs/s2_methods_20260921/lifecycle-full \
  --resident-count 1024 --adaptive --reuse
```

Omit flags for disabled mechanisms and use resident count zero without a list.
This check assumes ordinary two-layer prefill overlap remains enabled. It checks
configured counters, same-mode request isolation, streaming stops/output/context
limits, cancellation, repeated capacity admission, and real rebuilds 1536->1408
->rejected below-minimum->1536. It must not run alongside performance traffic.
Routing reuse must produce changed routes. Greedy lifecycle requests need not
trigger an adaptive stop; the separate stochastic check requires that coverage.

## Fixed adaptive sampling confirmation

Use two freshly started resident servers: fixed N=16 and adaptive maximum N=16,
both with the same hot list, measured cost profile where applicable, and reuse
off. Each arm performs eight fixed 224-token presampling requests, then exactly
256 original eight-token A/B requests, all at concurrency four. Only the latter
enter the histogram and counter deltas. There is no public seed/reseed API;
this is a fixed request-history check, not a claim of independent new seeds.

```bash
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python blackbox_tests/s2_methods/adaptive_sampling.py \
  collect fixed http://127.0.0.1:60387 /data2/servebig-envs/s2_methods_20260921/sampling-fixed.json
# After the coordinator starts the adaptive server:
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python blackbox_tests/s2_methods/adaptive_sampling.py \
  collect adaptive http://127.0.0.1:60387 /data2/servebig-envs/s2_methods_20260921/sampling-adaptive.json
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python blackbox_tests/s2_methods/adaptive_sampling.py \
  compare /data2/servebig-envs/s2_methods_20260921/sampling-fixed.json \
  /data2/servebig-envs/s2_methods_20260921/sampling-adaptive.json \
  /data2/servebig-envs/s2_methods_20260921/sampling-comparison.json
```

The predeclared rule is the full A-count/other histogram distance, exact
conditional permutation probability, alpha 0.001. Both arms must actually draft,
verify and retain drafts; scored adaptive requests must increment adaptive_stops.
Preserve all data, including failed/incomplete coverage. Do not change the fixed
counts, substitute old samples, or repeat until passing. This finite projection
cannot prove exact equality for every input or freedom from all sampling bias.

## Generated-KV cache isolation with verification reuse

Run `cache_isolation_http.py URL OUTPUT_DIRECTORY` on an idle reuse-enabled
server with radix prefix caching and `--enable-cache-report`. It sends one new
natural-language prompt ending in a colon, generates 64 tokens, then sends
prompt+generated-text+continuation and repeats that extended prompt. It verifies
the tokenizer prefix before interpreting cache counts. The first continuation
may cache at most the original prompt length; the repeat must cache beyond that
length, proving normal prompt prefill remains reusable. Responses are recorded
without textual-equality comparisons. Omitted zero-hit cache details mean zero;
the repeat still requires a reported positive hit beyond the original prompt.
Both boundary outcomes are retained even if
the first one fails, in `cache-isolation.json`.

The fixed prompt is outside the evaluation/calibration/lifecycle inputs. Use a
fresh service/cache history; the default does not rebuild an old live version
being diagnosed. For a same-instance repeat, `--rebuild-first` invokes the public
cache rebuild before these requests. Keep old and repaired evidence in separate
output directories. No existing scoring input or assertion is changed.

## Stop before an unaffordable draft

`test_early_draft_stop.py` starts one N16/K3 adaptive server, with reuse disabled,
naive cache, and `--moe-collect-stats`. Its legal cost profile has target cost
1 ms and draft cost 2 ms: every request must retain its first candidate, but no
prefix can justify executing the next draft. One 64-token request and one batch
with 8/17/33/64-token limits check complete outputs, sampling, verification,
retained drafts, adaptive stops, and return to idle.

For each idle-to-idle phase, the public service-log `decode_layer_calls` delta
must be at most `48 * (2 * verify_steps + request_count)`: one draft and one
verification forward per round, plus at most one ordinary tail decode per
request. Prefill is excluded by the public counter contract. This detects wasted
model execution without using wall-clock thresholds or exact text comparisons.
Both phases and their raw public counters are saved even if that bound fails.

Run only after the coordinator assigns an empty GPU:

```bash
FT_SD_GPU="$GPU_UUID" \
FT_SD_CANDIDATE="$PWD/python" \
FT_SD_ARTIFACTS=/data2/servebig-envs/sd_early_stop_20260922/after \
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/nengneng/miniconda3/envs/freetoken-dev/bin/python -m pytest -c /dev/null \
  -p no:cacheprovider --confcutdir=blackbox_tests/s2_methods -q -s \
  blackbox_tests/s2_methods/test_early_draft_stop.py
```

For the unfixed control, set `FT_SD_CANDIDATE` to
`/data2/servebig-envs/s2_methods_20260921/candidate/python` and use the separate
`/data2/servebig-envs/sd_early_stop_20260922/before` evidence directory. A useful
regression reproduction fails `necessary_execution_only` with otherwise valid
requests and counters; startup/input/snapshot failures do not establish it.
CPU compilation and collection do not establish GPU acceptance.

## Draft expert-count ablation

Use the same HTTP runner with `--part draft-ablation` on each ready ordinary or
SD server. The frozen prompts are `PERFORMANCE` in `corpus.py`: an explanation
of blue-sky scattering and a stable Python merge-sort implementation. Each run
has one separate 4x64-token warmup, followed by both prompts at concurrency 1
and 4, in two fixed rounds: 20 measured requests and 1,280 output tokens. All
requests use temperature 0 and `ignore_eos=true`; incomplete 64-token responses
fail after their public evidence is saved. No coding-quality judging is run.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/s2_methods/evaluate_http.py http://127.0.0.1:60387 \
  /data2/servebig-envs/draft_k_ablation_20260922/k1 --part draft-ablation
```

The coordinator starts all services with the same model, 1,536 dynamic expert
slots, context/KV/prefill budgets, cache policy and input history. Disable
adaptive drafting, verification-route reuse and permanent residency. SD uses
maximum 4 draft steps and separately sets draft experts to 1, 2 or 3; ordinary
generation uses SD disabled. Save each service command and output directory.

Before sending each batch, its label is written to the output's `phase.txt` for
the coordinator's separate observations. `http.json` keeps every repetition and
marks warmup `scored=false`. It records
text, usage, TTFT, request/batch completion time and full public stats/deltas,
including draft, accepted-draft and verification counts. Acceptance rate is the
observed accepted/drafted ratio, or null when no drafts occurred. Compare the
two-round mean for each prompt/concurrency and total tokens/total time, excluding
warmup. No expert count is assumed faster. Execution-route differences between
configurations must be reported separately; this workload cannot isolate their
cost from the effect of expert count. Fusion and new quality tasks are excluded.
