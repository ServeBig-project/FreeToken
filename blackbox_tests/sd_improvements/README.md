# Independent public SD improvement checks

Only public contracts, CLI/HTTP outputs, and this author's existing blackbox
helpers are used. The coordinator starts all servers. No production module is
imported. The supported main configuration is Qwen3-30B-A3B BF16, offload,
legacy, context1024, prefill512, KV4096 and1706 expert slots (read from
`/v1/cache/status`).

## Fixed performance and quality entry

```bash
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/sd_improvements/evaluate_http.py URL OUTPUT \
  --execution graph --concurrencies 1 4 16 \
  --warmup-tokens 16 --max-tokens 64 --repetitions 2 --quality
```

The displayed concurrency/token/repetition values are defaults. Each C uses
the first C entries of the existing32-prompt frozen list. At each C, one
independent warmup precedes the scored repetitions. Default scored output is
2688 tokens, plus336 warmup tokens. The input sequence is the same for every
server configuration; flags and maximum draft length belong to the public
server command, not the workload. Omit `--quality` when it is not scheduled.

`http.json` retains every request, text, usage, request start/first-text/end
timestamps, TTFT, completion latency, batch throughput and p50/p95. Percentiles
use nearest rank. Full public stats are saved before, at response exit and at
idle; replay differences preserve phase/realB/logicalQ/physicalQ separately.
`phase.txt` labels warmup-cC, performance-R-cC, and quality-c8. No measured
repetition is discarded after viewing its result.

The current entry checks HTTP completion, output/usage limits, main budgets,
idle recovery and declared eager/Graph execution. Its `passed` field does not
establish the new feature-specific paths or quality preservation. New public
flags, request-round draft histograms and load/prefetch counters remain in the
unmodified stats snapshots for later contract-specific checks and analysis.
Actual B/N and path coverage must be reported from those observations; queued
concurrency and physical padding are not substitutes for logical coverage.

`--quality` runs the original8 coding tasks together at C8 only after performance.
The256-token ceiling, five cases per task, three-second checker and truncation
rule are unchanged. Results and generated code are separate in quality.json
and quality/. Compare with the same round's ordinary target baseline; preserve
new failures and visible degeneration. Previously accepted numerical differences
are not blanket permission for new quality regressions.

CPU preparation compiles this client and checks its entry point only. GPU
correctness and performance remain pending coordinator execution. The bounded
default workload has3024 total output tokens per server configuration; elapsed
time depends on actual execution and cache behavior.

## Public CLI contracts

```bash
FT_SD_PACKAGE=/ABSOLUTE/CANDIDATE/python \
  /home/nengneng/miniconda3/envs/freetoken-dev/bin/python -m pytest -q \
  blackbox_tests/sd_improvements/test_cli_contract.py
```

These two checks cover option discoverability and load-missing without router.
Their outcomes change whether the advertised CLI is usable and the invalid
configuration rejects for the stated reason. CUDA devices are hidden from these subprocesses;
timeouts or unrelated CUDA errors do not count as correct rejection.

## New-feature HTTP contracts

On a coordinator-owned candidate service with `--moe-collect-stats`:

```bash
/home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/sd_improvements/check_http.py URL OUTPUT \
  --execution graph --part smoke --speculative-num-steps 8 \
  --speculative-draft-residency router --speculative-adaptive-cost \
  --speculative-draft-load-missing --speculative-verify-prefetch
```

The flag arguments describe expected server settings; this client does not
change them. Use `smoke` for each of the eight flag combinations: two distinct
17-token requests, one streamed and one plain. Also use prefetch alone with
residency off. Strict router plus prefetch may legitimately load nothing when
the pool is full; its coverage is then absent, not a claimed prefetch success.

The remaining parts provide focused additional coverage:

- `boundaries`: single-request limits1/2/3/7/8/9/17, then mixed tails at C8/C32,
  followed by four requests using the frozen stochastic A/B input. This checks
  sampling API behavior; it is not a statistical distribution proof.
- `lifecycle`: stop in streamed/plain requests, cancellation while other
  requests remain active, survivor completion and subsequent admission.
- `small-cache`: load-only or all three controls on; rebuild to256, two
  distinct17-token requests plus a repeated wave, then restore1706 and generate.
  With adaptive cost off, actual drafting, verification and demand loads must
  occur without strict residency stops. All-on may legally select ordinary
  decoding at256; it still checks outputs, resources and actual Graph replay
  after restoration. Missing prefetch activity remains uncovered. Rebuild is
  performed only at idle and uses the public API.

Both rebuilds preserve full before/after stats and require published process-
lifetime counters not to decrease: draft/accepted/verify, every N histogram bin,
cost AR/stopped/probe, control time, samples, GPU times, transfer predictions and
errors, and prefetch events/bytes. Cache occupancy and old epoch counters may
change normally. Cost estimates and strategy choices need not stay fixed;
overlapping parent/child time fields are not summed into a fabricated total.

Run detailed boundaries/lifecycle on the selected main eager/Graph controls,
not on every flag combination. The old maximum4/default-off comparison retains
the original baseline's missing new fields as version differences, not failures;
this new-field checker is for the candidate. Performance/raw-response evidence
can be collected with the same workload on both versions.

Each stage records full requests/responses, stats, logical/physical replay
shapes, request-round N histogram differences and per-feature path coverage.
Without cancellation or stop, actual N-weighted round counts must equal
completed draft tokens; Graph logical verify positions additionally confirm
that all candidates entered verification. Physical dummy positions are excluded.
Effective flags, maximum steps, budgets, prefetch first-use/unused-eviction
accounting, graph verification and idle recovery are checked separately.
Coverage booleans state which paths actually ran. No prefetch load/use or no
post-candidate cost stop means that path remains unobserved, even if the local
API checks pass. No wall-clock threshold or private cost formula is used.

## Frozen sampling confirmation and old-four comparison

After the main functional/performance work, run one AR Graph versus all-on
Graph pair on coordinator-owned services. Collection is separate from analysis:

```bash
python blackbox_tests/sd_improvements/sampling_http.py collect ar URL AR.json
python blackbox_tests/sd_improvements/sampling_http.py collect all-on URL ALL.json
python blackbox_tests/sd_improvements/sampling_http.py compare AR.json ALL.json RESULT.json
```

The original A/B prompt and temperature1.5/top_k2/top_p1/ignore-EOS parameters
are fixed. Each arm first produces8 unscored224-token requests at C4, then256
scored8-token requests at C4:3840 output tokens per arm,7680 total. For an output
of exactly8 spaced A/B choices, score the A count in choices2–8; everything
else is retained as `other`. Thus the A/B statistic excludes the first prefill
choice. Real-tokenizer CPU calibration found each spaced A/B is one token and
no vocabulary token can contain multiple complete spaced A/B choices.

The full categorical L1 distance uses the existing exact permutation function:
dynamic programming sums all category allocations with combination weights,
with total mass checked against comb(512,256). This is not finite random
permutation sampling. Alpha stays0.001. Keep this single round, raw responses,
all categories and results; do not alter sample counts or repeat until passing.
Fixed startup RNG and this finite workload limit what a passed check proves.

Each scored C4 batch retains full stats, including logical/physical graph
shapes. The all-on arm must actually draft, accept and verify after prefill;
N0 rounds and batches with no SD are reported. Without actual SD participation,
the distribution check cannot be labeled covered even if the histograms agree.
Old AR versions do not need the new per-SD-round histogram or feature fields.

For a fixed4-step baseline/candidate pair with new controls disabled, use:

```bash
python blackbox_tests/sd_improvements/compare_legacy.py BASELINE/http.json CANDIDATE/http.json comparison.json
```

Both inputs come from this same frozen performance client and execution mode.
The comparator preserves exact request/text/usage/finish differences and returns
failure for diagnostics; it does not silently grant new numerical exemptions.
Only the candidate is required to expose new fields and show the three controls
off. Absence of these new fields in baseline5cef96d is deliberately not an error.
