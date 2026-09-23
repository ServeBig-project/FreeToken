# Independent public SD improvement checks

Only public contracts, CLI/HTTP outputs, and this author's existing blackbox
helpers are used. The coordinator starts all servers. No production module is
imported. The supported main configuration is Qwen3-30B-A3B BF16, offload,
legacy, context1024, prefill512, KV4096 and1706 expert slots, with old adaptive,
permanent residency and approximate verify disabled.

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

These three checks cover option discoverability, load-missing without router,
and new adaptive cost combined with the old adaptive profile. Their outcomes
change whether the advertised CLI is usable and invalid configurations reject
for the stated reason. CUDA devices are hidden from these subprocesses;
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
- `small-cache`: adaptive cost off and load-missing on; rebuild to256, two
  distinct17-token requests plus a repeated wave, then restore1706 and generate.
  Actual drafting, verification and demand loads must occur without strict
  residency stops. Rebuild is performed only at idle and uses the public API.

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
