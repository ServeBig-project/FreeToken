# Independent serving acceptance

These tests were authored without reading FreeToken implementation, internal
tests, or diffs. Their only product inputs are the public self-speculative
decoding contract, public CLI/HTTP behavior, and real checkpoint metadata and
tokenizer data. They do not import FreeToken except to execute its public CLI.

## Run

Before GPU assignment, syntax and collection are safe:

```bash
PYTHONPYCACHEPREFIX=/tmp/sd-blackbox-pycache \
  /home/nengneng/miniconda3/envs/freetoken-dev/bin/python -m py_compile \
  blackbox_tests/self_speculative/test_serving.py
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/nengneng/miniconda3/envs/freetoken-dev/bin/python -m pytest \
  -c /dev/null -p no:cacheprovider --confcutdir=blackbox_tests/self_speculative \
  --collect-only -q blackbox_tests/self_speculative
```

After the coordinator assigns an RTX 4090 and the implementation is ready:

```bash
FT_SD_GPU='<assigned GPU UUID>' \
FT_SD_CANDIDATE=/home/nengneng/AIPrometheus/servebig/servebig-project/.sd-worktrees/freetoken-s2-sd/python \
FT_SD_BASELINE=/home/nengneng/AIPrometheus/servebig/servebig-project/FreeToken/python \
FT_SD_MODEL=/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B \
FT_SD_UNSUPPORTED_MODEL=/data1/lmcache_kv/models/Qwen3.6-35B-A3B-NVFP4 \
FT_SD_BASELINE_EVIDENCE=/data2/servebig-envs/self_speculative_20260921/run3/baseline.json \
FT_SD_CONCURRENT_REFERENCE_REDUCED=/data2/servebig-envs/self_speculative_20260921/eager_naive_sync_diagnostic/ordinary-eager-comparison.json \
FT_SD_CONCURRENT_REFERENCE_EQUAL=/data2/servebig-envs/self_speculative_20260921/eager_naive_sync_diagnostic/ordinary-eager-comparison.json \
FT_SD_CONCURRENT_REFERENCE_SINGLE=/data2/servebig-envs/self_speculative_20260921/eager_naive_sync_diagnostic/ordinary-eager-comparison.json \
FT_SD_ARTIFACTS=/tmp/self-speculative-acceptance \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/nengneng/miniconda3/envs/freetoken-dev/bin/python -m pytest \
  -c /dev/null -p no:cacheprovider --confcutdir=blackbox_tests/self_speculative \
  -q -s blackbox_tests/self_speculative
```

The suite starts servers sequentially; no server starts unless `FT_SD_GPU` is
set. Readiness requires `/health` JSON status `ok`; status `error` fails immediately.
It needs the full real checkpoints and their ordinary runtime dependencies.
Unsupported-model coverage is explicitly skipped if that checkpoint is absent.
Other GPU tests skip when no GPU has been assigned; skips are not acceptance.
Run without other traffic to these servers because stats are cumulative.

Set `FT_SD_BASELINE_EVIDENCE` to a previously completed short-context
`baseline.json` to reuse its recorded responses, including all 128 stochastic
samples, without starting the ordinary baseline server. Use this only with the
same frozen baseline, checkpoint, requests, and server parameters; the
coordinator is responsible for ensuring they are unchanged.

For a matched ordinary concurrent control, set
`FT_SD_CONCURRENT_REFERENCE_SINGLE` (or `_REDUCED`, `_EQUAL`, `_DISABLED`) to its
`ordinary-eager-comparison.json`. The coordinator selects the reference matching
the mode's execution settings and uses a new output directory. The test verifies
ordinary inference and the complete ordered request group, then compares the
five deterministic responses exactly, including text, usage and finish reason.
They must jointly equal either complete ordinary group frozen in
`ordinary_reference` or `ordinary_eager`, without mixing individual responses.
The reference path is recorded and failed evidence retained; no SD result or
text tolerance becomes an expectation. Freeze these groups before rerunning.

## Matrix and failure decisions

| Check | Failure detected; effect on acceptance |
|---|---|
| Public help and enabled startup errors | Missing flags, invalid values, unsupported architecture, policy, CPU/hybrid backend, CPU-layer selection or multi-GPU configuration accepted; block acceptance. |
| Baseline versus candidate disabled | Existing serving behavior changed with default speculation disabled; block acceptance. |
| Reduced experts, unchanged expert count, one-step drafting | Incorrect target greedy text, finish reason, or committed-token usage across real supported configurations; block acceptance. |
| Text prompts with calibrated token lengths, multilingual text, chat | Supported request formats produce different target behavior; block acceptance. |
| Output limits around four-token rounds, context clamping and errors | Drafts escape output/context limits or errors retain admission slots; block acceptance. |
| EOS, string/list stops, streaming usage and termination | Extra text/tokens, incorrect boundaries, or incomplete stream lifecycle; block acceptance. |
| Mixed concurrent requests and five requests for four admission slots | Requests interfere, queues stall, or target outputs change; block acceptance. |
| Concurrent 2K/3K public README excerpts, then reuse | Long document prefixes or cached history change greedy target output versus ordinary serving; block acceptance. |
| Top-k and top-p with single-token support | Requested filtering changes the target's deterministic result; block acceptance. |
| 128 eight-token A/B continuations per server, top-k=2 | Compare the count of A across each complete continuation, plus a separate non-symbol outcome, using a permutation test (p<0.001). This small outcome space avoids sparse full-string comparisons. The ordinary baseline must show variation, and enabled probes must increment draft, verification and retained-draft counters. Inspect recorded outputs before acceptance; no seed identity is assumed. |
| Isolated 64-token probes with EOS disabled | Accepted drafts or a meaningful rejection path were not exercised; mark coverage incomplete. Discards exceeding one complete draft round per request establish work discarded before final boundaries. |
| Repeated cache pressure and stream cancellation | Leaked active requests, exhausted fixed capacity, lost cancellation counters, or stale reusable history; block acceptance. |

The common resource limits are 256 context tokens, 1,024 KV tokens, four running
requests, and 512 offloaded expert slots. The ordinary baseline and the disabled,
reduced-expert, and unchanged-expert modes use identical settings. The one-step
mode also exercises the naive cache and automatic backend selection; its timing
is not a matched performance comparison. The real 30B checkpoint cannot fit as
fully resident BF16 experts on one 4090, so this matrix uses expert offload.

The single long-context check runs a separate matched baseline/enabled pair with
4,096 context tokens, 8,192 KV tokens, and a 2,048-token prefill limit. It uses
contiguous real README excerpts with explicit summarization tasks, at most 16
output tokens, and two concurrent requests repeated once. It runs before the
short-context server matrix and requires observed speculative verification.

Each server writes public process output plus JSON containing exact requests,
normalized responses, request latency, batch committed-token throughput, public
memory/cache snapshots, and speculative counters. EOS, rejection, or sampling
variation that is not actually observed is reported as missing coverage. The
finite sampling check can detect bias; it cannot prove distributional equality.
The unchanged-expert control does not demand identical stochastic text or zero
floating-point differences. Prompt-prefix retention and retained allocator
memory are allowed; recovery is judged by activity and subsequent real requests.

This suite does not assert a speedup or reproduction of full S2-MoE results.

## Fixed sampling confirmation

`sampling_confirmation.py` is a separate explicit entry; normal directory
collection still contains the original 56 cases. It starts ordinary serving and
N=4/K=3 sequentially and takes exactly 512 new samples from each, using the
original request and concurrency. It ignores `FT_SD_BASELINE_EVIDENCE`.
Before scoring, each arm makes exactly eight unscored requests with the same
prompt and sampling policy, `max_tokens=224`, and `ignore_eos=true`, at concurrency
four. All 1,792 output tokens must complete before the scored segment begins.
This fixed request history avoids restarting directly at the old scored prefix;
the public API does not expose RNG offsets or provide an independent new seed.

```bash
FT_SD_GPU='<assigned GPU UUID>' \
FT_SD_ARTIFACTS=/tmp/self-speculative-confirmation \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/nengneng/miniconda3/envs/freetoken-dev/bin/python -m pytest \
  -c /dev/null -p no:cacheprovider --confcutdir=blackbox_tests/self_speculative \
  -q -s blackbox_tests/self_speculative/sampling_confirmation.py
```

Use a new evidence directory and record the candidate version before running.
This is one fixed round: no interim stopping, old-sample pooling, automatic
retries, or resampling after failure. Its complete histogram distance is the
same as the main matrix, with exact conditional permutation probability and
alpha 0.001. Draft, verification and retained-draft counters must all increase.
Per-server JSON preserves requests, normalized responses and public statistics;
the separate `*-presampling-raw.json` and `*-scored-raw.json` retain every complete
POST response body and status for their respective segments. Only the scored
segment enters the histograms and speculative counter deltas.
`sampling-confirmation.json` records both histograms and the decision
inputs. Preserve these files together with the original evidence.

## Ordinary eager diagnostic

`diagnose_greedy_eager.py` runs only the current candidate with speculation off
and `--cuda-graph-max-bs 0`. It replays the recorded 64-token story, followed by
the original six mixed requests in their original submission order. It records
output comparisons with the run4 ordinary and single-step observations; these
are diagnostic observations, not a replacement for the unchanged assertions.
`FT_SD_DIAGNOSTIC_CACHE` selects the cache (default `radix`); set it to `naive`
to match the single-step mode's cache while keeping all other settings unchanged.

```bash
FT_SD_GPU='<assigned GPU UUID>' FT_SD_ARTIFACTS=/tmp/ordinary-eager-diagnostic \
  /home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/self_speculative/diagnose_greedy_eager.py
```
