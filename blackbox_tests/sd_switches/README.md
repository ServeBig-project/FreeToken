# SD switch cleanup: independent public checks

Written from `docs/self-speculative-decoding.md`, `docs/adaptive-loading.md`,
`docs/sd-cuda-graphs.md` and public CLI/HTTP output only; no implementation
source, diff or internal test was read.

## CPU: removed CLI surface

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/nengneng/miniconda3/envs/freetoken-dev/bin/python -m pytest -c /dev/null \
  -p no:cacheprovider --rootdir=. -q blackbox_tests/sd_switches/test_switch_cli.py
```

`ft serve` must reject `--speculative-adaptive-profile`,
`--speculative-reuse-expert-cap`, `--moe-resident-experts` and
`--moe-expert-profile` as unrecognized arguments, reject residency `affinity`,
and list only `{off,router}`. `ft bench experts` must not run on a valid old
profile; `ft bench bw` remains. The unknown-subcommand exit code is not asserted
because no public document states it.

## GPU: startup rules, Graph coverage and removed stats

Run on one coordinator-assigned idle GPU. The script starts and stops every
server itself (legacy scheduling, a free local port) and needs roughly 20–40
minutes for all cases, mostly model loading and Graph capture:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/nengneng/miniconda3/envs/freetoken-dev/bin/python \
  blackbox_tests/sd_switches/gpu_contract.py --gpu GPU_UUID_OR_INDEX --output OUTPUT_DIR
```

`--cases NAME...` reruns a subset. Each case writes `NAME.log`; `results.json`
keeps commands, stats snapshots, responses, checks and log tails. The exit code
is nonzero when any case fails.

| Case | Server options beyond the shared budget | Expected |
| --- | --- | --- |
| `fused-graph` | fused, N4, Graph default | startup fails naming `--cuda-graph-max-bs 0` |
| `fused-eager` | fused, N4, `--cuda-graph-max-bs 0` | starts and serves eagerly |
| `steps9-graph` | offload, N9, Graph 32 | startup fails naming `--cuda-graph-max-bs 0` |
| `steps9-eager` | offload, N9, `--cuda-graph-max-bs 0` | starts and serves eagerly |
| `k1`, `k5`, `k8` | N8, K, router, load-missing, adaptive cost, Graph 32 | Graph SD |
| `all-controls` | N8, default K, all three boolean controls, Graph 32 | Graph SD |
| `plain-sd` | N8, residency off, no boolean controls, Graph 32 | Graph SD |

The shared budget is offload cache 1706 (except fused), 32 running requests,
KV4096, context1024, prefill512 and radix cache. Served cases send one greedy
request, then six concurrent distinct greedy requests, all ignoring EOS.

Every served case requires complete outputs, drafting and verification, a
draft-length histogram of N+1 entries, and no removed stats fields
(`speculative.adaptive_enabled`, `adaptive_stops`, `reuse_enabled`,
`reuse_changed_routes`, `draft_expert_replacements`, top-level `moe_residency`).
Eager cases require disabled graphs with zero replay counters. Graph cases
require `cuda_graph.enabled`, draft and verify replays at batch 1 and above 4,
and `verify_steps == cuda_graph.verify` at every idle snapshot, reading one
verification round as one batched verification forward. With adaptive cost,
`cost_samples.ar/draft/verify` must equal `cuda_graph.target_decode/draft/verify`
at every idle snapshot, and target decode must replay.

All BF16 experts do not fit on one 24 GB RTX 4090. A fused case that fails to
start without naming `--cuda-graph-max-bs 0` is reported `INCONCLUSIVE`, not a
pass; inspect its log tail for the actual reason.
