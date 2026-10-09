# SD + batching black-box suite

Written from `docs/sd-batching-ship-public-contract.md` (design worktree), the `serve --help` text and
HTTP responses only. Each `test_<x>_*.py` module is one server launch (`SESSION`) serving several checks;
`test_m`, `test_n`, `test_k` launch per case. Only one server runs at a time; each launch first waits
until the test GPU has no compute process.

Run (one module or all):

    FT_RESULTS_DIR=/path/to/results \
    /home/nengneng/miniconda3/envs/freetoken-dev/bin/python -m pytest \
      -c blackbox_tests/sd_batching_ship/pytest.ini blackbox_tests/sd_batching_ship [-k test_a]

Environment: `FT_IMPL` (worktree under test), `FT_GPU` (GPU UUID), `FT_CPUS`, `FT_Q36_NVFP4`,
`FT_Q36_BF16`, `FT_Q3_BF16`, `FT_DFLASH`, `FREETOKEN_BENCHBW_PATH`, `FT_TIGHT_GDN_BYTES` (enables the
capacity pair). Server logs, stats snapshots and `observations.jsonl` (greedy drift, counters) go to
`FT_RESULTS_DIR`.

| module | configuration | contract checks |
|---|---|---|
| a | NVFP4, offload, defaults | effective defaults, real SD, outwave, shapes/tails, stop/EOS, SSE, Graph, hot prefix + cache_group, context edge, KV pressure, resource report, cancel, maintenance |
| b | NVFP4, layered + SD 0, ignored phase/draft path, Replay | SD off keeps layered, zero SD counters/allocation, greedy vs a, cancel, maintenance |
| c | NVFP4, hybrid, legacy + SD 4, Graph bs<=2 | legacy not switched, real SD, prefill overlap, out-of-coverage shape |
| d | NVFP4 + DFlash, layered, SD 8, all, Replay, host cache | drafter, phase all, N=8 tails, cold restore with SD, cancel |
| e | BF16 + DFlash, hybrid, layered, SD 4, inwave, Graph off | inwave phase, eager, maintenance then in-wave SD |
| f | BF16 defaults + host cache continuation | cold restore, waiters, multi-turn/fork, cancel during restore |
| g | Qwen3 MoE, legacy + 0 (+ ignored phase/path) | old default restored, AR only |
| h | Qwen3 MoE under a misleading dir name, defaults | name does not decide capability, maintenance |
| i | Qwen3 MoE, layered, SD 1, all, Graph off, multi-chunk | N=1, phase all, context edge |
| j | Qwen3 MoE, legacy, SD 2 | original 2-step path, Graph |
| k | Python `LLM` | omitted/0/positive semantics as CLI |
| l | Qwen3 MoE, CPU MoE, SD omitted | AR with reported reason |
| m | startup errors | explicit illegal/unsupported combos rejected before ready |
| n | NVFP4, tight GDN budget | auto -> AR + reason, explicit -> startup error |

`view.PATHS` maps contract information to the JSON paths seen in `/v1/stats`; a missing path fails
with near matches listed.
