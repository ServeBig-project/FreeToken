# DFlash mainline black-box acceptance

Independent tests for `docs/dflash-public-contract.md`. They use only the public
contract, the CLI, the HTTP API and the documented `freetoken.speculative.dflash_model`
entry; the author did not read production code, diffs or internal tests.

## Run

Environment (defaults in `harness.py`):

| Variable | Meaning |
| --- | --- |
| `FT_SOURCE` | `python/` directory of the implementation under test (put on `PYTHONPATH` of the server) |
| `FT_PYTHON` | interpreter with torch/flashinfer (`freetoken-dev`) |
| `FT_GPU` | GPU UUID for the service |
| `FT_LOG_DIR` | where service logs and result JSON go |
| `FT_PORT` | service port |

```sh
# CPU only: config rejection, public model entry, independent numeric reference
CUDA_VISIBLE_DEVICES='' PYTHONPATH=$FT_SOURCE pytest blackbox_tests/dflash_mainline/test_p1_cpu.py

python run.py --list                         # sessions (one service launch each) and their scenarios
python run.py --session nvfp4_lp_n8          # launch, run every planned scenario, write <session>.json
python run.py --session nvfp4_n8 --only pressure_admission,pressure_mixed   # subset, merged into the JSON
python run.py --compare                      # cross-session checks over $FT_LOG_DIR/*.json
./batch.sh <session>...                      # several sessions in sequence (joins the `perf` cpuset)
```

Scenarios live in `p1.py` (configuration, budget, math A/B, rejections), `p2.py` (window
reuse, sharing, rebuild), `p3.py` (long generation, pressure, cold cache, cancellation),
`p4.py` (control modes, termination, stats, layered phases, C16) and `p5.py` (compact vs
full performance pair). Thresholds are constants at the top of each module and were fixed
before the first run. Each result records the implementation commit it ran against.

## Archived results

Raw results are not kept in the branch. They are archived at
`/data2/servebig-envs/dflash_mainline_blackbox_20261009/results/`, one folder per round,
named by the implementation commit(s) tested (each JSON's `impl_head` is authoritative):

| Folder | Content |
| --- | --- |
| `batch1-2` | first legacy rounds (fcf2c67, a7e4ba1) |
| `r2-32646f6-94b8198` | rejections, AR, legacy/layered main matrix, smokes |
| `r2-2d8cf2e` | layered matrix up to the window-pool crash, legacy subset, cold cache, C16 |
| `r3-2295b39`, `r3-2295b39-perf-repeat` | layered remainder and stress, tool checkpoints, perf A-B-A |
| `r3-4e761f1-stats` | fixed-mode section 4 |
| `r4-e07fe48` | perf A-B-A-B with prefix-cache counters, adaptive section 4, stress and cold-cancel smokes |
