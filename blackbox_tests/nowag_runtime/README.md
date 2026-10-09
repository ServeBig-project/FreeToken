# NoWAG runtime black-box acceptance

Written only from `docs/nowag-runtime-public-contract.md`, public CLI help, public model
definitions (HF configs/transformers, DeepSeek-V4 `inference/`) and the public NoWAG
artifacts. No production code, design notes, diffs or internal tests were read.

| File | Group (contract §7) | Needs |
| --- | --- | --- |
| `test_reference.py` | independent reference self-checks, frozen bounds | CPU |
| `test_bind_contract.py` | CPU format/math + `bind_expert_method` call contract; GPU op + graph replay | CPU; `cuda` rows need GPU |
| `test_public_errors.py` | §6 startup errors, rename | GPU (TP2 row: two GPUs) |
| `test_ftw.py` | FTW round trip, isolation, rename, missing data, bias, TP2 | GPU + `NOWAG_SCRATCH` |
| `test_cache_status.py` | `/v1/cache/status` `geometry.experts` | GPU (TP2 row: two GPUs) |
| `test_service.py` | modes, cache sizes, batching, concurrency/cancel, HTTP, SD, TP2, DSV4, non-NoWAG regression, paired perf | GPU; baseline rows need `NOWAG_BASELINE_SOURCE` |

`tolerances.py` holds the numeric bounds (frozen before any candidate result; basis in its
docstring, reproduce with `python calibrate.py`); `harness.py` holds the frozen output
comparison protocol for service runs.

## Environment

| Variable | Meaning |
| --- | --- |
| `PYTHONPATH` | candidate `python/` for in-process tests (`test_bind_contract.py`) |
| `NOWAG_SOURCE` | candidate `python/` for `ft serve` / `ft checkpoint` subprocesses |
| `NOWAG_BASELINE_SOURCE` | pre-feature baseline `python/` (regression and paired perf) |
| `NOWAG_PYTHON` | interpreter (default freetoken-dev) |
| `NOWAG_GPU_OK=1`, `NOWAG_GPU` | approve single-GPU use; GPU UUID or index for `--gpu` |
| `NOWAG_TP2_OK=1`, `NOWAG_TP2_GPUS=a,b` | approve two-GPU runs |
| `CUDA_VISIBLE_DEVICES` | the approved GPU for in-process CUDA tests (`cuda` rows use `cuda:0`) |
| `NOWAG_SCRATCH` | writable multi-GB dir: synthetic full-geometry sidecars, FTW outputs |
| `NOWAG_QWEN36_BASE`, `NOWAG_QWEN36_SIDE`, `NOWAG_QWEN36_BF16` | Qwen3.6 inputs (defaults: /data1 paths) |
| `NOWAG_DSV4_BASE`, `NOWAG_DSV4_SIDE` | DSV4 inputs (base has no default) |
| `NOWAG_GPTOSS_BASE` | GPT-OSS-20B snapshot dir (synthetic NoWAG weights are generated) |
| `NOWAG_DENSE_BASE` | a model without experts (default Qwen3.5-4B) |
| `NOWAG_DFLASH_DRAFT`, `NOWAG_DFLASH_ARGS` | DFlash draft for Qwen3.6 and its flags |
| `NOWAG_QWEN36_CACHE[_SMALL/_LARGE/_SIZES]`, `NOWAG_DSV4_CACHE`, `NOWAG_GPTOSS_CACHE` | `--moe-cache-size` values |
| `NOWAG_LOG_DIR`, `NOWAG_PORT` | service logs / perf JSON; port |

```sh
P=/home/nengneng/miniconda3/envs/freetoken-dev/bin/python
# CPU now (reference rows run, candidate rows skip with the reason)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=$CAND/python $P -m pytest -rs blackbox_tests/nowag_runtime
# GPU
NOWAG_GPU_OK=1 NOWAG_GPU=<uuid> CUDA_VISIBLE_DEVICES=<uuid> NOWAG_SOURCE=$CAND/python \
  PYTHONPATH=$CAND/python NOWAG_SCRATCH=/data1/... $P -m pytest -rs blackbox_tests/nowag_runtime
```
