# NoWAG runtime black-box acceptance

Written only from `docs/nowag-runtime-public-contract.md`, public CLI help, public model
definitions (HF configs/transformers, DeepSeek-V4 `inference/`) and the public NoWAG
artifacts. No production code, design notes, diffs or internal tests were read.

| File | Group (contract §7) | Needs |
| --- | --- | --- |
| `test_reference.py` | independent reference self-checks, frozen bounds | CPU |
| `test_bind_contract.py` | `bind_expert_method` call contract, format/math, GPU op + graph replay | reference rows CPU; candidate (`cuda`) rows need GPU (§9: no CPU `run`; CPU expert compute is covered by `test_service.py`) |
| `test_public_errors.py` | §6 startup errors, rename | GPU (TP2 row: two GPUs) |
| `test_ftw.py` | FTW round trip, isolation, rename, missing data, bias, TP2 | GPU + `NOWAG_SCRATCH` |
| `test_cache_status.py` | `/v1/cache/status` `geometry.experts` | GPU (TP2 row: two GPUs) |
| `test_service.py` | modes, cache sizes, batching, concurrency/cancel, HTTP, SD, TP2, DSV4, non-NoWAG regression, paired perf | GPU; baseline rows need `NOWAG_BASELINE_SOURCE` |

`tolerances.py` holds the numeric bounds (frozen before any candidate result; basis in its
docstring, reproduce with `python calibrate.py`); `harness.py` holds the frozen output
comparison protocol for service runs.

## Prepared coverage and evidence (2026-10-09)

No candidate GPU matrix has run. The coordinator confirmed that no earlier candidate GPU
pass log exists. The available suite is preparation, not delivery acceptance.

| Prepared | Remaining material or observation |
| --- | --- |
| D4/D6 and both assignment layouts; real SiLU/DSV4, component SwiGLU-OAI/tanh-GELU, GPT-OSS bias | GPT-OSS BASE missing; component math is not a real-model service pass |
| Physical cache slot permutation, capacities 1/top-k/top-k+3, dirty buffers, empty/padded tail, two workspaces, concurrent CUDA streams | Candidate GPU run; service minimum-capacity rules still needed |
| Graph replay for full-layer and cache banks, changing tokens/routes/tails | Service Graph/SD execution-statistics protocol and candidate run |
| FTW D4/D6/word-major, rename, missing data, bias, TP2 | Current source deletion removes input symlink copies only; canonical source paths remain visible, so complete source isolation is not yet established |
| Real Qwen/DSV4 service, CPU/hybrid/resident/offload, layered scheduling, cancellation, self-SD/DFlash | Explicit legal-mode expectations; maintenance API; cache groups/multi-turn; profiler evidence for transport, HBM, Graph and SD |
| TP2 service and independent TP boundary math | Two approved GPUs; public per-rank bind recipe; actual collective observation is unavailable under contract §9 |
| Paired baseline/candidate timings and non-NoWAG regression | Baseline e6f6d90 source environment, quiet resource window; HBM and transfer-byte measurements |

Flash-Next has no qualified BASE/weight pair for this acceptance. Its reference geometry
does not count as a served-model pass. Offline quantization/training and GGUF are excluded.

This handoff ran only independent single-thread CPU checks with `CUDA_VISIBLE_DEVICES=''`:
runner transport/token accounting **2 passed** (0.96 s), cache-slot/workspace reference rows
**4 passed** (2.34 s), and pack/layout/loop-projection/E4M3 self-checks **35 passed** (0.97 s).
These results establish the test/reference behavior only. Frozen numeric bounds are unchanged.

The coordinator supplied these public inputs (set them explicitly in the run environment):

```sh
export NOWAG_QWEN36_SIDE=/data1/lmcache_kv/goodput_campaign/qwen36_general_mix_joint_recovery_v1
export NOWAG_DSV4_BASE=/data1/lmcache_kv/models/DeepSeek-V4-Flash-0731
export NOWAG_DFLASH_DRAFT=/data2/servebig-envs/dflash_models/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/f181eece646affea2c38b2765f1aaa01a9734ccd
```

The Qwen manifest declares generic v1, D6/B12, H=2048/I=512/E=256, 40 MoE layers;
the DSV4 BASE config declares H=4096/I=2048, 43 decoder layers. These are input checks,
not evidence that the candidate can load or serve them.

## Environment

| Variable | Meaning |
| --- | --- |
| `PYTHONPATH` | candidate `python/` for in-process tests (`test_bind_contract.py`) |
| `NOWAG_SOURCE` | candidate `python/` for `ft serve` / `ft checkpoint` subprocesses |
| `NOWAG_BASELINE_SOURCE` | pre-feature baseline `python/` (regression and paired perf) |
| `NOWAG_PYTHON` | interpreter (default freetoken-dev) |
| `NOWAG_GPU_OK=1`, `NOWAG_GPU` | approve single-GPU use; GPU UUID or index for `--gpu` |
| `NOWAG_TP2_OK=1`, `NOWAG_TP2_GPUS=a,b` | approve two-GPU runs |
| `NOWAG_TP_DELIVERED=1` | count TP>1 NoWAG rows (contract §9: not before phase P4) |
| `CUDA_VISIBLE_DEVICES` | the approved GPU for in-process CUDA tests (`cuda` rows use `cuda:0`) |
| `NOWAG_SCRATCH` | writable multi-GB dir (e.g. `/dev/shm/...`): synthetic full-geometry sidecars (~8-12 GB each), FTW outputs |
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
