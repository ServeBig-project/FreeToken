# NoWAG runtime black-box acceptance

Written only from `docs/nowag-runtime-public-contract.md`, public CLI help, public model
definitions (HF configs/transformers, DeepSeek-V4 `inference/`) and the public NoWAG
artifacts. No production code, design notes, diffs or internal tests were read.

The current contract is the [candidate public contract](/home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees/nowag-runtime/docs/nowag-runtime-public-contract.md).
The test worktree's old design documents are not a contract source.

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
| Physical cache slot permutation, capacities 1/top-k/top-k+3, dirty buffers, empty/padded tail, two workspaces, concurrent CUDA streams | Candidate GPU run |
| Graph replay for full-layer/cache banks; service AR/SD replay-counter differences | Candidate GPU run; independent profiler corroboration |
| FTW D4/D6/word-major, rename, missing data, bias, TP2; standalone source-absent container entry | Coordinator must mount only DEST/environment/code/reference JSON for the isolation row |
| Real Qwen/DSV4 modes, exact cache lower boundaries, streaming/overlap/D2D, maintenance, groups/continuations, legacy/layered SD | Candidate run; profiler evidence for compressed transport and total memory; prolonged resource-reuse coverage |
| Two-process bind/NCCL math at I512/D4/D6, H4096/I2048 DSV4 math, H2560/I640 component shape; small Qwen3MoE engine TP2 offload/cpu/hybrid | Two approved GPUs; GPT-OSS TP bias still needs weights. Qwen3.6/DSV4 real BASE remain TP1 |
| Paired timings and non-NoWAG regression against e6f6d90 | Quiet resource window; HBM and transfer-byte measurements |

Flash-Next has no qualified BASE/weight pair for this acceptance. Its reference geometry
does not count as a served-model pass. Offline quantization/training and GGUF are excluded.

This handoff ran only independent single-thread CPU checks with `CUDA_VISIBLE_DEVICES=''`:
runner transport/token accounting **2 passed** (0.96 s), cache-slot/workspace reference rows
**4 passed** (2.34 s), and pack/layout/loop-projection/E4M3 self-checks **35 passed** (0.97 s).
These results establish the test/reference behavior only. Frozen numeric bounds are unchanged.
The small HF Qwen3MoE BASE and D4/D6 sidecars were also built on CPU and checked through
public config/tokenizer/format readers (H256/I512/E8, 2 layers, 48 matrices; about 16 MB per pair).
The larger component geometries are prepared in code but have not been generated.

The coordinator supplied these public inputs (set them explicitly in the run environment):

```sh
export NOWAG_QWEN36_SIDE=/data1/lmcache_kv/goodput_campaign/qwen36_general_mix_joint_recovery_v1
export NOWAG_DSV4_BASE=/data1/lmcache_kv/models/DeepSeek-V4-Flash-0731
export NOWAG_DFLASH_DRAFT=/data2/servebig-envs/dflash_models/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/f181eece646affea2c38b2765f1aaa01a9734ccd
```

The Qwen manifest declares generic v1, D6/B12, H=2048/I=512/E=256, 40 MoE layers;
the DSV4 BASE config declares H=4096/I=2048, 43 decoder layers. These are input checks,
not evidence that the candidate can load or serve them.

## Staged coordinator commands

Run from this test worktree. Set the real input variables above, plus:

```sh
P=/home/nengneng/miniconda3/envs/freetoken-dev/bin/python
T=blackbox_tests/nowag_runtime
export NOWAG_SOURCE=/home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees/nowag-runtime/python
export NOWAG_BASELINE_SOURCE=/home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees/nowag-runtime-baseline/python
export PYTHONPATH="$NOWAG_SOURCE"
export NOWAG_SCRATCH=/dev/shm/nowag-runtime-acceptance-20261009
export NOWAG_LOG_DIR=/data2/servebig-envs/nowag_runtime_acceptance_20261009
```

1. CPU/reference: 51 self-check rows plus 65 reference-bind rows, no candidate acceptance.

   ```sh
   CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $P -m pytest -q "$T/test_reference.py" "$T/test_harness.py"
   CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $P -m pytest -q "$T/test_bind_contract.py" -k 'reference and not cuda'
   ```

2. Approved single GPU: set `NOWAG_GPU_OK=1`, `NOWAG_GPU=<approved UUID>` and
   `CUDA_VISIBLE_DEVICES=<same UUID>`. Component selection has 75 GPU rows, including
   5 GPT-OSS rows that skip while `NOWAG_GPTOSS_BASE` is missing.

   ```sh
   $P -m pytest -rs "$T/test_bind_contract.py" -k 'cuda or graph_replay or unsupported_math'
   ```

3. Real-service and lifecycle selection: 78 GPU rows. Baseline/performance is a separate
   5-row quiet-resource run; GPT-OSS service adds 4 rows when its BASE is available.

   ```sh
   $P -m pytest -rs "$T/test_service.py" "$T/test_service_paths.py" "$T/test_cache_status.py" "$T/test_public_errors.py" -k 'not tp2 and not gptoss and not baseline and not default_backend and not paired_performance'
   $P -m pytest -rs "$T/test_service.py" -k 'baseline or default_backend or paired_performance'
   ```

   FTW must run one variant at a time; archive needed results and reclaim that generated
   output before the next full model conversion. The native reference JSON is written
   alongside service logs. Inside the source-isolated container set `NOWAG_ISOLATED_FTW`
   and `NOWAG_ISOLATED_REFERENCE`, then select `test_ftw_with_original_sources_absent`.

   ```sh
   $P -m pytest -rs "$T/test_ftw.py::test_conversion_leaves_sources_untouched[qwen36]" "$T/test_ftw.py::test_ftw_roundtrip_matches_native[qwen36]" "$T/test_ftw.py::test_ftw_renamed_gives_identical_output" "$T/test_ftw.py::test_ftw_missing_data_rejected"
   ```

4. Approved TP2 only: set `NOWAG_TP_DELIVERED=1`, `NOWAG_TP2_OK=1`,
   `NOWAG_TP2_GPUS=<uuid0>,<uuid1>` and the same ordered `CUDA_VISIBLE_DEVICES`.
   Keep single-GPU approval set for the paired TP1 runs. This selects 14 real GPU rows;
   the large component fixtures may need about 1 GB of scratch/host RAM during creation.

   ```sh
   $P -m pytest -rs "$T/test_tp_numeric.py" "$T/test_service.py" "$T/test_ftw.py" "$T/test_cache_status.py" "$T/test_public_errors.py" -k tp2
   ```

No broad suite run should hide skips: missing GPT-OSS currently affects 12 GPU rows;
source-isolated FTW has its own container row. DSV4/Qwen/DFlash inputs are available.

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
