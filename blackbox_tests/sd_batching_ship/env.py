"""Run configuration taken from the coordinator's public run configurations."""
import os

PY = os.environ.get("FT_PYTHON", "/home/nengneng/miniconda3/envs/freetoken-dev/bin/python")
IMPL = os.environ.get("FT_IMPL", "/home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees/sd-batching-ship")
GPU = os.environ.get("FT_GPU", "GPU-b8a2a927-a7dd-4a70-5fca-aa2f74a142cd")
CPUS = os.environ.get("FT_CPUS", "0-15")
RESULTS = os.environ.get("FT_RESULTS_DIR", os.path.join(os.path.dirname(__file__), "_results"))
BENCHBW = os.environ.get("FREETOKEN_BENCHBW_PATH", "/data2/servebig-envs/fp4_dflash_layered_20261007/benchbw.json")
STARTUP_TIMEOUT = float(os.environ.get("FT_STARTUP_TIMEOUT", "900"))
REQ_TIMEOUT = float(os.environ.get("FT_REQ_TIMEOUT", "600"))

Q36_NVFP4 = os.environ.get("FT_Q36_NVFP4", "/data1/lmcache_kv/models/Qwen3.6-35B-A3B-NVFP4")
Q36_BF16 = os.environ.get("FT_Q36_BF16", "/data1/lmcache_kv/models/Qwen3.6-35B-A3B")
Q3_BF16 = os.environ.get("FT_Q3_BF16", "/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B")
DFLASH = os.environ.get(
    "FT_DFLASH",
    "/data2/servebig-envs/dflash_models/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/"
    "f181eece646affea2c38b2765f1aaa01a9734ccd",
)
# A GDN budget that holds the AR state pool but not the extra SD state (section 5).
# Coordinator-provided public value for Qwen3.6 NVFP4 with ReplaySSM off and the default cache type.
TIGHT_GDN_BYTES = os.environ.get("FT_TIGHT_GDN_BYTES", "1250000000")

BUDGET = ["--max-running-requests", "4", "--num-tokens", "16384", "--max-seq-len-override", "4096",
          "--cuda-graph-max-bs", "4"]
CTX = 4096


def model_args(model):
    if model == "q36_nvfp4":
        return ["--model-path", Q36_NVFP4, "--dtype", "bfloat16", "--attention-backend", "fi",
                "--nvfp4-backend", "triton", "--moe-cache-size", "2500",
                "--gdn-state-budget-bytes", "3000000000"]
    if model == "q36_bf16":
        return ["--model-path", Q36_BF16, "--dtype", "bfloat16", "--attention-backend", "fi",
                "--moe-cache-size", "1000", "--gdn-state-budget-bytes", "3000000000"]
    if model in ("q3", "q3_renamed"):
        path = renamed(Q3_BF16, "Llama-3-8B-dense") if model == "q3_renamed" else Q3_BF16
        return ["--model-path", path, "--dtype", "bfloat16", "--attention-backend", "fi",
                "--moe-cache-size", "1200"]
    raise ValueError(model)


def tokenizer_path(model):
    return {"q36_nvfp4": Q36_NVFP4, "q36_bf16": Q36_BF16, "q3": Q3_BF16, "q3_renamed": Q3_BF16}[model]


def renamed(src, name):
    """A directory of per-file symlinks under a misleading name: the name must not change capability."""
    dst = os.path.join(RESULTS, "renamed", name)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if not os.path.exists(os.path.join(dst, f)):
            os.symlink(os.path.join(src, f), os.path.join(dst, f))
    return dst
