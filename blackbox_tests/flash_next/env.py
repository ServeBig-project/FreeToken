"""Run configuration from the coordinator's run materials (flash-next round 1: I1 + I2)."""
import os

PY = os.environ.get("FT_PYTHON", "/home/nengneng/miniconda3/envs/freetoken-dev/bin/python")
IMPL = os.environ.get("FT_IMPL", "/home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees/flash-next")
REF_FT = os.environ.get("FT_REF_FT", "/opt/freetoken/.venv/bin/ft")  # upstream image freetoken:555efd8
MODEL = os.environ.get("FT_MODEL", "/data1/yuchen/models/Qwen3.8-Flash-Next-NVFP4")
WORK = os.environ.get("FT_WORK_DIR", "/tmp/claude-1003/-home-nengneng-AIPrometheus-servebig-servebig-project/"
                                     "d365ae8e-28b7-42d8-a065-bfd44418b1fa/scratchpad/bb")
RESULTS = os.environ.get("FT_RESULTS_DIR", os.path.join(WORK, "results"))
FTW_DIR = os.environ.get("FT_FTW_DIR", "")  # needs ~130 GB free; empty skips the FTW module
STARTUP_TIMEOUT = float(os.environ.get("FT_STARTUP_TIMEOUT", "1500"))
REQ_TIMEOUT = float(os.environ.get("FT_REQ_TIMEOUT", "900"))

CTX = 16384  # --max-seq-len-override for every session
KV_RESERVE = 32768
# Shared by every candidate session; per-session flags add the dimension under test.
BASE = ["--moe-backend", "offload", "--moe-cache-auto", "--kv-reserve-tokens", str(KV_RESERVE),
        "--max-seq-len-override", str(CTX), "--enable-cache-report", "--speculative-num-steps", "0"]
REF_BASE = ["--moe-backend", "offload", "--moe-cache-auto", "--kv-reserve-tokens", str(KV_RESERVE),
            "--max-seq-len-override", str(CTX), "--enable-cache-report", "--text-model-only",
            "--cuda-graph-max-bs", "4", "--max-prefill-length", "1024", "--cache-type", "radix"]


def renamed_model(name="Qwen3.8-Flash-Next-FP8-dense"):
    """Per-file symlinks under a misleading name: a directory name must not choose precision."""
    dst = os.path.join(WORK, "renamed", name)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(MODEL):
        if not f.startswith(".") and not os.path.lexists(os.path.join(dst, f)):
            os.symlink(os.path.join(MODEL, f), os.path.join(dst, f))
    return dst
