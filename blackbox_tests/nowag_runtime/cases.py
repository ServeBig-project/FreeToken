"""Public inputs: model math families, real artifact locations, environment gates.

Math families are transcribed from the public model definitions:
  Qwen3.6 (qwen3_5_moe, HF transformers): SiLU(gate)*up, route weight on expert output.
  GPT-OSS (HF transformers GptOssExperts): gate<=7, -7<=up<=7, (up+1)*gate*sigmoid(1.702*gate),
      gate/up/down bias, route weight on output; HF gate_up is interleaved (gate=::2, up=1::2).
  DeepSeek-V4 (checkpoint inference/model.py): E4M3 block-128 act_quant (pow2 scale) on the
      gate/up input, gate<=10, -10<=up<=10, SiLU(gate)*up, route weight times that product,
      cast BF16, E4M3 act_quant again on the down input.
  GELU-tanh (Gemma4 text experts, HF transformers): gelu_pytorch_tanh(gate)*up. No base is
      available locally, so it is a component-level family with Qwen3.6 geometry.
"""

import os
from pathlib import Path

import pytest

FAMILIES = {
    #             H     I     top_k  math
    "qwen36_silu": (2048, 512, 8, {"family": "silu", "route": "output"}),
    "gptoss": (2880, 2880, 4, {"family": "gptoss", "alpha": 1.702, "limit": 7.0,
                               "route": "output", "bias": True}),
    "dsv4": (4096, 2048, 6, {"family": "swiglu_limit", "limit": 10.0, "route": "down_input",
                             "dsv4_round": True}),
    "gelu_tanh": (2048, 512, 8, {"family": "gelu_tanh", "route": "output"}),
}

HERE = Path(__file__).resolve().parent


def env_path(name, default=None):
    value = os.environ.get(name, default)
    return Path(value) if value else None


QWEN36_BASE = env_path("NOWAG_QWEN36_BASE", "/data1/lmcache_kv/models/Qwen3.6-35B-A3B-NVFP4")
QWEN36_BF16 = env_path("NOWAG_QWEN36_BF16", "/data1/lmcache_kv/models/Qwen3.6-35B-A3B")
QWEN36_SIDE = env_path(
    "NOWAG_QWEN36_SIDE",
    "/data1/lmcache_kv/nowag_qwen36_experiment/quantized/"
    "qwen36_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5")
DSV4_BASE = env_path("NOWAG_DSV4_BASE")
DSV4_SIDE = env_path(
    "NOWAG_DSV4_SIDE",
    "/data1/lmcache_kv/nowag_4090_experiment/quantized/"
    "dsv4_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5")
GPTOSS_BASE = env_path("NOWAG_GPTOSS_BASE")
DENSE_BASE = env_path("NOWAG_DENSE_BASE", "/data1/lmcache_kv/models/Qwen3.5-4B")
DFLASH_DRAFT = env_path("NOWAG_DFLASH_DRAFT")
SCRATCH = env_path("NOWAG_SCRATCH")


def need_path(path, what):
    if path is None or not Path(path).exists():
        pytest.skip(f"{what} not available (set the NOWAG_* variable; got {path})")
    return Path(path)


def need_scratch():
    if SCRATCH is None:
        pytest.skip("NOWAG_SCRATCH unset: synthetic full-geometry sidecars / FTW outputs need "
                    "a multi-GB writable directory")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return SCRATCH


def need_gpu():
    if os.environ.get("NOWAG_GPU_OK") != "1":
        pytest.skip("GPU use not approved (set NOWAG_GPU_OK=1 and NOWAG_GPU=<uuid or index>)")
    gpu = os.environ.get("NOWAG_GPU")
    if not gpu:
        pytest.skip("NOWAG_GPU unset")
    return gpu


def need_tp2():
    if os.environ.get("NOWAG_TP2_OK") != "1":
        pytest.skip("two-GPU use not approved (set NOWAG_TP2_OK=1 and NOWAG_TP2_GPUS=a,b)")
    gpus = os.environ.get("NOWAG_TP2_GPUS", "")
    if len(gpus.split(",")) != 2:
        pytest.skip("NOWAG_TP2_GPUS must name two GPUs")
    return gpus
