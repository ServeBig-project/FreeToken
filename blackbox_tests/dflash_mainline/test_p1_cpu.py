"""P1 CPU-only checks: configuration rejection, public model entry on real and tiny checkpoints.

Run with CUDA hidden:
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=<impl>/python pytest blackbox_tests/dflash_mainline/test_p1_cpu.py
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from harness import DRAFTER, NVFP4, PYTHON, SOURCE  # noqa: E402
from tiny_drafter import WINDOW, build  # noqa: E402

from freetoken.speculative.dflash_model import DFlashModel, read_dflash_config  # noqa: E402

DF = ["--speculative-draft-model-path", DRAFTER]


def launch_error(args):
    """Start the CLI with no visible GPU; it must refuse the config before touching a device."""
    env = {**os.environ, "PYTHONPATH": SOURCE, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run([PYTHON, "-m", "freetoken", "--model-path", NVFP4, "--port", "31799", *args],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode != 0, f"accepted invalid config {args}"
    tail = proc.stderr.strip().splitlines()[-1]
    assert "cannot use CUDA device" not in tail, f"config {args} was not rejected before device use: {tail}"
    return tail


@pytest.mark.parametrize("args, words", [
    (["--dflash-recent-acceptance"], ["dflash-recent-acceptance"]),
    (["--speculative-num-steps", "9", *DF], ["8"]),
    (["--speculative-num-steps", "-1"], ["1..8", ">= 0"]),
    (["--speculative-num-steps", "8", *DF, "--dflash-adaptive-observe-only"], ["adaptive"]),
    (["--speculative-num-steps", "4", "--speculative-adaptive-cost", "--dflash-adaptive-observe-only"],
     ["observe-only"]),
    (["--speculative-num-steps", "8", *DF, "--dflash-attention-window", "-1"], ["window"]),
    (["--speculative-num-steps", "8", *DF, "--batching-policy", "layered"], ["legacy"]),
    (["--speculative-num-steps", "8", *DF, "--batching-policy", "mixed"], ["legacy"]),  # others: GPU rejections
    (["--speculative-num-steps", "8", *DF, "--page-size", "16"], ["page size"]),
])
def test_rejected_before_ready(args, words):
    tail = launch_error(args)
    assert any(w in tail for w in words), tail


def test_real_checkpoint_public_attributes():
    config = read_dflash_config(DRAFTER)
    assert config.hidden_size == 2048 and config.num_hidden_layers == 6
    model = DFlashModel(DRAFTER, dtype=torch.bfloat16, device="cpu")
    assert list(model.target_layer_ids) == [1, 6, 11, 16, 22, 27, 32, 37]
    assert model.mask_token_id == 248077 and model.block_size == 16 and model.hidden_size == 2048
    assert [tuple(m) for m in model.attention_modes] == [(True, 4095)] * 5 + [(False, -1)]
    with safe_open(str(Path(DRAFTER) / "model.safetensors"), "pt") as f:
        stored = 2 * sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())  # all BF16
    assert model.weight_bytes == stored


def test_tiny_no_gdn_checkpoint_loads(tmp_path):
    _, stored = build(tmp_path / "tiny")
    model = DFlashModel(tmp_path / "tiny", dtype=torch.bfloat16, device="cpu")
    assert model.weight_bytes == stored
    assert [tuple(m) for m in model.attention_modes] == [(True, WINDOW - 1), (False, -1)]
    features = torch.randn(5, 3 * 2048, dtype=torch.bfloat16)
    stored_kv = {}
    model.project_context(features, torch.arange(5), lambda layer, k, v: stored_kv.setdefault(layer, (k, v)))
    assert sorted(stored_kv) == [0, 1] and stored_kv[0][0].shape == (5, 4, 128)


def test_independent_numeric_reference_regression():
    """Re-run the existing independent FP32/BF16 reference suite against this revision."""
    runner = Path(__file__).parents[1] / "dflash_numeric" / "run_numeric.py"
    env = {**os.environ, "PYTHONPATH": SOURCE, "CUDA_VISIBLE_DEVICES": "",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    proc = subprocess.run([PYTHON, str(runner), "--output", os.devnull], env=env,
                          capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
