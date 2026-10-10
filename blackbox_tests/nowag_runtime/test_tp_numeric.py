"""Actual two-GPU reductions of D4/D6, I=512 compressed experts (no model-support expansion)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import tiny_model as tiny
from cases import need_tp2
from harness import LOG_DIR, PYTHON, env


@pytest.mark.parametrize("d,shape,math", [(4, "qwen3", "silu"), (6, "qwen3", "silu"),
                                       (6, "dsv4-math", "dsv4"), (6, "flashnext-shape", "silu")])
def test_tp2_bind_numeric(d, shape, math):
    gpus = need_tp2()
    base, side = tiny.paths(d, shape)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = LOG_DIR / f"tp2-bind-{shape}-d{d}"
    for rank in range(2):
        Path(f"{result}.rank{rank}.json").unlink(missing_ok=True)
    command = [PYTHON, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
               str(Path(__file__).with_name("tp_numeric.py")), "--base", str(base), "--side", str(side),
               "--result", str(result), "--math", math]
    with result.with_suffix(".log").open("w") as log:
        process = subprocess.run(command, env={**env(), "CUDA_VISIBLE_DEVICES": gpus},
                                 stdout=log, stderr=subprocess.STDOUT, timeout=900)
    assert process.returncode == 0, result.with_suffix(".log").read_text()[-6000:]
    records = [json.loads(Path(f"{result}.rank{rank}.json").read_text()) for rank in range(2)]
    assert [record["rank"] for record in records] == [0, 1]
    assert [record["local_rank"] for record in records] == [0, 1]
    assert all(r["world_size"] == 2 and r["backend"] == "nccl" and r["all_reduce_calls"] == 4
               for r in records)
