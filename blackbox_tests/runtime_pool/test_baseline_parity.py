"""Compact layout: the implementation matches the pre-change release bit for bit.

The release is imported in a subprocess from FREETOKEN_BASELINE_PYTHONPATH (default: the
dflash-mainline worktree next to this one)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from parity_cases import run_all

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parents[2] / "dflash-mainline" / "python"
BASELINE = os.environ.get("FREETOKEN_BASELINE_PYTHONPATH", str(DEFAULT))


@pytest.mark.skipif(not Path(BASELINE).is_dir(), reason="baseline release not available")
def test_compact_layout_matches_release(tmp_path):
    out = tmp_path / "baseline.pt"
    env = dict(os.environ, PYTHONPATH=BASELINE)
    subprocess.run([sys.executable, str(HERE / "parity_cases.py"), str(out)], cwd=HERE, env=env,
                   check=True)
    base = torch.load(out)
    impl = run_all()
    assert base.keys() == impl.keys()
    mismatched = [n for n in base if not torch.equal(base[n], impl[n])]
    assert not mismatched, mismatched
