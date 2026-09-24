"""Public CLI after the SD switch cleanup; CUDA is hidden and no server starts."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "python"
MODEL = "/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B"
SERVE = ("serve", "--model-path", MODEL, "--batching-policy", "legacy", "--speculative-num-steps", "4")
REMOVED = {"--speculative-adaptive-profile": "profile.json", "--speculative-reuse-expert-cap": "14",
           "--moe-resident-experts": "resident.json", "--moe-expert-profile": "counts.json"}


def ft(*args):
    completed = subprocess.run([sys.executable, "-c", "from freetoken.cli import main; main()", *args],
                               env={**os.environ, "PYTHONPATH": str(PACKAGE), "CUDA_VISIBLE_DEVICES": ""},
                               capture_output=True, text=True, timeout=60)
    return completed.returncode, completed.stdout + completed.stderr


@pytest.mark.parametrize("option", REMOVED)
def test_removed_serve_option_rejected(option):
    """An old command that still parses would run with silently different behavior."""
    code, output = ft(*SERVE, option, REMOVED[option])
    assert code != 0 and f"unrecognized arguments: {option}" in output, output


def test_affinity_residency_rejected():
    code, output = ft(*SERVE, "--speculative-draft-residency", "affinity")
    assert code != 0 and re.search(r"invalid choice: '?affinity", output), output


def test_serve_help_lists_only_current_switches():
    code, output = ft("serve", "--help")
    assert code == 0, output
    assert "--speculative-draft-residency {off,router}" in output, output
    assert not [option for option in REMOVED if option in output], output


def test_bench_experts_removed(tmp_path):
    """A valid old profile proves the subcommand did not run, not just that input was bad."""
    profile, hot = tmp_path / "counts.json", tmp_path / "hot.json"
    profile.write_text(json.dumps({"num_layers": 2, "num_experts": 4, "counts": [[3, 1, 8, 0], [2, 6, 4, 1]]}))
    _, output = ft("bench", "experts", "--profile", str(profile), "--count", "4", "--output", str(hot))
    assert re.search(r"unknown.*experts", output, re.I) and not hot.exists(), output
    code, output = ft("bench", "--help")
    assert code == 0 and re.search(r"^\s+bw\s", output, re.M), output
    assert not re.search(r"^\s+experts\s", output, re.M), output


def test_bench_bw_remains():
    code, output = ft("bench", "bw", "--help")
    assert code == 0 and "usage: ft bench bw" in output, output
