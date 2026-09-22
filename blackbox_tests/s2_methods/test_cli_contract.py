"""Public CLI contracts; only startup rejection cases require an assigned GPU."""

from contextlib import suppress
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "python"
MODEL = "/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B"


def run_cli(args):
    env = {**os.environ, "PYTHONPATH": str(PACKAGE)}
    command = [sys.executable, "-c", "from freetoken.cli import main; main()", *args]
    process = subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=45)
        return process.returncode, output
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def test_public_help():
    """Missing advertised options prevents reproducing the configured experiment."""
    code, output = run_cli(["serve", "--help"])
    assert code == 0, output
    for option in ("--moe-resident-experts", "--moe-expert-profile",
                   "--speculative-adaptive-profile", "--speculative-reuse-expert-cap"):
        assert option in output


@pytest.mark.parametrize("count", [0, 4, 8])
def test_expert_selection(tmp_path, count):
    """The public documentation's profile must produce its deterministic ranking."""
    profile, output = tmp_path / "counts.json", tmp_path / "hot.json"
    profile.write_text(json.dumps({"num_layers": 2, "num_experts": 4,
                                   "counts": [[3, 1, 8, 0], [2, 6, 4, 1]]}))
    code, message = run_cli(["bench", "experts", "--profile", str(profile), "--count", str(count), "--output", str(output)])
    assert code == 0, message
    ranked = [[0, 2], [1, 1], [1, 2], [0, 0], [1, 0], [0, 1], [1, 3], [0, 3]]
    assert json.loads(output.read_text())["gpu_experts"] == ranked[:count]


@pytest.mark.parametrize("count", [-1, 9])
def test_expert_selection_bounds(tmp_path, count):
    """Out-of-range requested cardinality must fail, rather than silently truncate."""
    profile = tmp_path / "counts.json"
    profile.write_text(json.dumps({"num_layers": 2, "num_experts": 4, "counts": [[0] * 4] * 2}))
    code, message = run_cli(["bench", "experts", "--profile", str(profile), "--count", str(count),
                            "--output", str(tmp_path / "hot.json")])
    assert code != 0 and "count" in message.lower(), message


COST = {"target_token_ms": 45.0, "draft_step_ms": 27.0, "expert_bandwidth_gib_s": 24.419}
STARTUP_CASES = [
    (["--moe-resident-experts", "INPUT"], {"gpu_experts": [[0, 0], [0, 0]]}, r"resident|duplicate|unique"),
    (["--moe-resident-experts", "INPUT"], {"gpu_experts": [[48, 0]]}, r"resident|layer|range"),
    (["--moe-resident-experts", "INPUT"], {"gpu_experts": [[0, 128]]}, r"resident|expert|range"),
    (["--moe-resident-experts", "INPUT", "--moe-cache-size", "1279"],
     {"gpu_experts": [[layer, expert] for layer in range(8) for expert in range(128)]}, r"resident|cache|slot|prefill"),
    (["--moe-resident-experts", "INPUT", "--moe-backend", "fused"], {"gpu_experts": [[0, 0]]}, r"resident|fused|offload"),
    (["--moe-resident-experts", "INPUT", "--batching-policy", "mixed"], {"gpu_experts": [[0, 0]]}, r"legacy|batching-policy"),
    (["--moe-expert-profile", "INPUT", "--speculative-num-steps", "4"], {}, r"profile|profiling|ordinary|speculative"),
    (["--speculative-adaptive-profile", "INPUT"], COST, r"adaptive|speculative"),
    *[(["--speculative-num-steps", "16", "--speculative-adaptive-profile", "INPUT"], {**COST, key: value}, rf"{key}|positive|finite|adaptive|profile")
      for key, value in [("target_token_ms", 0), ("draft_step_ms", -1), ("expert_bandwidth_gib_s", float("inf"))]],
    (["--speculative-reuse-expert-cap", "14"], {}, r"reuse|speculative"),
    (["--speculative-num-steps", "16", "--speculative-reuse-expert-cap", "7"], {}, r"reuse|cap|range|experts"),
    (["--speculative-num-steps", "16", "--speculative-reuse-expert-cap", "129"], {}, r"reuse|cap|range|experts"),
]


@pytest.mark.parametrize("options,payload,reason", STARTUP_CASES)
def test_startup_rejection(tmp_path, options, payload, reason):
    """Malformed controls must reject clearly before serving requests."""
    gpu = os.environ.get("FT_SD_GPU")
    if not gpu:
        pytest.skip("Coordinator must assign GPU before startup rejection checks")
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    options = [str(path) if value == "INPUT" else value for value in options]
    code, output = run_cli(["serve", "--model-path", MODEL, "--gpu", gpu,
                           "--moe-backend", "offload", "--batching-policy", "legacy",
                           "--moe-cache-size", "1536", "--num-tokens", "4096", *options])
    assert code != 0, options
    assert re.search(reason, output, re.I), output
