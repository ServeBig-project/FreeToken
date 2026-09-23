"""Public option discovery and the two explicitly rejected combinations."""

from contextlib import suppress
import json
import os
import re
import signal
import subprocess
import sys

import pytest

FLAGS = ("--speculative-adaptive-cost", "--speculative-draft-load-missing", "--speculative-verify-prefetch")
MODEL = "/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B"


@pytest.fixture
def public_cli(tmp_path):
    package = os.environ.get("FT_SD_PACKAGE")
    assert package, "Set FT_SD_PACKAGE to the candidate's public Python package directory"

    def run(args):
        command = [sys.executable, "-c", "from freetoken.cli import main; main()", "serve", *args]
        env = {**os.environ, "PYTHONPATH": package, "CUDA_VISIBLE_DEVICES": ""}
        process = subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            output, _ = process.communicate(timeout=45)
            (tmp_path / "public-cli-output.txt").write_text(output)
            return process.returncode, output
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    return run


def test_new_flags_in_public_help(public_cli):
    code, text = public_cli(["--help"])
    assert code == 0, text
    assert all(flag in text for flag in FLAGS), text


@pytest.mark.parametrize("case", ["load-requires-router", "adaptive-mutual-exclusion"])
def test_declared_startup_rejections(public_cli, tmp_path, case):
    args = ["--model-path", MODEL, "--moe-backend", "offload", "--batching-policy", "legacy",
            "--cuda-graph-max-bs", "0", "--speculative-num-steps", "8", "--speculative-draft-experts", "3"]
    if case == "load-requires-router":
        args += ["--speculative-draft-residency", "off", "--speculative-draft-load-missing"]
        reason = r"router"
    else:
        profile = tmp_path / "old-cost-profile.json"
        profile.write_text(json.dumps({"target_token_ms": 45.0, "draft_step_ms": 27.0,
                                       "expert_bandwidth_gib_s": 24.419}))
        args += ["--speculative-adaptive-cost", "--speculative-adaptive-profile", str(profile)]
        reason = r"mutual|exclusive|cannot.*(?:both|together|combin|with)|incompatible|conflict"
    code, text = public_cli(args)
    assert code != 0 and re.search(reason, text, re.I), text
