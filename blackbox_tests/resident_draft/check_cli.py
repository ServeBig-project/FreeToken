"""CPU-only startup errors; run from the checkout with PYTHONPATH=python."""

import os
import subprocess
import sys


command = [sys.executable, "-c", "from freetoken.cli import main; main()", "serve",
           "--model-path", "/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B"]
cases = [
    (["--speculative-draft-residency", "invalid"], ("invalid choice", "residency")),
    (["--speculative-draft-residency", "router", "--speculative-num-steps", "0"],
     ("residency", "speculative")),
]
for flags, markers in cases:
    completed = subprocess.run(command + flags, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                               capture_output=True, text=True, timeout=60)
    output = completed.stdout + completed.stderr
    if completed.returncode == 0 or not all(marker in output.lower() for marker in markers):
        raise AssertionError(f"expected a clear startup rejection for {flags}:\n{output}")
    if any(marker in output.lower() for marker in ("unrecognized arguments", "no cuda gpus")):
        raise AssertionError(f"startup failed before validating residency: {output}")
    print(f"PASS: {' '.join(flags)}")
