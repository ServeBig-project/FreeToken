#!/usr/bin/env python3
"""Small public CLI rejection matrix; the coordinator supplies a usable GPU."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


MODELS = {
    "qwen3": "/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B",
    "qwen36": "/data1/lmcache_kv/models/Qwen3.6-35B-A3B",
}


def invoke(command, env):
    started = time.monotonic()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=env, start_new_session=True) as process:
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return {"command": command, "stdout": stdout, "stderr": stderr,
                "returncode": process.returncode, "timed_out": timed_out,
                "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Server's Python executable")
    parser.add_argument("--pythonpath", required=True, help="Candidate's public python directory")
    parser.add_argument("--gpu", required=True, help="An actually usable device UUID or index")
    parser.add_argument("--port", required=True, type=int, help="An unused local service port")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=args.pythonpath)
    prefix = [args.python, "-c", "from freetoken.cli import main; main()", "serve"]
    results = []
    help_result = invoke(prefix + ["--help"], env)
    text = help_result["stdout"] + help_result["stderr"]
    flags = ("--speculative-num-steps", "--speculative-draft-experts",
             "--speculative-draft-residency", "--speculative-draft-load-missing",
             "--speculative-adaptive-cost", "--speculative-verify-prefetch",
             "--cuda-graph-max-bs", "--moe-backend")
    descriptions = re.findall(r"(?ms)^\s{2}--speculative-[^\n]*\n(?:(?!\s{2}--).)*", text)
    narrow_scope = any("qwen3" in block.lower() and "qwen3.6" not in block.lower()
                       for block in descriptions)
    help_result["case"] = "serve_help"
    help_result["checks"] = {"successful_help": help_result["returncode"] == 0,
                              "all_existing_parameters": all(flag in text for flag in flags),
                              "no_qwen3_only_sd_description": not narrow_scope,
                              "within_timeout": not help_result["timed_out"]}
    results.append(help_result)
    common = ["--gpu", args.gpu, "--host", "127.0.0.1", "--port", str(args.port),
              "--dtype", "bfloat16", "--attention-backend", "fi", "--moe-cache-size", "512",
              "--num-tokens", "16384", "--max-seq-len-override", "4096",
              "--max-prefill-length", "2048", "--max-running-requests", "4",
              "--cuda-graph-max-bs", "4", "--cache-type", "radix", "--batching-policy", "legacy",
              "--sampling-defaults", "none", "--reasoning-parser", "off",
              "--speculative-draft-experts", "3"]
    for model, checkpoint in MODELS.items():
        for kind, backend, steps in (("fused_graph", "fused", 3), ("graph_n9", "offload", 9)):
            command = prefix + ["--model-path", checkpoint] + common + [
                "--moe-backend", backend, "--speculative-num-steps", str(steps)]
            result = invoke(command, env)
            lower = (result["stdout"] + result["stderr"]).lower()
            unavailable = re.search(r"no (?:cuda )?gpus?|cuda (?:is )?not available|"
                                    r"invalid device ordinal|no cuda-capable device|"
                                    r"unrecognized arguments|no such option", lower)
            hint = "eager" in lower or re.search(r"--cuda-graph-max-bs[ =]+0", lower)
            rejection = re.search(r"not support|unsupported|does not|cannot|"
                                  r"requires|missing|maximum|at most|<=\s*8|1\.\.8|1[–-]8", lower)
            result["case"] = model + "_" + kind
            result["checks"] = {
                "nonzero_exit": result["returncode"] != 0,
                "within_timeout": not result["timed_out"],
                "not_device_or_argument_failure": unavailable is None,
                "explicit_graph_rejection": "graph" in lower and rejection is not None,
                "explicit_eager_guidance": bool(hint),
            }
            results.append(result)
    report = {"pythonpath": args.pythonpath, "gpu": args.gpu, "cases": results,
              "passed": sum(all(case["checks"].values()) for case in results),
              "total": len(results)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output), "passed": report["passed"], "total": len(results)}))
    return 0 if report["passed"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
