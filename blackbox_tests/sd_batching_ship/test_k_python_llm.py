"""Python `LLM` entry keeps CLI semantics: omitted / 0 / positive steps are distinct (section 2)."""
import json
import os
import subprocess

import pytest

from . import env
from .client import record
from .server import free_port, launch_env, wait_gpu_free

BASE = dict(max_running_req=4, num_token_override=16384, max_seq_len_override=4096, cuda_graph_max_bs=4,
            moe_backend="offload", moe_cache_size=1200, attention_backend="fi")


def run(name, kwargs):
    wait_gpu_free()
    p = subprocess.run(["taskset", "-c", env.CPUS, env.PY, os.path.join(os.path.dirname(__file__), "py_llm.py"),
                        env.Q3_BF16, str(free_port()), json.dumps({**BASE, **kwargs})],
                       capture_output=True, text=True, env=launch_env(), timeout=env.STARTUP_TIMEOUT)
    os.makedirs(env.RESULTS, exist_ok=True)
    with open(os.path.join(env.RESULTS, f"K_{name}.log"), "w") as f:
        f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    wait_gpu_free(300)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    res = json.loads(lines[-1][7:]) if lines else {"error": f"exit {p.returncode}: {p.stderr[-1500:]}"}
    record("K_" + name, res)
    return p.returncode, res


@pytest.mark.parametrize("name,kwargs", [
    ("legacy_sd4_all", dict(batching_policy="legacy", speculative_num_steps=4, speculative_phase="all")),
    ("legacy_inwave_omitted", dict(batching_policy="legacy", speculative_phase="inwave")),
])
def test_explicit_illegal_raises(name, kwargs):
    code, res = run(name, kwargs)
    assert code != 0 and "error" in res, res
    assert "phase" in res["error"].lower(), res


def test_zero_wins_over_phase_and_draft_path():
    code, res = run("legacy_sd0", dict(batching_policy="legacy", speculative_num_steps=0, speculative_phase="all",
                                       speculative_draft_model_path="/nonexistent/dflash-dir"))
    assert code == 0 and "outputs" in res, res
    for o in res["outputs"]:
        assert o.get("token_ids", 24) == 24, res
