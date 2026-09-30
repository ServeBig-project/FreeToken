import gc
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent
RESULTS = {}


@pytest.fixture(scope="session", autouse=True)
def _gpu_budget():
    # the card may be shared with a serving process; keep this process near 1 GiB of tensors
    torch.cuda.set_per_process_memory_fraction(2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    yield


@pytest.fixture
def log(request):
    gc.collect()  # a failed test's tensors sit in reference cycles until collected
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = RESULTS.setdefault(request.node.nodeid, {})
    yield rec.setdefault("entries", [])
    rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()


@pytest.fixture
def extra(request):
    return RESULTS.setdefault(request.node.nodeid, {}).setdefault("extra", {})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    rep = (yield).get_result()
    if rep.when == "call" or rep.failed:
        rec = RESULTS.setdefault(item.nodeid, {})
        rec["outcome"] = rep.outcome
        if rep.failed:
            rec["failure"] = str(rep.longrepr)[-4000:]


def _summary(entries):
    by_kind = defaultdict(list)
    for e in entries:
        by_kind[e["kind"]].append(e)
    out = {}
    for kind, es in by_kind.items():
        s = {"count": len(es)}
        for key in ("tol_ratio", "max_abs", "max_rel", "rel_rms", "tol_ratio_old"):
            vals = [e[key] for e in es if key in e]
            if vals:
                s["max_" + key if not key.startswith("max_") else key] = max(vals)
        if kind != "negative_control":
            bad = sorted((e for e in es if e.get("tol_ratio", 0.0) > 1.0), key=lambda e: -e["tol_ratio"])
            s["violations"] = len(bad)
            s["worst_violations"] = bad[:8]
            if any("tol_ratio_old" in e for e in es):
                s["violations_under_old_formula"] = sum(e["tol_ratio_old"] > 1.0 for e in es)
        if kind == "negative_control":
            s["min_tol_ratio"] = min(e["tol_ratio"] for e in es)
            s["names"] = sorted({e["name"] for e in es})
        if kind == "fold_count0_bitwise_copy":
            s["all_bitwise"] = all(e["value"] for e in es)
        out[kind] = s
    return out


def pytest_sessionfinish(session, exitstatus):
    if not RESULTS:
        return
    import freetoken.kernel.triton.gdn_replay as op

    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
    doc = {
        "env": {
            "worktree_head": git("rev-parse", "HEAD"),
            "operator_file": op.__file__,  # PYTHONPATH decides which tree is under test
            "operator_file_blob": git("hash-object", op.__file__),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "tests": {nid: {"outcome": r.get("outcome"), "failure": r.get("failure"),
                        "peak_allocated_bytes": r.get("peak_allocated_bytes"),
                        "summary": _summary(r.get("entries", [])), "extra": r.get("extra", {})}
                  for nid, r in RESULTS.items()},
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "gdn_replay_numeric.json").write_text(json.dumps(doc, indent=1))
