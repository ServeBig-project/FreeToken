"""FTW conversion (`ft checkpoint --nowag-expert-path`): round trip, source isolation, rename,
missing data, bias carried, TP2. Needs GPU approval and NOWAG_SCRATCH (multi-GB outputs).

Isolation is shown by converting from symlink copies of BASE and SIDE and deleting those
copies before serving the FTW directory.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
import access as A  # noqa: E402
import sidecar as S  # noqa: E402
from cases import (QWEN36_BASE, QWEN36_SIDE, DSV4_BASE, DSV4_SIDE, GPTOSS_BASE,  # noqa: E402
                   need_gpu, need_path, need_scratch, need_tp2)
from harness import (Server, expect_rejected, experts, ft, run_prompts, same_execution,  # noqa: E402
                     cross_path)

CACHE = {"qwen36": os.environ.get("NOWAG_QWEN36_CACHE", "1536"),
         "dsv4": os.environ.get("NOWAG_DSV4_CACHE", "640"),
         "gptoss": os.environ.get("NOWAG_GPTOSS_CACHE", "256")}
VARIANTS = ["qwen36", "qwen36-d4-random", "qwen36-d6-wordmajor", "dsv4", "gptoss"]


def inventory(root):
    """(relative path, size, mtime) of everything reachable, following symlinks."""
    out = []
    for p in sorted(Path(root).rglob("*")):
        st = p.stat()
        out.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns))
    return out


def sources(which):
    if which == "qwen36":
        return need_path(QWEN36_BASE, "Qwen3.6 base"), need_path(QWEN36_SIDE, "Qwen3.6 sidecar")
    if which == "dsv4":
        return need_path(DSV4_BASE, "DSV4 base"), need_path(DSV4_SIDE, "DSV4 sidecar")
    if which.startswith("qwen36-"):
        return need_path(QWEN36_BASE, "Qwen3.6 base"), A.sidecar_dir(which)
    return need_path(GPTOSS_BASE, "GPT-OSS base"), A.sidecar_dir("gptoss-d6-random")


def serve_args(model, which, side=None, *extra):
    cache = CACHE[which.split("-")[0]]
    args = ["--model", model, "--moe-backend", "offload", "--moe-cache-size", cache, *extra]
    return args + (["--nowag-expert-path", side] if side else [])


_FTW = {}


def converted(which):
    """Convert once per session from symlink copies, check the sources, delete the copies."""
    if which in _FTW:
        return _FTW[which]
    gpu = need_gpu()
    base, side = sources(which)
    scratch = need_scratch()
    work = scratch / f"ftw-src-{which}"
    shutil.rmtree(work, ignore_errors=True)
    base_copy = S.link_copy(base, work / "base", mutable=())
    side_copy = S.link_copy(side, work / "side", mutable=())
    before = (inventory(base_copy), inventory(side_copy))
    dest = scratch / f"ftw-{which}"
    shutil.rmtree(dest, ignore_errors=True)
    proc = ft(["checkpoint", "--model", base_copy, "--nowag-expert-path", side_copy, "--out", dest,
               "--moe-backend", "offload", "--gpu", gpu], label=f"ftw_convert_{which}")
    assert proc.returncode == 0, proc.stdout[-3000:]
    after = (inventory(base_copy), inventory(side_copy))
    shutil.rmtree(work)                                   # FTW must not need them any more
    _FTW[which] = {"dest": dest, "sources_unchanged": before == after, "base": base, "side": side}
    return _FTW[which]


@pytest.mark.parametrize("which", VARIANTS)
def test_conversion_leaves_sources_untouched(which):
    assert converted(which)["sources_unchanged"]


@pytest.mark.parametrize("which", VARIANTS)
def test_ftw_runs_alone_and_matches_native(which):
    """Native and FTW carry the same compressed weights; for GPT-OSS the native run reads the
    expert biases from BASE, so a FTW that dropped them diverges from it."""
    gpu = need_gpu()
    info = converted(which)
    task = which in ("qwen36", "dsv4")                        # only calibrated real weights
    with Server(f"ftw_native_{which}", serve_args(info["base"], which, info["side"]), gpu) as s:
        native, native_status = run_prompts(s), experts(s.status())
    with Server(f"ftw_{which}", serve_args(info["dest"], which), gpu) as s:
        ftw, ftw_status = run_prompts(s), experts(s.status())
    assert ftw_status["format"] == native_status["format"]
    assert ftw_status["format_parameters"] == native_status["format_parameters"]
    cross_path(native, ftw, f"{which} native vs FTW", task=task)


def test_ftw_renamed_gives_identical_output():
    gpu = need_gpu()
    info = converted("qwen36")
    with Server("ftw_named", serve_args(info["dest"], "qwen36"), gpu) as s:
        a = run_prompts(s)
    renamed = info["dest"].with_name("unrelated-name-ftw")
    info["dest"].rename(renamed)
    try:
        with Server("ftw_renamed", serve_args(renamed, "qwen36"), gpu) as s:
            b = run_prompts(s)
    finally:
        renamed.rename(info["dest"])
    same_execution(a, b, "renamed FTW")


def test_ftw_missing_data_rejected(tmp_path):
    gpu = need_gpu()
    dest = converted("qwen36")["dest"]
    broken = S.link_copy(dest, tmp_path / "broken", mutable=())
    shards = sorted(broken.rglob("*.safetensors"), key=lambda p: p.stat().st_size)
    assert shards, f"no safetensors in {dest}"
    shards[-1].unlink()
    expect_rejected("ftw_missing_shard", serve_args(broken, "qwen36"), gpu)


@pytest.mark.parametrize("which", ["qwen36", "qwen36-d4-random", "dsv4"])
def test_ftw_tp2_matches_tp1(which):
    gpus = need_tp2()
    gpu = need_gpu()
    dest = converted(which)["dest"]
    with Server(f"ftw_tp1_{which}", serve_args(dest, which), gpu) as s:
        tp1 = run_prompts(s)
    with Server(f"ftw_tp2_{which}", serve_args(dest, which, None, "--tensor-parallel-size", "2"),
                gpus) as s:
        tp2, ranks = run_prompts(s), experts(s.status())["ranks"]
    assert sorted(r["rank"] for r in ranks) == [0, 1]
    assert len({str(r["device"]) for r in ranks}) == 2
    cross_path(tp1, tp2, f"{which} FTW TP1 vs TP2", task=which in ("qwen36", "dsv4"))


def test_gptoss_native_without_base_bias_rejected():
    """Native NoWAG has no bias; a missing BASE expert bias must not be read as zero."""
    gpu = need_gpu()
    base, side = sources("gptoss")
    work = need_scratch() / "gptoss-nobias"
    shutil.rmtree(work, ignore_errors=True)
    copy = S.link_copy(base, work, mutable=("model.safetensors.index.json",))
    index_path = copy / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    dropped = {k for k in index["weight_map"] if ".experts." in k and k.endswith("_bias")}
    assert dropped
    for shard in sorted({index["weight_map"][k] for k in dropped}):
        with safe_open(str(base / shard), "pt") as f:
            keep = {k: f.get_tensor(k) for k in f.keys() if k not in dropped}
        (copy / shard).unlink()
        save_file(keep, str(copy / shard), metadata={"format": "pt"})
    index["weight_map"] = {k: v for k, v in index["weight_map"].items() if k not in dropped}
    index_path.write_text(json.dumps(index))
    expect_rejected("gptoss_no_bias", serve_args(copy, "gptoss", side), gpu)
