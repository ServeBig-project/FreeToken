"""FTW conversion (`ft checkpoint --nowag-expert-path`): round trip, source isolation, rename,
missing data, bias carried, TP2. Needs GPU approval and NOWAG_SCRATCH (multi-GB outputs).

Round trips delete their input symlink copies. Complete source isolation is checked by
test_ftw_with_original_sources_absent inside the coordinator's separate container.
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
from harness import (LOG_DIR, Server, expect_rejected, experts, ft, run_prompts, same_execution,  # noqa: E402
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
    if which.startswith("tiny-qwen3-"):
        import tiny_model as tiny
        return tiny.paths(int(which[-1]))
    if which == "qwen36":
        return need_path(QWEN36_BASE, "Qwen3.6 base"), need_path(QWEN36_SIDE, "Qwen3.6 sidecar")
    if which == "dsv4":
        return need_path(DSV4_BASE, "DSV4 base"), need_path(DSV4_SIDE, "DSV4 sidecar")
    if which.startswith("qwen36-"):
        return need_path(QWEN36_BASE, "Qwen3.6 base"), A.sidecar_dir(which)
    return need_path(GPTOSS_BASE, "GPT-OSS base"), A.sidecar_dir("gptoss-d6-random")


def serve_args(model, which, side=None, *extra):
    if which.startswith("tiny-qwen3-"):
        import tiny_model as tiny
        args = tiny.serve_args(model, side)
        return args + list(extra)
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
def test_ftw_roundtrip_matches_native(which):
    """Native and FTW carry the same compressed weights; for GPT-OSS the native run reads the
    expert biases from BASE, so a FTW that dropped them diverges from it."""
    gpu = need_gpu()
    info = converted(which)
    task = which in ("qwen36", "dsv4")                        # only calibrated real weights
    with Server(f"ftw_native_{which}", serve_args(info["base"], which, info["side"]), gpu) as s:
        native, native_status = run_prompts(s), experts(s.status())
    (LOG_DIR / f"ftw-native-reference-{which}.json").write_text(json.dumps(
        {"which": which, "outputs": native, "status": native_status, "task": task,
         "source_paths": [str(info["base"]), str(info["side"])]}, indent=2))
    with Server(f"ftw_{which}", serve_args(info["dest"], which), gpu) as s:
        ftw, ftw_status = run_prompts(s), experts(s.status())
    assert ftw_status["format"] == native_status["format"]
    assert ftw_status["format_parameters"] == native_status["format_parameters"]
    cross_path(native, ftw, f"{which} native vs FTW", task=task)


def test_ftw_with_original_sources_absent():
    """The coordinator runs this inside a container mounting only DEST, code and environment.
    The small reference JSON comes from the native/FTW round-trip run on the host."""
    gpu = need_gpu()
    dest = need_path(os.environ.get("NOWAG_ISOLATED_FTW"), "isolated FTW directory")
    reference_path = need_path(os.environ.get("NOWAG_ISOLATED_REFERENCE"), "native reference JSON")
    reference = json.loads(reference_path.read_text())
    assert all(not Path(path).exists() for path in reference["source_paths"]), \
        "BASE and SIDE must be absent from the serving process's filesystem"
    with Server("ftw_isolated", serve_args(dest, reference["which"]), gpu) as server:
        outputs, block = run_prompts(server), experts(server.status())
    assert block["format"] == reference["status"]["format"]
    assert block["format_parameters"] == reference["status"]["format_parameters"]
    cross_path(reference["outputs"], outputs, "source-isolated FTW", task=reference["task"])


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
    # FTW = freetoken-NNNNN.ftw shards + freetoken_weight.json index + copied non-weight files
    shards = sorted(broken.glob("freetoken-*.ftw"), key=lambda p: p.stat().st_size)
    assert shards, f"no freetoken-*.ftw shards in {dest}"
    shards[-1].unlink()
    expect_rejected("ftw_missing_shard", serve_args(broken, "qwen36"), gpu)


@pytest.mark.parametrize("which", ["tiny-qwen3-d4", "tiny-qwen3-d6"])
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


def base_copy_without(base, work, drop):
    """Symlink copy of an HF BASE whose shards no longer contain the tensors `drop` selects."""
    shutil.rmtree(work, ignore_errors=True)
    copy = S.link_copy(base, work, mutable=("model.safetensors.index.json",))
    index_path = copy / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    dropped = {k for k in index["weight_map"] if drop(k)}
    assert dropped
    for shard in sorted({index["weight_map"][k] for k in dropped}):
        with safe_open(str(base / shard), "pt") as f:
            keep = {k: f.get_tensor(k) for k in f.keys() if k not in dropped}
        (copy / shard).unlink()
        save_file(keep, str(copy / shard), metadata={"format": "pt"})
    index["weight_map"] = {k: v for k, v in index["weight_map"].items() if k not in dropped}
    index_path.write_text(json.dumps(index))
    return copy


def expert_bias(key, layer=None):
    hit = ".experts." in key and key.endswith("_bias")
    return hit and (layer is None or f".layers.{layer}." in key)


def test_gptoss_native_without_base_bias_rejected():
    """Native NoWAG has no bias; a missing BASE expert bias must not be read as zero."""
    gpu = need_gpu()
    base, side = sources("gptoss")
    copy = base_copy_without(base, need_scratch() / "gptoss-nobias", expert_bias)
    expect_rejected("gptoss_no_bias", serve_args(copy, "gptoss", side), gpu)


def test_gptoss_one_layer_without_bias_rejected_native_and_converted():
    """Only the last layer's expert biases are missing from BASE: the native load must refuse,
    and conversion must either refuse or yield a FTW that the server refuses before ready."""
    gpu = need_gpu()
    base, side = sources("gptoss")
    last = json.loads((base / "config.json").read_text())["num_hidden_layers"] - 1
    copy = base_copy_without(base, need_scratch() / "gptoss-nobias-last-layer",
                             lambda k: expert_bias(k, last))
    expect_rejected("gptoss_no_bias_last_layer", serve_args(copy, "gptoss", side), gpu)
    dest = need_scratch() / "ftw-gptoss-nobias-last-layer"
    shutil.rmtree(dest, ignore_errors=True)
    proc = ft(["checkpoint", "--model", copy, "--nowag-expert-path", side, "--out", dest,
               "--moe-backend", "offload", "--gpu", gpu], label="ftw_convert_gptoss_nobias_last")
    if proc.returncode == 0:
        expect_rejected("ftw_gptoss_no_bias_last_layer", serve_args(dest, "gptoss"), gpu)


def ftw_with_index(dest, work, edit):
    """FTW copy (shards symlinked) whose freetoken_weight.json index went through `edit`."""
    copy = S.link_copy(dest, work, mutable=("freetoken_weight.json",))
    path = copy / "freetoken_weight.json"
    index = json.loads(path.read_text())
    edit(index)
    path.write_text(json.dumps(index))
    return copy


def test_gptoss_ftw_rewritten_index_still_serves(tmp_path):
    """Control for the bias-group rows: re-serialising the index alone is accepted, so their
    rejections come from the missing bias, not from touching the index."""
    gpu = need_gpu()
    copy = ftw_with_index(converted("gptoss")["dest"], tmp_path / "ftw", lambda index: None)
    with Server("ftw_gptoss_index_control", serve_args(copy, "gptoss"), gpu) as s:
        assert all(o.strip() for o in run_prompts(s))


@pytest.mark.parametrize("group", ["gate", "up", "down"])
def test_gptoss_ftw_missing_one_bias_group_rejected(tmp_path, group):
    gpu = need_gpu()

    def drop(index):
        names = {t["name"] for t in index["tensors"]
                 if t["kind"] == "experts_bank" and t["name"].split("#")[0] == f"{group}_bias"}
        assert names, f"no {group}_bias bank in the FTW index"
        index["tensors"] = [t for t in index["tensors"] if t["name"] not in names]
        index["counts"]["experts_bank"] -= len(names)
    copy = ftw_with_index(converted("gptoss")["dest"], tmp_path / "ftw", drop)
    expect_rejected(f"ftw_gptoss_no_{group}_bias", serve_args(copy, "gptoss"), gpu)
