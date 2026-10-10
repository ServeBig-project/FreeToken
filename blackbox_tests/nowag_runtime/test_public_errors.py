"""Public errors of contract §6, reported before the server is ready. Needs GPU approval.

Every invalid input is built from the real public artifacts as a symlink copy with an edited
manifest (or a removed file), so nothing large is written and the originals stay untouched.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sidecar as S  # noqa: E402
from cases import (QWEN36_BASE, QWEN36_SIDE, DSV4_SIDE, DENSE_BASE,  # noqa: E402
                   need_gpu, need_path, need_tp2)
from harness import Server, expect_rejected, run_prompts, same_execution  # noqa: E402

CACHE = os.environ.get("NOWAG_QWEN36_CACHE", "1536")


def qwen_args(side, *extra):
    return ["--model", need_path(QWEN36_BASE, "Qwen3.6 base"), "--nowag-expert-path", side,
            "--moe-backend", "offload", "--moe-cache-size", CACHE, *extra]


@pytest.fixture
def qwen_side():
    return need_path(QWEN36_SIDE, "Qwen3.6 NoWAG sidecar")


def copy(src, tmp_path, name):
    return S.link_copy(src, tmp_path / name)


def test_wrong_model_weights_under_a_matching_name(tmp_path):
    gpu = need_gpu()
    dsv4 = need_path(DSV4_SIDE, "DSV4 sidecar")
    disguised = copy(dsv4, tmp_path, "qwen36_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5")
    expect_rejected("err_wrong_model", qwen_args(disguised), gpu)


def test_layer_file_swapped(tmp_path, qwen_side):
    gpu = need_gpu()
    side = copy(qwen_side, tmp_path, "swapped")

    def swap(m):
        a, b = m["layers"][0], m["layers"][1]
        a["file"], b["file"] = b["file"], a["file"]
        a["index"], b["index"] = b["index"], a["index"]
    S.edit_manifest(side, swap)
    expect_rejected("err_layer_swap", qwen_args(side), gpu)


def test_layer_not_in_model(tmp_path, qwen_side):
    gpu = need_gpu()
    side = copy(qwen_side, tmp_path, "extra_layer")
    S.edit_manifest(side, lambda m: m["layers"][-1].update(layer=1000))
    expect_rejected("err_layer_out_of_model", qwen_args(side), gpu)


def test_partial_layer_coverage(tmp_path, qwen_side):
    """Contract §9: the sidecar must cover exactly every MoE decoder layer."""
    gpu = need_gpu()
    side = copy(qwen_side, tmp_path, "partial")
    S.edit_manifest(side, lambda m: m.update(layers=m["layers"][:-1]))
    expect_rejected("err_partial_layers", qwen_args(side), gpu)


def test_declared_d_disagrees_with_tensors(tmp_path, qwen_side):
    gpu = need_gpu()
    side = copy(qwen_side, tmp_path, "d_mismatch")
    S.edit_manifest(side, lambda m: m.update(d=4))
    expect_rejected("err_d_mismatch", qwen_args(side), gpu)


def test_missing_shared_codebook(tmp_path, qwen_side):
    gpu = need_gpu()
    side = copy(qwen_side, tmp_path, "no_codebook")
    (side / S.manifest(side)["codebook"]["file"]).unlink()
    expect_rejected("err_no_codebook", qwen_args(side), gpu)


def test_missing_layer_file(tmp_path, qwen_side):
    gpu = need_gpu()
    side = copy(qwen_side, tmp_path, "no_layer")
    (side / S.manifest(side)["layers"][5]["file"]).unlink()
    expect_rejected("err_no_layer_file", qwen_args(side), gpu)


def test_model_without_experts(qwen_side):
    gpu = need_gpu()
    dense = need_path(DENSE_BASE, "dense base")
    assert "num_experts" not in json.dumps(json.loads((dense / "config.json").read_text()))
    expect_rejected("err_dense_model",
                    ["--model", dense, "--nowag-expert-path", qwen_side], gpu)


def test_zero_expert_capacity(qwen_side):
    """A cache that cannot hold a single expert cannot run any routed token."""
    gpu = need_gpu()
    args = qwen_args(qwen_side)
    args[args.index("--moe-cache-size") + 1] = "0"
    expect_rejected("err_zero_capacity", args, gpu)


def test_cpu_experts_with_speculation_rejected(qwen_side):
    """Contract §3: SD with all experts on CPU stays rejected; NoWAG must not lift that."""
    gpu = need_gpu()
    args = ["--model", need_path(QWEN36_BASE, "Qwen3.6 base"), "--nowag-expert-path", qwen_side,
            "--moe-backend", "cpu", "--speculative-num-steps", "4"]
    expect_rejected("err_cpu_sd", args, gpu)


def test_tp2_with_speculation_rejected():
    """Contract §3: mainline SD is TP=1 only."""
    import tiny_model as tiny
    gpus = need_tp2()
    base, side = tiny.paths(6)
    expect_rejected("err_tp2_sd", tiny.serve_args(base, side, tp=2) +
                    ["--speculative-num-steps", 4], gpus)


def test_renamed_valid_directory_gives_identical_output(tmp_path, qwen_side):
    gpu = need_gpu()
    renamed = copy(qwen_side, tmp_path, "some_unrelated_name")
    with Server("rename_orig", qwen_args(qwen_side), gpu) as s:
        a = run_prompts(s)
    with Server("rename_new", qwen_args(renamed), gpu) as s:
        b = run_prompts(s)
    same_execution(a, b, "renamed sidecar")
