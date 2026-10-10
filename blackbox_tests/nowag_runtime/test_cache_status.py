"""`/v1/cache/status` geometry.experts (contract §6). Needs GPU approval.

Byte checks use sizes computable from the public artifacts: compressed expert bytes = the
layer tensor files listed in the manifest; a full BF16 expert copy = sum of N*K*2 over the
logical shapes; the shared codebook = 4096*D*2.
"""

import json
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
import sidecar as S  # noqa: E402
from cases import QWEN36_BASE, QWEN36_SIDE, DSV4_BASE, DSV4_SIDE, need_gpu, need_path, need_tp2  # noqa: E402
from harness import Server, experts, run_prompts, same_execution  # noqa: E402

RANK_FIELDS = ["rank", "device", "compute_backend", "kernel_backend", "storage_mode",
               "expert_host_bytes", "expert_device_bytes", "shared_host_bytes",
               "shared_device_bytes", "workspace_device_bytes"]
BYTE_FIELDS = [f for f in RANK_FIELDS if f.endswith("_bytes")]
SMALL, LARGE = (os.environ.get("NOWAG_QWEN36_CACHE_SMALL", "1024"),
                os.environ.get("NOWAG_QWEN36_CACHE_LARGE", "3072"))


def sizes(side):
    m = S.manifest(side)
    compressed = sum((Path(side) / e["file"]).stat().st_size for e in m["layers"])
    bf16 = 0
    for e in m["layers"]:
        index = json.loads((Path(side) / e["index"]).read_text())
        bf16 += sum(v["logical_shape"][0] * v["logical_shape"][1] * 2
                    for v in index["matrices"].values())
    return {"compressed": compressed, "bf16": bf16, "codebook": 4096 * m["d"] * 2, "d": m["d"]}


def check_block(block, d, n_ranks=1):
    assert block["format"] == "nowag"
    assert block["format_parameters"] == {"d": d, "assignment_bits": 12}
    ranks = block["ranks"]
    assert sorted(r["rank"] for r in ranks) == list(range(n_ranks))
    for r in ranks:
        missing = [f for f in RANK_FIELDS if f not in r]
        assert not missing, f"rank fields missing: {missing}"
        for f in BYTE_FIELDS:   # workspace_device_bytes is 0 until shared-runtime reservation (§9)
            assert isinstance(r[f], int) and r[f] >= 0, (f, r[f])
        kb = r["kernel_backend"]
        assert (isinstance(kb, str) and kb) or (isinstance(kb, list) and kb
                                                and all(isinstance(k, str) and k for k in kb))
    return ranks


def qwen(side, cache, *extra):
    return ["--model", need_path(QWEN36_BASE, "Qwen3.6 base"), "--nowag-expert-path", side,
            "--moe-backend", "offload", "--moe-cache-size", cache, *extra]


def test_qwen_offload_fields_bytes_and_read_only():
    gpu = need_gpu()
    side = need_path(QWEN36_SIDE, "Qwen3.6 sidecar")
    size = sizes(side)
    blocks = {}
    for cache in (SMALL, LARGE):
        with Server(f"status_qwen_{cache}", qwen(side, cache), gpu) as s:
            before = run_prompts(s)
            reads = [experts(s.status()) for _ in range(5)]
            after = run_prompts(s)
            same_execution(before, after, "status reads must not change execution")
            assert all(r == reads[0] for r in reads), "status changed between plain reads"
            blocks[cache] = check_block(reads[0], size["d"])[0]
    small, large = blocks[SMALL], blocks[LARGE]
    # one codebook per rank, independent of the number of cache slots
    assert small["shared_device_bytes"] == large["shared_device_bytes"]
    assert size["codebook"] <= small["shared_device_bytes"] < 64 * size["codebook"]
    assert large["expert_device_bytes"] > small["expert_device_bytes"]
    # offload keeps the compressed experts on the host, never a full BF16 copy
    assert small["expert_host_bytes"] >= 0.9 * size["compressed"]
    assert small["expert_host_bytes"] < 0.5 * size["bf16"]


def test_dsv4_offload_fields():
    gpu = need_gpu()
    side = need_path(DSV4_SIDE, "DSV4 sidecar")
    args = ["--model", need_path(DSV4_BASE, "DSV4 base"), "--nowag-expert-path", side,
            "--moe-backend", "offload", "--moe-cache-size", os.environ.get("NOWAG_DSV4_CACHE", "640")]
    with Server("status_dsv4", args, gpu) as s:
        r = check_block(experts(s.status()), 6)[0]
    size = sizes(side)
    assert r["expert_host_bytes"] < 0.5 * size["bf16"]
    assert size["codebook"] <= r["shared_device_bytes"] < 64 * size["codebook"]


def test_non_nowag_model_reports_a_different_format():
    gpu = need_gpu()
    side = need_path(QWEN36_SIDE, "Qwen3.6 sidecar")
    base = need_path(QWEN36_BASE, "Qwen3.6 base")
    with Server("status_plain", ["--model", base, "--moe-backend", "offload",
                                 "--moe-cache-size", SMALL], gpu) as s:
        plain = experts(s.status())
    with Server("status_nowag", qwen(side, SMALL), gpu) as s:
        nowag = experts(s.status())
    assert plain["format"] not in ("", "nowag") and nowag["format"] == "nowag"
    assert plain["format_parameters"] == {} and "ranks" in plain


def test_tp2_one_codebook_per_rank():
    import tiny_model as tiny
    gpus = need_tp2()
    base, side = tiny.paths(6)
    size = sizes(side)
    with Server("status_tp2", tiny.serve_args(base, side, tp=2), gpus) as s:
        ranks = check_block(experts(s.status()), size["d"], n_ranks=2)
    assert len({str(r["device"]) for r in ranks}) == 2
    for r in ranks:
        assert size["codebook"] <= r["shared_device_bytes"] < 64 * size["codebook"]
