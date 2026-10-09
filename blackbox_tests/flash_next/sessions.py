"""Server configurations. One pytest module per entry; pairs differ only in the dimensions under test."""
from . import env

LEGACY_GRAPH = ["--batching-policy", "legacy", "--cuda-graph-max-bs", "4", "--max-prefill-length", "1024"]
LAYERED_EAGER = ["--batching-policy", "layered-pipeline", "--cuda-graph-max-bs", "0"]
REPLAY = ["--enable-gdn-replayssm", "--gdn-replay-buffer-len", "8"]  # 8 records: decode loops it many times


def cand(name, extra, dense, kv, batching, graph, replay, radix, host, model=None):
    args = env.BASE + ["--model", model or env.MODEL] + extra + REPLAY * replay + [
        "--cache-type", "radix" if radix else "naive", "--prefix-cache-host-gib", str(host)]
    return dict(name=name, args=args, dense=dense, kv=kv, batching=batching, graph=graph, replay=replay,
                radix=radix, host=host)


def session(key):
    return {
        # upstream image freetoken:555efd8, same weights: output-quality reference only
        "R": dict(name="R_reference", args=env.REF_BASE + ["--model", env.MODEL], reference=True, radix=True),
        "A": cand("A_legacy_auto_graph", LEGACY_GRAPH, "bf16", "bf16", "legacy", True, False, True, 0),
        "B": cand("B_layered_auto_eager_replay_host_renamed", LAYERED_EAGER, "bf16", "bf16", "layered", False,
                  True, True, 16, model=env.renamed_model()),
        "C": cand("C_legacy_fp8_int8_graph_replay_host", LEGACY_GRAPH + ["--dense-quant", "fp8", "--kv-dtype",
                  "int8"], "fp8", "int8", "legacy", True, True, True, 16),
        "D": cand("D_layered_fp8_int8_eager_naive", LAYERED_EAGER + ["--dense-quant", "fp8", "--kv-dtype", "int8"],
                  "fp8", "int8", "layered", False, False, False, 0),
        "E": cand("E_legacy_bf16_int8_graph", LEGACY_GRAPH + ["--dense-quant", "bf16", "--kv-dtype", "int8"],
                  "bf16", "int8", "legacy", True, False, True, 0),
        "F": cand("F_legacy_fp8_kvbf16_graph", LEGACY_GRAPH + ["--dense-quant", "fp8", "--kv-dtype", "bf16"],
                  "fp8", "bf16", "legacy", True, False, True, 0),
    }[key]


# (left, right, what differs) — same precision, different execution paths
PATH_PAIRS = [("A", "B", "legacy/layered, Graph/eager, Replay off/on, host 0/16, chunk 1024/default, dir name"),
              ("C", "D", "legacy/layered, Graph/eager, Replay on/off, radix/naive, host 16/0, chunk 1024/default")]
# (baseline, quantized, what is switched on)
QUANT_PAIRS = [("A", "E", "int8 KV"), ("A", "F", "fp8 dense"), ("E", "C", "fp8 dense under int8 KV"),
               ("F", "C", "int8 KV under fp8 dense")]
