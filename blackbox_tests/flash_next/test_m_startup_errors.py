"""Public startup errors (contract 7): broken vocabulary input and explicitly requested QSA SD.

Each case launches the candidate and must fail before ready with a readable reason in its log.
"""
import json
import os
import re
import struct

import pytest

from . import env
from .server import expect_startup_error
from .sessions import LEGACY_GRAPH

ARGS = env.BASE + LEGACY_GRAPH


def linked_copy(name, skip):
    """Symlink every model file except `skip` into WORK/broken/<name>."""
    dst = os.path.join(env.WORK, "broken", name)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(env.MODEL):
        if f.startswith(".") or f in skip or os.path.lexists(os.path.join(dst, f)):
            continue
        os.symlink(os.path.join(env.MODEL, f), os.path.join(dst, f))
    return dst


def drop_tensor(src, dst, tensor):
    """Rewrite a safetensors file without one tensor (data of the others copied unchanged)."""
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        meta = hdr.pop("__metadata__", None)
        assert tensor in hdr, tensor
        hdr.pop(tensor)
        new, off, pieces = {}, 0, []
        for k, v in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = v["data_offsets"]
            new[k] = dict(v, data_offsets=[off, off + b - a])
            pieces.append((base + a, b - a))
            off += b - a
        if meta is not None:
            new["__metadata__"] = meta
        h = json.dumps(new).encode()
        h += b" " * (-len(h) % 8)
        with open(dst + ".tmp", "wb") as out:
            out.write(struct.pack("<Q", len(h)) + h)
            for start, size in pieces:
                f.seek(start)
                while size:
                    chunk = f.read(min(size, 1 << 26))
                    out.write(chunk)
                    size -= len(chunk)
    os.replace(dst + ".tmp", dst)


def check(name, args, pattern):
    state, secs, log = expect_startup_error(name, args)
    tail = log[-2500:]
    assert state == "error", f"expected a startup error before ready, got {state} after {secs:.0f}s\n{tail}"
    assert re.search(pattern, log, re.I), f"no readable reason matching {pattern!r}\n{tail}"


def test_missing_vocabulary_shard():
    d = linked_copy("missing-ple-shard", {"model-plefp8-00003.safetensors"})
    check("M_missing_ple_shard", ARGS + ["--model", d], r"plefp8-00003|missing|not found|no such file")


def test_missing_vocabulary_scale():
    shard, idx = "model-plefp8-00009.safetensors", "model.safetensors.index.json"
    with open(os.path.join(env.MODEL, idx)) as f:
        index = json.load(f)
    scale = [k for k in index["weight_map"] if k.endswith("ngram_embedding.weight_scale")]
    assert len(scale) == 1 and index["weight_map"][scale[0]] == shard, scale
    d = linked_copy("missing-ple-scale", {shard, idx})
    if not os.path.exists(os.path.join(d, shard)):
        drop_tensor(os.path.join(env.MODEL, shard), os.path.join(d, shard), scale[0])
    index["weight_map"].pop(scale[0])
    with open(os.path.join(d, idx), "w") as f:
        json.dump(index, f)
    check("M_missing_ple_scale", ARGS + ["--model", d], r"scale")


@pytest.mark.parametrize("steps", ["4"])
def test_explicit_qsa_speculation_rejected(steps):
    args = [a for a in ARGS] + ["--model", env.MODEL]
    i = args.index("--speculative-num-steps")
    args[i + 1] = steps
    check("M_explicit_sd", args, r"specul|draft|\bSD\b|QSA|not support|unsupported")
