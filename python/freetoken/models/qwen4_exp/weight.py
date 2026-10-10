"""Qwen3.8-Flash-Next checkpoint reader (NVFP4 routed experts, bf16 everything else).

Three paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, prefix-stripped, fused where the
  model expects one buffer, and quantized to the resolved dense precision.
* :func:`load_ple_table` -- the FP8 n-gram table, ``split_ngram_parts`` checkpoint shards
  concatenated into one pinned :class:`HostBank`; :func:`ftw_side_files` copies those shards
  next to an FTW checkpoint so the converted directory serves on its own.
* :func:`load_nvfp4_expert_sources` -- the routed NVFP4 experts, through the common bank reader.

Dropped: ``mtp.*`` (speculative head) and ``model.visual.*`` (served text-only).
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    load_nvfp4_expert_source_banks_parallel,
)
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.quant.dense import plan_dense, quant_fp8_per_row
from freetoken.utils import download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

# Routed NVFP4 experts (modelopt layout): per-expert, un-fused. The ``model.language_model.``
# anchor excludes the MTP head's ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# Per-tensor modelopt scales: consumed with their expert ``.weight`` by the bank reader.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$")
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_ST_DTYPE = "F8_E4M3"
_PLE_SIDE_FILE = "ple-table-{:05d}.safetensors"
_PLE_SIDE_FILE_BYTES = 4 << 30

# Checkpoint parts concatenated along dim 0 into one model buffer, in this order.
_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj.weight": (
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ),
    ".linear_attn.in_proj.weight": (
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ),
    ".mlp.shared_expert.gate_up_proj.weight": (
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ),
    # per-layer hyper-connections only: the top-level mixer has no block_inject and stays unfused
    ".input_mix_weight_down_block_inject.weight": (
        ".input_mix_weight_down.weight", ".block_inject_weight.weight",
    ),
}
_HC_WITH_INJECT = (".attn_hyper_connection.", ".mlp_hyper_connection.")
# Operator role of the 2-D weights, for the public dense plan: everything else is a projection.
_ROLES = {"embed_tokens.weight": "embedding", ".mlp.gate.weight": "router",
          ".mlp.shared_expert_gate.weight": "router"}


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith(("mtp.", "model.visual.")):
        return None
    if _PLE_TABLE_INFIX in raw_name or _EXPERT_RE.search(raw_name):
        return None  # n-gram table: load_ple_table; routed experts: the offload banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model."):]
    return raw_name


def _try_fuse(name: str, tensor: torch.Tensor, buf: dict,
              fusions: dict[str, tuple[str, ...]] = _FUSIONS) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; the merged ``(name, tensor)`` once every part arrived, ``()``
    while incomplete, ``None`` if ``name`` is not a part. The hyper-connection merge pads to a
    multiple of 16 rows with zeros (vLLM's skinny-GEMM alignment)."""
    for fused_suffix, parts in fusions.items():
        for idx, part in enumerate(parts):
            if not name.endswith(part):
                continue
            if fused_suffix.startswith(".input_mix") and not any(h in name for h in _HC_WITH_INJECT):
                return None
            key = name[: -len(part)] + fused_suffix
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(parts):
                return ()
            del buf[key]
            rows = [slots[i] for i in range(len(parts))]
            if fused_suffix.startswith(".input_mix"):
                pad = (-sum(t.shape[0] for t in rows)) % 16
                rows.append(torch.zeros(pad, rows[0].shape[1], dtype=rows[0].dtype, device=rows[0].device))
            return key, torch.cat(rows, dim=0)
    return None


def _weight_map(model_path: str) -> tuple[str, dict[str, str]]:
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as fh:
        return folder, json.load(fh)["weight_map"]


def _emit(name: str, tensor: torch.Tensor, dense_precision: str):
    """One model buffer: a projection under the fp8 plan becomes ``.weight`` (fp8) plus its
    per-row ``.weight_scale``. Per-row scales commute with the output-row fusion above, so
    quantizing the fused matrix equals quantizing each part; zero pad rows quantize to zero."""
    role = next((r for suffix, r in _ROLES.items() if name.endswith(suffix)), "projection")
    if tensor.dim() == 2 and tensor.is_floating_point() and plan_dense(dense_precision, role) == "fp8":
        q, scale = quant_fp8_per_row(tensor)
        yield name, q
        yield name[: -len(".weight")] + ".weight_scale", scale
    else:
        yield name, tensor


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    dense_precision: str = "source",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers,
    at the resolved ``dense_precision`` (``freetoken.quant.dense``). Zero-centered norms are
    yielded RAW (the model applies ``1 + w`` at runtime). The routed experts are NVFP4 and
    always come from the offload banks, so ``include_moe_experts`` never yields anything."""
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return
    folder, weight_map = _weight_map(model_path)
    files = sorted({shard for name, shard in weight_map.items() if _rename(name) is not None})
    fuse_buf: dict = {}
    for file in tqdm(files, desc="Loading weights", disable=not get_tp_info().is_primary()):
        with safetensors.safe_open(os.path.join(folder, file), framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                tensor = f.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is None:
                    yield from _emit(name, tensor, dense_precision)
                elif fused != ():
                    yield from _emit(*fused, dense_precision)
    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"


# The MTP layer keeps q, k and v as separate projections (its history update runs k and v alone).
_MTP_FUSIONS = {k: v for k, v in _FUSIONS.items() if k != ".self_attn.qkv_proj.weight"}


def iter_mtp_weights(model_path: str, config, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The ``mtp.*`` tensors as ``Qwen4ExpMTP`` buffers, BF16 as stored: ``layers.0`` is the
    module's ``layer``, and the indexer's merged ``index_qk_proj`` splits into its q and k
    projections."""
    args = config.qwen4_args
    folder, weight_map = _weight_map(model_path)
    files = sorted({shard for name, shard in weight_map.items() if name.startswith("mtp.")})
    if not files:
        raise ValueError(f"{model_path} has no mtp.* tensors for native MTP drafting")
    index_split = [args.index_n_heads * args.index_head_dim, args.index_head_dim]
    fuse_buf: dict = {}
    for file in files:
        with safetensors.safe_open(os.path.join(folder, file), framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                if not raw_name.startswith("mtp."):
                    continue
                name = raw_name[len("mtp."):].replace("layers.0.", "layer.", 1)
                tensor = f.get_tensor(raw_name)
                if name.endswith(".indexer.index_qk_proj.weight"):
                    q, k = tensor.split(index_split, dim=0)
                    yield name.replace("index_qk_proj", "index_q_proj"), q
                    yield name.replace("index_qk_proj", "index_k_proj"), k
                    continue
                fused = _try_fuse(name, tensor, fuse_buf, _MTP_FUSIONS)
                if fused is None:
                    yield name, tensor
                elif fused != ():
                    yield fused
    assert not fuse_buf, f"Incomplete MTP fusions: {sorted(fuse_buf)}"


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's scalar FP8 scale."""

    bank: HostBank
    weight_scale: torch.Tensor


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(model_path: str) -> list[str]:
    """Shards holding a piece of the n-gram table: from the index of an HF checkpoint, or the
    ``ple-table-*.safetensors`` side files of an FTW directory."""
    folder = download_hf_weight(model_path)
    if os.path.exists(os.path.join(folder, "model.safetensors.index.json")):
        _, weight_map = _weight_map(folder)
        return sorted({os.path.join(folder, s) for n, s in weight_map.items() if _PLE_TABLE_INFIX in n})
    return sorted(glob.glob(os.path.join(folder, _PLE_SIDE_FILE.format(0)[:-len("00000.safetensors")] + "*.safetensors")))


def ftw_side_files(model_path: str, out_dir: str) -> list[str]:
    """Write the n-gram table tensors (and their scale), and only those, into ``ple-table-*``
    safetensors files next to an FTW checkpoint; returns the file names written."""
    from safetensors.torch import save_file

    written: list[str] = []
    batch: dict[str, torch.Tensor] = {}
    size = 0

    def flush() -> None:
        nonlocal batch, size
        if batch:
            name = _PLE_SIDE_FILE.format(len(written))
            save_file(batch, os.path.join(out_dir, name))
            written.append(name)
            batch, size = {}, 0

    for path in _ple_table_files(model_path):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _PLE_TABLE_INFIX not in key:
                    continue
                t = f.get_tensor(key)
                batch[key] = t
                size += t.numel() * t.element_size()
                if size >= _PLE_SIDE_FILE_BYTES:
                    flush()
    flush()
    return written


def load_ple_table(model_path: str, qwen4_args, *, workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned bank.

    The table is split into ``split_ngram_parts`` equal row blocks named by shard index, so
    the bank is filled at ``shard_index * rows_per_shard`` regardless of file order. Reads are
    O_DIRECT: a 47.7 GiB table must not also sit in the page cache next to the bank.
    """
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    for path in _ple_table_files(model_path):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")
            if rows and tuple(meta["shape"]) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {meta['shape']}, expected {[rows, cols]}")
            rows, cols = meta["shape"]
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}")
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")

    bank = HostBank((expected * rows, cols), torch.float8_e4m3fn)
    shard_bytes = rows * cols
    bar = byte_bar(expected * shard_bytes, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
            read_range_into(buf, path, file_offset=offset, nbytes=nbytes,
                            dest_offset=shard * shard_bytes, workers=workers, chunk=chunk)
            bar.update(nbytes)
    finally:
        bar.close()
    if torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None) -> dict[str, torch.Tensor]:
    return load_nvfp4_expert_source_banks(
        model_path, config, _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache, primary=get_tp_info().is_primary(), layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    return load_nvfp4_expert_source_banks_parallel(
        model_path, config, _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache, primary=get_tp_info().is_primary(),
        workers=workers, chunk=chunk, layer_sink=layer_sink,
    )


__all__ = [
    "PleTable",
    "ftw_side_files",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
]
