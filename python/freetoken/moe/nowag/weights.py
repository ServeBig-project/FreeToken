"""Read NoWAG v1 expert-only weights (native sidecar) into pinned host banks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


FORMAT = "nowag_expert_sidecar_v1"
LEGACY_DSV4_FORMAT = "deepseek_v4_nowag_expert_sidecar_v1"
CODEBOOK_KEY = "global_all.codebook"
RUNTIME_ASSIGNMENT_LAYOUT = "word_major"

# The files keep the projection names used by the first DSV4 quantizer.  Their
# meaning is model-independent: w1 is gate, w3 is up, and w2 is down.
_PROJECTION_BANK = {"w1": "gate", "w3": "up", "w2": "down"}


@dataclass(frozen=True)
class NowagState:
    """Encoding parameters the NoWAG method interprets; opaque to common code."""

    d: int
    assignment_bits: int
    # This TP rank's Down input width and the lanes of its first (global) codeword
    # that belong to the previous rank.
    intermediate_size: int
    down_start_lane: int = 0


def expert_intermediate_range(model_config, rank: int, size: int) -> tuple[int, int]:
    """This rank's slice of the expert intermediate axis: the model's own partition if it
    declares one, else the contiguous chunks the dense loader uses."""
    from freetoken.moe.expert_banks import _model_hook

    width = int(model_config.moe_intermediate_size)
    model_range = _model_hook(model_config, "expert_intermediate_range")
    if model_range is not None:
        return model_range(width, rank=rank, world_size=size)
    per_rank = -(-width // size)
    start = min(rank * per_rank, width)
    return start, min(start + per_rank, width)


@dataclass(frozen=True)
class _Shard:
    """This TP rank's view of the global expert encoding."""

    start: int
    end: int
    first_group: int
    last_group: int
    rank: int

    @classmethod
    def of(cls, model_config, d: int) -> "_Shard":
        from freetoken.distributed import get_tp_info

        tp = get_tp_info()
        start, end = expert_intermediate_range(model_config, tp.rank, tp.size)
        return cls(start, end, start // d, -(-end // d), tp.rank)

    def state(self, d: int, bits: int) -> NowagState:
        return NowagState(d, bits, self.end - self.start, self.start - self.first_group * d)

    def apply(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Gate/Up keep their output rows, Down its input lanes (global codeword groups)."""
        if name == "down_assignments":
            return _regroup(tensor, self.first_group, self.last_group)
        if name in _INTERMEDIATE_LAST:
            return tensor[..., self.start:self.end]
        return tensor


# Banks whose last axis is the expert intermediate axis.
_INTERMEDIATE_LAST = (
    "gate_assignments", "up_assignments", "gate_output_norm", "up_output_norm",
    "down_input_norm", "gate_bias", "up_bias",
)


def _regroup(words: torch.Tensor, first: int, last: int) -> torch.Tensor:
    """Re-pack the 12-bit ids of global codeword groups ``[first, last)`` of word-major
    ``[..., W, N]`` assignments from bit 0, keeping the global grouping."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    w = torch.cat((w, torch.zeros_like(w[..., :1, :])), dim=-2)
    out = torch.zeros(
        (*w.shape[:-2], -(-(last - first) * 12 // 32) + 1, w.shape[-1]), dtype=torch.int64
    )
    for k, group in enumerate(range(first, last)):
        word, shift = divmod(group * 12, 32)
        ids = ((w[..., word, :] >> shift) | (w[..., word + 1, :] << (32 - shift))) & 0xFFF
        word, shift = divmod(k * 12, 32)
        out[..., word, :] |= (ids << shift) & 0xFFFFFFFF
        if shift + 12 > 32:
            out[..., word + 1, :] |= ids >> (32 - shift)
    out = out[..., :-1, :]
    return torch.where(out >= 1 << 31, out - (1 << 32), out).to(torch.int32)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _words(width: int, group_size: int, assignment_bits: int) -> int:
    groups = math.ceil(width / group_size)
    return math.ceil(groups * assignment_bits / 32)


def _tensor_key(layer: int, expert: int, projection: str, suffix: str) -> str:
    return f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"


def _validate_manifest_model(
    manifest: dict[str, Any],
    manifest_path: Path,
    model_type: str,
    *,
    layers: int,
    experts: int,
    hidden: int,
    intermediate: int,
) -> None:
    output_format = manifest.get("format")
    if output_format == LEGACY_DSV4_FORMAT:
        if model_type != "deepseek_v4":
            raise ValueError(
                f"{manifest_path}: legacy DeepSeek-V4 NoWAG weights cannot serve "
                f"{model_type}"
            )
        return
    if output_format != FORMAT:
        raise ValueError(f"{manifest_path}: unsupported NoWAG output format")
    if manifest.get("model_type") != model_type:
        raise ValueError(
            f"{manifest_path}: NoWAG weights are for model_type "
            f"{manifest.get('model_type')!r}, expected {model_type!r}"
        )
    expected_dims = {
        "num_moe_layers": layers,
        "num_experts": experts,
        "hidden_size": hidden,
        "moe_intermediate_size": intermediate,
    }
    for name, expected in expected_dims.items():
        if int(manifest.get(name, -1)) != expected:
            raise ValueError(
                f"{manifest_path}: {name} must be {expected}, got {manifest.get(name)!r}"
            )


def load_nowag_expert_sources(
    output_path: str | Path,
    model_config,
    *,
    dtype: torch.dtype = torch.bfloat16,
    layer_sink=None,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, torch.Tensor], NowagState]:
    """Load and pin the nine per-expert banks plus one model-wide codebook.

    With ``layer_sink``, each completed layer's unpinned host banks go to
    ``layer_sink(layer, {name: HostBank})`` instead, which owns (and may release) them.

    Assignment banks are always returned as contiguous ``[E, W, N]`` tensors.
    Existing sidecars store each expert as ``[N, W]`` and are transposed while
    copying into the final pinned bank.  A sidecar that declares
    ``assignment_layout="word_major"`` is copied directly, so a quantizer can
    write the runtime layout and avoid the transpose altogether.
    """
    if dtype != torch.bfloat16:
        raise ValueError("NoWAG expert serving currently requires bfloat16")

    root = Path(output_path).resolve()
    manifest_path = root / "manifest.json" if root.is_dir() else root
    root = manifest_path.parent
    manifest = _read_json(manifest_path)

    layers = int(model_config.num_moe_layers)
    experts = int(model_config.num_experts)
    hidden = int(model_config.hidden_size)
    intermediate = int(model_config.moe_intermediate_size)
    _validate_manifest_model(
        manifest,
        manifest_path,
        model_config.model_type,
        layers=layers,
        experts=experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    if manifest.get("scope") != "expert_only" or manifest.get("codebook_sharing") != "global_all":
        raise ValueError("NoWAG runtime requires expert-only quantization with one global codebook")
    group_size = int(manifest.get("d", 0))
    assignment_bits = int(manifest.get("assignment_bits", 0))
    if group_size not in (4, 6) or assignment_bits != 12:
        raise ValueError("NoWAG GPU runtime supports only D4/B12 or D6/B12")
    if manifest.get("assignments_packed") is not True:
        raise ValueError("NoWAG runtime requires packed assignments")
    source_assignment_layout = manifest.get("assignment_layout", "row_major")
    if source_assignment_layout not in ("row_major", RUNTIME_ASSIGNMENT_LAYOUT):
        raise ValueError(
            "NoWAG assignment_layout must be 'row_major' or 'word_major'"
        )
    if int(manifest.get("matrix_count", -1)) != layers * experts * 3:
        raise ValueError("NoWAG output does not cover every routed expert matrix")

    layer_entries = manifest.get("layers")
    if not isinstance(layer_entries, list) or len(layer_entries) != layers:
        raise ValueError("NoWAG output has the wrong number of expert layers")
    files_by_layer: dict[int, Path] = {}
    for entry in layer_entries:
        if not isinstance(entry, dict):
            raise TypeError("NoWAG layer entry must be a JSON object")
        layer = int(entry["layer"])
        file = entry.get("file")
        if not isinstance(file, str):
            raise TypeError("NoWAG layer file must be a string")
        files_by_layer[layer] = root / file
    # Layer numbers are decoder layers; models with leading dense layers start
    # their MoE layers at first_k_dense_replace.
    first = int(getattr(model_config, "first_k_dense_replace", 0))
    if set(files_by_layer) != set(range(first, first + layers)):
        raise ValueError(
            f"NoWAG layers must be the MoE decoder layers [{first}, {first + layers})"
        )

    shard = _Shard.of(model_config, group_size)
    local = shard.end - shard.start
    gate_words = _words(hidden, group_size, assignment_bits)
    down_words = _words(intermediate, group_size, assignment_bits)
    local_down_words = -(-(shard.last_group - shard.first_group) * assignment_bits // 32)
    specs = {
        "gate_assignments": ((experts, gate_words, local), torch.int32),
        "gate_input_norm": ((experts, hidden), dtype),
        "gate_output_norm": ((experts, local), dtype),
        "up_assignments": ((experts, gate_words, local), torch.int32),
        "up_input_norm": ((experts, hidden), dtype),
        "up_output_norm": ((experts, local), dtype),
        "down_assignments": ((experts, local_down_words, hidden), torch.int32),
        "down_input_norm": ((experts, local), dtype),
        "down_output_norm": ((experts, hidden), dtype),
    }
    full_shapes = {
        "gate_assignments": (gate_words, intermediate),
        "up_assignments": (gate_words, intermediate),
        "down_assignments": (down_words, hidden),
        "gate_output_norm": (intermediate,),
        "up_output_norm": (intermediate,),
        "down_input_norm": (intermediate,),
    }
    sliced = local != intermediate

    from freetoken.moe.host_banks import PinPipeline, alloc_layer_banks

    host_banks = alloc_layer_banks(specs, layers)
    sources = {
        name: [bank.tensor for bank in per_layer]
        for name, per_layer in host_banks.items()
    }
    with PinPipeline() as pins:
        for layer in range(layers):
            path = files_by_layer[first + layer]
            if not path.is_file():
                raise FileNotFoundError(path)
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for expert in range(experts):
                    for projection, bank in _PROJECTION_BANK.items():
                        keys = {
                            "assignments": _tensor_key(
                                first + layer, expert, projection, "assignments"
                            ),
                            "input_norm": _tensor_key(
                                first + layer, expert, projection, "normalizer.norms.0"
                            ),
                            "output_norm": _tensor_key(
                                first + layer, expert, projection, "normalizer.norms.1"
                            ),
                        }
                        missing = [name for name in keys.values() if name not in available]
                        if missing:
                            raise KeyError(f"{path}: missing {missing[0]}")
                        for kind, key in keys.items():
                            name = f"{bank}_{kind}"
                            target = sources[name][layer][expert]
                            loaded = handle.get_tensor(key)
                            expected_shape = full_shapes.get(name, tuple(target.shape))
                            if kind == "assignments" and source_assignment_layout == "row_major":
                                expected_shape = expected_shape[::-1]
                            if tuple(loaded.shape) != expected_shape:
                                raise ValueError(
                                    f"{key}: expected {expected_shape}, "
                                    f"got {tuple(loaded.shape)}"
                                )
                            if kind == "assignments" and loaded.dtype != torch.int32:
                                raise TypeError(f"{key}: assignments must use int32")
                            if kind != "assignments" and loaded.dtype != dtype:
                                raise TypeError(
                                    f"{key}: normalizer must use {dtype}, got {loaded.dtype}"
                                )
                            if kind == "assignments" and source_assignment_layout == "row_major":
                                loaded = loaded.transpose(0, 1)
                            target.copy_(shard.apply(name, loaded) if sliced else loaded)
            layer_banks = {name: per[layer] for name, per in host_banks.items()}
            if layer_sink is not None:
                layer_sink(layer, layer_banks)
            else:
                pins(layer, layer_banks)

    codebook_entry = manifest.get("codebook")
    if not isinstance(codebook_entry, dict):
        raise ValueError("NoWAG output is missing the global codebook")
    codebook_file = codebook_entry.get("file")
    codebook_name = codebook_entry.get("tensor")
    if not isinstance(codebook_file, str) or not isinstance(codebook_name, str):
        raise TypeError("NoWAG codebook file and tensor must be strings")
    if codebook_entry.get("dtype") not in (None, "bfloat16"):
        raise TypeError("NoWAG expert serving requires a bfloat16 codebook")
    with safe_open(root / codebook_file, framework="pt", device="cpu") as handle:
        if codebook_name != CODEBOOK_KEY or codebook_name not in handle.keys():
            raise KeyError(f"NoWAG output is missing {CODEBOOK_KEY}")
        codebook = handle.get_tensor(codebook_name)
    if codebook.dtype != dtype:
        raise TypeError(f"NoWAG codebook must use {dtype}, got {codebook.dtype}")
    codebook = codebook.contiguous()
    expected_codebook_shape = (1 << assignment_bits, group_size)
    if tuple(codebook.shape) != expected_codebook_shape:
        raise ValueError(
            f"NoWAG codebook must be {list(expected_codebook_shape)}, "
            f"got {tuple(codebook.shape)}"
        )
    return sources, {"codebook": codebook}, shard.state(group_size, assignment_bits)


def prepare_ftw_banks(stored_state, model_config):
    """Select this rank's encoding before FTW locks each loaded bank in host memory."""
    stored = NowagState(**stored_state)
    shard = _Shard.of(model_config, stored.d)
    state = shard.state(stored.d, stored.assignment_bits)
    if state.intermediate_size == stored.intermediate_size:
        return state, None

    def transform(name, tensor):
        if name == "down_bias" and shard.rank != 0:
            return None
        return shard.apply(name, tensor)

    return state, transform
