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

    gate_words = _words(hidden, group_size, assignment_bits)
    down_words = _words(intermediate, group_size, assignment_bits)
    specs = {
        "gate_assignments": ((experts, gate_words, intermediate), torch.int32),
        "gate_input_norm": ((experts, hidden), dtype),
        "gate_output_norm": ((experts, intermediate), dtype),
        "up_assignments": ((experts, gate_words, intermediate), torch.int32),
        "up_input_norm": ((experts, hidden), dtype),
        "up_output_norm": ((experts, intermediate), dtype),
        "down_assignments": ((experts, down_words, hidden), torch.int32),
        "down_input_norm": ((experts, intermediate), dtype),
        "down_output_norm": ((experts, hidden), dtype),
    }

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
                            target = sources[f"{bank}_{kind}"][layer][expert]
                            loaded = handle.get_tensor(key)
                            expected_shape = tuple(target.shape)
                            if kind == "assignments" and source_assignment_layout == "row_major":
                                expected_shape = (target.shape[1], target.shape[0])
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
                                target.copy_(loaded.transpose(0, 1))
                            else:
                                target.copy_(loaded)
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
    return sources, {"codebook": codebook}, NowagState(group_size, assignment_bits)
