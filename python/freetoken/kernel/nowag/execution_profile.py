"""Offline-profiled backend dispatch for NoWag dense and MoE kernels.

The tables in ``execution_profiles/`` are evidence, not tuning
logic.  A serving call must match the complete hardware, wire-format, dtype,
and local shape key before an exact-K48 range is considered.  Everything else
uses the generic Triton implementation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal, Mapping, Sequence

from .moe_activation import (
    DOWN_PROLOGUE_NORM,
    GATE_UP_EPILOGUE_NORM,
    NO_ACTIVATION_ROUNDING,
)


ExecutionBackend = Literal["triton", "cuda_exact_k48"]


_ProfileRange = tuple[int, int, int | None, int | None]

_HARDWARE_KEYS = ("backend", "device_name", "compute_capability")
_FORMAT_KEYS = ("dtype", "group_size", "assignment_bits", "codebook_size")
_DENSE_SHAPE_KEYS = (
    "out_features",
    "in_features",
    "input_group_start_lane",
)
_DENSE_MULTI_SHAPE_KEYS = (
    "output_widths",
    "in_features",
    "input_group_start_lane",
)
_MOE_SHAPE_KEYS = (
    "num_experts",
    "physical_expert_rows",
    "hidden_size",
    "intermediate_size",
    "physical_intermediate_size",
    "pad_down_to_k48",
    "down_input_group_start_lane",
    "assignment_layout",
    "structural_down",
    "top_k",
    "activation_kind",
    "gate_up_input_rounding",
    "swiglu_limit",
    "down_input_rounding",
    "down_norm_placement",
)


def _legacy_moe_shape_defaults(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Fill fields that older execution-profile schemas did not record."""
    shape = dict(raw)
    shape.setdefault("physical_expert_rows", shape.get("num_experts"))
    shape.setdefault("down_input_group_start_lane", 0)
    shape.setdefault("assignment_layout", "word_major")
    shape.setdefault("structural_down", True)
    shape.setdefault("activation_kind", "silu_mul")
    shape.setdefault("gate_up_input_rounding", NO_ACTIVATION_ROUNDING)
    shape.setdefault("swiglu_limit", None)
    shape.setdefault("down_input_rounding", NO_ACTIVATION_ROUNDING)
    shape.setdefault("down_norm_placement", GATE_UP_EPILOGUE_NORM)
    return shape


def _canonical_value(key: str, value: Any) -> Any:
    if key == "compute_capability":
        capability = tuple(int(part) for part in value)
        if len(capability) != 2:
            raise ValueError("compute_capability must contain two integers")
        return capability
    if key == "output_widths":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("output_widths must be a sequence of positive integers")
        try:
            widths = tuple(int(width) for width in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "output_widths must be a sequence of positive integers"
            ) from exc
        if (
            not widths
            or any(
                isinstance(raw, bool) or parsed != raw
                for raw, parsed in zip(value, widths)
            )
            or any(width <= 0 for width in widths)
        ):
            raise ValueError("output_widths must be a sequence of positive integers")
        return widths
    if key in ("pad_down_to_k48", "structural_down"):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a bool")
        return value
    if key == "down_input_group_start_lane":
        value = int(value)
        if not 0 <= value < 6:
            raise ValueError("down_input_group_start_lane must be in [0, 5]")
        return value
    if key == "assignment_layout":
        if value not in ("row_major", "word_major"):
            raise ValueError(
                "assignment_layout must be 'row_major' or 'word_major'"
            )
        return value
    if key == "activation_kind":
        if value != "silu_mul":
            raise ValueError("activation_kind must be 'silu_mul'")
        return value
    if key in ("gate_up_input_rounding", "down_input_rounding"):
        if value not in (
            NO_ACTIVATION_ROUNDING,
            "dynamic_e4m3_per_token_group128_ue8m0",
        ):
            raise ValueError(f"unsupported {key} {value!r}")
        return value
    if key == "swiglu_limit":
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            raise ValueError("swiglu_limit must be positive")
        return value
    if key == "down_norm_placement":
        if value not in (GATE_UP_EPILOGUE_NORM, DOWN_PROLOGUE_NORM):
            raise ValueError(
                "down_norm_placement must be 'gate_up_epilogue' or "
                "'down_prologue'"
            )
        return value
    if key == "input_group_start_lane":
        value = int(value)
        if not 0 <= value < 6:
            raise ValueError("input_group_start_lane must be in [0, 5]")
    elif key in {
        "group_size",
        "assignment_bits",
        "codebook_size",
        "out_features",
        "in_features",
        "num_experts",
        "physical_expert_rows",
        "hidden_size",
        "intermediate_size",
        "physical_intermediate_size",
        "top_k",
    }:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{key} must be positive")
    return value


def _canonical_mapping(
    raw: Mapping[str, Any], *, keys: tuple[str, ...], source: str
) -> tuple[tuple[str, Any], ...]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source} must be an object")
    missing = [key for key in keys if key not in raw]
    if missing:
        raise ValueError(f"{source} is missing {', '.join(missing)}")
    return tuple((key, _canonical_value(key, raw[key])) for key in keys)


def _parse_ranges(
    raw: Any,
    *,
    source: str,
    allow_dense_tiles: bool,
) -> tuple[_ProfileRange, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{source} must be a list")
    ranges: list[_ProfileRange] = []
    previous_max = 0
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"{source}[{index}] must be an object")
        minimum = int(item.get("min_tokens", 0))
        maximum = int(item.get("max_tokens", 0))
        if minimum <= 0 or maximum < minimum:
            raise ValueError(f"{source}[{index}] has an invalid token range")
        if minimum <= previous_max:
            raise ValueError(f"{source} ranges overlap")
        raw_block_m = item.get("block_m")
        raw_block_n = item.get("block_n")
        if (raw_block_m is None) != (raw_block_n is None):
            raise ValueError(
                f"{source}[{index}] must specify block_m and block_n together"
            )
        block_m = None if raw_block_m is None else int(raw_block_m)
        block_n = None if raw_block_n is None else int(raw_block_n)
        if block_m is not None:
            if not allow_dense_tiles:
                raise ValueError(
                    f"{source}[{index}] launch tiles are only valid for dense"
                )
            if block_m not in (16, 32, 64, 128):
                raise ValueError(f"{source}[{index}] has an invalid block_m")
            if block_n not in (64, 128):
                raise ValueError(f"{source}[{index}] has an invalid block_n")
        ranges.append((minimum, maximum, block_m, block_n))
        previous_max = maximum
    if not ranges:
        raise ValueError(f"{source} must not be empty")
    return tuple(ranges)


def _parse_profile(raw: Mapping[str, Any], source: str) -> dict[str, Any]:
    schema_version = raw.get("schema_version")
    if schema_version not in (1, 2, 3, 4):
        raise ValueError(f"{source}: unsupported execution-profile schema")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError(f"{source}: profile name must be non-empty")
    parsed: dict[str, Any] = {
        "name": name,
        "hardware": _canonical_mapping(
            raw.get("hardware", {}), keys=_HARDWARE_KEYS, source=f"{source}.hardware"
        ),
        "format": _canonical_mapping(
            raw.get("format", {}), keys=_FORMAT_KEYS, source=f"{source}.format"
        ),
    }
    for kind, keys in (
        ("dense", _DENSE_SHAPE_KEYS),
        ("dense_multi", _DENSE_MULTI_SHAPE_KEYS),
        ("moe", _MOE_SHAPE_KEYS),
    ):
        entries = raw.get(kind, [])
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"{source}.{kind} must be a list")
        parsed_entries = []
        seen_shapes: set[tuple[tuple[str, Any], ...]] = set()
        for index, entry in enumerate(entries):
            entry_source = f"{source}.{kind}[{index}]"
            if not isinstance(entry, Mapping):
                raise ValueError(f"{entry_source} must be an object")
            raw_shape = entry.get("shape", {})
            # Schema v1 predates TP boundary-aware dense dispatch.  Preserve
            # those profiles as lane-0 evidence; schema v2 records the lane
            # explicitly so a tuned shard is never reused for another lane.
            if (
                schema_version == 1
                and kind in ("dense", "dense_multi")
                and isinstance(raw_shape, Mapping)
                and "input_group_start_lane" not in raw_shape
            ):
                raw_shape = {**raw_shape, "input_group_start_lane": 0}
            if (
                schema_version < 4
                and kind == "moe"
                and isinstance(raw_shape, Mapping)
            ):
                raw_shape = _legacy_moe_shape_defaults(raw_shape)
            shape = _canonical_mapping(
                raw_shape, keys=keys, source=f"{entry_source}.shape"
            )
            if shape in seen_shapes:
                raise ValueError(f"{entry_source}: duplicate shape")
            seen_shapes.add(shape)
            parsed_entries.append(
                (
                    shape,
                    _parse_ranges(
                        entry.get("exact_token_ranges"),
                        source=f"{entry_source}.exact_token_ranges",
                        allow_dense_tiles=kind in ("dense", "dense_multi"),
                    ),
                )
            )
        parsed[kind] = tuple(parsed_entries)
    return parsed


@lru_cache(maxsize=1)
def _bundled_profile_dicts() -> tuple[dict[str, Any], ...]:
    profile_dir = files(__package__).joinpath("execution_profiles")
    try:
        resources = sorted(profile_dir.iterdir(), key=lambda item: item.name)
    except FileNotFoundError:
        return ()
    return tuple(
        json.loads(resource.read_text(encoding="utf-8"))
        for resource in resources
        if resource.name.endswith(".json")
    )


@lru_cache(maxsize=1)
def _bundled_profiles() -> tuple[dict[str, Any], ...]:
    return tuple(
        _parse_profile(raw, f"bundled execution profile {index}")
        for index, raw in enumerate(_bundled_profile_dicts())
    )


def cuda_hardware_key(
    device: object,
) -> tuple[str, str, tuple[int, int]]:
    """Return the process-cached, hashable execution-profile hardware key."""
    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"execution profile requires a CUDA device, got {resolved}")
    index = resolved.index
    if index is None:
        index = torch.cuda.current_device()
    return _cached_cuda_hardware_identity(index)


@lru_cache(maxsize=None)
def _cached_cuda_hardware_identity(
    device_index: int,
) -> tuple[str, str, tuple[int, int]]:
    import torch

    properties = torch.cuda.get_device_properties(device_index)
    return "cuda", properties.name, (properties.major, properties.minor)


def moe_shape_identity(
    *,
    num_experts: int,
    physical_expert_rows: int | None = None,
    hidden_size: int,
    intermediate_size: int,
    physical_intermediate_size: int,
    pad_down_to_k48: bool,
    top_k: int,
    down_input_group_start_lane: int = 0,
    assignment_layout: str = "word_major",
    structural_down: bool = True,
    activation_kind: str = "silu_mul",
    gate_up_input_rounding: str = NO_ACTIVATION_ROUNDING,
    swiglu_limit: float | None = None,
    down_input_rounding: str = NO_ACTIVATION_ROUNDING,
    down_norm_placement: str = GATE_UP_EPILOGUE_NORM,
) -> dict[str, Any]:
    if physical_expert_rows is None:
        physical_expert_rows = num_experts
    return {
        "num_experts": int(num_experts),
        "physical_expert_rows": int(physical_expert_rows),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "physical_intermediate_size": int(physical_intermediate_size),
        "pad_down_to_k48": bool(pad_down_to_k48),
        "down_input_group_start_lane": int(down_input_group_start_lane),
        "assignment_layout": assignment_layout,
        "structural_down": bool(structural_down),
        "top_k": int(top_k),
        "activation_kind": activation_kind,
        "gate_up_input_rounding": gate_up_input_rounding,
        "swiglu_limit": swiglu_limit,
        "down_input_rounding": down_input_rounding,
        "down_norm_placement": down_norm_placement,
    }


@lru_cache(maxsize=None)
def _bundled_matching_ranges(
    kind: Literal["dense", "dense_multi", "moe"],
    hardware_key: tuple[tuple[str, Any], ...],
    format_key: tuple[tuple[str, Any], ...],
    shape_key: tuple[tuple[str, Any], ...],
) -> tuple[_ProfileRange, ...]:
    return _find_matching_ranges(
        kind=kind,
        hardware_key=hardware_key,
        format_key=format_key,
        shape_key=shape_key,
        parsed_profiles=_bundled_profiles(),
    )


def _find_matching_ranges(
    *,
    kind: Literal["dense", "dense_multi", "moe"],
    hardware_key: tuple[tuple[str, Any], ...],
    format_key: tuple[tuple[str, Any], ...],
    shape_key: tuple[tuple[str, Any], ...],
    parsed_profiles: Sequence[Mapping[str, Any]],
) -> tuple[_ProfileRange, ...]:
    match: tuple[_ProfileRange, ...] | None = None
    for profile in parsed_profiles:
        if profile["hardware"] != hardware_key or profile["format"] != format_key:
            continue
        for candidate_shape, ranges in profile[kind]:
            if candidate_shape != shape_key:
                continue
            if match is not None:
                raise RuntimeError(
                    f"multiple execution profiles match one {kind} shape"
                )
            match = ranges
    return match or ()


@lru_cache(maxsize=4096)
def select_cuda_moe_backend(
    *,
    device: object,
    dtype: object,
    group_size: int,
    assignment_bits: int,
    codebook_size: int,
    num_experts: int,
    physical_expert_rows: int | None = None,
    hidden_size: int,
    intermediate_size: int,
    physical_intermediate_size: int,
    pad_down_to_k48: bool,
    top_k: int,
    num_tokens: int,
    down_input_group_start_lane: int = 0,
    assignment_layout: str = "word_major",
    structural_down: bool = True,
    activation_kind: str = "silu_mul",
    gate_up_input_rounding: str = NO_ACTIVATION_ROUNDING,
    swiglu_limit: float | None = None,
    down_input_rounding: str = NO_ACTIVATION_ROUNDING,
    down_norm_placement: str = GATE_UP_EPILOGUE_NORM,
) -> ExecutionBackend:
    """Hashable warm-hit selector shared by every same-shape MoE layer."""
    if physical_expert_rows is None:
        physical_expert_rows = num_experts
    return _select_bundled_backend_cached(
        "moe",
        _hardware_mapping_key(cuda_hardware_key(device)),
        _format_tuple(dtype, group_size, assignment_bits, codebook_size),
        (
            ("num_experts", int(num_experts)),
            ("physical_expert_rows", int(physical_expert_rows)),
            ("hidden_size", int(hidden_size)),
            ("intermediate_size", int(intermediate_size)),
            ("physical_intermediate_size", int(physical_intermediate_size)),
            ("pad_down_to_k48", bool(pad_down_to_k48)),
            (
                "down_input_group_start_lane",
                int(down_input_group_start_lane),
            ),
            ("assignment_layout", assignment_layout),
            ("structural_down", bool(structural_down)),
            ("top_k", int(top_k)),
            ("activation_kind", activation_kind),
            ("gate_up_input_rounding", gate_up_input_rounding),
            (
                "swiglu_limit",
                None if swiglu_limit is None else float(swiglu_limit),
            ),
            ("down_input_rounding", down_input_rounding),
            ("down_norm_placement", down_norm_placement),
        ),
        int(num_tokens),
    )


def _hardware_mapping_key(
    key: tuple[str, str, tuple[int, int]],
) -> tuple[tuple[str, Any], ...]:
    backend, name, capability = key
    return (
        ("backend", backend),
        ("device_name", name),
        ("compute_capability", capability),
    )


def _format_tuple(
    dtype: object,
    group_size: int,
    assignment_bits: int,
    codebook_size: int,
) -> tuple[tuple[str, Any], ...]:
    return (
        ("dtype", str(dtype).removeprefix("torch.")),
        ("group_size", int(group_size)),
        ("assignment_bits", int(assignment_bits)),
        ("codebook_size", int(codebook_size)),
    )


@lru_cache(maxsize=4096)
def _select_bundled_backend_cached(
    kind: Literal["dense", "moe"],
    hardware_key: tuple[tuple[str, Any], ...],
    format_key: tuple[tuple[str, Any], ...],
    shape_key: tuple[tuple[str, Any], ...],
    num_tokens: int,
) -> ExecutionBackend:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    ranges = _bundled_matching_ranges(kind, hardware_key, format_key, shape_key)
    return (
        "cuda_exact_k48"
        if any(
            minimum <= num_tokens <= maximum
            for minimum, maximum, _, _ in ranges
        )
        else "triton"
    )


