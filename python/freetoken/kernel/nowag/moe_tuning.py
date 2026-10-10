"""Offline-tuned launch plans for the NoWag MoE kernels.

Gate/Up and Down are different matrix multiplications and therefore own
independent launch configurations.  They share one expert-sorted intermediate
layout, so a plan must keep ``down.block_m`` divisible by
``gate_up.block_m``.  Profiles are measured offline; serving only performs a
small host-side lookup and never benchmarks a live request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Mapping, Sequence

from .moe_activation import (
    DOWN_PROLOGUE_NORM,
    GATE_UP_EPILOGUE_NORM,
    NO_ACTIVATION_ROUNDING,
)


_HARDWARE_KEYS = ("backend", "device_name", "compute_capability")
_SHAPE_KEYS = (
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
    "codebook_size",
    "dtype",
    "group_size",
    "assignment_bits",
    "activation_kind",
    "gate_up_input_rounding",
    "swiglu_limit",
    "down_input_rounding",
    "down_norm_placement",
)

ADAPTIVE_RESIDUAL_BM16 = "bm16"
ADAPTIVE_RESIDUAL_TAIL64 = "tail64"
_ADAPTIVE_RESIDUAL_POLICIES = (
    ADAPTIVE_RESIDUAL_BM16,
    ADAPTIVE_RESIDUAL_TAIL64,
)


def _legacy_shape_defaults(raw: Mapping[str, Any]) -> dict[str, Any]:
    shape = dict(raw)
    shape.setdefault("physical_expert_rows", shape.get("num_experts"))
    shape.setdefault("structural_down", True)
    shape.setdefault("activation_kind", "silu_mul")
    shape.setdefault("gate_up_input_rounding", NO_ACTIVATION_ROUNDING)
    shape.setdefault("swiglu_limit", None)
    shape.setdefault("down_input_rounding", NO_ACTIVATION_ROUNDING)
    shape.setdefault("down_norm_placement", GATE_UP_EPILOGUE_NORM)
    return shape


@dataclass(frozen=True)
class MoeCudaStageConfig:
    """Compile-time launch choices for one MoE projection stage."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    shared_codebook_entries: int = 0
    assignment_l2_only: bool = False
    tasks_per_cta: int = 2

    def __post_init__(self) -> None:
        if self.shared_codebook_entries not in (0, 4096):
            raise ValueError(
                "shared_codebook_entries must be 0 or 4096"
            )
        if not isinstance(self.assignment_l2_only, bool):
            raise TypeError("assignment_l2_only must be a bool")
        if (
            not isinstance(self.tasks_per_cta, int)
            or isinstance(self.tasks_per_cta, bool)
            or self.tasks_per_cta not in (1, 2, 4, 8)
        ):
            raise ValueError("tasks_per_cta must be one of 1, 2, 4, or 8")
        if self.shared_codebook_entries and not self.assignment_l2_only:
            raise ValueError(
                "shared codebook entries require assignment_l2_only=True"
            )
        if (
            self.shared_codebook_entries or self.assignment_l2_only
        ) and self.block_m != 16:
            raise ValueError(
                "codebook cache policies require block_m=16"
            )


@dataclass(frozen=True)
class MoeCudaLaunchPlan:
    """Independent Gate/Up and Down choices with dispatch provenance."""

    gate_up: MoeCudaStageConfig
    down: MoeCudaStageConfig
    source: str
    profile_name: str | None
    adaptive_m_tiles: bool = False
    adaptive_residual_policy: str = ADAPTIVE_RESIDUAL_BM16

    def __post_init__(self) -> None:
        if not isinstance(self.adaptive_m_tiles, bool):
            raise TypeError("adaptive_m_tiles must be a bool")
        if self.adaptive_residual_policy not in _ADAPTIVE_RESIDUAL_POLICIES:
            raise ValueError(
                "adaptive_residual_policy must be 'bm16' or 'tail64'"
            )
        if not self.adaptive_m_tiles:
            if self.adaptive_residual_policy != ADAPTIVE_RESIDUAL_BM16:
                raise ValueError(
                    "adaptive_residual_policy is configurable only when "
                    "adaptive_m_tiles=True"
                )
            for name, stage in (("gate_up", self.gate_up), ("down", self.down)):
                if stage.tasks_per_cta != 2:
                    raise ValueError(
                        f"{name}.tasks_per_cta is configurable only when "
                        "adaptive_m_tiles=True"
                    )
            return
        has_valid_provenance = (
            self.source == "manual" and self.profile_name is None
        ) or (
            self.source == "profile"
            and isinstance(self.profile_name, str)
            and bool(self.profile_name)
        )
        if not has_valid_provenance:
            raise ValueError(
                "adaptive_m_tiles requires an explicit manual plan or a "
                "named measured profile"
            )
        expected = (16, 128, 8, 2)
        for name, stage in (("gate_up", self.gate_up), ("down", self.down)):
            actual = (
                stage.block_m,
                stage.block_n,
                stage.num_warps,
                stage.num_stages,
            )
            if actual != expected:
                raise ValueError(
                    f"adaptive_m_tiles requires {name} "
                    "BM16/BN128/8 warps/2 stages"
                )
            if stage.shared_codebook_entries or stage.assignment_l2_only:
                raise ValueError(
                    "adaptive_m_tiles does not implement codebook cache "
                    "controls"
                )


@dataclass(frozen=True)
class _TokenBucket:
    min_tokens: int
    max_tokens: int
    gate_up: MoeCudaStageConfig
    down: MoeCudaStageConfig
    adaptive_m_tiles: bool = False
    adaptive_residual_policy: str = ADAPTIVE_RESIDUAL_BM16


@dataclass(frozen=True)
class _MoeTuningProfile:
    name: str
    hardware: tuple[tuple[str, Any], ...]
    shape: tuple[tuple[str, Any], ...]
    token_buckets: tuple[_TokenBucket, ...]


def _positive_int(value: Any, *, field: str, source: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {field} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{source}: {field} must be positive")
    return parsed


def _parse_stage(
    raw: Mapping[str, Any],
    *,
    stage: str,
    source: str,
    adaptive_m_tiles: bool,
) -> MoeCudaStageConfig:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: {stage} must be an object")
    if "tasks_per_cta" in raw and not adaptive_m_tiles:
        raise ValueError(
            f"{source}: {stage}.tasks_per_cta is valid only when "
            "adaptive_m_tiles=true"
        )
    config = MoeCudaStageConfig(
        block_m=_positive_int(
            raw.get("block_m"), field=f"{stage}.block_m", source=source
        ),
        block_n=_positive_int(
            raw.get("block_n"), field=f"{stage}.block_n", source=source
        ),
        num_warps=_positive_int(
            raw.get("num_warps"), field=f"{stage}.num_warps", source=source
        ),
        num_stages=_positive_int(
            raw.get("num_stages"), field=f"{stage}.num_stages", source=source
        ),
        shared_codebook_entries=int(raw.get("shared_codebook_entries", 0)),
        assignment_l2_only=raw.get("assignment_l2_only", False),
        tasks_per_cta=raw.get("tasks_per_cta", 2),
    )
    if config.block_m % 16 or config.block_n % 16:
        raise ValueError(
            f"{source}: {stage} BM/BN must be multiples of the MMA tile (16)"
        )
    if config.block_m & (config.block_m - 1) or config.block_n & (
        config.block_n - 1
    ):
        raise ValueError(f"{source}: {stage} BM/BN must be powers of two")
    if config.num_warps != config.block_n // 16:
        raise ValueError(
            f"{source}: {stage} requires one warp per N=16 output tile"
        )
    if config.num_stages != 2:
        raise ValueError(
            f"{source}: {stage} profiled kernels currently use two stages"
        )
    if stage == "down" and (
        config.shared_codebook_entries or config.assignment_l2_only
    ):
        raise ValueError(
            f"{source}: Down does not support a codebook cache policy"
        )
    return config


def _canonical_value(key: str, value: Any) -> Any:
    if key == "compute_capability":
        try:
            capability = tuple(int(part) for part in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "compute_capability must contain two integers"
            ) from exc
        if len(capability) != 2:
            raise ValueError("compute_capability must contain two integers")
        return capability
    if key in ("pad_down_to_k48", "structural_down"):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a bool")
        return value
    if key in ("physical_expert_rows", "physical_intermediate_size"):
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{key} must be positive")
        return parsed
    if key == "down_input_group_start_lane":
        parsed = int(value)
        if not 0 <= parsed < 6:
            raise ValueError("down_input_group_start_lane must be in [0, 5]")
        return parsed
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
        parsed = float(value)
        if parsed <= 0:
            raise ValueError("swiglu_limit must be positive")
        return parsed
    if key == "down_norm_placement":
        if value not in (GATE_UP_EPILOGUE_NORM, DOWN_PROLOGUE_NORM):
            raise ValueError(
                "down_norm_placement must be 'gate_up_epilogue' or "
                "'down_prologue'"
            )
        return value
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


def _parse_profile(raw: Mapping[str, Any], source: str) -> _MoeTuningProfile:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: profile must be an object")
    schema_version = raw.get("schema_version")
    if schema_version not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"{source}: unsupported profile schema")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError(f"{source}: profile name must be non-empty")
    hardware = _canonical_mapping(
        raw.get("hardware", {}),
        keys=_HARDWARE_KEYS,
        source=f"{source}.hardware",
    )
    raw_shape = raw.get("shape", {})
    if schema_version == 1 and isinstance(raw_shape, Mapping):
        # Safely migrate legacy unpadded profiles.  A legacy profile cannot
        # claim padded/lane-specific evidence that its schema never recorded.
        raw_shape = dict(raw_shape)
        raw_shape.setdefault(
            "physical_intermediate_size",
            raw_shape.get("intermediate_size"),
        )
        raw_shape.setdefault("pad_down_to_k48", False)
        raw_shape.setdefault("down_input_group_start_lane", 0)
        raw_shape.setdefault("assignment_layout", "word_major")
    if schema_version < 4 and isinstance(raw_shape, Mapping):
        raw_shape = _legacy_shape_defaults(raw_shape)
    shape = _canonical_mapping(
        raw_shape, keys=_SHAPE_KEYS, source=f"{source}.shape"
    )

    raw_buckets = raw.get("token_buckets")
    if not isinstance(raw_buckets, Sequence) or isinstance(
        raw_buckets, (str, bytes)
    ):
        raise ValueError(f"{source}: token_buckets must be a list")
    buckets: list[_TokenBucket] = []
    previous_max = 0
    for index, raw_bucket in enumerate(raw_buckets):
        bucket_source = f"{source}.token_buckets[{index}]"
        if not isinstance(raw_bucket, Mapping):
            raise ValueError(f"{bucket_source} must be an object")
        min_tokens = _positive_int(
            raw_bucket.get("min_tokens"),
            field="min_tokens",
            source=bucket_source,
        )
        max_tokens = _positive_int(
            raw_bucket.get("max_tokens"),
            field="max_tokens",
            source=bucket_source,
        )
        if min_tokens > max_tokens:
            raise ValueError(f"{bucket_source}: min_tokens exceeds max_tokens")
        if min_tokens <= previous_max:
            raise ValueError(f"{bucket_source}: token buckets overlap")
        adaptive_m_tiles = raw_bucket.get("adaptive_m_tiles", False)
        if not isinstance(adaptive_m_tiles, bool):
            raise ValueError(
                f"{bucket_source}: adaptive_m_tiles must be a bool"
            )
        adaptive_residual_policy = raw_bucket.get(
            "adaptive_residual_policy", ADAPTIVE_RESIDUAL_BM16
        )
        if adaptive_residual_policy not in _ADAPTIVE_RESIDUAL_POLICIES:
            raise ValueError(
                f"{bucket_source}: adaptive_residual_policy must be "
                "'bm16' or 'tail64'"
            )
        if (
            adaptive_residual_policy != ADAPTIVE_RESIDUAL_BM16
            and not adaptive_m_tiles
        ):
            raise ValueError(
                f"{bucket_source}: adaptive_residual_policy is valid only "
                "when adaptive_m_tiles=true"
            )
        gate_up = _parse_stage(
            raw_bucket.get("gate_up", {}),
            stage="gate_up",
            source=bucket_source,
            adaptive_m_tiles=adaptive_m_tiles,
        )
        down = _parse_stage(
            raw_bucket.get("down", {}),
            stage="down",
            source=bucket_source,
            adaptive_m_tiles=adaptive_m_tiles,
        )
        if down.block_m % gate_up.block_m:
            raise ValueError(
                f"{bucket_source}: down BM must be divisible by Gate/Up BM"
            )
        buckets.append(
            _TokenBucket(
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                gate_up=gate_up,
                down=down,
                adaptive_m_tiles=adaptive_m_tiles,
                adaptive_residual_policy=adaptive_residual_policy,
            )
        )
        previous_max = max_tokens
    if not buckets:
        raise ValueError(f"{source}: token_buckets must not be empty")
    return _MoeTuningProfile(
        name=name,
        hardware=hardware,
        shape=shape,
        token_buckets=tuple(buckets),
    )


@lru_cache(maxsize=1)
def _bundled_profile_dicts() -> tuple[dict[str, Any], ...]:
    profile_dir = files(__package__).joinpath("moe_profiles")
    try:
        resources = sorted(profile_dir.iterdir(), key=lambda item: item.name)
    except FileNotFoundError:
        return ()
    profiles = []
    for resource in resources:
        if resource.name.endswith(".json"):
            profiles.append(json.loads(resource.read_text(encoding="utf-8")))
    return tuple(profiles)


@lru_cache(maxsize=1)
def _bundled_profiles() -> tuple[_MoeTuningProfile, ...]:
    return tuple(
        _parse_profile(raw, f"bundled profile {index}")
        for index, raw in enumerate(_bundled_profile_dicts())
    )


def moe_shape_identity(
    *,
    num_experts: int,
    physical_expert_rows: int | None = None,
    hidden_size: int,
    intermediate_size: int,
    physical_intermediate_size: int,
    pad_down_to_k48: bool,
    down_input_group_start_lane: int,
    assignment_layout: str,
    top_k: int,
    codebook_size: int,
    dtype: object,
    group_size: int,
    assignment_bits: int,
    structural_down: bool = True,
    activation_kind: str = "silu_mul",
    gate_up_input_rounding: str = NO_ACTIVATION_ROUNDING,
    swiglu_limit: float | None = None,
    down_input_rounding: str = NO_ACTIVATION_ROUNDING,
    down_norm_placement: str = GATE_UP_EPILOGUE_NORM,
) -> dict[str, Any]:
    """Build a TP-local problem key without importing model-specific types."""
    dtype_name = str(dtype).removeprefix("torch.")
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
        "codebook_size": int(codebook_size),
        "dtype": dtype_name,
        "group_size": int(group_size),
        "assignment_bits": int(assignment_bits),
        "activation_kind": activation_kind,
        "gate_up_input_rounding": gate_up_input_rounding,
        "swiglu_limit": swiglu_limit,
        "down_input_rounding": down_input_rounding,
        "down_norm_placement": down_norm_placement,
    }


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def fallback_moe_cuda_launch_plan(
    *, shape: Mapping[str, Any], num_tokens: int
) -> MoeCudaLaunchPlan:
    """Return a conservative shape rule when no measured profile matches.

    The rule may select BM16/32, which deliberately makes the current exact
    CUDA backend fall back to the generic Triton implementation.  This avoids
    forcing route-sparse decode through a large CUDA macro-tile merely because
    a large-M specialization exists.
    """
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    canonical_shape = dict(
        _canonical_mapping(
            _legacy_shape_defaults(shape), keys=_SHAPE_KEYS, source="shape"
        )
    )
    num_experts = _positive_int(
        canonical_shape["num_experts"], field="num_experts", source="shape"
    )
    top_k = _positive_int(
        canonical_shape["top_k"], field="top_k", source="shape"
    )
    hidden_size = _positive_int(
        canonical_shape["hidden_size"], field="hidden_size", source="shape"
    )
    intermediate_size = _positive_int(
        canonical_shape["intermediate_size"],
        field="intermediate_size",
        source="shape",
    )
    routes_per_expert = (
        num_tokens * top_k + num_experts - 1
    ) // num_experts
    down_bm = min(128, max(16, _next_power_of_two(routes_per_expert)))
    gate_bm = min(64, max(16, down_bm // 2))
    while down_bm % gate_bm:
        gate_bm //= 2

    # Pick an exact-kernel tile only when it divides the local projection.
    # Widths divisible by 64 but not 128 must retain the smaller candidate.
    gate_bn = (
        128
        if intermediate_size % 128 == 0
        else 64
        if intermediate_size % 64 == 0
        else 128
        if intermediate_size >= 128
        else 64
    )
    max_down_bn = max(16, min(128, 8192 // down_bm))
    down_bn = next(
        (
            candidate
            for candidate in (128, 64)
            if candidate <= max_down_bn and hidden_size % candidate == 0
        ),
        min(max_down_bn, max(16, _next_power_of_two(hidden_size))),
    )
    return MoeCudaLaunchPlan(
        gate_up=MoeCudaStageConfig(
            block_m=gate_bm,
            block_n=gate_bn,
            num_warps=max(1, gate_bn // 16),
            num_stages=2,
        ),
        down=MoeCudaStageConfig(
            block_m=down_bm,
            block_n=down_bn,
            num_warps=max(1, down_bn // 16),
            num_stages=2,
        ),
        source="fallback",
        profile_name=None,
    )


def match_moe_cuda_launch_plan(
    *,
    hardware: Mapping[str, Any],
    shape: Mapping[str, Any],
    num_tokens: int,
    profiles: Sequence[Mapping[str, Any]] | None = None,
) -> MoeCudaLaunchPlan | None:
    """Return only a measured hardware/shape/token-bucket match.

    Unlike :func:`select_moe_cuda_launch_plan`, this API never synthesizes a
    fallback plan.  Auto-selected Triton uses it so an unmeasured shape cannot
    accidentally inherit Exact CUDA's conservative fallback launch choices.
    """
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    hardware_key = _canonical_mapping(
        hardware, keys=_HARDWARE_KEYS, source="hardware"
    )
    shape_key = _canonical_mapping(
        _legacy_shape_defaults(shape), keys=_SHAPE_KEYS, source="shape"
    )
    parsed_profiles = (
        _bundled_profiles()
        if profiles is None
        else tuple(
            _parse_profile(raw, f"profile[{index}]")
            for index, raw in enumerate(profiles)
        )
    )

    selected: tuple[_MoeTuningProfile, _TokenBucket] | None = None
    for profile in parsed_profiles:
        if profile.hardware != hardware_key or profile.shape != shape_key:
            continue
        for bucket in profile.token_buckets:
            if bucket.min_tokens <= num_tokens <= bucket.max_tokens:
                if selected is not None:
                    raise RuntimeError(
                        "multiple MoE tuning profiles match the same problem"
                    )
                selected = (profile, bucket)

    if selected is None:
        return None
    profile, bucket = selected
    return MoeCudaLaunchPlan(
        gate_up=bucket.gate_up,
        down=bucket.down,
        source="profile",
        profile_name=profile.name,
        adaptive_m_tiles=bucket.adaptive_m_tiles,
        adaptive_residual_policy=bucket.adaptive_residual_policy,
    )


def select_moe_cuda_launch_plan(
    *,
    hardware: Mapping[str, Any],
    shape: Mapping[str, Any],
    num_tokens: int,
    profiles: Sequence[Mapping[str, Any]] | None = None,
) -> MoeCudaLaunchPlan:
    """Select a measured row or Exact CUDA's conservative fallback."""
    matched = match_moe_cuda_launch_plan(
        hardware=hardware,
        shape=shape,
        num_tokens=num_tokens,
        profiles=profiles,
    )
    if matched is not None:
        return matched
    return fallback_moe_cuda_launch_plan(
        shape=shape,
        num_tokens=num_tokens,
    )


@lru_cache(maxsize=4096)
def select_cuda_moe_launch_plan(
    *,
    hardware_key: tuple[str, str, tuple[int, int]],
    dtype: object,
    group_size: int,
    assignment_bits: int,
    codebook_size: int,
    num_experts: int,
    physical_expert_rows: int,
    hidden_size: int,
    intermediate_size: int,
    physical_intermediate_size: int,
    pad_down_to_k48: bool,
    down_input_group_start_lane: int,
    assignment_layout: str,
    structural_down: bool,
    top_k: int,
    activation_kind: str,
    gate_up_input_rounding: str,
    swiglu_limit: float | None,
    down_input_rounding: str,
    down_norm_placement: str,
    num_tokens: int,
) -> MoeCudaLaunchPlan:
    """Return one process-cached launch plan for the complete profile key.

    The hardware and shape dictionaries, profile lookup, and returned frozen
    dataclasses are all constructed only on a cache miss.  ``hardware_key`` is
    the same process-cached identity used by execution backend selection.
    """
    backend, device_name, compute_capability = hardware_key
    hardware = {
        "backend": backend,
        "device_name": device_name,
        "compute_capability": list(compute_capability),
    }
    shape = moe_shape_identity(
        num_experts=num_experts,
        physical_expert_rows=physical_expert_rows,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        physical_intermediate_size=physical_intermediate_size,
        pad_down_to_k48=pad_down_to_k48,
        down_input_group_start_lane=down_input_group_start_lane,
        assignment_layout=assignment_layout,
        structural_down=structural_down,
        top_k=top_k,
        codebook_size=codebook_size,
        dtype=dtype,
        group_size=group_size,
        assignment_bits=assignment_bits,
        activation_kind=activation_kind,
        gate_up_input_rounding=gate_up_input_rounding,
        swiglu_limit=swiglu_limit,
        down_input_rounding=down_input_rounding,
        down_norm_placement=down_norm_placement,
    )
    return select_moe_cuda_launch_plan(
        hardware=hardware,
        shape=shape,
        num_tokens=num_tokens,
    )


