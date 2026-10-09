"""Model-independent activation math for routed NoWag experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ActivationRounding = Literal[
    "none",
    "dynamic_e4m3_per_token_group128_ue8m0",
]
DownNormPlacement = Literal["gate_up_epilogue", "down_prologue"]

NO_ACTIVATION_ROUNDING: ActivationRounding = "none"
DYNAMIC_E4M3_GROUP128_UE8M0: ActivationRounding = (
    "dynamic_e4m3_per_token_group128_ue8m0"
)
GATE_UP_EPILOGUE_NORM: DownNormPlacement = "gate_up_epilogue"
DOWN_PROLOGUE_NORM: DownNormPlacement = "down_prologue"


@dataclass(frozen=True)
class MoeActivationMath:
    """The arithmetic around the three expert matrix multiplications.

    The main lookup/MMA kernel always consumes restored BF16 values.  The two
    rounding fields record whether those values passed through dynamic E4M3
    storage before Gate/Up or before Down.  Keeping this identity separate
    from model names lets offline profiles distinguish mathematically
    different calls that happen to share E/H/I/top-k.
    """

    gate_up_input_rounding: ActivationRounding = NO_ACTIVATION_ROUNDING
    swiglu_limit: float | None = None
    down_input_rounding: ActivationRounding = NO_ACTIVATION_ROUNDING
    down_norm_placement: DownNormPlacement = GATE_UP_EPILOGUE_NORM

    def __post_init__(self) -> None:
        for field, value in (
            ("gate_up_input_rounding", self.gate_up_input_rounding),
            ("down_input_rounding", self.down_input_rounding),
        ):
            if value not in (
                NO_ACTIVATION_ROUNDING,
                DYNAMIC_E4M3_GROUP128_UE8M0,
            ):
                raise ValueError(f"unsupported {field} {value!r}")
        if self.swiglu_limit is not None and self.swiglu_limit <= 0:
            raise ValueError("swiglu_limit must be positive")
        if self.down_norm_placement not in (
            GATE_UP_EPILOGUE_NORM,
            DOWN_PROLOGUE_NORM,
        ):
            raise ValueError(
                "down_norm_placement must be 'gate_up_epilogue' or "
                "'down_prologue'"
            )
        if (
            self.down_input_rounding != NO_ACTIVATION_ROUNDING
            and self.down_norm_placement != DOWN_PROLOGUE_NORM
        ):
            raise ValueError(
                "Down input rounding must run before its normalizer, so it "
                "requires down_norm_placement='down_prologue'"
            )

    def profile_identity(self) -> dict[str, Any]:
        return {
            "activation_kind": "silu_mul",
            "gate_up_input_rounding": self.gate_up_input_rounding,
            "swiglu_limit": self.swiglu_limit,
            "down_input_rounding": self.down_input_rounding,
            "down_norm_placement": self.down_norm_placement,
        }


