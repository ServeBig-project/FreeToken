from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .prefill import ChunkedReq

if TYPE_CHECKING:
    from freetoken.core import Batch


GateBlockReason = Literal["decode", "queue", "cooldown"]

_UNKNOWN_ENV_WARNED = False


def adaptive_gate_enabled(
    value: str,
    *,
    warn: Callable[[str], None] | None = None,
) -> bool:
    """Parse the narrow on/off experiment switch, defaulting unknown values to off."""
    global _UNKNOWN_ENV_WARNED

    normalized = value.strip().lower()
    if normalized == "off":
        return False
    if normalized == "on":
        return True
    if not _UNKNOWN_ENV_WARNED:
        _UNKNOWN_ENV_WARNED = True
        if warn is not None:
            warn(
                f"Unknown FREETOKEN_LP_ADAPTIVE_GATE={value!r}; expected 'on' "
                "or 'off', disabling the adaptive gate"
            )
    return False


@dataclass
class AdaptiveFastPathGate:
    """Decode-load hysteresis and post-wave cooldown for direct prefills."""

    max_running_req: int
    enabled: bool = True
    pending_depth_cap: int = 2
    cooldown_seconds: float = 0.5
    low_open: int = field(init=False)
    high_close: int = field(init=False)
    _open: bool = False
    _last_wave_closed_at: float | None = None

    def __post_init__(self) -> None:
        if self.max_running_req < 1:
            raise ValueError("max_running_req must be positive")
        if self.pending_depth_cap < 0:
            raise ValueError("pending_depth_cap must be non-negative")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        # Sixth/third capacity keeps a broad hysteresis band: small decode sets use
        # eager TTFT, while a sustained batch large enough to amortize resident expert
        # staging cannot flap back to eager execution around a single boundary.
        self.low_open = max(2, self.max_running_req // 6)
        self.high_close = max(4, self.max_running_req // 3)

    def note_wave_closed(self, now: float) -> None:
        if now < 0:
            raise ValueError("monotonic timestamp must be non-negative")
        self._open = False
        self._last_wave_closed_at = now

    def should_use_fast_path(
        self,
        *,
        running_decode_count: int,
        pending_complete_prefill_depth: int,
        wave_active: bool,
        now: float,
    ) -> tuple[bool, GateBlockReason | None]:
        if running_decode_count < 0 or pending_complete_prefill_depth < 0:
            raise ValueError("runtime load counts must be non-negative")
        if now < 0:
            raise ValueError("monotonic timestamp must be non-negative")
        if not self.enabled or wave_active:
            return False, None

        if self._open:
            if running_decode_count > self.high_close:
                self._open = False
        elif running_decode_count < self.low_open:
            self._open = True
        if not self._open:
            return False, "decode"
        if pending_complete_prefill_depth > self.pending_depth_cap:
            return False, "queue"
        if (
            self._last_wave_closed_at is not None
            and now - self._last_wave_closed_at < self.cooldown_seconds
        ):
            return False, "cooldown"
        return True, None


@dataclass
class AdaptiveFastPathStats:
    fast_path_forwards: int = 0
    fast_path_prefills: int = 0
    wave_opens: int = 0
    gate_blocks_by_decode: int = 0
    gate_blocks_by_queue: int = 0
    gate_blocks_by_cooldown: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "fast_path_forwards": self.fast_path_forwards,
            "fast_path_prefills": self.fast_path_prefills,
            "wave_opens": self.wave_opens,
            "gate_blocks_by_decode": self.gate_blocks_by_decode,
            "gate_blocks_by_queue": self.gate_blocks_by_queue,
            "gate_blocks_by_cooldown": self.gate_blocks_by_cooldown,
        }


def pending_complete_prefill_depth(
    pending_reqs: Iterable[object], token_budget: int
) -> int:
    """Count fresh, unimodal FIFO entries conservatively known to fit one prefill."""
    return sum(
        1
        for pending in pending_reqs
        if getattr(pending, "chunked_req", None) is None
        and getattr(pending, "layered_cached_len", None) is None
        and getattr(pending, "mm_embeds", None) is None
        and getattr(pending, "input_len") <= token_budget
    )


def direct_prefill_batch_is_eligible(
    batch: Batch,
    *,
    token_budget: int,
    continuation_uids: Collection[int],
) -> bool:
    """Reject paths whose wave-specific continuation semantics must be retained."""
    if not batch.has_prefill:
        return False
    if sum(req.extend_len for req in batch.reqs) > token_budget:
        return False
    admitted_uids = {uid for uid, _, _ in batch.prompt_admissions}
    return all(
        not isinstance(req, ChunkedReq)
        and req.uid not in continuation_uids
        and req.mm_embeds is None
        and req.uid in admitted_uids
        for req in batch.prefill_reqs
    )


__all__ = [
    "AdaptiveFastPathGate",
    "AdaptiveFastPathStats",
    "GateBlockReason",
    "adaptive_gate_enabled",
    "direct_prefill_batch_is_eligible",
    "pending_complete_prefill_depth",
]
