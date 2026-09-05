from contextlib import nullcontext
from types import SimpleNamespace

import torch

from freetoken.core import Batch
from freetoken.engine import BatchSamplingArgs, ForwardOutput
from freetoken.scheduler.adaptive_gate import (
    AdaptiveFastPathGate,
    direct_prefill_batch_is_eligible,
)
from freetoken.scheduler.forward import ForwardInput
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.scheduler import Scheduler


def test_gate_opens_at_zero_and_low_decode_load() -> None:
    gate = AdaptiveFastPathGate(max_running_req=60)

    assert gate.should_use_fast_path(
        running_decode_count=0,
        pending_complete_prefill_depth=1,
        wave_active=False,
        now=1.0,
    ) == (True, None)
    assert gate.should_use_fast_path(
        running_decode_count=9,
        pending_complete_prefill_depth=2,
        wave_active=False,
        now=1.1,
    ) == (True, None)


def test_gate_hysteresis_queue_cap_and_cooldown() -> None:
    gate = AdaptiveFastPathGate(max_running_req=60)
    assert gate.low_open == 10
    assert gate.high_close == 20

    assert gate.should_use_fast_path(
        running_decode_count=9,
        pending_complete_prefill_depth=1,
        wave_active=False,
        now=10.0,
    ) == (True, None)
    assert gate.should_use_fast_path(
        running_decode_count=15,
        pending_complete_prefill_depth=1,
        wave_active=False,
        now=10.1,
    ) == (True, None)
    assert gate.should_use_fast_path(
        running_decode_count=15,
        pending_complete_prefill_depth=3,
        wave_active=False,
        now=10.2,
    ) == (False, "queue")
    assert gate.should_use_fast_path(
        running_decode_count=21,
        pending_complete_prefill_depth=1,
        wave_active=False,
        now=10.3,
    ) == (False, "decode")
    assert gate.should_use_fast_path(
        running_decode_count=15,
        pending_complete_prefill_depth=1,
        wave_active=False,
        now=10.4,
    ) == (False, "decode")

    gate.note_wave_closed(20.0)
    assert gate.should_use_fast_path(
        running_decode_count=0,
        pending_complete_prefill_depth=1,
        wave_active=False,
        now=20.49,
    ) == (False, "cooldown")
    assert gate.should_use_fast_path(
        running_decode_count=0,
        pending_complete_prefill_depth=1,
        wave_active=False,
        now=20.5,
    ) == (True, None)


def _batch(req: object, *, admitted: bool = True) -> Batch:
    batch = Batch(reqs=[req], decode_size=0)
    if admitted:
        batch.prompt_admissions = [(req.uid, 4, 0)]
    return batch


def test_chunked_continuation_and_multimodal_prefills_are_ineligible() -> None:
    complete = SimpleNamespace(uid=1, extend_len=4, mm_embeds=None)
    assert direct_prefill_batch_is_eligible(
        _batch(complete), token_budget=4, continuation_uids=set()
    )

    chunked = object.__new__(ChunkedReq)
    chunked.uid = 2
    chunked.cached_len = 0
    chunked.device_len = 4
    chunked.mm_embeds = None
    assert not direct_prefill_batch_is_eligible(
        _batch(chunked), token_budget=4, continuation_uids=set()
    )

    continuation = SimpleNamespace(uid=3, extend_len=4, mm_embeds=None)
    assert not direct_prefill_batch_is_eligible(
        _batch(continuation, admitted=False),
        token_budget=4,
        continuation_uids={3},
    )

    multimodal = SimpleNamespace(uid=4, extend_len=4, mm_embeds=object())
    assert not direct_prefill_batch_is_eligible(
        _batch(multimodal), token_budget=4, continuation_uids=set()
    )


def test_direct_mixed_output_keeps_decode_deferred_and_prefill_immediate() -> None:
    decode = SimpleNamespace(uid=10, extend_len=1, mm_embeds=None)
    prefill = SimpleNamespace(uid=11, extend_len=4, mm_embeds=None)
    batch = Batch(reqs=[decode, prefill], decode_size=1)
    batch.prompt_admissions = [(11, 4, 0)]
    mapping = (torch.tensor([0, 1]), torch.tensor([0, 4]))
    args = BatchSamplingArgs(None, None, None)
    forward_input = ForwardInput(batch, args, mapping, mapping)
    event = object()
    output = ForwardOutput(torch.tensor([20, 21]), torch.tensor([20, 21]), event)

    scheduler = object.__new__(Scheduler)
    scheduler.stream = object()
    scheduler.engine = SimpleNamespace(
        stream=SimpleNamespace(wait_stream=lambda stream: None)
    )
    scheduler.engine_stream_ctx = nullcontext()
    scheduler.cache_manager = SimpleNamespace(
        reserve_next_decode=lambda reqs: setattr(scheduler, "reserved", list(reqs))
    )
    scheduler._prepare_resident_batch = lambda current: forward_input
    scheduler._report_prompt_admissions = lambda current: setattr(
        scheduler, "reported", current
    )
    scheduler._restore_linear_states = lambda current: setattr(
        scheduler, "restored", current
    )
    scheduler._forward = lambda current: output

    decode_data, prefill_data = scheduler._forward_direct_resident_batch(batch)

    assert decode_data[0].batch.is_decode_only
    assert prefill_data[0].batch.has_prefill
    assert decode_data[1].next_tokens_cpu.tolist() == [20]
    assert prefill_data[1].next_tokens_cpu.tolist() == [21]
    assert decode_data[1].copy_done_event is event
    assert prefill_data[1].copy_done_event is event
    assert scheduler.reserved == [decode]
    assert scheduler.reported is batch
    assert scheduler.restored is batch
