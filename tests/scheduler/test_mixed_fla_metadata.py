from types import SimpleNamespace

import pytest
import torch

import freetoken.core as core
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Context, Req, SamplingParams


@pytest.fixture(autouse=True)
def linear_context(monkeypatch):
    ctx = Context(page_size=1)
    ctx.linear_state_pool = SimpleNamespace(
        replay=None, padding_slot=0, conv_states=torch.zeros(1, 10, 3, 3),
    )
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)


def _req(*, extend_len: int, cached_len: int, state_slot: int):
    req = Req(
        input_ids=torch.arange(cached_len + extend_len, dtype=torch.int32),
        cached_len=cached_len, table_idx=state_slot + 100, uid=state_slot,
        output_len=4, sampling_params=SamplingParams(), cache_handle=None,
    )
    req.linear_slot_idx = state_slot
    return req


def test_mixed_fla_metadata_keeps_per_request_boundaries_and_state():
    decode = _req(extend_len=1, cached_len=32, state_slot=3)
    prefill = _req(extend_len=4, cached_len=0, state_slot=7)
    batch = Batch(reqs=[decode, prefill], decode_size=1)
    batch.padded_reqs = batch.reqs

    metadata = build_fla_metadata(batch, torch.device("cpu"))
    assert metadata.verify is None

    assert metadata.decode is not None
    assert metadata.decode.cu_seqlens.tolist() == [0, 1]
    assert metadata.decode.cache_indices.tolist() == [3]

    assert metadata.prefill is not None
    assert metadata.prefill.cu_seqlens.tolist() == [0, 4]
    assert metadata.prefill.cache_indices.tolist() == [7]
    assert metadata.prefill.has_initial_state.tolist() == [False]
    assert metadata.prefill.fresh_state_indices.tolist() == [7]


def test_mixed_fla_metadata_rebases_multiple_prefill_requests():
    decode_a = _req(extend_len=1, cached_len=32, state_slot=3)
    decode_b = _req(extend_len=1, cached_len=48, state_slot=4)
    continued = _req(extend_len=5, cached_len=64, state_slot=7)
    fresh = _req(extend_len=3, cached_len=0, state_slot=8)
    batch = Batch(
        reqs=[decode_a, decode_b, continued, fresh],
        decode_size=2,
    )
    batch.padded_reqs = batch.reqs

    metadata = build_fla_metadata(batch, torch.device("cpu"))
    assert metadata.verify is None

    assert metadata.decode is not None
    assert metadata.decode.cu_seqlens.tolist() == [0, 1, 2]
    assert metadata.decode.cache_indices.tolist() == [3, 4]

    assert metadata.prefill is not None
    assert metadata.prefill.cu_seqlens.tolist() == [0, 5, 8]
    assert metadata.prefill.cache_indices.tolist() == [7, 8]
    assert metadata.prefill.has_initial_state.tolist() == [True, False]
    assert metadata.prefill.fresh_state_indices.tolist() == [8]


def test_fla_metadata_builds_only_the_active_execution_path():
    decode = _req(extend_len=1, cached_len=32, state_slot=3)
    decode_batch = Batch(reqs=[decode], decode_size=1)
    decode_batch.padded_reqs = decode_batch.reqs
    decode_batch.linear_table_idx = torch.tensor([3], dtype=torch.int32)

    decode_metadata = build_fla_metadata(decode_batch, torch.device("cpu"))

    assert decode_metadata.decode is not None
    assert decode_metadata.prefill is None

    prefill = _req(extend_len=4, cached_len=0, state_slot=7)
    prefill_batch = Batch(reqs=[prefill], decode_size=0)
    prefill_batch.padded_reqs = prefill_batch.reqs

    prefill_metadata = build_fla_metadata(prefill_batch, torch.device("cpu"))

    assert prefill_metadata.decode is None
    assert prefill_metadata.prefill is not None
