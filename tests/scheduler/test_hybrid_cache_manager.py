"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    # int32 to match production Req.input_ids dtype (fast_compare_key needs consistent dtype)
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None, cache_group="")


def test_hybrid_cache_manager_donate_then_hit():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    # cold match on an empty tree
    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None

    # The running request owns its working state and one pending prefix snapshot.
    live, snapshot = pool.alloc(2)
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    reqA = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
               cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
               cache_handle=mr.cuda_handle)
    reqA.linear_slot_idx, reqA.mamba_snapshot_slot = live, snapshot
    reqA.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(reqA, finished=False)
    assert pool.num_free_slots == free_before  # ownership moves; no replacement allocation
    assert reqA.mamba_snapshot_slot is None
    assert reqA.linear_slot_idx == live

    # req B shares the [1,2,3,4] prefix -> HIT: returns the donated snapshot + reused KV
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value == snapshot
    assert mrB.cuda_handle.get_matched_indices().tolist() == [100, 101, 102, 103]

    cm._free_req_slots(reqA)
    assert pool.num_free_slots == free_before + 1  # only the private working slot is released
    assert cm.match_req(_pend([1, 2, 3, 4, 8])).mamba_value == snapshot


def test_hybrid_finish_donates_live_slot():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live = pool.alloc(1)[0]
    page_table[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1,
              cached_len=3, output_len=1, uid=1, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx = live
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(req, finished=True)
    assert req.linear_slot_idx is None
    assert req.mamba_snapshot_slot is None
    assert pool.num_free_slots == free_before  # the tree keeps the donated working state
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value == live


@pytest.mark.parametrize("with_snapshot", [False, True])
def test_free_req_slots_idempotent(with_snapshot):
    """C2: a finish/abort double-free of the same request must NOT push its GDN slots twice."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live = pool.alloc(1)[0]
    snapshot = pool.alloc(1)[0] if with_snapshot else None
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0, cached_len=2,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx, req.mamba_snapshot_slot = live, snapshot
    base = pool.num_free_slots
    cm._free_req_slots(req)
    assert pool.num_free_slots == base + 1 + with_snapshot
    assert req.linear_slot_idx is None and req.mamba_snapshot_slot is None
    cm._free_req_slots(req)                        # second free (abort/finish race)
    assert pool.num_free_slots == base + 1 + with_snapshot


def test_rebuild_reclaims_donated_gdn_slots():
    """C5: a runtime cache rebuild must return the discarded tree's GDN snapshot slots (idle)."""
    pool = _pool(num_slots=16)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live = pool.alloc(1)[0]
    pt[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1, cached_len=3,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx = live
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)
    assert pool.num_free_slots < pool.num_slots - 1   # a slot is now tree-owned
    cm.rebuild(64, pt)                            # idle rebuild discards the tree
    assert pool.num_free_slots == pool.num_slots - 1  # all GDN slots reclaimed (no leak)


@pytest.mark.parametrize("prompt_len,chunked,snapshot_needed", [
    (32, False, False), (64, False, False), (70, False, True), (128, True, False),
])
def test_prefill_allocates_only_the_requested_snapshot(prompt_len, chunked, snapshot_needed):
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(3, 160, dtype=torch.int32)
    cm = CacheManager(160, 1, pt, "hybrid_radix", linear_state_pool=pool)
    pending = PendingReq(0, torch.arange(prompt_len, dtype=torch.int32), SamplingParams(max_tokens=4))
    pm = PrefillManager(cm, TableManager(2, pt), DecodeManager(1), [pending])
    free_before = pool.num_free_slots
    batch = pm.schedule_next_batch(64 if chunked else prompt_len)
    (req,) = batch.reqs
    assert isinstance(req, ChunkedReq) == chunked
    assert req.linear_slot_idx is not None and req.mamba_snapshot_slot is None
    assert pool.num_free_slots == free_before - 1

    cm.prepare_prefill_snapshots(batch.reqs)
    assert (req.mamba_snapshot_slot is not None) == snapshot_needed
    assert pool.num_free_slots == free_before - 1 - snapshot_needed
    if snapshot_needed:
        assert req.mamba_snapshot_slot != req.linear_slot_idx

    cm._free_req_slots(req)
    assert pool.num_free_slots == free_before


def test_optional_prefill_snapshot_does_not_require_a_second_free_slot():
    pool = _pool(num_slots=2)
    pt = torch.zeros(2, 128, dtype=torch.int32)
    cm = CacheManager(128, 1, pt, "hybrid_radix", linear_state_pool=pool)
    req = Req(input_ids=torch.arange(70, dtype=torch.int32), table_idx=0, cached_len=0,
              output_len=4, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx = pool.alloc(1)[0]

    cm.prepare_prefill_snapshots([req])

    assert req.mamba_snapshot_slot is None
    assert req.linear_slot_idx is not None
    assert pool.num_free_slots == 0


def test_pool_sizing_covers_4mr_floor():
    """Pool capacity retains its configured 4*max_running_req floor at a tiny ratio."""
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1, gdn_state_budget_bytes=None,
                            enable_gdn_replayssm=False)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))
