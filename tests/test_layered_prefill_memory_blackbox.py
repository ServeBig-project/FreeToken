"""Public memory-accounting contract for bounded layered-prefill waves."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.prefill_memory import PrefillMemoryBudget
from freetoken.core import SamplingParams
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq


KIB = 1024
TILE = 64
MAX_TOKENS = 1024


@pytest.fixture
def memory(monkeypatch):
    state = SimpleNamespace(allocated=1024 * KIB, reserved=1024 * KIB,
                            peak=1024 * KIB, free=16 * 1024 * KIB, resets=0)
    for name, field in (("memory_allocated", "allocated"), ("memory_reserved", "reserved"),
                        ("max_memory_allocated", "peak")):
        monkeypatch.setattr(torch.cuda, name, lambda *args, field=field, **kwargs: getattr(state, field))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (state.free, 32 * 1024 * KIB))

    def reset_peak(*args, **kwargs):
        state.peak = state.allocated
        state.resets += 1

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
    return state


def _observe(budget, memory, rows=TILE, retained=64 * KIB, peak=256 * KIB, first_stage=True):
    before = budget.start()
    memory.allocated += retained
    memory.peak = before + peak
    growth = max(0, memory.allocated - memory.reserved)
    memory.reserved += growth
    memory.free -= growth
    budget.record(before, rows, first_stage=first_stage)


def _available(memory, free, unused_reserved=0):
    memory.free = free
    memory.reserved = memory.allocated + unused_reserved


def _calibrated(memory):
    budget = PrefillMemoryBudget(torch.device("cuda:0"))
    assert budget.token_budget(TILE, MAX_TOKENS) == TILE
    _observe(budget, memory)
    return budget


def test_uncalibrated_budget_only_allows_one_tile(memory):
    budget = PrefillMemoryBudget(torch.device("cuda:0"))
    assert budget.token_budget(TILE, MAX_TOKENS) == TILE
    assert budget.token_budget(TILE, TILE) == TILE


def test_start_resets_peak_and_returns_current_allocated_bytes(memory):
    budget = PrefillMemoryBudget(torch.device("cuda:0"))
    memory.peak += 256 * KIB
    assert budget.start() == memory.allocated
    assert memory.peak == memory.allocated
    assert memory.resets == 1


def test_partial_tile_observation_does_not_prove_full_tile_was_measured(memory):
    budget = PrefillMemoryBudget(torch.device("cuda:0"))
    assert budget.token_budget(TILE, MAX_TOKENS) == TILE
    _observe(budget, memory, rows=TILE - 1, retained=(TILE - 1) * KIB)
    _available(memory, 1024 * KIB)
    assert budget.token_budget(TILE, MAX_TOKENS) == TILE
    _observe(budget, memory, retained=0, first_stage=False)
    assert budget.token_budget(TILE, MAX_TOKENS) == 768


def test_full_tile_without_observed_retained_state_stays_at_one_tile(memory):
    budget = PrefillMemoryBudget(torch.device("cuda:0"))
    _observe(budget, memory, retained=0, first_stage=False)
    assert budget.token_budget(TILE, MAX_TOKENS) == TILE
    _observe(budget, memory, retained=0)
    assert budget.token_budget(TILE, MAX_TOKENS) == TILE


def test_free_plus_unused_reserved_memory_sets_budget_and_maximum_caps_it(memory):
    budget = _calibrated(memory)
    _available(memory, 512 * KIB, unused_reserved=256 * KIB)
    assert budget.token_budget(TILE, MAX_TOKENS) == 512
    assert budget.token_budget(TILE, 256) == 256


def test_less_available_memory_shrinks_budget_but_keeps_one_runnable_tile(memory):
    budget = _calibrated(memory)
    _available(memory, 1024 * KIB)
    assert budget.token_budget(TILE, MAX_TOKENS) == 768
    _available(memory, 448 * KIB)
    assert budget.token_budget(TILE, MAX_TOKENS) == 192
    _available(memory, 288 * KIB)
    assert budget.token_budget(TILE, MAX_TOKENS) == TILE


def test_fractional_tile_capacity_rounds_down(memory):
    budget = _calibrated(memory)
    _available(memory, 256 * KIB + 230 * KIB + 511)
    assert budget.token_budget(TILE, MAX_TOKENS) == 192


def test_largest_observed_workspace_controls_later_budgets(memory):
    budget = _calibrated(memory)
    _available(memory, 1024 * KIB)
    assert budget.token_budget(TILE, MAX_TOKENS) == 768
    _observe(budget, memory, retained=0, peak=512 * KIB, first_stage=False)
    assert budget.token_budget(TILE, MAX_TOKENS) == 512
    _observe(budget, memory, retained=0, peak=128 * KIB, first_stage=False)
    assert budget.token_budget(TILE, MAX_TOKENS) == 512


def test_only_first_stage_growth_updates_maximum_retained_bytes_per_row(memory):
    budget = _calibrated(memory)
    _observe(budget, memory, retained=128 * KIB, first_stage=False)
    _available(memory, 1024 * KIB)
    assert budget.token_budget(TILE, MAX_TOKENS) == 768
    _observe(budget, memory, retained=128 * KIB)
    _available(memory, 1024 * KIB)
    assert budget.token_budget(TILE, MAX_TOKENS) == 384
    _observe(budget, memory, retained=64 * KIB)
    _available(memory, 1024 * KIB)
    assert budget.token_budget(TILE, MAX_TOKENS) == 384


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_memory_measurement_does_not_modify_cuda_tensor_contents():
    expected = torch.arange(512, dtype=torch.int32).reshape(64, 8)
    original = expected.cuda()
    budget = PrefillMemoryBudget(original.device)
    before = budget.start()
    retained = torch.ones(64, 8, dtype=torch.float32, device=original.device)
    budget.record(before, TILE, first_stage=True)
    allowed = budget.token_budget(TILE, 4 * TILE)
    assert TILE <= allowed <= 4 * TILE
    assert allowed % TILE == 0
    torch.testing.assert_close(original.cpu(), expected)
    torch.testing.assert_close(retained.cpu(), torch.ones_like(retained, device="cpu"))


class WindowCache:
    available_size = 4096
    is_hybrid = False
    swa_paged = True
    page_size = 8
    sliding_window_size = 16
    swa_available_size = 4096

    def __init__(self):
        self.active_locks = 0

    def match_req(self, pending):
        return SimpleNamespace(cuda_handle=SimpleNamespace(cached_len=0), mamba_value=None)

    def lock(self, handle):
        self.active_locks += 1

    def unlock(self, handle):
        self.active_locks -= 1

    def decode_reserved_tokens_for(self, reqs):
        return 0

    def decode_swa_reservation(self, reqs):
        return 0

    def incremental_prefill_window_reservation(self, remain_len):
        return min(remain_len, self.sliding_window_size)


def _window_manager(length):
    cache = WindowCache()
    table = TableManager(2, torch.zeros((3, 128), dtype=torch.int32))
    prompt = torch.arange(length, dtype=torch.int32)
    pending = PendingReq(42, prompt, SamplingParams(max_tokens=4))
    return PrefillManager(cache, table, DecodeManager(cache.page_size), [pending]), cache, table, prompt


@pytest.mark.parametrize("length,budget,chunk_limit,ends", [
    (35, 12, None, [8, 16, 24, 35]),
    (35, 24, 12, [8, 16, 24, 35]),
    (17, 8, None, [8, 16, 17]),
    (8, 8, None, [8]),
])
def test_window_prefill_segments_end_on_pages_without_shortening_prompt(length, budget, chunk_limit, ends):
    manager, cache, table, prompt = _window_manager(length)
    available = table.available_size
    start = 0
    table_idx = None
    for end in ends:
        batch = manager.schedule_next_batch(
            budget, chunk_token_limit=chunk_limit, incremental_window_prefill=True,
        )
        assert batch is not None and len(batch.reqs) == 1
        req = batch.reqs[0]
        assert req.cached_len == start
        assert req.device_len == end
        assert req.extend_len == end - start
        assert req.extend_len <= (budget if chunk_limit is None else min(budget, chunk_limit))
        torch.testing.assert_close(req.input_ids[:end], prompt[:end])
        table_idx = req.table_idx if table_idx is None else table_idx
        assert req.table_idx == table_idx
        assert table.available_size == available - 1
        if end < length:
            assert isinstance(req, ChunkedReq)
            assert end % cache.page_size == 0
            assert not req.can_decode
            assert len(manager.pending_list) == 1
            torch.testing.assert_close(manager.pending_list[0].input_ids, prompt)
            req.commit_prefill_kv()
        else:
            assert not isinstance(req, ChunkedReq)
            assert req.can_decode
            assert manager.pending_list == []
        start = end


def test_subpage_budget_defers_without_leaks_and_preserves_existing_request_state():
    manager, cache, table, prompt = _window_manager(17)
    available = table.available_size
    assert manager.schedule_next_batch(7, incremental_window_prefill=True) is None
    assert table.available_size == available
    assert cache.active_locks == 0
    assert len(manager.pending_list) == 1
    assert manager.pending_list[0].chunked_req is None
    torch.testing.assert_close(manager.pending_list[0].input_ids, prompt)

    req = manager.schedule_next_batch(8, incremental_window_prefill=True).reqs[0]
    assert isinstance(req, ChunkedReq)
    req.linear_slot_idx = 5
    req.mamba_ping_pong = (6, 7)
    req.commit_prefill_kv()
    table_idx = req.table_idx
    cache_handle = req.cache_handle
    occupied = table.available_size
    locks = cache.active_locks
    assert manager.schedule_next_batch(7, incremental_window_prefill=True) is None
    assert table.available_size == occupied
    assert cache.active_locks == locks
    assert req.cached_len == 8
    assert req.linear_slot_idx == 5
    assert req.mamba_ping_pong == (6, 7)

    for budget, end in ((8, 16), (1, 17)):
        req = manager.schedule_next_batch(budget, incremental_window_prefill=True).reqs[0]
        assert req.device_len == end
        assert req.table_idx == table_idx
        assert req.cache_handle is cache_handle
        assert req.linear_slot_idx == 5
        assert req.mamba_ping_pong == (6, 7)
        torch.testing.assert_close(req.input_ids[:end], prompt[:end])
        if isinstance(req, ChunkedReq):
            req.commit_prefill_kv()
    assert req.can_decode
    assert manager.pending_list == []
