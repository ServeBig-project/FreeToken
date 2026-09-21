"""Source-blind checks of expert-cache queries and their observable copies."""

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
PINNED_USAGE = torch.iinfo(torch.int64).max


def _make_cache(layers=3, experts=4, slots=6):
    sources = {"gate_up": [], "down": []}
    for layer in range(layers):
        for name, rows, columns in (("gate_up", 128, 128), ("down", 128, 64)):
            source = torch.empty(experts, rows, columns, dtype=torch.bfloat16, pin_memory=True)
            source.copy_(torch.arange(rows * columns).reshape(rows, columns).remainder(31) / 32)
            source[..., 0] = layer + 1
            source[..., 1] = torch.arange(experts).reshape(experts, 1)
            sources[name].append(source)
    cache = OffloadMoeCache(layers, experts, slots, torch.device("cuda"), cache_policy="lru")
    cache.set_bank_sources(sources)
    cache.bank_views()
    return cache, sources


def _ids(values):
    return torch.tensor(values, dtype=torch.int32).reshape(1, -1)


def _check_request(cache, sources, layer, logical, device_ids, execute, policy="lru"):
    before_map = cache.slot_for_id.cpu().long()
    before_ids = cache.id_of_slot.cpu().long()
    before_usage = cache.usage.cpu()
    before_step = int(cache.step)
    layers, experts = before_map.shape
    requested = sorted(set(logical.flatten().tolist()))
    hits = {int(before_map[layer, expert]) for expert in requested if before_map[layer, expert] >= 0}
    missing = [expert for expert in requested if before_map[layer, expert] < 0]
    eligible = [slot for slot in range(before_ids.numel())
                if slot not in hits and before_usage[slot] != PINNED_USAGE]

    def replacement_order(slot):
        if policy == "lru":
            return int(before_usage[slot]), slot
        occupant = int(before_ids[slot])
        distance = (occupant // experts - layer) % layers or layers
        return occupant >= 0, -distance if occupant >= 0 else 0, int(before_usage[slot]), slot

    replacements = sorted(eligible, key=replacement_order)[:len(missing)]
    assert len(replacements) == len(missing)
    expected_map = before_map.clone()
    expected_ids = before_ids.clone()
    expected_usage = before_usage.clone()
    evicted = []
    for expert, slot in zip(missing, replacements):
        occupant = int(before_ids[slot])
        if occupant >= 0:
            evicted.append((occupant // experts, occupant % experts))
        expected_map[layer, expert] = slot
        expected_ids[slot] = layer * experts + expert
    for expert in requested:
        slot = int(expected_map[layer, expert])
        if expected_usage[slot] != PINNED_USAGE:
            expected_usage[slot] = before_step + 1
    expected_routes = expected_map[layer, logical.long()]

    device_ids.copy_(logical)
    torch.cuda.reset_peak_memory_stats()
    before_bytes = torch.cuda.memory_allocated()
    execute()
    torch.cuda.synchronize()
    extra_bytes = torch.cuda.max_memory_allocated() - before_bytes

    torch.testing.assert_close(device_ids.cpu().long(), expected_routes)
    assert int(cache.step) == before_step + 1
    assert cache.usage.dtype == torch.int64
    count = int(cache.num_indices)
    assert count == len(missing)
    torch.testing.assert_close(cache.src_indices[:count].cpu().long(), torch.tensor(missing, dtype=torch.int64))
    torch.testing.assert_close(cache.evict_slots[:count].cpu().long(), torch.tensor(replacements, dtype=torch.int64))
    actual_map = cache.slot_for_id.cpu().long()
    for old_layer, old_expert in evicted:
        assert actual_map[old_layer, old_expert] < 0
        expected_map[old_layer, old_expert] = actual_map[old_layer, old_expert]
    torch.testing.assert_close(actual_map, expected_map)
    torch.testing.assert_close(cache.id_of_slot.cpu().long(), expected_ids)
    torch.testing.assert_close(cache.usage.cpu(), expected_usage)
    requested_slots = expected_map[layer, requested].cuda()
    for name, bank in zip(("gate_up", "down"), cache.bank_views()):
        torch.testing.assert_close(bank[requested_slots].cpu(), sources[name][layer][requested], rtol=0, atol=0)
    return extra_bytes


def _request(cache, sources, layer, values, decode=False):
    logical = _ids(values)
    device_ids = logical.cuda()

    def execute():
        method = cache.ensure_decode_experts if decode else cache.ensure_experts
        method(layer, device_ids)
        cache.copy_missing()

    if decode:
        assert logical.numel() * len(sources["gate_up"]) > cache.decode_cache_size
    return _check_request(cache, sources, layer, logical, device_ids, execute,
                          policy="distance" if decode else "lru")


def test_repeated_routes_copy_each_missing_expert_once():
    cache, sources = _make_cache()
    _request(cache, sources, 0, [3, 1, 1, 3, 0, 3, 0, 1])
    _request(cache, sources, 0, [3, 3, 1, 0])
    _request(cache, sources, 0, [2, 1, 2, 1])


def test_lru_cross_layer_eviction_protects_this_requests_hits():
    cache, sources = _make_cache(slots=4)
    for layer, logical in ((0, [0, 1, 2, 3]), (1, [0, 1]), (0, [2, 0, 2, 0]),
                           (2, [1, 3]), (1, [0, 1])):
        _request(cache, sources, layer, logical)


def test_fixed_resident_slot_keeps_its_data_and_usage_marker():
    cache, sources = _make_cache()
    _request(cache, sources, 0, [0, 1, 2, 3])
    fixed_slot = int(cache.slot_for_id[0, 0])
    cache.usage[fixed_slot] = PINNED_USAGE
    _request(cache, sources, 0, [0, 2])
    _request(cache, sources, 1, [0, 1, 2, 3])
    _request(cache, sources, 2, [0, 1, 2, 3])
    assert int(cache.slot_for_id[0, 0]) == fixed_slot
    for name, bank in zip(("gate_up", "down"), cache.bank_views()):
        torch.testing.assert_close(bank[fixed_slot].cpu(), sources[name][0][0], rtol=0, atol=0)


@pytest.mark.parametrize("queries", [8, 640, 641, 1024, 65536])
def test_large_queries_keep_temporary_cuda_memory_small(queries):
    cache, sources = _make_cache(layers=16, experts=256, slots=3092)
    assert cache.id_of_slot.numel() == 3092
    distinct = 7 if queries == 641 else 256
    topk = 1 if queries == 641 else 8
    logical = (torch.arange(queries).remainder(distinct) * 37).remainder(256).to(torch.int32).reshape(-1, topk)
    device_ids = logical.cuda()

    def execute():
        cache.ensure_experts(0, device_ids)
        cache.copy_missing()

    extra_bytes = _check_request(cache, sources, 0, logical, device_ids, execute)
    assert extra_bytes < 128 * 1024 * 1024, f"Q={queries}: extra CUDA peak was {extra_bytes / 2**20:.1f} MiB"


def test_graph_replay_uses_new_logical_routes_and_copies_new_misses():
    cache, sources = _make_cache()
    _request(cache, sources, 1, [0, 1, 2, 3])
    initial = _ids([0, 0, 1, 1, 0, 0, 1, 1]).cuda()
    device_ids = initial.clone()

    def execute():
        cache.ensure_experts(0, device_ids)
        cache.copy_missing()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            device_ids.copy_(initial)
            execute()
    torch.cuda.current_stream().wait_stream(stream)
    device_ids.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    for values in ([2, 3, 2, 3, 2, 3, 2, 3], [0, 2, 0, 2, 0, 2, 0, 2],
                   [1, 3, 1, 3, 1, 3, 1, 3], [0, 1, 0, 1, 0, 1, 0, 1]):
        _check_request(cache, sources, 0, _ids(values), device_ids, graph.replay)


@pytest.mark.parametrize("pin_current_layer_candidate", [False, True])
def test_decode_distance_policy_prefers_empty_slots_and_protects_hits(pin_current_layer_candidate):
    cache, sources = _make_cache(layers=4, slots=8)
    for layer, logical in ((1, [0, 1]), (2, [0, 1]), (3, [0, 1]), (0, [0])):
        _request(cache, sources, layer, logical)
    _request(cache, sources, 0, [0, 1, 2, 3], decode=True)
    if pin_current_layer_candidate:
        cache.usage[int(cache.slot_for_id[1, 1])] = PINNED_USAGE
    _request(cache, sources, 1, [0, 2, 3, 2], decode=True)
