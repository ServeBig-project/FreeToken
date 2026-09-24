"""Public-contract checks for logical-to-physical expert mapping."""

import pytest
import torch
import torch.nn.functional as F

from freetoken.moe.fused import fused_experts_impl, moe_align_block_size
from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
NUM_EXPERTS = 256
PHYSICAL_ROWS = 3092
HIDDEN = 128
INTERMEDIATE = 64


def _routes(tokens, pattern, topk=2):
    positions = torch.arange(tokens * topk)
    if pattern == "spread":
        ids = positions.remainder(NUM_EXPERTS)
    else:
        ids = torch.tensor([0, 0, 255, 3, 3, 128, 255, 0])[positions.remainder(8)]
    return ids.reshape(tokens, topk).to(torch.int32)


def _mapping(kind):
    if kind == "none":
        return None
    logical = torch.arange(NUM_EXPERTS, dtype=torch.int32)
    return (logical * 17 + 13).remainder(NUM_EXPERTS) if kind == "permuted" else logical * 12 + 31


def _check_alignment(result, route_ids, block_size, mapping):
    sorted_ids, block_experts, padded_count = result
    for tensor in result:
        assert tensor.is_cuda
        assert tensor.dtype == torch.int32
        assert tensor.ndim == 1
    assert padded_count.shape == (1,)
    count = padded_count.item()
    routes = route_ids.flatten().long()
    counts = torch.bincount(routes, minlength=NUM_EXPERTS)
    assert count == int(((counts + block_size - 1) // block_size).sum()) * block_size
    assert sorted_ids.numel() >= count
    assert block_experts.numel() >= count // block_size
    sorted_ids = sorted_ids[:count].cpu().long().reshape(-1, block_size)
    valid = sorted_ids < routes.numel()
    assert (sorted_ids >= 0).all()
    assert valid.any(dim=1).all()
    torch.testing.assert_close(sorted_ids[valid].sort().values, torch.arange(routes.numel()))
    logical = routes[sorted_ids.clamp(max=routes.numel() - 1)]
    first_valid = valid.to(torch.int64).argmax(dim=1)
    per_block = logical[torch.arange(logical.shape[0]), first_valid]
    assert (logical[valid] == per_block[:, None].expand_as(logical)[valid]).all()
    expected = per_block if mapping is None else mapping[per_block].long()
    torch.testing.assert_close(block_experts[:expected.numel()].cpu().long(), expected)


@pytest.mark.parametrize("tokens", [1, 80, 8192])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("mapping_kind", ["none", "permuted", "far"])
@pytest.mark.parametrize("pattern", ["spread", "repeated"])
def test_alignment_routes_and_physical_rows(tokens, block_size, mapping_kind, pattern):
    route_ids = _routes(tokens, pattern)
    mapping = _mapping(mapping_kind)
    device_ids = route_ids.cuda()
    device_map = None if mapping is None else mapping.cuda()
    result = moe_align_block_size(device_ids, block_size, NUM_EXPERTS, expert_map=device_map)
    _check_alignment(result, route_ids, block_size, mapping)
    torch.testing.assert_close(device_ids.cpu(), route_ids)
    if device_map is not None:
        torch.testing.assert_close(device_map.cpu(), mapping)


def _capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    return graph, result


@pytest.mark.parametrize("block_size", [16, 32])
def test_alignment_graph_replay_reads_current_map(block_size):
    route_ids = _routes(80, "repeated")
    device_ids = route_ids.cuda()
    mapping = _mapping("far")
    device_map = mapping.cuda()
    graph, result = _capture(
        lambda: moe_align_block_size(device_ids, block_size, NUM_EXPERTS, expert_map=device_map)
    )
    for current_map in (mapping, mapping.roll(1), mapping):
        device_map.copy_(current_map)
        graph.replay()
        _check_alignment(result, route_ids, block_size, current_map)
        torch.testing.assert_close(device_ids.cpu(), route_ids)
        torch.testing.assert_close(device_map.cpu(), current_map)


def _scatter_bank(compact, mapping):
    physical = torch.zeros((PHYSICAL_ROWS, *compact.shape[1:]), dtype=compact.dtype, device="cuda")
    physical.view(torch.uint8)[mapping.cuda().long()] = compact.view(torch.uint8)
    return physical


@pytest.fixture(scope="module")
def bf16_banks():
    generator = torch.Generator(device="cuda").manual_seed(4701)
    w1 = (torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator,
                      device="cuda") / HIDDEN ** 0.5).bfloat16()
    w2 = (torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator=generator,
                      device="cuda") / INTERMEDIATE ** 0.5).bfloat16()
    mapping = _mapping("far")
    return (w1, w2), (_scatter_bank(w1, mapping), _scatter_bank(w2, mapping))


def _inputs(tokens):
    generator = torch.Generator(device="cuda").manual_seed(8301 + tokens)
    hidden = torch.randn(tokens, HIDDEN, device="cuda", generator=generator).bfloat16()
    weights = torch.rand(tokens, 2, device="cuda", generator=generator) + 0.2
    return hidden, weights / weights.sum(dim=1, keepdim=True), _routes(tokens, "repeated").cuda()


def _bf16_reference(hidden, w1, w2, weights, ids, mapping, weight_on_input):
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    for logical in ids.unique().tolist():
        tokens, routes = (ids == logical).nonzero(as_tuple=True)
        selected = hidden[tokens].float()
        route_weights = weights[tokens, routes, None]
        if weight_on_input:
            selected = selected * route_weights
        physical = logical if mapping is None else int(mapping[logical])
        gate, up = F.linear(selected, w1[physical].float()).chunk(2, dim=-1)
        output = F.linear(F.silu(gate) * up, w2[physical].float())
        if not weight_on_input:
            output = output * route_weights
        expected.index_add_(0, tokens, output)
    return expected


def _check_bf16(actual, hidden, expected):
    assert actual.shape == hidden.shape
    assert actual.dtype == hidden.dtype == torch.bfloat16
    assert actual.device == hidden.device
    torch.testing.assert_close(actual.float(), expected, rtol=0.025, atol=0.015)
    torch.testing.assert_close(hidden.float(), expected, rtol=0.025, atol=0.015)


@pytest.mark.parametrize("tokens", [1, 80, 8192])
@pytest.mark.parametrize("use_mapping", [False, True])
@pytest.mark.parametrize("weight_on_input", [False, True])
def test_bf16_matches_independent_formula(bf16_banks, tokens, use_mapping, weight_on_input):
    hidden, weights, ids = _inputs(tokens)
    original = hidden.clone()
    w1, w2 = bf16_banks[int(use_mapping)]
    mapping = _mapping("far") if use_mapping else None
    expected = _bf16_reference(original, w1, w2, weights, ids, mapping, weight_on_input)
    actual = fused_experts_impl(
        hidden, w1, w2, weights, ids, activation="silu",
        apply_router_weight_on_input=weight_on_input,
        expert_map=None if mapping is None else mapping.cuda(),
    )
    _check_bf16(actual, hidden, expected)


@pytest.mark.parametrize("weight_on_input", [False, True])
def test_bf16_graph_replay_reads_current_map(bf16_banks, weight_on_input):
    original, weights, ids = _inputs(80)
    hidden = original.clone()
    w1, w2 = bf16_banks[1]
    mapping = _mapping("far")
    device_map = mapping.cuda()

    def call():
        hidden.copy_(original)
        return fused_experts_impl(
            hidden, w1, w2, weights, ids,
            apply_router_weight_on_input=weight_on_input, expert_map=device_map,
        )

    graph, actual = _capture(call)
    for current_map in (mapping, mapping.roll(1), mapping):
        device_map.copy_(current_map)
        graph.replay()
        expected = _bf16_reference(original, w1, w2, weights, ids, current_map, weight_on_input)
        _check_bf16(actual, hidden, expected)


@pytest.fixture(scope="module")
def nvfp4_banks():
    generator = torch.Generator(device="cuda").manual_seed(6811)
    shapes = [(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN), (NUM_EXPERTS, HIDDEN, INTERMEDIATE)]
    compact = []
    for experts, rows, columns in shapes:
        compact.extend((
            torch.randint(256, (experts, rows, columns // 2), generator=generator,
                          dtype=torch.uint8, device="cuda"),
            torch.full((experts, rows, columns // 16), 0.125, device="cuda").to(torch.float8_e4m3fn),
            torch.ones((experts, rows), dtype=torch.float16, device="cuda"),
        ))
    return compact, [_scatter_bank(bank, _mapping("far")) for bank in compact]


@pytest.mark.parametrize("tokens", [1, 80, 8192])
@pytest.mark.parametrize("weight_on_input", [False, True])
def test_nvfp4_scattered_banks_match_compact_banks(nvfp4_banks, tokens, weight_on_input):
    original, weights, ids = _inputs(tokens)
    compact, physical = nvfp4_banks
    expected = fused_experts_nvfp4(
        original.clone(), *compact, weights, ids, NUM_EXPERTS,
        apply_router_weight_on_input=weight_on_input,
    )
    actual = fused_experts_nvfp4(
        original.clone(), *physical, weights, ids, NUM_EXPERTS,
        apply_router_weight_on_input=weight_on_input, expert_map=_mapping("far").cuda(),
    )
    assert expected.abs().max().item() > 0.01
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.005)


def test_nvfp4_graph_replay_reads_current_map(nvfp4_banks):
    original, weights, ids = _inputs(80)
    compact, physical = nvfp4_banks
    hidden = original.clone()
    mapping = _mapping("far")
    device_map = mapping.cuda()

    def call():
        hidden.copy_(original)
        return fused_experts_nvfp4(
            hidden, *physical, weights, ids, NUM_EXPERTS, expert_map=device_map,
        )

    graph, actual = _capture(call)
    for shift in (0, 1, 0):
        device_map.copy_(mapping.roll(shift))
        graph.replay()
        expected = fused_experts_nvfp4(
            original.clone(), *compact, weights, (ids - shift).remainder(NUM_EXPERTS), NUM_EXPERTS,
        )
        torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.005)
