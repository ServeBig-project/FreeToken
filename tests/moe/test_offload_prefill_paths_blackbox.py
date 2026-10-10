"""Black-box tests for the offload MoE prefill paths, written from the public contract only.

A prefill chunk of T tokens with T * K < E * ln(2E) whose routes touch fewer than E experts
loads only its routed experts into the GPU slot cache, like decode. A longer chunk, or one
routed to every expert, streams whole layers into the prefill buffers instead.

Prefill chunks are checked against a float32 reference with the expert path's bf16 roundings.
A one-token decode step forms its products in bf16 before reducing them, so it is checked bit
for bit against the same decode rerun with its experts resident, plus a norm bound.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import freetoken.core as core
from freetoken.core import Batch, Context, Req
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.expert_format import ExpertLayout, bind_expert_method, expert_math
from freetoken.moe.offload_cache import OffloadMoeCache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the offload MoE path runs a CUDA kernel"
)

DEVICE = torch.device("cuda")
E, K, H, I = 16, 4, 256, 128
NUM_LAYERS = 3
# The shortest chunk with T * K >= E * ln(2E); it streams whole layers whatever its routes.
LONG_TOKENS = math.ceil(E * math.log(2 * E) / K)
LAYER_BYTES = E * (2 * I * H + H * I) * 2  # one layer's gate_up and down weights in bf16
# Bound on ||decode out - ref|| / ||ref|| per row: twice the worst (0.059) seen when emulating
# the decode precision with its least precise summation order, half the least (0.24) that one
# wrong expert causes.
DECODE_REL_ERR = 0.12


def _routes(tokens, experts):
    """Token t routes to the next K of `experts` (cyclic), so no row repeats an expert."""
    experts = torch.tensor(list(experts), dtype=torch.int32)
    return experts[(torch.arange(tokens)[:, None] * K + torch.arange(K)) % len(experts)]


def _random_routes(tokens, seed):
    """K distinct random experts per token, like a router's top-k."""
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(tokens, E, generator=gen).argsort(dim=1)[:, :K].int()


def _build(prefill_overlap):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    gen = torch.Generator().manual_seed(0)

    # 1/sqrt(fan-in) keeps every expert's output of order 1; pinned because the cache's
    # GPU copy reads the host banks directly.
    def weight(*shape):
        return (torch.randn(*shape, generator=gen) / shape[-1] ** 0.5).bfloat16().pin_memory()

    w_gu = [weight(E, 2 * I, H) for _ in range(NUM_LAYERS)]
    w_d = [weight(E, H, I) for _ in range(NUM_LAYERS)]
    # 2E slots: the fewest prefill_overlap=True accepts, and too few to keep every layer on the GPU.
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=2 * E,
        device=DEVICE,
        prefill_overlap=prefill_overlap,
    )
    cache.set_bank_sources({"gate_up": w_gu, "down": w_d})
    layers = [
        OffloadMoELayer(layer_id=i, num_experts=E, top_k=K, hidden_size=H, intermediate_size=I)
        for i in range(NUM_LAYERS)
    ]
    for layer in layers:
        layer.offload_cache = cache
        # The engine binds the offload layers' expert method; these tests build layers directly.
        layer.expert_method = bind_expert_method(
            expert_math(layer),
            ExpertLayout("bf16", layer.hidden_size, layer.intermediate_size, layer.num_experts),
            None,
            device=DEVICE,
            backend="offload",
        )
    return SimpleNamespace(cache=cache, layers=layers, w_gu=w_gu, w_d=w_d, gen=gen)


def _counters(cache):
    torch.cuda.synchronize()
    stats = cache.decode_miss_stats()
    return cache.prefill_layer_prepares, cache.prefill_h2d_bytes, stats["layer_calls"]


def _bf16(t):
    return t.bfloat16().float()


def _reference(x, ids, weights, w_gu, w_d):
    """sum_k w_k * W_d[e_k] @ (silu(gate) * up), where [gate; up] = W_gu[e_k] @ x, in float32
    rounded to bf16 where the expert path rounds: gate/up, the activation, each weighted
    expert output and the sum."""
    ids = ids.long()
    h = _bf16(torch.einsum("tkoh,th->tko", w_gu[ids].float(), x.float()))
    a = _bf16(F.silu(h[..., :I]) * h[..., I:])
    y = _bf16(weights[..., None] * torch.einsum("tkhi,tki->tkh", w_d[ids].float(), a))
    return _bf16(y.sum(dim=1))


def _rms_norm(x):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True))).bfloat16()


def _forward(ctx, model, x, weights, routes, decode):
    """Run x through every layer in order, as a model forward does; return each layer's input
    and output."""
    tokens = len(routes)
    req = Req(
        input_ids=torch.tensor([0] * tokens, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=0,
        sampling_params=None,
        cache_handle=None,
    )
    batch = Batch(reqs=[req], decode_size=1 if decode else 0)
    batch.positions = torch.arange(tokens, dtype=torch.int32, device=DEVICE)
    batch.num_token_non_padded = None
    if decode:
        batch.num_token_non_padded = torch.tensor(1, dtype=torch.int32, device=DEVICE)
    inputs, outputs = [], []
    with ctx.forward_batch(batch):
        for layer in model.layers:
            # The layer may write into the tensors it is given, so it only gets copies.
            out = layer.routed_forward(
                x.clone(), weights.to(DEVICE, copy=True), routes.to(DEVICE, copy=True)
            )
            inputs.append(x)
            outputs.append(out.clone())
            x = _rms_norm(out)  # stands in for the norm between decoder layers
    return inputs, outputs


def _step(ctx, model, routes, decode=False):
    """Run one batch through the layers and check every layer against the reference for the
    input that layer received. Returns how much the prefill layer prepares, the prefill H2D
    bytes and the on-demand layer calls (counted only with collect_stats) grew."""
    weights = torch.rand(len(routes), K, generator=model.gen) * 0.5 + 0.25
    x = torch.randn(len(routes), H, generator=model.gen).to(DEVICE, torch.bfloat16)
    before = _counters(model.cache)
    inputs, outputs = _forward(ctx, model, x, weights, routes, decode)
    after = _counters(model.cache)
    got = torch.stack(outputs).float().cpu()
    ref = torch.stack([
        _reference(inp.cpu(), routes, weights, model.w_gu[i], model.w_d[i])
        for i, inp in enumerate(inputs)
    ])
    if decode:
        _, resident = _forward(ctx, model, x, weights, routes, decode)
        assert [torch.equal(a, b) for a, b in zip(outputs, resident)] == [True] * NUM_LAYERS
        rel = (got - ref).norm(dim=-1) / ref.norm(dim=-1)
        assert (rel < DECODE_REL_ERR).all(), rel.flatten().tolist()
    else:
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    return tuple(a - b for a, b in zip(after, before))


@pytest.fixture
def ctx(monkeypatch):
    ctx = Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    return ctx


@pytest.mark.parametrize("prefill_overlap", [False, True])
@pytest.mark.parametrize("tokens", [3, LONG_TOKENS - 1])
def test_short_chunk_loads_only_routed_experts(ctx, prefill_overlap, tokens):
    model = _build(prefill_overlap)
    model.cache.collect_stats = True  # layer_calls is opt-in; the other tests keep the default
    prepares, h2d_bytes, layer_calls = _step(ctx, model, _routes(tokens, range(10)))
    assert (prepares, h2d_bytes) == (0, 0)
    assert layer_calls > 0


@pytest.mark.parametrize("prefill_overlap", [False, True])
def test_requests_after_on_demand_load_stay_correct(ctx, prefill_overlap):
    model = _build(prefill_overlap)
    _step(ctx, model, _routes(3, range(6)))  # loads experts 0-5 of every layer
    _step(ctx, model, _routes(1, [2, 0, 5, 3]), decode=True)  # all already loaded
    _step(ctx, model, _routes(1, [9, 1, 12, 4]), decode=True)  # 9 and 12 not loaded yet
    _step(ctx, model, _routes(3, [0, 7, 3, 10, 5, 11, 1]))  # 7, 10 and 11 not loaded yet


@pytest.mark.parametrize("prefill_overlap", [False, True])
@pytest.mark.parametrize(
    "routes",
    [_random_routes(32, seed=1), _routes(LONG_TOKENS, range(8)), _routes(4, range(E))],
    ids=["long", "shortest-long-few-experts", "short-every-expert"],
)
def test_whole_layer_prefill_streams_each_layer_once(ctx, prefill_overlap, routes):
    model = _build(prefill_overlap)
    prepares, h2d_bytes, _ = _step(ctx, model, routes)
    # Only the overlapped path specifies these counters; without overlap the outputs are the check.
    if prefill_overlap:
        assert prepares == NUM_LAYERS
        assert h2d_bytes == NUM_LAYERS * LAYER_BYTES


@pytest.mark.parametrize("prefill_overlap", [False, True])
def test_mixed_short_and_long_chunks_stay_correct(ctx, prefill_overlap):
    model = _build(prefill_overlap)
    _step(ctx, model, _routes(3, range(6)))  # short, on demand
    _step(ctx, model, _random_routes(32, seed=1))  # long after short
    _step(ctx, model, _routes(1, [3, 1, 4, 0]), decode=True)  # decode after long
    _step(ctx, model, _routes(3, range(6)))  # short after long, same experts as the first
    _step(ctx, model, _random_routes(32, seed=2))  # long after on-demand loads
