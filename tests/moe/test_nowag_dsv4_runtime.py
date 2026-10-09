from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


def _key(layer: int, expert: int, projection: str, suffix: str) -> str:
    return f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"


def _dsv4_math(swiglu_limit):
    from freetoken.moe.expert_format import E4M3_GROUP128_UE8M0, ExpertMath

    return ExpertMath(
        activation_limit=swiglu_limit,
        gate_up_input_rounding=E4M3_GROUP128_UE8M0,
        down_input_rounding=E4M3_GROUP128_UE8M0,
    )


def _run_nowag(math, x, slots, weights, codebook, *banks):
    """Bind the NoWAG method for these bank shapes and run it once."""
    from freetoken.moe.expert_format import _BANK_SCHEMAS, ExpertLayout, bind_expert_method
    from freetoken.moe.nowag.weights import NowagState

    layout = ExpertLayout("nowag", x.shape[1], banks[2].shape[1], banks[0].shape[0])
    method = bind_expert_method(
        math, layout, NowagState(codebook.shape[1], 12), device=x.device, backend="offload"
    )
    return method.run(
        x, slots, weights, dict(zip(_BANK_SCHEMAS["nowag"], banks)), {"codebook": codebook}
    )


def test_nowag_sidecar_maps_three_projections_to_nine_banks(tmp_path, monkeypatch):
    from freetoken.moe.host_banks import HostBank
    from freetoken.moe.nowag.weights import NowagState, load_nowag_expert_sources

    monkeypatch.setattr(HostBank, "pin", lambda self: setattr(self, "_pinned", True))
    output = tmp_path / "nowag"
    output.mkdir()

    hidden, intermediate, experts = 12, 6, 2
    tensors = {}
    projection_shapes = {
        "w1": (intermediate, 1, hidden, intermediate),
        "w3": (intermediate, 1, hidden, intermediate),
        "w2": (hidden, 1, intermediate, hidden),
    }
    for expert in range(experts):
        for projection, (rows, words, in_size, out_size) in projection_shapes.items():
            value = {"w1": 10, "w3": 20, "w2": 30}[projection] + expert
            tensors[_key(0, expert, projection, "assignments")] = torch.full(
                (rows, words), value, dtype=torch.int32
            )
            tensors[_key(0, expert, projection, "normalizer.norms.0")] = torch.full(
                (in_size,), float(value), dtype=torch.bfloat16
            )
            tensors[_key(0, expert, projection, "normalizer.norms.1")] = torch.full(
                (out_size,), float(value + 1), dtype=torch.bfloat16
            )
    save_file(tensors, output / "layer-000.safetensors")
    save_file(
        {"global_all.codebook": torch.zeros(4096, 6, dtype=torch.bfloat16)},
        output / "global_codebook.safetensors",
    )
    manifest = {
        "format": "deepseek_v4_nowag_expert_sidecar_v1",
        "scope": "expert_only",
        "codebook_sharing": "global_all",
        "source_model": "/cluster/models/DeepSeek-V4-Flash-0731",
        "d": 6,
        "assignment_bits": 12,
        "assignments_packed": True,
        "matrix_count": experts * 3,
        "codebook": {
            "file": "global_codebook.safetensors",
            "tensor": "global_all.codebook",
        },
        "layers": [{"layer": 0, "file": "layer-000.safetensors"}],
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = SimpleNamespace(
        model_type="deepseek_v4",
        num_moe_layers=1,
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
    )

    banks, shared, state = load_nowag_expert_sources(output, config)

    assert set(banks) == {
        "gate_assignments",
        "gate_input_norm",
        "gate_output_norm",
        "up_assignments",
        "up_input_norm",
        "up_output_norm",
        "down_assignments",
        "down_input_norm",
        "down_output_norm",
    }
    assert banks["gate_assignments"][0][1, 0, 0].item() == 11
    assert banks["up_assignments"][0][1, 0, 0].item() == 21
    assert banks["down_assignments"][0][1, 0, 0].item() == 31
    assert shared["codebook"].shape == (4096, 6)
    assert state == NowagState(d=6, assignment_bits=12)


@pytest.mark.parametrize("source_layout", [None, "row_major", "word_major"])
def test_qwen_nowag_sidecar_uses_common_manifest_and_real_shapes(
    tmp_path, monkeypatch, source_layout
):
    from freetoken.moe.host_banks import HostBank
    from freetoken.moe.nowag.weights import load_nowag_expert_sources

    monkeypatch.setattr(HostBank, "pin", lambda self: setattr(self, "_pinned", True))
    output = tmp_path / "qwen-nowag"
    output.mkdir()

    hidden, intermediate, experts = 13, 7, 2
    gate_words = 2  # ceil(ceil(13 / 6) * 12 / 32)
    down_words = 1  # ceil(ceil(7 / 6) * 12 / 32)
    tensors = {}
    logical_assignments = {}
    projection_shapes = {
        "w1": (intermediate, gate_words, hidden, intermediate),
        "w3": (intermediate, gate_words, hidden, intermediate),
        "w2": (hidden, down_words, intermediate, hidden),
    }
    for expert in range(experts):
        for projection, (rows, words, in_size, out_size) in projection_shapes.items():
            value = {"w1": 10, "w3": 20, "w2": 30}[projection] + expert
            key = _key(0, expert, projection, "assignments")
            logical = torch.arange(rows * words, dtype=torch.int32).view(rows, words)
            logical.add_(value * 100)
            logical_assignments[key] = logical
            tensors[key] = (
                logical
                if source_layout in (None, "row_major")
                else logical.t().contiguous()
            )
            tensors[_key(0, expert, projection, "normalizer.norms.0")] = torch.full(
                (in_size,), float(value), dtype=torch.bfloat16
            )
            tensors[_key(0, expert, projection, "normalizer.norms.1")] = torch.full(
                (out_size,), float(value + 1), dtype=torch.bfloat16
            )
    save_file(tensors, output / "layer-000.safetensors")
    save_file(
        {"global_all.codebook": torch.zeros(4096, 6, dtype=torch.bfloat16)},
        output / "global_codebook.safetensors",
    )
    manifest = {
        "format": "nowag_expert_sidecar_v1",
        "model_type": "qwen3_5_moe",
        "scope": "expert_only",
        "codebook_sharing": "global_all",
        "d": 6,
        "assignment_bits": 12,
        "assignments_packed": True,
        "num_moe_layers": 1,
        "num_experts": experts,
        "hidden_size": hidden,
        "moe_intermediate_size": intermediate,
        "matrix_count": experts * 3,
        "codebook": {
            "file": "global_codebook.safetensors",
            "tensor": "global_all.codebook",
        },
        "layers": [{"layer": 0, "file": "layer-000.safetensors"}],
    }
    if source_layout is not None:
        manifest["assignment_layout"] = source_layout
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = SimpleNamespace(
        model_type="qwen3_5_moe",
        is_moe=True,
        hidden_act="silu",
        num_moe_layers=1,
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
    )

    banks, shared, _ = load_nowag_expert_sources(output, config)

    assert banks["gate_assignments"][0].shape == (experts, gate_words, intermediate)
    assert banks["down_assignments"][0].shape == (experts, down_words, hidden)
    for projection, bank in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
        for expert in range(experts):
            expected = logical_assignments[
                _key(0, expert, projection, "assignments")
            ].t().contiguous()
            torch.testing.assert_close(
                banks[f"{bank}_assignments"][0][expert], expected
            )
    assert shared["codebook"].shape == (4096, 6)


def test_dsv4_expert_math_keeps_fp8_roundtrips_and_clamped_swiglu(monkeypatch):
    import nowag_vllm.moe_ops as nowag_moe_ops
    from freetoken.kernel.triton.dsv4 import fp8_linear

    calls = []

    def round_input(x, block=128, output=None):
        calls.append(("input", block))
        assert output is None
        return x + 1

    def round_middle(x, block):
        calls.append(("middle", block))
        x.add_(2)
        return x

    result = torch.full((2, 12), 7, dtype=torch.bfloat16)

    def fake_nowag_fused_moe(**kwargs):
        assert kwargs["hidden_states"].eq(1).all()
        assert kwargs["assignment_layout"] == "word_major"
        assert kwargs["gate_up_backend"] == "auto"
        assert kwargs["down_backend"] == "auto"
        assert kwargs["structural_down"] is True
        assert kwargs["swiglu_limit"] == 10.0
        assert (
            kwargs["gate_up_input_rounding"]
            == "dynamic_e4m3_per_token_group128_ue8m0"
        )
        assert (
            kwargs["down_input_rounding"]
            == "dynamic_e4m3_per_token_group128_ue8m0"
        )
        assert kwargs["down_norm_placement"] == "down_prologue"
        assert kwargs["validate_route_ids"] is False
        assert callable(kwargs["align_routes"])
        assert kwargs["gate_codebook"] is kwargs["up_codebook"]
        assert kwargs["gate_codebook"] is kwargs["down_codebook"]
        rounded_input = kwargs["gate_up_input_transform"](
            kwargs["hidden_states"]
        )
        assert rounded_input.eq(2).all()
        middle = torch.zeros(2, 6, dtype=torch.bfloat16)
        assert kwargs["middle_transform"](middle) is middle
        assert middle.eq(2).all()
        return result

    monkeypatch.setattr(fp8_linear, "act_quant_fp8_roundtrip", round_input)
    monkeypatch.setattr(fp8_linear, "act_quant_fp8_inplace", round_middle)
    monkeypatch.setattr(nowag_moe_ops, "nowag_fused_moe", fake_nowag_fused_moe)

    x = torch.ones(2, 12, dtype=torch.bfloat16)
    slots = torch.zeros(2, 2, dtype=torch.int32)
    weights = torch.ones(2, 2, dtype=torch.float32)
    codebook = torch.zeros(4096, 6, dtype=torch.bfloat16)
    assignments = torch.zeros(2, 6, 1, dtype=torch.int32)
    hidden_norm = torch.ones(2, 12, dtype=torch.bfloat16)
    intermediate_norm = torch.ones(2, 6, dtype=torch.bfloat16)

    actual = _run_nowag(
        _dsv4_math(10.0),
        x,
        slots,
        weights,
        codebook,
        assignments,
        hidden_norm,
        intermediate_norm,
        assignments,
        hidden_norm,
        intermediate_norm,
        torch.zeros(2, 12, 1, dtype=torch.int32),
        intermediate_norm,
        hidden_norm,
    )

    assert actual is result
    assert calls == [("input", 128), ("middle", 128)]


def test_qwen_expert_math_keeps_bf16_input_and_unclamped_swiglu(monkeypatch):
    import nowag_vllm.moe_ops as nowag_moe_ops
    from freetoken.moe.expert_format import ExpertMath

    result = torch.full((2, 12), 7, dtype=torch.bfloat16)
    x = torch.ones(2, 12, dtype=torch.bfloat16)

    def fake_nowag_fused_moe(**kwargs):
        assert kwargs["hidden_states"] is x
        assert kwargs["assignment_layout"] == "word_major"
        assert kwargs["gate_up_backend"] == "auto"
        assert kwargs["down_backend"] == "auto"
        assert kwargs["structural_down"] is True
        assert kwargs["swiglu_limit"] is None
        assert kwargs["gate_up_input_rounding"] == "none"
        assert kwargs["down_input_rounding"] == "none"
        assert kwargs["down_norm_placement"] == "gate_up_epilogue"
        assert kwargs["gate_up_input_transform"] is None
        assert kwargs["middle_transform"] is None
        assert kwargs["validate_route_ids"] is False
        assert callable(kwargs["align_routes"])
        assert kwargs["gate_codebook"] is kwargs["up_codebook"]
        assert kwargs["gate_codebook"] is kwargs["down_codebook"]
        return result

    monkeypatch.setattr(nowag_moe_ops, "nowag_fused_moe", fake_nowag_fused_moe)

    slots = torch.zeros(2, 2, dtype=torch.int32)
    weights = torch.ones(2, 2, dtype=torch.float32)
    codebook = torch.zeros(4096, 6, dtype=torch.bfloat16)
    assignments = torch.zeros(2, 6, 1, dtype=torch.int32)
    hidden_norm = torch.ones(2, 12, dtype=torch.bfloat16)
    intermediate_norm = torch.ones(2, 6, dtype=torch.bfloat16)

    actual = _run_nowag(
        ExpertMath(),
        x,
        slots,
        weights,
        codebook,
        assignments,
        hidden_norm,
        intermediate_norm,
        assignments,
        hidden_norm,
        intermediate_norm,
        torch.zeros(2, 12, 1, dtype=torch.int32),
        intermediate_norm,
        hidden_norm,
    )

    assert actual is result


def test_nowag_backend_environment_override(monkeypatch):
    from freetoken.moe.expert_format import ExpertMath

    monkeypatch.setenv("FREETOKEN_NOWAG_BACKEND", "triton")
    monkeypatch.setattr(
        "nowag_vllm.moe_ops.nowag_fused_moe",
        lambda **kwargs: (kwargs["gate_up_backend"], kwargs["down_backend"]),
    )
    x = torch.ones(1, 12, dtype=torch.bfloat16)
    assignments = torch.zeros(2, 1, 6, dtype=torch.int32)
    input_norm = torch.ones(2, 12, dtype=torch.bfloat16)
    output_norm = torch.ones(2, 6, dtype=torch.bfloat16)

    actual = _run_nowag(
        ExpertMath(),
        x,
        torch.zeros(1, 1, dtype=torch.int32),
        torch.ones(1, 1),
        torch.ones(4096, 6, dtype=torch.bfloat16),
        assignments,
        input_norm,
        output_norm,
        assignments,
        input_norm,
        output_norm,
        assignments,
        output_norm,
        input_norm,
    )

    assert actual == ("triton", "triton")


def test_qwen_offload_layer_dispatches_unrounded_unclamped_math(monkeypatch):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import make_moe_layer
    from freetoken.moe.expert_format import (
        _BANK_SCHEMAS,
        ExpertLayout,
        ExpertMath,
        bind_expert_method,
        expert_math,
    )
    from freetoken.moe.nowag import method as nowag_method
    from freetoken.moe.nowag.weights import NowagState

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    config = SimpleNamespace(
        moe_backend="offload",
        expert_quant="nowag",
        model_type="qwen3_5_moe",
        moe_router="softmax",
        num_experts=2,
        num_experts_per_tok=1,
        hidden_size=12,
        moe_intermediate_size=6,
        norm_topk_prob=True,
    )
    layer = make_moe_layer(config, layer_id=0, renormalize=True)
    x = torch.ones(1, 12, dtype=torch.bfloat16)
    called = {}

    def fake_run(math, layout, state, kernel_backend, *args, **kwargs):
        called.update(math=math, layout=layout, state=state, banks=args[3], shared=args[4])
        return x

    monkeypatch.setattr(nowag_method, "_run", fake_run)
    layer.expert_method = bind_expert_method(
        expert_math(layer),
        ExpertLayout("nowag", 12, 6, 2),
        NowagState(6, 12),
        device=torch.device("cpu"),
        backend="offload",
    )
    cache = SimpleNamespace(
        quant_format="nowag",
        bank_schema=_BANK_SCHEMAS["nowag"],
        shared={"codebook": torch.zeros(4096, 6, dtype=torch.bfloat16)},
    )
    views = tuple(torch.zeros(1, dtype=torch.bfloat16) for _ in range(9))

    actual = layer._expert_gemm(
        cache,
        x,
        torch.ones(1, 1, dtype=torch.float32),
        torch.zeros(1, 1, dtype=torch.int32),
        views=views,
        n=None,
        alphas=None,
        is_prefill=False,
    )

    assert actual is x
    assert called["math"] == ExpertMath()
    assert called["layout"].num_experts == 2
    assert called["state"] == NowagState(6, 12)
    assert list(called["banks"]) == list(_BANK_SCHEMAS["nowag"])
    assert called["shared"] is cache.shared


def test_shared_codebook_is_not_part_of_the_per_expert_slot():
    from freetoken.engine.cache_budget import expert_bytes_per_slot
    from freetoken.moe.offload_cache import OffloadMoeCache, _BANK_SCHEMAS

    sources = {
        "assignments": [torch.empty(2, 3, dtype=torch.int32)],
        "normalizer": [torch.empty(2, 4, dtype=torch.bfloat16)],
    }
    codebook = torch.empty(4096, 6, dtype=torch.bfloat16)

    assert expert_bytes_per_slot(sources) == 3 * 4 + 4 * 2
    assert codebook.numel() * codebook.element_size() == 48 * 1024

    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=2,
        cache_size=2,
        device=torch.device("cpu"),
        quant_format="nowag",
    )
    cache.set_bank_sources(
        {
            name: [torch.zeros(2, 1, dtype=torch.int32)]
            for name in _BANK_SCHEMAS["nowag"]
        }
    )
    cache.set_shared({"codebook": codebook})
    resident = cache.shared["codebook"]
    cache.rebuild(3)
    assert cache.shared["codebook"] is resident


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nowag_triton_aligner_handles_auto_cache_above_1024_slots():
    from freetoken.kernel.triton.moe_align import moe_align_block_size

    slots = torch.tensor(
        [[0, 1, 1023, 1024, 1200, 1458]], dtype=torch.int32, device="cuda"
    )
    sorted_tickets, expert_ids, num_tickets = moe_align_block_size(slots, 16, 1459)
    torch.cuda.synchronize()

    assert num_tickets.cpu().item() == 96
    valid = sorted_tickets[:96].cpu()
    assert sorted(valid[valid < slots.numel()].tolist()) == list(range(slots.numel()))
    assert set(expert_ids[:6].cpu().tolist()) == set(slots.cpu().view(-1).tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dsv4_nowag_method_compiles_on_cuda():
    import torch.nn.functional as F

    from freetoken.kernel.triton.dsv4.fp8_linear import (
        act_quant_fp8_inplace,
        act_quant_fp8_roundtrip,
    )
    from nowag_vllm.ops import pack_assignments

    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(20260822)
    num_experts = 4
    hidden = intermediate = 128
    codebook = (
        torch.randn((4096, 6), generator=generator) * 0.5
    ).to(device=device, dtype=torch.bfloat16)

    def projection(out_features: int, in_features: int):
        groups = (in_features + 5) // 6
        ids = torch.randint(
            0,
            4096,
            (num_experts, out_features, groups),
            generator=generator,
            dtype=torch.int64,
        )
        packed = torch.stack(
            [pack_assignments(ids[expert], 12) for expert in range(num_experts)]
        ).transpose(1, 2).contiguous().to(device)
        input_norm = torch.ones(
            (num_experts, in_features), dtype=torch.bfloat16, device=device
        )
        output_norm = torch.ones(
            (num_experts, out_features), dtype=torch.bfloat16, device=device
        )
        codewords = codebook[ids.to(device)]
        dense = codewords.reshape(num_experts, out_features, -1)[
            ..., :in_features
        ].contiguous()
        return (packed, input_norm, output_norm), dense

    gate, dense_gate = projection(intermediate, hidden)
    up, dense_up = projection(intermediate, hidden)
    down, dense_down = projection(hidden, intermediate)
    hidden_states = (
        torch.randn((3, hidden), generator=generator) * 0.2
    ).to(device=device, dtype=torch.bfloat16)
    slots = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75], [0.5, 0.5]],
        dtype=torch.float32,
        device=device,
    )

    output = _run_nowag(
        _dsv4_math(0.25),
        hidden_states,
        slots,
        topk_weights,
        codebook,
        *gate,
        *up,
        *down,
    )
    torch.cuda.synchronize()

    rounded_input = act_quant_fp8_roundtrip(hidden_states, 128)
    route_middle = []
    route_experts = []
    for token in range(slots.shape[0]):
        for route in range(slots.shape[1]):
            expert = int(slots[token, route])
            gate_value = torch.mv(
                dense_gate[expert].float(), rounded_input[token].float()
            ).to(torch.bfloat16).float()
            up_value = torch.mv(
                dense_up[expert].float(), rounded_input[token].float()
            ).to(torch.bfloat16).float()
            gate_value = torch.minimum(gate_value, gate_value.new_tensor(0.25))
            up_value = torch.clamp(up_value, -0.25, 0.25)
            route_middle.append(
                (F.silu(gate_value) * up_value).to(torch.bfloat16)
            )
            route_experts.append(expert)
    route_middle = torch.stack(route_middle)
    act_quant_fp8_inplace(route_middle, 128)
    route_output = []
    for ticket, expert in enumerate(route_experts):
        weight = topk_weights.reshape(-1)[ticket]
        route_output.append(
            (
                torch.mv(dense_down[expert].float(), route_middle[ticket].float())
                * weight
            ).to(torch.bfloat16)
        )
    expected = torch.stack(route_output).reshape(3, 2, hidden).float().sum(1)
    expected = expected.to(torch.bfloat16)

    assert output.shape == hidden_states.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) > 0
    torch.testing.assert_close(output, expected, rtol=8e-2, atol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_nowag_method_matches_plain_swiglu_on_cuda():
    import torch.nn.functional as F

    from freetoken.moe.expert_format import ExpertMath
    from nowag_vllm.ops import pack_assignments

    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(20260824)
    num_experts = 4
    hidden = intermediate = 128
    codebook = (
        torch.randn((4096, 6), generator=generator) * 0.5
    ).to(device=device, dtype=torch.bfloat16)

    def projection(out_features: int, in_features: int):
        groups = (in_features + 5) // 6
        ids = torch.randint(
            0,
            4096,
            (num_experts, out_features, groups),
            generator=generator,
            dtype=torch.int64,
        )
        packed = torch.stack(
            [pack_assignments(ids[expert], 12) for expert in range(num_experts)]
        ).transpose(1, 2).contiguous().to(device)
        input_norm = torch.ones(
            (num_experts, in_features), dtype=torch.bfloat16, device=device
        )
        output_norm = torch.ones(
            (num_experts, out_features), dtype=torch.bfloat16, device=device
        )
        dense = codebook[ids.to(device)].reshape(num_experts, out_features, -1)[
            ..., :in_features
        ].contiguous()
        return (packed, input_norm, output_norm), dense

    gate, dense_gate = projection(intermediate, hidden)
    up, dense_up = projection(intermediate, hidden)
    down, dense_down = projection(hidden, intermediate)
    hidden_states = (
        torch.randn((3, hidden), generator=generator) * 0.2
    ).to(device=device, dtype=torch.bfloat16)
    slots = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75], [0.5, 0.5]],
        dtype=torch.float32,
        device=device,
    )

    output = _run_nowag(
        ExpertMath(),
        hidden_states,
        slots,
        topk_weights,
        codebook,
        *gate,
        *up,
        *down,
    )
    torch.cuda.synchronize()

    route_output = []
    for token in range(slots.shape[0]):
        for route in range(slots.shape[1]):
            expert = int(slots[token, route])
            gate_value = torch.mv(
                dense_gate[expert].float(), hidden_states[token].float()
            )
            up_value = torch.mv(
                dense_up[expert].float(), hidden_states[token].float()
            )
            middle = (F.silu(gate_value) * up_value).to(torch.bfloat16)
            route_output.append(
                (
                    torch.mv(dense_down[expert].float(), middle.float())
                    * topk_weights[token, route]
                ).to(torch.bfloat16)
            )
    expected = torch.stack(route_output).reshape(3, 2, hidden).float().sum(1)
    expected = expected.to(torch.bfloat16)

    assert output.shape == hidden_states.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) > 0
    torch.testing.assert_close(output, expected, rtol=8e-2, atol=8e-2)
