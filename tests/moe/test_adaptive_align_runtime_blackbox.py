from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")


def test_uses_large_moe_align_public_boundary() -> None:
    from freetoken.kernel.triton import moe_align as align

    assert align.uses_large_moe_align(1024) is False
    assert align.uses_large_moe_align(1025) is True


def _method_module():
    from freetoken.moe.nowag import method

    return method


def _run_routed(positional, output):
    from freetoken.moe.expert_format import (
        _BANK_SCHEMAS,
        ExpertLayout,
        ExpertMath,
        bind_expert_method,
    )
    from freetoken.moe.nowag.weights import NowagState

    x, slots, weights, codebook, *banks = positional
    layout = ExpertLayout("nowag", x.shape[1], banks[2].shape[1], banks[0].shape[0])
    method = bind_expert_method(
        ExpertMath(),
        layout,
        NowagState(codebook.shape[1], 12),
        device=x.device,
        backend="offload",
    )
    banks = dict(zip(_BANK_SCHEMAS["nowag"], banks))
    return method.run(x, slots, weights, banks, {"codebook": codebook}, out=output)


def _routed_args(num_tokens: int):
    # Real Qwen3.5 shapes; meta tensors carry the shapes without the memory, since
    # the fused kernel entry is replaced in every test.
    top_k = 8
    hidden = 2048
    intermediate = 512
    experts = 256

    def tensor(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    positional = (
        tensor((num_tokens, hidden)),
        tensor((num_tokens, top_k), torch.int32),
        tensor((num_tokens, top_k), torch.float32),
        tensor((4096, 6)),
        tensor((experts, 128, intermediate), torch.int32),
        tensor((experts, hidden)),
        tensor((experts, intermediate)),
        tensor((experts, 128, intermediate), torch.int32),
        tensor((experts, hidden)),
        tensor((experts, intermediate)),
        tensor((experts, 32, hidden), torch.int32),
        tensor((experts, intermediate)),
        tensor((experts, hidden)),
    )
    return positional, tensor((num_tokens, hidden))


@pytest.mark.parametrize(
    ("num_tokens", "expect_adaptive"), ((128, False), (129, True))
)
def test_routed_experts_selects_adaptive_callback_only_above_1024_routes(
    monkeypatch, num_tokens: int, expect_adaptive: bool
) -> None:
    method = _method_module()
    plain_result = (object(), object(), object())
    adaptive_result = (object(), object(), object())
    plain_calls = []
    adaptive_calls = []

    def plain(*args, **kwargs):
        plain_calls.append((args, kwargs))
        return plain_result

    def adaptive(*args, **kwargs):
        adaptive_calls.append((args, kwargs))
        return adaptive_result

    monkeypatch.setattr(method, "moe_align_block_size", plain)
    monkeypatch.setattr(method, "moe_align_block_size_adaptive", adaptive)
    captured = {}
    result_marker = object()

    def capture_nowag(**kwargs):
        captured.update(kwargs)
        captured["builder_selection"] = (
            "adaptive_callback"
            if kwargs["align_routes_adaptive"] is not None
            else "plugin_builder"
        )
        return result_marker

    monkeypatch.setattr(method, "nowag_fused_moe", capture_nowag)
    positional, output = _routed_args(num_tokens)
    assert _run_routed(positional, output) is result_marker

    ordinary_args = (positional[1], 16, 256)
    assert captured["align_routes"](*ordinary_args) is plain_result
    assert plain_calls == [(ordinary_args, {"alignment_storage": None})]
    if expect_adaptive:
        assert captured["builder_selection"] == "adaptive_callback"
        callback = captured["align_routes_adaptive"]
        assert callback is adaptive
        adaptive_args = tuple(object() for _ in range(6))
        assert callback(*adaptive_args) is adaptive_result
        assert adaptive_calls == [(adaptive_args, {})]
    else:
        assert captured["builder_selection"] == "plugin_builder"
        assert captured["align_routes_adaptive"] is None
        assert adaptive_calls == []


def test_adaptive_callback_error_propagates_through_routed_experts(monkeypatch) -> None:
    method = _method_module()
    expected = RuntimeError("adaptive builder failed")

    def fail_adaptive(*args, **kwargs):
        raise expected

    monkeypatch.setattr(method, "moe_align_block_size_adaptive", fail_adaptive)

    def invoke_callback(**kwargs):
        callback = kwargs["align_routes_adaptive"]
        return callback(*(object() for _ in range(6)))

    monkeypatch.setattr(method, "nowag_fused_moe", invoke_callback)
    positional, output = _routed_args(129)
    with pytest.raises(RuntimeError) as raised:
        _run_routed(positional, output)
    assert raised.value is expected


def test_routed_experts_propagates_plugin_error(monkeypatch) -> None:
    method = _method_module()
    expected = RuntimeError("plugin failed")

    def fail_nowag(**kwargs):
        raise expected

    monkeypatch.setattr(method, "nowag_fused_moe", fail_nowag)
    positional, output = _routed_args(129)
    with pytest.raises(RuntimeError) as raised:
        _run_routed(positional, output)
    assert raised.value is expected
