"""Public row-selection and native-weight contracts of the NVFP4 output head."""

from types import SimpleNamespace

import pytest
import torch

import freetoken.core as core
from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead, nvfp4_dense_linear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _native_weights(outputs, width=128):
    generator = torch.Generator(device="cuda").manual_seed(5701 + outputs)
    return {
        "weight": torch.randint(256, (outputs, width // 2), generator=generator,
                                device="cuda", dtype=torch.uint8),
        "weight_scale": (0.0625 * (1 + torch.arange(outputs * width // 16, device="cuda") % 3))
                        .reshape(outputs, width // 16).to(torch.float8_e4m3fn),
        "weight_global": (0.5 + torch.arange(outputs, device="cuda") % 7 / 8).half(),
    }


def _head(weights, loaded):
    with torch.device("cuda"):
        head = Nvfp4LMHead(weights["weight"].shape[0], weights["weight"].shape[1] * 2)
    if loaded:
        head.load_state_dict({name: tensor.clone() for name, tensor in weights.items()})
    else:
        for name, tensor in weights.items():
            setattr(head, name, torch.nn.Parameter(tensor.clone(), requires_grad=False))
    return head


def _inputs(rows, dtype):
    generator = torch.Generator(device="cuda").manual_seed(7201 + rows)
    return torch.randn(rows, 128, device="cuda", generator=generator).to(dtype)


def _check_projection(actual, expected, inputs, original):
    assert actual.shape == expected.shape
    assert actual.device == inputs.device
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.015, atol=0.015)
    torch.testing.assert_close(inputs, original, rtol=0, atol=0)


@pytest.mark.parametrize("loaded,rows,dtype", [
    (False, 1, torch.bfloat16), (False, 3, torch.bfloat16),
    (True, 1, torch.bfloat16), (True, 3, torch.bfloat16), (True, 3, torch.float16),
])
def test_forward_selected_projects_every_row_without_context(monkeypatch, loaded, rows, dtype):
    def forbidden_context():
        raise AssertionError("forward_selected must not consult global context")

    monkeypatch.setattr(core, "get_global_ctx", forbidden_context)
    weights = _native_weights(128)
    inputs = _inputs(rows, dtype)
    original = inputs.clone()
    expected = nvfp4_dense_linear(inputs, weights["weight"], weights["weight_scale"], weights["weight_global"])
    actual = _head(weights, loaded).forward_selected(inputs)
    _check_projection(actual, expected, inputs, original)


@pytest.mark.parametrize("loaded", [False, True])
@pytest.mark.parametrize("extend_path", [False, True])
def test_forward_keeps_extend_last_row_selection(monkeypatch, loaded, extend_path):
    weights = _native_weights(128)
    inputs = _inputs(3, torch.bfloat16)
    original = inputs.clone()
    last_indices = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    queried = []

    def get_last_indices(size):
        assert size == 2
        queried.append(size)
        return last_indices

    batch = SimpleNamespace(uses_extend_path=extend_path, size=2,
                            attn_metadata=SimpleNamespace(get_last_indices=get_last_indices))
    monkeypatch.setattr(core, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    selected = inputs[last_indices] if extend_path else inputs
    expected = nvfp4_dense_linear(selected, weights["weight"], weights["weight_scale"], weights["weight_global"])
    actual = _head(weights, loaded).forward(inputs)
    assert bool(queried) == extend_path
    _check_projection(actual, expected, inputs, original)
