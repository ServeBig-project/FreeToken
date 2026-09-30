"""DFlash weights and block computation; cache ownership stays with the runtime.

Reference: https://github.com/z-lab/dflash/blob/main/dflash/model.py
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


def read_dflash_config(path: str | Path) -> SimpleNamespace:
    raw = json.loads((Path(path) / "config.json").read_text())
    raw.update(raw.get("dflash_config", {}))
    c = SimpleNamespace(**raw)
    c.head_dim = raw.get("head_dim", c.hidden_size // c.num_attention_heads)
    c.attention_bias = raw.get("attention_bias", False)
    c.target_layer_ids = raw.get("target_layer_ids")
    if c.target_layer_ids is None:
        n = c.num_hidden_layers
        c.target_layer_ids = (
            [c.num_target_layers // 2] if n == 1 else
            [round(1 + i * (c.num_target_layers - 4) / (n - 1)) for i in range(n)]
        )
    c.mask_token_id = raw.get("mask_token_id")
    c.block_size = int(raw.get("block_size", 16))
    c.input_embedding_scale = float(raw.get("input_embedding_scale", 1.0))
    c.output_multiplier = float(raw.get("output_multiplier", 1.0))
    c.final_logit_softcapping = raw.get("final_logit_softcapping")
    rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    if c.hidden_act != "silu" or rope.get("rope_type", "default") != "default":
        raise ValueError("DFlash requires SiLU and default RoPE")
    if c.mask_token_id is None:
        raise ValueError("DFlash checkpoint must specify mask_token_id")
    c.rope_theta = rope.get("rope_theta", raw.get("rope_theta", 10000.0))
    c.attention_modes = []
    for kind in raw.get("layer_types") or ["full_attention"] * c.num_hidden_layers:
        if kind not in ("sliding_attention", "full_attention"):
            raise ValueError(f"Unsupported DFlash attention type: {kind}")
        sliding = kind == "sliding_attention"
        causal = raw.get("is_causal")
        c.attention_modes.append((
            sliding if causal is None else bool(causal),
            c.sliding_window - 1 if sliding else -1,
        ))
    return c


class _Norm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match the checkpoint's normalization, including its rounding before scale.
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * value.to(x.dtype)


class _Attention(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.head_dim = config.head_dim
        q_width = config.num_attention_heads * self.head_dim
        kv_width = config.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=config.attention_bias)
        self.q_norm = _Norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _Norm(self.head_dim, config.rms_norm_eps)

    def kv(self, hidden: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]):
        shape = (hidden.shape[0], -1, self.head_dim)
        key = self.k_norm(self.k_proj(hidden).view(shape))
        return _rotate(key, rotary), self.v_proj(hidden).view(shape)


class _MLP(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.self_attn = _Attention(config)
        self.mlp = _MLP(config)
        self.input_layernorm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _Norm(config.hidden_size, config.rms_norm_eps)


def _rotate(x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rotary
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos + rotated * sin


class DFlashModel(nn.Module):
    """Load a DFlash checkpoint without a second embedding or output vocabulary.

    ``project_context`` passes normalized, rotated K and V shaped [T, Hkv, D]
    to ``store(layer, k, v)``. ``forward`` passes Q/K/V to
    ``attend(layer, q, k, v)``; the runtime applies each ``attention_modes``
    (causal, window_left) pair and the usual head_dim**-0.5 attention scale.
    Inputs use packed token rows; callbacks own request boundaries and padding.
    """

    def __init__(self, path: str | Path, *, dtype: torch.dtype, device: torch.device | str):
        super().__init__()
        path = Path(path)
        self.config = c = read_dflash_config(path)
        self.target_layer_ids = c.target_layer_ids
        self.hidden_size = c.hidden_size
        self.mask_token_id = c.mask_token_id
        self.block_size = c.block_size
        self.input_embedding_scale = c.input_embedding_scale
        self.output_multiplier = c.output_multiplier
        self.final_logit_softcapping = c.final_logit_softcapping
        self.attention_modes = c.attention_modes

        # Meta construction avoids holding an uninitialized second weight copy.
        with torch.device("meta"):
            self.layers = nn.ModuleList(_Layer(c) for _ in range(c.num_hidden_layers))
            self.fc = nn.Linear(len(self.target_layer_ids) * c.hidden_size, c.hidden_size, bias=False)
            self.hidden_norm = _Norm(c.hidden_size, c.rms_norm_eps)
            self.norm = _Norm(c.hidden_size, c.rms_norm_eps)
        weights = {}
        for file in sorted(path.glob("*.safetensors")):
            weights.update({name: value.to(device=device, dtype=dtype)
                            for name, value in load_file(file).items()})
        self.load_state_dict(weights, strict=True, assign=True)
        frequency = 1.0 / (c.rope_theta ** (
            torch.arange(0, c.head_dim, 2, dtype=torch.float32, device=device) / c.head_dim
        ))
        self.register_buffer("inv_freq", frequency, persistent=False)
        self.requires_grad_(False)
        self.eval()
        self.weight_bytes = sum(t.numel() * t.element_size() for t in self.parameters())

    def _rotary(self, positions: torch.Tensor, dtype: torch.dtype):
        frequencies = positions.float().unsqueeze(-1) * self.inv_freq
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies.cos().to(dtype).unsqueeze(1), frequencies.sin().to(dtype).unsqueeze(1)

    def project_context(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        store: Callable[[int, torch.Tensor, torch.Tensor], None],
    ) -> None:
        hidden = self.hidden_norm(self.fc(features))
        rotary = self._rotary(positions, hidden.dtype)
        for index, layer in enumerate(self.layers):
            # Context is shared across layers and does not use input_layernorm.
            key, value = layer.self_attn.kv(hidden, rotary)
            store(index, key, value)

    def forward(
        self,
        noise_embeddings: torch.Tensor,
        positions: torch.Tensor,
        attend: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        hidden = noise_embeddings
        rotary = self._rotary(positions, hidden.dtype)
        for index, layer in enumerate(self.layers):
            attention = layer.self_attn
            normalized = layer.input_layernorm(hidden)
            key, value = attention.kv(normalized, rotary)
            query = attention.q_proj(normalized).view(hidden.shape[0], -1, attention.head_dim)
            query = _rotate(attention.q_norm(query), rotary)
            output = attend(index, query, key, value).reshape(hidden.shape[0], -1)
            hidden = hidden + attention.o_proj(output)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.norm(hidden)
