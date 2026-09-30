"""Small checkpoint with the public current-checkpoint geometry ratios."""

import json

import numpy as np
import torch
from safetensors.torch import save_file


def checkpoint(directory):
    config = {
        "architectures": ["DFlashDraftModel"], "model_type": "qwen3",
        "hidden_size": 32, "intermediate_size": 96, "head_dim": 8,
        "num_attention_heads": 8, "num_key_value_heads": 2, "num_hidden_layers": 6,
        "num_target_layers": 40, "vocab_size": 47, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "attention_bias": False, "attention_dropout": 0.0,
        "max_position_embeddings": 262144, "use_sliding_window": True,
        "sliding_window": 4096, "max_window_layers": 6,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "dflash_config": {"block_size": 16, "mask_token_id": 46,
                          "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37]},
        "tie_word_embeddings": False, "dtype": "float32", "use_cache": True,
    }
    random = np.random.default_rng(20260930)
    weights = {}

    def matrix(name, rows, columns):
        value = (random.standard_normal((rows, columns)) * (0.45 / np.sqrt(columns))).astype(np.float32)
        weights[name] = torch.from_numpy(value.copy())
        return value.astype(np.float64)

    def norm(name, size):
        value = random.uniform(0.85, 1.15, size).astype(np.float32)
        weights[name] = torch.from_numpy(value.copy())
        return value.astype(np.float64)

    parameters = {"fc": matrix("fc.weight", 32, 256),
                  "hidden_norm": norm("hidden_norm.weight", 32), "layers": []}
    for index in range(6):
        prefix = f"layers.{index}."
        layer = {"input_norm": norm(prefix + "input_layernorm.weight", 32),
                 "post_norm": norm(prefix + "post_attention_layernorm.weight", 32),
                 "q_norm": norm(prefix + "self_attn.q_norm.weight", 8),
                 "k_norm": norm(prefix + "self_attn.k_norm.weight", 8)}
        for name, rows, columns in (("q", 64, 32), ("k", 16, 32), ("v", 16, 32), ("o", 32, 64)):
            layer[name] = matrix(prefix + f"self_attn.{name}_proj.weight", rows, columns)
        for name, rows, columns in (("gate", 96, 32), ("up", 96, 32), ("down", 32, 96)):
            layer[name] = matrix(prefix + f"mlp.{name}_proj.weight", rows, columns)
        parameters["layers"].append(layer)
    parameters["final_norm"] = norm("norm.weight", 32)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(weights, str(directory / "model.safetensors"))
    return config, parameters, sum(value.numel() for value in weights.values())


def quantized(parameters, convert):
    return {key: [{name: convert(value) for name, value in layer.items()} for layer in value]
            if key == "layers" else convert(value) for key, value in parameters.items()}
