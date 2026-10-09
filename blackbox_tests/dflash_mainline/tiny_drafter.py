"""Legal public-format DFlash checkpoint for a target without GDN (Qwen3-30B-A3B geometry).

Random weights: it exercises capability and resource behaviour only, never model quality.
A 64-token native window keeps window wrap-around reachable with short prompts.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

WINDOW = 64


def build(directory, hidden=2048, target_layers=48, vocab=151936, mask_token=151662, seed=7):
    heads, kv_heads, head_dim, inter = 16, 4, 128, 512
    config = {
        "architectures": ["DFlashDraftModel"], "model_type": "qwen3",
        "hidden_size": hidden, "intermediate_size": inter, "head_dim": head_dim,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "num_hidden_layers": 2,
        "num_target_layers": target_layers, "vocab_size": vocab, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "attention_bias": False, "attention_dropout": 0.0,
        "max_position_embeddings": 40960, "use_sliding_window": True, "sliding_window": WINDOW,
        "max_window_layers": 2, "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        "dflash_config": {"block_size": 16, "mask_token_id": mask_token,
                          "target_layer_ids": [1, 23, 46]},
        "tie_word_embeddings": False, "dtype": "bfloat16", "use_cache": True,
    }
    gen = torch.Generator().manual_seed(seed)

    def matrix(rows, cols):
        return (torch.randn(rows, cols, generator=gen) * (0.5 / cols ** 0.5)).to(torch.bfloat16)

    def ones(n):
        return torch.ones(n, dtype=torch.bfloat16)

    w = {"fc.weight": matrix(hidden, 3 * hidden), "hidden_norm.weight": ones(hidden), "norm.weight": ones(hidden)}
    for i in range(2):
        p = f"layers.{i}."
        w.update({p + "input_layernorm.weight": ones(hidden), p + "post_attention_layernorm.weight": ones(hidden),
                  p + "self_attn.q_norm.weight": ones(head_dim), p + "self_attn.k_norm.weight": ones(head_dim),
                  p + "self_attn.q_proj.weight": matrix(heads * head_dim, hidden),
                  p + "self_attn.k_proj.weight": matrix(kv_heads * head_dim, hidden),
                  p + "self_attn.v_proj.weight": matrix(kv_heads * head_dim, hidden),
                  p + "self_attn.o_proj.weight": matrix(hidden, heads * head_dim),
                  p + "mlp.gate_proj.weight": matrix(inter, hidden), p + "mlp.up_proj.weight": matrix(inter, hidden),
                  p + "mlp.down_proj.weight": matrix(hidden, inter)})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(w, str(directory / "model.safetensors"))
    return config, sum(t.numel() * t.element_size() for t in w.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    print(build(parser.parse_args().directory)[1])
