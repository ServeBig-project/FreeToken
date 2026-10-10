"""Independent legal HF Qwen3MoE fixture: I=512 gives the D6/TP2 boundary at lane 256.

Only the published transformers model/config and tokenizer APIs construct BASE. No
FreeToken code is imported; NoWAG matrices come from the independent format writer.
The random model is for format/TP execution, never a task-quality or speed claim.
"""

from pathlib import Path

import torch

import sidecar as S
from cases import need_scratch

HIDDEN, INTER, EXPERTS, LAYERS, TOP_K = 256, 512, 8, 2, 2
GEOMETRIES = {"qwen3": (HIDDEN, INTER, 4, TOP_K),
              "dsv4-math": (4096, 2048, 64, 6),
              "flashnext-shape": (2560, 640, 40, TOP_K)}


def paths(d=6, shape="qwen3"):
    hidden, inter, heads, top_k = GEOMETRIES[shape]
    root = need_scratch() / ("tiny-qwen3-moe" if shape == "qwen3" else f"component-{shape}")
    base = root / "base"
    if not (base / "config.json").exists():
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

        config = Qwen3MoeConfig(vocab_size=256, hidden_size=hidden, intermediate_size=inter,
                               moe_intermediate_size=inter, num_hidden_layers=LAYERS,
                               num_attention_heads=heads, num_key_value_heads=2, head_dim=64,
                               num_local_experts=EXPERTS, num_experts_per_tok=top_k,
                               decoder_sparse_step=1, max_position_embeddings=4096,
                               bos_token_id=1, eos_token_id=2, pad_token_id=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(61009)
            model = Qwen3MoeForCausalLM(config).bfloat16()
        model.save_pretrained(base, safe_serialization=True)
        vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        vocab.update({f"word{i}": i for i in range(4, 256)})
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>",
                               bos_token="<s>", eos_token="</s>").save_pretrained(base)
    template = {"format": "nowag_expert_sidecar_v1", "model_type": "qwen3_moe",
                "hidden_size": hidden, "moe_intermediate_size": inter,
                "num_experts": EXPERTS, "num_moe_layers": LAYERS}
    geometry = {"template": template, "layers": list(range(LAYERS)), "experts": EXPERTS,
                "hidden": hidden, "inter": inter}
    side = S.synth_dir(geometry, root / f"side-d{d}", d, "random")
    return Path(base), side


def serve_args(base, side=None, tp=1, backend="offload"):
    args = ["--model", base, "--moe-backend", backend, "--moe-cache-size", 2 * EXPERTS,
            "--batching-policy", "legacy", "--attention-backend", "fi", "--num-pages", 1024,
            "--max-running-requests", 4, "--cuda-graph-max-bs", 4, "--tensor-parallel-size", tp]
    return args + (["--nowag-expert-path", side] if side else [])
