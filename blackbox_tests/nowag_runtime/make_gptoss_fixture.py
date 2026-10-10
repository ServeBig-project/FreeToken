"""Make an untrained BF16 GPT-OSS BASE and independent NoWAG D6 input, using CPU only.

Sources: https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json and the public
transformers GptOssForCausalLM weight definition. This is a bias/format fixture, not a
replacement for trained GPT-OSS weights or a quality/performance acceptance result.
"""

import argparse
import gc
import json
import resource
from pathlib import Path

import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GptOssConfig, GptOssForCausalLM, PreTrainedTokenizerFast

import reference as R
import sidecar as S
import tolerances as TOL
from access import with_base_bias


def make_base(base):
    config = GptOssConfig(
        hidden_size=2880, intermediate_size=2880, num_local_experts=4, num_experts_per_tok=4,
        num_hidden_layers=2, num_attention_heads=64, num_key_value_heads=8, head_dim=64,
        vocab_size=256, max_position_embeddings=4096, sliding_window=128,
        layer_types=["sliding_attention", "full_attention"], swiglu_alpha=1.702, swiglu_limit=7.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        rope_parameters={"rope_type": "yarn", "rope_theta": 150000.0, "factor": 32.0,
                         "beta_fast": 32.0, "beta_slow": 1.0,
                         "original_max_position_embeddings": 4096, "truncate": False})
    config._attn_implementation = "eager"
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # avoid a full FP32 copy of the random checkpoint
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(610102880)
            model = GptOssForCausalLM(config).eval()
    finally:
        torch.set_default_dtype(previous)
    bias_stats = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".experts." not in name or not name.endswith("_bias"):
                continue
            layer = int(name.split(".")[2])
            expert = torch.arange(1, 5).float()[:, None]
            lanes = torch.arange(2880).float()[None, :]
            if name.endswith("gate_up_proj_bias"):
                parameter[:, ::2] = (layer + 1) * (expert / 8 + (lanes % 7 - 3) / 128)
                parameter[:, 1::2] = (layer + 1) * (-expert / 4 + (lanes % 5 - 2) / 128)
            else:
                parameter.copy_((layer + 1) * expert * (1 + lanes % 4 / 8) * (2 * (lanes % 2) - 1))
            assert bool(torch.isfinite(parameter).all()) and bool((parameter != 0).all())
            bias_stats[name] = {"shape": list(parameter.shape), "nonzero": parameter.numel(),
                                "minimum": float(parameter.min()), "maximum": float(parameter.max())}
        logits = model(torch.tensor([[4, 5, 6]]), use_cache=False).logits
        assert logits.shape == (1, 3, 256) and bool(torch.isfinite(logits).all())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model.save_pretrained(base, safe_serialization=True, max_shard_size="64MB")
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocab.update({f"word{i}": i for i in range(4, 256)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
                                       bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    tokenizer.save_pretrained(base)
    assert tokenizer.encode("word4 word5") == [4, 5]
    return {"parameter_bytes": parameter_bytes, "bias": bias_stats,
            "hf_cpu_forward": {"shape": list(logits.shape), "finite": True}}


def bias_observability(base, side):
    weights = with_base_bias(S.read_experts(side, 0, [0])[0], base, 0, 0)
    codebook = S.codebook(side)
    weights = {p: dict(w, dense=R.codeword_matrix(w["assignments"], codebook, w["input_norm"].numel()))
               for p, w in weights.items()}
    x = torch.randn(4, 2880, generator=torch.Generator().manual_seed(441)).bfloat16()
    route = torch.tensor([0.125, 0.25, 0.5, 0.75])
    math = {"family": "gptoss", "alpha": 1.702, "limit": 7.0, "route": "output"}
    expected = R.expert(x, weights, codebook, math, route)
    metrics = {}
    for projection in ("w1", "w3", "w2"):
        removed = {p: dict(w, bias=None) if p == projection else w for p, w in weights.items()}
        wrong = R.expert(x, removed, codebook, math, route).bfloat16()
        metrics[projection] = TOL.metrics(wrong, expected)
        assert any(metrics[projection][key] > bound for key, bound in TOL.BOUNDS["bf16"].items()), \
            f"fixture does not expose a missing {projection} bias under the frozen bounds"
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    base, scratch = args.out / "base", args.out / "scratch"
    report = make_base(base)
    gc.collect()
    side = S.synth_dir(S.gptoss_geometry(base), scratch / "synth-gptoss-d6-random-row_major", 6, "random")
    report["missing_bias_errors"] = bias_observability(base, side)
    report.update({"kind": "synthetic_random_gpt_oss", "trained": False,
                   "base": str(base.resolve()), "scratch": str(scratch.resolve()), "side": str(side.resolve()),
                   "seed": 610102880, "torch": torch.__version__, "transformers": transformers.__version__,
                   "threads": torch.get_num_threads(), "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   "artifact_bytes": sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())})
    assert report["peak_rss_bytes"] <= 2 * 1024 ** 3, report["peak_rss_bytes"]
    (args.out / "fixture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
