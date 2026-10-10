"""Independent legal HF Qwen3MoE fixture: I=512 gives the D6/TP2 boundary at lane 256.

Only the published transformers model/config and tokenizer APIs construct BASE. No
FreeToken code is imported; NoWAG matrices come from the independent format writer.
The random model is for format/TP execution, never a task-quality or speed claim.
"""

from pathlib import Path

import torch

import reference as R
import sidecar as S
from cases import need_scratch

HIDDEN, INTER, EXPERTS, LAYERS, TOP_K = 256, 512, 8, 2, 2
# 48 prompts in the tiny vocabulary (word4..word255); TP2 rows generate 12 greedy tokens each
PROMPTS = [" ".join(f"word{4 + (17 * i + 29 * j) % 252}" for j in range(6)) for i in range(48)]
GREEDY_TOKENS = 12
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
    side = S.synth_dir(geometry(hidden, inter), root / f"side-d{d}", d, "random")
    return Path(base), side


def geometry(hidden, inter):
    template = {"format": "nowag_expert_sidecar_v1", "model_type": "qwen3_moe",
                "hidden_size": hidden, "moe_intermediate_size": inter,
                "num_experts": EXPERTS, "num_moe_layers": LAYERS}
    return {"template": template, "layers": list(range(LAYERS)), "experts": EXPERTS,
            "hidden": hidden, "inter": inter}


# ------------------------------------------------------------------ sidecar fitted to BASE
# The random side-d4/d6 above are unrelated to BASE (expert outputs ~32x the BF16 experts'),
# which makes greedy text hypersensitive to rounding. These sidecars approximate BASE's own
# BF16 experts instead, so NoWAG and BF16 runs of the same model are comparable.

def base_experts(base):
    from safetensors import safe_open
    out = {}
    with safe_open(str(Path(base) / "model.safetensors"), "pt") as f:
        for layer in range(LAYERS):
            for e in range(EXPERTS):
                pre = f"model.layers.{layer}.mlp.experts.{e}."
                out[layer, e] = {p: f.get_tensor(pre + name + ".weight").float()
                                 for p, name in (("w1", "gate_proj"), ("w3", "up_proj"),
                                                 ("w2", "down_proj"))}
    return out


def normalise(w, d):
    """W ~= diag(out) * Wn * diag(in): in = column RMS, out = row RMS of W/in; Wn split into
    D-lane groups along K (tail lanes zero) for the codebook fit."""
    inn = w.pow(2).mean(0).sqrt().clamp_min(1e-8).bfloat16().float()
    out = (w / inn).pow(2).mean(1).sqrt().clamp_min(1e-8).bfloat16().float()
    wn = w / inn / out[:, None]
    count = R.ids_per_row(w.shape[1], d)
    groups = torch.nn.functional.pad(wn, (0, count * d - w.shape[1])).reshape(w.shape[0], count, d)
    return groups, inn, out


def nearest(groups, codebook, valid):
    """Codeword ids minimising squared error over the first `valid` lanes."""
    flat = groups.reshape(-1, groups.shape[-1])[:, :valid]
    cb = codebook[:, :valid]
    ids = torch.empty(len(flat), dtype=torch.long)
    for i in range(0, len(flat), 16384):
        chunk = flat[i:i + 16384]
        dist = chunk.pow(2).sum(1, keepdim=True) - 2 * chunk @ cb.t() + cb.pow(2).sum(1)
        ids[i:i + 16384] = dist.argmin(1)
    return ids


def kmeans(points, gen, iters=8):
    centers = points[torch.randperm(len(points), generator=gen)[:R.CODEBOOK_SIZE]].clone()
    for _ in range(iters):
        ids = nearest(points[:, None, :], centers, points.shape[1])
        total = torch.zeros_like(centers).index_add_(0, ids, points)
        count = torch.zeros(len(centers)).index_add_(0, ids, torch.ones(len(points)))
        centers = torch.where(count[:, None] > 0, total / count.clamp_min(1)[:, None], centers)
    return centers


def fit_projection(w, codebook, d):
    groups, inn, out = normalise(w, d)
    k = w.shape[1]
    ids = nearest(groups, codebook, d).reshape(groups.shape[:2])
    if k % d:                                  # last codeword: only its valid lanes count
        ids[:, -1] = nearest(groups[:, -1:], codebook, k % d)
    return {"assignments": R.pack(ids), "input_norm": inn.bfloat16(),
            "output_norm": out.bfloat16(), "bias": None}


def fitted_side(d, kind="fit"):
    """kind "fit": BASE's experts approximated (12-bit global codebook, k-means, 8 iterations).
    kind "shuffled": the fit with the assignments of gate/up rows I/2..I (TP rank 1's half)
    randomly permuted within each row -- a public stand-in for a wrong rank-1 shard."""
    base, _ = paths(d)
    root = base.parent
    out = root / f"side-{kind}-d{d}"
    if (out / "manifest.json").exists():
        return out
    tmp = out.with_name(out.name + ".partial")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        gen = torch.Generator().manual_seed(1009 + d)
        if kind == "fit":
            experts = base_experts(base)
            points = torch.cat([normalise(w, d)[0].reshape(-1, d)
                                for ws in experts.values() for w in ws.values()])
            codebook = kmeans(points, gen).bfloat16()
            weights = {key: {p: fit_projection(w, codebook.float(), d) for p, w in ws.items()}
                       for key, ws in experts.items()}
        else:
            source = fitted_side(d, "fit")
            codebook = S.codebook(source)
            weights = {}
            for layer in range(LAYERS):
                for e, w in S.read_experts(source, layer, range(EXPERTS)).items():
                    for p in ("w1", "w3"):
                        ids = R.unpack(w[p]["assignments"], R.ids_per_row(HIDDEN, d))
                        half = ids[INTER // 2:]
                        order = torch.argsort(torch.rand(half.shape, generator=gen), dim=1)
                        ids[INTER // 2:] = half.gather(1, order)
                        w[p] = dict(w[p], assignments=R.pack(ids))
                    weights[layer, e] = w
        entries = [S.write_layer(tmp, layer, {e: weights[layer, e] for e in range(EXPERTS)})
                   for layer in range(LAYERS)]
        S.write_head(tmp, geometry(HIDDEN, INTER)["template"], d, codebook, entries)
    finally:
        torch.set_num_threads(threads)
    tmp.rename(out)
    return out


def serve_args(base, side=None, tp=1, backend="offload"):
    args = ["--model", base, "--moe-backend", backend, "--moe-cache-size", 2 * EXPERTS,
            "--batching-policy", "legacy", "--attention-backend", "fi", "--num-pages", 1024,
            "--max-running-requests", 4, "--cuda-graph-max-bs", 4, "--tensor-parallel-size", tp]
    return args + (["--nowag-expert-path", side] if side else [])
