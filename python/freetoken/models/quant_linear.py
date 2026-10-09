"""Quant-aware dense-linear factories, shared by the models that serve quantized
dense projections (qwen3_5_moe, muse_glimmer).

Maps the resolved precision to the right ``BaseOP`` linear: an explicit public plan
(``dense_precision`` "fp8" or "bf16", see ``freetoken.quant.dense``) decides outright; under
"source" a checkpoint's own quant config does (``expert_quant`` for the dense MLP /
shared-expert path, ``attn_quant`` for attention + GatedDeltaNet projections). Block-FP8, per-tensor-FP8 and NVFP4 implementations live under
``freetoken.kernel.triton``; the bf16 fallback is the framework's TP-aware ``freetoken.layers``.
Only the *dispatch* (config -> layer class) lives here.
"""

from __future__ import annotations


def make_col_merged_quant(expert_quant: str, attn_quant: str, in_f: int,
                          output_sizes: list[int], has_bias: bool = False,
                          dense_precision: str = "source"):
    """Column-merged linear for a dense projection: block-fp8 / per-tensor-fp8 / nvfp4 / bf16."""
    if dense_precision == "fp8":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged

        return Fp8PerTensorColMerged(in_f, output_sizes, has_bias)
    if dense_precision == "bf16":
        from freetoken.layers import LinearColParallelMerged

        return LinearColParallelMerged(in_f, output_sizes, has_bias=has_bias)
    if expert_quant == "fp8_block":
        from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged

        return Fp8BlockColMerged(in_f, output_sizes, has_bias)
    if attn_quant == "fp8_pertensor":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged

        return Fp8PerTensorColMerged(in_f, output_sizes, has_bias)
    if attn_quant == "nvfp4":  # compressed-tensors W4A16 attention (q/k/v fused)
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged

        return Nvfp4DenseColMerged(in_f, output_sizes, has_bias)
    from freetoken.layers import LinearColParallelMerged

    return LinearColParallelMerged(in_f, output_sizes, has_bias=has_bias)


def make_replicated_quant(expert_quant: str, attn_quant: str, in_f: int, out_f: int,
                          has_bias: bool = False, dense_precision: str = "source"):
    """Replicated linear for a dense projection: block-fp8 / per-tensor-fp8 / nvfp4 / bf16."""
    if dense_precision == "fp8":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorLinear

        return Fp8PerTensorLinear(in_f, out_f, has_bias)
    if dense_precision == "bf16":
        from freetoken.layers import LinearReplicated

        return LinearReplicated(in_f, out_f, has_bias=has_bias)
    if expert_quant == "fp8_block":
        from freetoken.kernel.triton.fp8_block_linear import Fp8BlockLinear

        return Fp8BlockLinear(in_f, out_f, has_bias)
    if attn_quant == "fp8_pertensor":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorLinear

        return Fp8PerTensorLinear(in_f, out_f, has_bias)
    if attn_quant == "nvfp4":  # compressed-tensors W4A16 attention o_proj / GDN out_proj
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseLinear

        return Nvfp4DenseLinear(in_f, out_f, has_bias)
    from freetoken.layers import LinearReplicated

    return LinearReplicated(in_f, out_f, has_bias=has_bias)


def make_replicated(config, in_f: int, out_f: int, has_bias: bool = False):
    """Config-driven replicated linear: ``Fp8BlockLinear`` under block-fp8, ``Fp8PerTensorLinear``
    under per-tensor-fp8 attention, ``Nvfp4DenseLinear`` under nvfp4, else ``LinearReplicated``."""
    return make_replicated_quant(
        getattr(config, "expert_quant", "none"), getattr(config, "attn_quant", "none"),
        in_f, out_f, has_bias, getattr(config, "dense_precision", "source"),
    )


def make_col_merged(config, in_f: int, output_sizes: list[int], has_bias: bool = False):
    """Config-driven column-merged linear: ``Fp8BlockColMerged`` under block-fp8,
    ``Fp8PerTensorColMerged`` under per-tensor-fp8 attention, ``Nvfp4DenseColMerged`` under
    nvfp4, else ``LinearColParallelMerged``."""
    return make_col_merged_quant(
        getattr(config, "expert_quant", "none"), getattr(config, "attn_quant", "none"),
        in_f, output_sizes, has_bias, getattr(config, "dense_precision", "source"),
    )


def make_lm_head(config, embed_tokens):
    """The output head for the resolved dense precision: per-row FP8 under ``fp8``, else the
    bf16 ``ParallelLMHead`` (tied to ``embed_tokens`` when the checkpoint ties them)."""
    if getattr(config, "dense_precision", "source") == "fp8":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8LMHead

        assert not config.tie_word_embeddings, "FP8 lm_head assumes untied embeddings"
        return Fp8LMHead(config.vocab_size, config.hidden_size)
    from freetoken.layers import ParallelLMHead

    return ParallelLMHead(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        tie_word_embeddings=config.tie_word_embeddings,
        tied_embedding=embed_tokens if config.tie_word_embeddings else None,
    )


__all__ = [
    "make_col_merged_quant",
    "make_replicated_quant",
    "make_replicated",
    "make_col_merged",
    "make_lm_head",
]
