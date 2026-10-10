from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SlotStateSpec,
    detect_expert_quant,
)
from freetoken.models.qwen3_5_moe.config import _expert_quant as _modelopt_expert_quant


@dataclass(frozen=True)
class Qwen4ExpArgs:
    """Qwen3.8-Flash-Next geometry beyond the generic ModelConfig fields (ModelConfig.qwen4_args)."""

    hidden_size: int
    # Hyper-connections: every layer reads/writes hc_count residual streams [T, hc_count*hidden].
    hc_count: int
    hc_lowrank: int
    # PLE n-gram embedding; layer ids are zero-based decoder layers.
    ple_layer_ids: Tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    split_ngram_parts: int
    # n-gram hash windows never cross this token (the eos id); they restart after it.
    ngram_boundary_token_id: int
    # QSA indexer scoring geometry (the slab/ratio geometry lives on the attention group).
    index_n_heads: int
    index_head_dim: int
    index_budget: int
    index_ratio: int

    @property
    def num_ngram_heads(self) -> int:
        # one head group per n-gram order 2..ngram_size (Qwen3.8: 8 x 2-gram + 8 x 3-gram)
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_head_dim(self) -> int:
        return self.ple_embed_dim // self.num_ngram_heads

    @property
    def ple_conv_dilation(self) -> int:
        # HF Qwen4ExpTextPLELayer sets the depthwise conv dilation to ngram_size
        return self.ngram_size

    @property
    def ple_conv_state_len(self) -> int:
        return (self.ple_conv_kernel_size - 1) * self.ple_conv_dilation

    @property
    def stream_width(self) -> int:
        return self.hc_count * self.hidden_size


PLE_CONV_STATE = "ple_conv"
PLE_NGRAM_STATE = "ple_ngram_ctx"
QSA_PENDING_STATE = "qsa_pending"
MTP_PENDING_STATE = "mtp_pending"
MTP_TAIL_STATE = "mtp_tail"


def slot_states(args: Qwen4ExpArgs, qsa_layer_ids: Tuple[int, ...],
                mtp_layer_ids: Tuple[int, ...] = ()) -> Tuple[SlotStateSpec, ...]:
    """Per-request state beyond the GDN recurrence, riding the linear-state slots: the PLE
    conv history and n-gram context, the raw index keys of each QSA layer's open group, and
    the MTP layer's open group and the target's last streams it pairs with the next token."""
    specs = []
    if args.ple_layer_ids:
        specs += [
            SlotStateSpec(
                name=PLE_CONV_STATE,
                shape=(args.stream_width, args.ple_conv_state_len),
                layer_ids=args.ple_layer_ids,
            ),
            # last ngram_size-1 token ids, shared by every PLE layer; eos = hash boundary
            SlotStateSpec(
                name=PLE_NGRAM_STATE,
                shape=(args.ngram_size - 1,),
                dtype=torch.int32,
                fill_value=float(args.ngram_boundary_token_id),
            ),
        ]
    specs.append(
        SlotStateSpec(
            name=QSA_PENDING_STATE,
            shape=(args.index_ratio, args.index_head_dim),
            layer_ids=qsa_layer_ids,
        )
    )
    if mtp_layer_ids:
        specs += [
            SlotStateSpec(name=MTP_PENDING_STATE, shape=(args.index_ratio, args.index_head_dim),
                          layer_ids=mtp_layer_ids, draft=True),
            SlotStateSpec(name=MTP_TAIL_STATE, shape=(args.stream_width,), draft=True),
        ]
    return tuple(specs)


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        # Transformers rewrites full_attention to qwen_sparse_attention in __post_init__.
        return ["full_attention" if t == "qwen_sparse_attention" else t for t in layer_types]
    interval = int(getattr(text, "full_attention_interval", 4))
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(int(text.num_hidden_layers))
    ]


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    # int(), not round(): HF configuration_qwen4_exp truncates head_dim * partial.
    rotary_dim = int(head_dim * partial)
    # Text-only serving with the default rope type: the mRoPE sections reduce to standard
    # partial rope, so no scaling dict reaches get_rope's cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    # Routed experts: plain NVFP4 (quant_algo NVFP4) or modelopt MIXED_PRECISION (per-module
    # quantized_layers). Every other projection of the supported checkpoints is bf16.
    expert_quant = detect_expert_quant(hf_config)
    if "mixed" in expert_quant:
        expert_quant = _modelopt_expert_quant(hf_config)
    if expert_quant != "nvfp4":
        raise ValueError(
            f"qwen4_exp serves NVFP4 routed experts; the checkpoint declares {expert_quant!r}"
        )

    # HF stores ple_layer_ids one-indexed.
    ple_layer_ids = tuple(int(i) - 1 for i in (getattr(text, "ple_layer_ids", None) or ()))
    for lid in ple_layer_ids:
        if layer_types[lid] != "linear_attention":
            raise ValueError(f"PLE must sit on a linear_attention layer, got layer {lid}")

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
        index_head_dim=int(text.indexer_head_dim),
        num_index_layers=len(full_ids),
        index_ratio=int(text.indexer_compress_ratio),
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        # HF resolves a null output_gate_type to hidden_act.
        output_gate=str(getattr(text, "output_gate_type", None) or text.hidden_act),
    )
    groups = tuple(sorted((full_group, linear_group), key=lambda g: g.layer_ids[0]))

    # HF accepts int | list here and uses the first entry (Qwen4ExpTextNGramEmbedding).
    eos_token_id = text.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]

    mtp_layers = int(getattr(text, "mtp_num_hidden_layers", 0) or 0)
    qwen4_args = Qwen4ExpArgs(
        hidden_size=text.hidden_size,
        hc_count=int(text.hc_count),
        hc_lowrank=int(text.hc_lowrank),
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=int(text.ple_embed_dim),
        ple_conv_kernel_size=int(text.ple_conv_kernel_size),
        ngram_size=int(text.ngram_size),
        heads_per_ngram=int(text.heads_per_ngram),
        split_ngram_parts=int(text.split_ngram_parts),
        ngram_boundary_token_id=int(eos_token_id),
        index_n_heads=int(text.indexer_n_heads),
        index_head_dim=int(text.indexer_head_dim),
        index_budget=int(text.indexer_budget),
        index_ratio=int(text.indexer_compress_ratio),
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0) or 0,
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=int(text.num_experts),
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.moe_intermediate_size),
        shared_expert_intermediate_size=int(text.shared_expert_intermediate_size),
        moe_router="softmax",
        # Absent from some shipped configs; HF defaults it True.
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        moe_enabled=True,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen4_exp"),
        architectures=getattr(hf_config, "architectures", ["Qwen4ExpForConditionalGeneration"]),
        attention_groups=groups,
        expert_quant=expert_quant,
        qwen4_args=qwen4_args,
        slot_states=slot_states(qwen4_args, full_ids, tuple(range(
            text.num_hidden_layers, text.num_hidden_layers + mtp_layers))),
        mtp_layers=mtp_layers,
    )


__all__ = [
    "MTP_PENDING_STATE",
    "MTP_TAIL_STATE",
    "PLE_CONV_STATE",
    "PLE_NGRAM_STATE",
    "QSA_PENDING_STATE",
    "Qwen4ExpArgs",
    "parse_config",
    "slot_states",
]
