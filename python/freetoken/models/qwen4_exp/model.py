"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over
the ``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and
injects its output back, and the top-level mixer collapses them once before ``lm_head``.
There is no input/post layernorm and no final ``model.norm``::

    R  = R + ple(R, batch)                 # the PLE layer only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)

Layer-group execution (layered pipeline, decode layer-range graphs) carries ``R`` and the
next layer as its opaque state; the plain forward is the same path over all layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, VocabParallelEmbedding
from freetoken.layers.gated_delta import GatedDeltaNet
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.quant_linear import make_lm_head
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .ple import PLELayer

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


@dataclass
class StreamState:
    """Layer-group state: the four residual streams of every row and the next layer to run."""

    streams: torch.Tensor  # [rows, hc_count*hidden]
    next_layer: int = 0


class Qwen4ExpDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                output_gate=g.output_gate,
                dense_precision=config.dense_precision,
            )
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen3_5MoE(config, layer_id)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.ple = PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, R: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.ple is not None:
            R = R + self.ple.forward(R, batch)
        x, s = self.attn_hyper_connection.mix(R)
        y = self.linear_attn.forward(x) if self._is_linear else self.self_attn.forward(x, batch)
        R = self.attn_hyper_connection.combine(R, y, s)
        x, s = self.mlp_hyper_connection.mix(R)
        return self.mlp_hyper_connection.combine(R, self.mlp.forward(x), s)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    def embed(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        """The streams before layer 0; also the moment a capture at a chunk start keeps the
        state before this forward (the GDN layers copy their own, the declared slot states
        are copied once here)."""
        if batch.fla_metadata is None:  # direct-op callers; the engine builds it before forward
            from freetoken.attention.linear import build_fla_metadata

            batch.fla_metadata = build_fla_metadata(batch, input_ids.device)
        prefill = batch.fla_metadata.prefill
        if prefill is not None and prefill.track_start_dst is not None:
            for t in get_global_ctx().linear_state_pool.slot_states.values():
                t.index_copy_(1, prefill.track_start_dst, t.index_select(1, prefill.track_start_src))
        return self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        self.lm_head = make_lm_head(config, self.model.embed_tokens)
        super().__init__()

    def load_host_tables(self, engine_config) -> None:
        """Attach the PLE n-gram table: the pinned checkpoint bank, or zeros for dummy weights."""
        ple_layers = [layer.ple for layer in self.model.layers.op_list if layer.ple is not None]
        if not ple_layers:
            return
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        args = self._config.qwen4_args
        if engine_config.use_dummy_weight:
            # dummy fill leaves the int64 hash buffers zero (a zero vocab size divides by zero)
            for ple in ple_layers:
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size, ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=20_000_000, ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(args.ngram_head_dim))
            return
        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, args)
        self._ple_table = table  # owns the pinned HostBank
        device = self.model.embed_tokens.weight.device
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale), device)
            )

    def create_layered_execution_adapter(self, engine):
        from freetoken.engine.layered_execution import LinearStateLayeredExecutionAdapter

        return LinearStateLayeredExecutionAdapter(engine)

    # ----- layer groups -----------------------------------------------------------------
    @property
    def layer_group_num_layers(self) -> int:
        return len(self.model.layers.op_list)

    def begin_layer_group_prefill(self, input_ids: torch.Tensor) -> StreamState:
        return StreamState(self.model.embed(input_ids, get_global_ctx().batch))

    @staticmethod
    def layer_group_state_layer(state: StreamState) -> int:
        return state.next_layer

    @staticmethod
    def layer_group_merge_states(decode: StreamState, prefill: StreamState) -> StreamState:
        if decode.next_layer != prefill.next_layer:
            raise RuntimeError("decode and prefill states are at different layers")
        return StreamState(torch.cat((decode.streams, prefill.streams), dim=0), decode.next_layer)

    @staticmethod
    def layer_group_split_state(state: StreamState, decode_rows: int) -> tuple[StreamState, StreamState]:
        return (
            StreamState(state.streams[:decode_rows], state.next_layer),
            StreamState(state.streams[decode_rows:], state.next_layer),
        )

    @staticmethod
    def create_layer_range_graph_inputs(seed: StreamState) -> StreamState:
        return StreamState(torch.zeros_like(seed.streams))

    @staticmethod
    def make_layer_range_graph_state(inputs: StreamState, start_layer: int, rows: int) -> StreamState:
        return StreamState(inputs.streams[:rows], start_layer)

    @staticmethod
    def stage_layer_range_graph_inputs(
        inputs: StreamState, state: StreamState, rows: int, start_layer: int
    ) -> None:
        if state.next_layer != start_layer:
            raise ValueError(f"layer-range replay expected state at layer {start_layer}")
        inputs.streams[:rows].copy_(state.streams)

    @staticmethod
    def finish_layer_range_graph_replay(captured: StreamState, rows: int, end_layer: int) -> StreamState:
        return StreamState(captured.streams[:rows], end_layer)

    def advance_layer_group_prefill(self, state: StreamState, end_layer: int) -> StreamState:
        if not state.next_layer < end_layer <= self.layer_group_num_layers:
            raise ValueError(
                f"invalid layer-group range [{state.next_layer}, {end_layer}) for "
                f"{self.layer_group_num_layers} layers"
            )
        batch = get_global_ctx().batch
        for layer_id in range(state.next_layer, end_layer):
            state.streams = self.model.layers.op_list[layer_id].forward(state.streams, batch)
        state.next_layer = end_layer
        return state

    def finish_layer_group_prefill(
        self, state: StreamState, output_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        if state.next_layer != self.layer_group_num_layers:
            raise ValueError("cannot finish layer-group prefill before every decoder layer ran")
        if output_indices is None:
            hidden = self.model.hyper_connection_mixer.mix(state.streams)[0]
            return self.lm_head.forward(hidden)
        hidden = self.model.hyper_connection_mixer.mix(state.streams[output_indices].contiguous())[0]
        return self.lm_head.forward_selected(hidden)

    def forward(self) -> torch.Tensor:
        state = self.begin_layer_group_prefill(get_global_ctx().batch.input_ids)
        return self.finish_layer_group_prefill(
            self.advance_layer_group_prefill(state, self.layer_group_num_layers)
        )


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "StreamState"]
