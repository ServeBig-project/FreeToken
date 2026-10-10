"""QSA full-attention layer for Qwen3.8-Flash-Next (12 of 48 layers).

Gated GQA (24 q heads / 2 kv heads / head_dim 256, zero-centered q/k norms, partial NeoX rope
over 64 dims, ``q_proj`` twice as wide for the output gate) plus the weights of the QSA
indexer (``index_qk_proj`` = 4 index q heads x 128 then 1 index k head x 128, and the two
per-head index norms). The layer owns the weights and hands the backend the RAW index
projections; the backend owns everything stateful (compressed slab, pending keys, scoring,
top-k, sparse attend). The index k norm runs AFTER the fp32 mean over each group of
``index_ratio`` raw keys, so both index norm weights travel with the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.models.quant_linear import make_col_merged, make_replicated
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


@dataclass(frozen=True)
class QSAIndexerInputs:
    """Raw ``index_qk_proj`` slices (no norm, no rope). The backend applies, per HF
    ``Qwen4ExpTextQSAIndexer``::

        q_h    = rope64(rmsnorm(q_h) * (1 + q_norm_weight), pos = query position)
        kbar_b = rope64(rmsnorm(mean_fp32(k[4b:4b+4])) * (1 + k_norm_weight), pos = 4b)
        s_b    = sum_h relu(<q_h, kbar_b>) / sqrt(index_head_dim)
    """

    q: torch.Tensor  # [T, index_n_heads, index_head_dim]
    k: torch.Tensor  # [T, index_head_dim]
    q_norm_weight: torch.Tensor  # [index_head_dim], zero-centered
    k_norm_weight: torch.Tensor  # [index_head_dim], zero-centered
    eps: float


class Qwen4ExpIndexer(BaseOP):
    """QSA indexer weights (checkpoint prefix ``self_attn.indexer``); the scoring lives in the backend.

    ``split`` keeps index q and index k as separate projections (``index_q_proj``,
    ``index_k_proj``): the MTP's history update needs index k alone."""

    def __init__(self, config: ModelConfig, split: bool = False) -> None:
        args = config.qwen4_args
        self.num_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.eps = config.rms_norm_eps
        self._split = [self.num_heads * self.head_dim, self.head_dim]
        if split:
            self.index_q_proj = make_replicated(config, args.hidden_size, self._split[0])
            self.index_k_proj = make_replicated(config, args.hidden_size, self._split[1])
        else:
            self.index_qk_proj = make_replicated(config, args.hidden_size, sum(self._split))
        self.q_layernorm = GemmaPlusOneRMSNorm(self.head_dim, eps=self.eps)
        self.k_layernorm = GemmaPlusOneRMSNorm(self.head_dim, eps=self.eps)

    def forward(self, x: torch.Tensor) -> QSAIndexerInputs:
        if hasattr(self, "index_qk_proj"):
            q, k = self.index_qk_proj.forward(x).split(self._split, dim=-1)
        else:
            q, k = self.index_q_proj.forward(x), self.index_k_proj.forward(x)
        return QSAIndexerInputs(
            q=q.reshape(-1, self.num_heads, self.head_dim).contiguous(),
            k=k.contiguous(),
            q_norm_weight=self.q_layernorm.weight,
            k_norm_weight=self.k_layernorm.weight,
            eps=self.eps,
        )

    def keys(self, x: torch.Tensor) -> QSAIndexerInputs:
        """The raw index keys alone (``split`` only): a history update scores nothing."""
        return QSAIndexerInputs(None, self.index_k_proj.forward(x).contiguous(),
                                self.q_layernorm.weight, self.k_layernorm.weight, self.eps)


class Qwen4ExpAttention(BaseOP):
    """Gated GQA with a QSA indexer::

        q, gate = chunk(q_proj(x).view(-1, num_q, 2*head_dim), 2, -1)
        q, k    = rope(q_norm(q), k_norm(k_proj(x)))          # first rotary_dim dims
        o       = backend.qsa_forward(q, k, v_proj(x), indexer(x), layer_id, batch)
        out     = o_proj(o * sigmoid(gate))

    q/k/v are one merged GEMM (``qkv_proj``, split ``[num_q*head_dim*2, kv, kv]``); the loader
    concatenates the checkpoint's ``q_proj``/``k_proj``/``v_proj`` along dim 0. ``split`` keeps
    them as three projections instead (the MTP layer: its history update runs k and v alone).
    ``q_norm`` / ``k_norm`` are zero-centered and loaded RAW. ``attend(q, k, v, index)`` owns
    the history: the target's global backend, or the MTP's own.
    """

    def __init__(self, config: ModelConfig, layer_id: int, split: bool = False) -> None:
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.qo_attn_dim = self.num_q * self.head_dim
        self.kv_attn_dim = self.num_kv * self.head_dim
        self._qkv_split = [self.qo_attn_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        if split:
            self.q_proj = make_replicated(config, config.hidden_size, self._qkv_split[0])
            self.k_proj = make_replicated(config, config.hidden_size, self.kv_attn_dim)
            self.v_proj = make_replicated(config, config.hidden_size, self.kv_attn_dim)
        else:
            self.qkv_proj = make_col_merged(config, config.hidden_size, self._qkv_split)
        self.o_proj = make_replicated(config, self.qo_attn_dim, config.hidden_size)
        self.q_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary = config.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rotary.base,
            rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
        )
        self.indexer = Qwen4ExpIndexer(config, split)

    @nvtx_annotate("QSA")
    def forward(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        backend = get_global_ctx().attn_backend
        return self.attend(x, batch.positions, lambda q, k, v, index: backend.qsa_forward(
            q, k, v, index, self.layer_id, batch))

    def attend(self, x: torch.Tensor, positions: torch.Tensor,
               attend: Callable[..., torch.Tensor]) -> torch.Tensor:
        if hasattr(self, "qkv_proj"):
            qg, k, v = self.qkv_proj.forward(x).split(self._qkv_split, dim=-1)
        else:
            qg, k, v = self.q_proj.forward(x), self.k_proj.forward(x), self.v_proj.forward(x)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.contiguous().view(-1, self.num_kv, self.head_dim)
        v = v.contiguous()
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)
        q, k = self.rotary.forward(
            positions, q.view(-1, self.qo_attn_dim), k.view(-1, self.kv_attn_dim)
        )
        o = attend(q.view(-1, self.num_q, self.head_dim), k, v, self.indexer.forward(x))
        return self.o_proj.forward(o.reshape(-1, self.qo_attn_dim) * torch.sigmoid(gate))

    def write_history(self, x: torch.Tensor, positions: torch.Tensor,
                      write: Callable[..., None]) -> None:
        """What later queries read of ``x``'s rows: K (normed, roped at ``positions``), V and
        the raw index keys, handed to ``write(k, v, index)`` (``split`` only; no query)."""
        k = self.k_proj.forward(x).view(-1, self.num_kv, self.head_dim)
        self.k_norm.forward_inplace(k)
        # The rope kernel rotates a query too; this one is a throwaway single head.
        _, k = self.rotary.forward(positions, k.new_empty(k.shape[0], self.head_dim),
                                   k.view(-1, self.kv_attn_dim))
        write(k, self.v_proj.forward(x).contiguous(), self.indexer.keys(x))


__all__ = ["QSAIndexerInputs", "Qwen4ExpAttention", "Qwen4ExpIndexer"]
