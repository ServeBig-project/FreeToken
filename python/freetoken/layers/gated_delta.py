"""GatedDeltaNet layer shared by the hybrid-linear Qwen models (Qwen3.5/3.6 silu gate,
Qwen3.8-Flash-Next sigmoid gate): mixed decode/prefill batches, speculative verify,
ReplaySSM and the chunk-boundary state snapshots over ``ctx.linear_state_pool``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged
from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged
from freetoken.kernel.triton.gdn_replay import (
    gdn_replay, gdn_replay_advance, gdn_replay_conv, gdn_replay_fold)
from freetoken.models.quant_linear import make_col_merged_quant, make_replicated_quant

from .base import BaseOP
from .linear import LinearColParallelMerged


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class _GatedRMSNorm(BaseOP):
    """RMSNorm of x followed by an ``activation(z)`` gate (HF ``RMSNormGated``), as the fused
    fla ``rms_norm_gated`` kernel."""

    def __init__(self, dim: int, eps: float, activation: str):
        self.weight = torch.empty(dim)
        self.eps = eps
        self.activation = activation

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x, weight=self.weight, bias=None, z=z, eps=self.eps,
            is_rms_norm=True, norm_before_gate=True, activation=self.activation,
        )


class GatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool``.

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.
    ``output_gate`` is the gated norm's activation (``LinearGatedDeltaGroupConfig.output_gate``).
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, expert_quant: str = "none",
        attn_quant: str = "none", output_gate: str = "silu", dense_precision: str = "source",
    ):
        self.layer_id = layer_id
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        # qkv|z carry a weight scale (block-fp8 weight_scale_inv, or per-tensor FP8
        # weight_scale); b|a stay bf16. Both quant modes therefore split the four-way
        # fusion into an fp8 qkvz GEMM + a bf16 ba GEMM (matches sglang/vLLM).
        self._block_fp8 = expert_quant == "fp8_block"
        self._pertensor_fp8 = attn_quant == "fp8_pertensor"
        self._fp8 = self._block_fp8 or self._pertensor_fp8

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        if dense_precision != "source":
            # an explicit plan decides every GDN projection: one fused GEMM, FP8 or BF16
            self._fp8 = False
            self.in_proj = make_col_merged_quant(
                "none", "none", hidden_size, self._in_proj_split, dense_precision=dense_precision
            )
        elif self._fp8:
            ColMerged = Fp8BlockColMerged if self._block_fp8 else Fp8PerTensorColMerged
            self.in_proj_qkvz = ColMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            self.in_proj = LinearColParallelMerged(hidden_size, self._in_proj_split, has_bias=False)
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        self.dt_bias = torch.empty(num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(num_v_heads, dtype=torch.float32)
        self.norm = _GatedRMSNorm(head_v_dim, rms_norm_eps, output_gate)
        # out_proj follows the checkpoint quant: block-fp8 / per-tensor-fp8 / compressed-tensors
        # NVFP4 (W4A16) / bf16. in_proj_* stay bf16 in every mode (above), so a compressed-tensors
        # NVFP4 checkpoint (attn_quant=="nvfp4") only makes out_proj native FP4.
        self.out_proj = make_replicated_quant(
            expert_quant, attn_quant, self.value_dim, hidden_size, has_bias=False,
            dense_precision=dense_precision,
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (h is [V,K], the
        state pool is [K,V]; they coincide because GDN requires head_k_dim == head_v_dim).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def _run_decode(
        self, conv_in: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
        pool, li: int, fla, dtype: torch.dtype, positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the fused single-token conv and recurrence for a decode sub-batch."""
        if fla.rows is not None:
            return self._run_replay(conv_in, a, b, pool, li, fla, positions)
        mixed = self._conv_decode(conv_in, fla.cache_indices, pool)
        size = mixed.shape[0]
        qf, kf, vf = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = qf.reshape(1, size, self.num_k_heads, self.head_k_dim).to(dtype)
        k = kf.reshape(1, size, self.num_k_heads, self.head_k_dim).to(dtype)
        v = vf.reshape(1, size, self.num_v_heads, self.head_v_dim).to(dtype)
        return gdn_decode_fla(
            q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
            state_source=pool.recurrent_states[li], indices=fla.cache_indices,
            cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
        )

    def _run_replay(self, conv_in, a, b, pool, li: int, fla, positions) -> torch.Tensor:
        """ReplaySSM: target decode, draft steps and verify windows start from each request's
        checkpoint plus records. Before its first GDN layer, target decode folds full rings for
        every layer and advances the checkpoint positions, so no record is overwritten while
        another layer or head still reads it; draft and verify never fold and use the replay
        conv window."""
        replay = pool.replay
        if not fla.speculative and li == 0:
            gdn_replay_fold(pool.recurrent_states, replay.u, replay.k, replay.g, replay.start,
                            fla.rows, fla.cache_indices, fla.cache_indices, positions, 1)
            gdn_replay_advance(replay.start, replay.stats, fla.rows, positions, 1, replay.ring)
        if fla.speculative:
            mixed = gdn_replay_conv(conv_in, self._conv_weight(), replay.window[li],
                                    fla.cu_seqlens, fla.rows, positions)
        else:
            mixed = self._conv_decode(conv_in, fla.cache_indices, pool)
        return gdn_replay(
            mixed, a, b, self.A_log, self.dt_bias, pool.recurrent_states[li], replay.u[li],
            replay.k[li], replay.g[li], replay.start, fla.cu_seqlens, fla.cache_indices,
            fla.rows, positions, self.head_k_dim ** -0.5,
        )

    def _run_verify(
        self, conv_in: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
        pool, li: int, steps, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Speculative verify: one single-token recurrence per position, each starting from a
        copy of the previous position's state, so every intermediate state survives for the
        commit and the live slot is never touched."""
        rec, cv = pool.recurrent_states[li], pool.conv_states[li]
        core_out = None
        for step in steps:
            dst = step.path.cache_indices.long()
            rec.index_copy_(0, dst, rec.index_select(0, step.prev))
            cv.index_copy_(0, dst, cv.index_select(0, step.prev))
            out = self._run_decode(conv_in[step.rows], a[step.rows], b[step.rows], pool, li, step.path, dtype)
            if core_out is None:  # one extra row absorbs the inert requests' writes
                core_out = out.new_empty((conv_in.shape[0] + 1, *out.shape[1:]))
            core_out.index_copy_(0, step.write, out)
        return core_out[:-1]

    def _run_prefill(
        self, conv_in: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
        pool, li: int, fla, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run varlen conv and chunked recurrence for a prefill sub-batch."""
        if fla.track_start_dst is not None:
            for states in (pool.recurrent_states[li], pool.conv_states[li]):
                states.index_copy_(0, fla.track_start_dst, states.index_select(0, fla.track_start_src))
        mixed = self._conv_prefill(
            conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state
        )
        size = mixed.shape[0]
        qf, kf, vf = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = qf.reshape(1, size, self.num_k_heads, self.head_k_dim).to(dtype)
        k = kf.reshape(1, size, self.num_k_heads, self.head_k_dim).to(dtype)
        v = vf.reshape(1, size, self.num_v_heads, self.head_v_dim).to(dtype)
        g, beta = self._gate_params(a, b)
        g = g.reshape(1, size, self.num_v_heads)
        beta = beta.float().reshape(1, size, self.num_v_heads)
        # The chunk kernel reads and writes state_source[cache_indices] in place.
        if fla.fresh_state_indices is not None:
            pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
        track = fla.track_dst is not None
        result = gdn_prefill_chunk_fla(
            q, k, v, g, beta,
            state_source=pool.recurrent_states[li], indices=fla.cache_indices,
            cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
            return_h=track,
        )
        if not track:
            return result
        core_out, h = result
        self._write_track_snapshot(pool, li, conv_in, h, fla)
        return core_out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._fp8:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        li = pool.local_index(self.layer_id)

        if fla.verify is not None:
            core_out = self._run_verify(conv_in, a, b, pool, li, fla.verify, dtype)
        elif fla.prefill is None:
            assert fla.decode is not None
            core_out = self._run_decode(conv_in, a, b, pool, li, fla.decode, dtype,
                                        batch.positions)
        elif fla.decode is not None:
            split = batch.decode_size
            decode_out = self._run_decode(
                conv_in[:split], a[:split], b[:split], pool, li, fla.decode, dtype,
                batch.positions[:split],
            )
            prefill_out = self._run_prefill(
                conv_in[split:], a[split:], b[split:], pool, li, fla.prefill, dtype
            )
            core_out = torch.cat((decode_out, prefill_out), dim=0)
        else:
            core_out = self._run_prefill(conv_in, a, b, pool, li, fla.prefill, dtype)

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        return self.out_proj.forward(out)


def gdn_prefill_chunk_fla(
    q: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, total, num_v_heads, head_v_dim] bf16
    g: torch.Tensor,        # [1, total, num_v_heads] log-decay (<=0), fp32
    beta: torch.Tensor,     # [1, total, num_v_heads] fp32
    *,
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32 (in place)
    indices: torch.Tensor,       # [num_seqs] slot id per sequence
    cu_seqlens: torch.Tensor,    # [num_seqs+1] int64
    scale: float,
    return_h: bool = False,
) -> torch.Tensor:
    """Chunked gated-delta-rule prefill via the vendored fla kernel. GQA is handled
    in-kernel (q/k at num_k_heads), q/k l2norm is done in-kernel, and the per-sequence
    recurrent state is read from and written back to ``state_source[indices]`` IN PLACE.
    Fresh sequences must have their ``state_source`` slot pre-zeroed by the caller.
    Returns ``o`` of shape ``[total, num_v_heads, head_v_dim]`` (bf16).

    When ``return_h=True`` also returns the per-chunk hidden-state buffer ``h`` of shape
    ``[1, NT_total, num_v_heads, head_v_dim, head_k_dim]`` (bf16). ``h[0, boh_i + c]`` is the
    recurrent state after ``c*64`` tokens of packed sequence ``i`` (chunk granularity 64), where
    ``boh_i = prepare_chunk_offsets(cu_seqlens, 64)[i]``. Note the last two dims are ``[V, K]`` --
    transposed vs ``state_source``'s ``[K, V]``. Used by the hybrid-radix track-checkpoint path."""
    from freetoken.kernel.fla import chunk_gated_delta_rule

    o, _, h = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=state_source, initial_state_indices=indices.to(torch.int32),
        cu_seqlens=cu_seqlens.to(torch.int64), head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    if return_h:
        return o[0], h
    return o[0]  # [total, num_v_heads, head_v_dim]


def gdn_decode_fla(
    q: torch.Tensor,        # [1, B, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, B, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, B, num_v_heads, head_v_dim] bf16
    a: torch.Tensor,        # [B, num_v_heads] raw
    b: torch.Tensor,        # [B, num_v_heads] raw
    *,
    A_log: torch.Tensor,        # [num_v_heads]
    dt_bias: torch.Tensor,      # [num_v_heads]
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32 (in place)
    indices: torch.Tensor,      # [B] int32 slot id per request
    cu_seqlens: torch.Tensor,   # [B+1] query indptr (arange) from FLAMetadata
    scale: float,
) -> torch.Tensor:
    """Fused sigmoid-gating gated-delta-rule decode (vendored fla triton kernel): gating +
    in-kernel l2norm + recurrent update + state read/write-by-index in one kernel. Returns
    [B, num_v, V]."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    o = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias,  # already fp32 (stored fp32)
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=state_source,
        initial_state_indices=indices,  # already int32 (built int32 in the scheduler)
        scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
    )
    # kernel returns o = [NK, *v.shape] then squeeze(NK) -> [1, B, num_v, V].
    return o[0]


__all__ = ["GatedDeltaNet", "gdn_decode_fla", "gdn_prefill_chunk_fla"]
