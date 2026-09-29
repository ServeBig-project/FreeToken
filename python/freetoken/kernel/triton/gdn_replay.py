# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ReplaySSM a84849410ab56cc2b23432969eb2ecfc42a13d9c:
#   vllm/model_executor/layers/fla/ops/fused_recurrent_replayssm.py
#   vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py
"""GatedDeltaNet ReplaySSM kernels: recurrent state = checkpoint + update records.

Per value head the recurrent state is a ``[V, K]`` matrix ``S`` (``y = S @ q``). A request
keeps the checkpoint ``S_b`` covering its inputs before position ``b`` and, for every later
consumed position ``p``, the record ``(u_p, k_p, g_p)``: the delta-rule update vector, the
l2-normalized key and the log-decay. Records live in a per-request ring of ``R`` entries
addressed by absolute position, ``p & (R - 1)``, so rolling the valid range forward never
moves data. The state after the records ``[b, b + n)`` is

    S = exp(sum g) * S_b + sum_i exp(sum of the g after i) * u_i k_i^T.

Precision: every product and accumulation runs in fp32; the one matrix product (records
into the state) uses 3xTF32, whose error is close to fp32 and which runs ~18x faster than
IEEE fp32 dots on the 4090. The records u and k are stored in the activation dtype and g in
fp32; the checkpoint keeps its storage dtype. Within one call the recurrence over the call's own inputs uses the unrounded
fp32 u and k, so the only rounding relative to the plain recurrent kernel is the stored
u/k of earlier calls.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _replayed_state(state, u, k, g, slot, row, start, count, i_hv, i_h, o_v, o_k,
                    mask_v, mask_k, stride_state, stride_u, stride_k, stride_g,
                    HV: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    R: tl.constexpr, BC: tl.constexpr):
    """One ``[BV, BK]`` tile of the state after the records ``[start, start + count)``."""
    p_s = state + slot.to(tl.int64) * stride_state + i_hv * V * K
    S = tl.load(p_s + o_v[:, None] * K + o_k[None, :],
                mask=mask_v[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    o_c = tl.arange(0, BC)
    valid = o_c < count
    ring = (start + o_c) & (R - 1)
    row = row.to(tl.int64)
    b_g = tl.load(g + row * stride_g + i_hv * R + ring, mask=valid, other=0.0)
    total = tl.sum(b_g, axis=0)
    coef = tl.where(valid, tl.exp(total - tl.cumsum(b_g, axis=0)), 0.0)
    b_u = tl.load(u + row * stride_u + (i_hv * R + ring[None, :]) * V + o_v[:, None],
                  mask=mask_v[:, None] & valid[None, :], other=0.0).to(tl.float32)
    b_k = tl.load(k + row * stride_k + (i_h * R + ring[:, None]) * K + o_k[None, :],
                  mask=valid[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
    return S * tl.exp(total) + tl.dot(b_u * coef[None, :], b_k, input_precision="tf32x3")


@triton.jit
def _replay_kernel(qkv, a, b, A_log, dt_bias, out, state, u, k, g, cu_seqlens, slots, cursors,
                   scale, stride_qkv, stride_a, stride_b, stride_state, stride_u, stride_k,
                   stride_g, HV: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, R: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                   BC: tl.constexpr):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = tl.load(cursors + i_n * 3)
    if row < 0:  # padding: the output was allocated zeroed, records stay untouched
        return
    start = tl.load(cursors + i_n * 3 + 1)
    pos = tl.load(cursors + i_n * 3 + 2)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    i_h = i_hv // (HV // H)
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, BK)
    mask_v, mask_k = o_v < V, o_k < K
    S = _replayed_state(state, u, k, g, tl.load(slots + i_n), row, start, pos - start, i_hv, i_h,
                        o_v, o_k, mask_v, mask_k, stride_state, stride_u, stride_k, stride_g,
                        HV, H, K, V, R, BC)

    neg_a = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    bias = tl.load(dt_bias + i_hv).to(tl.float32)
    # One program per key head writes its shared key record.
    owns_k = (i_v == 0) & (i_hv % (HV // H) == 0)
    row = row.to(tl.int64)
    rec_u = u + row * stride_u + i_hv * R * V
    rec_k = k + row * stride_k + i_h * R * K
    rec_g = g + row * stride_g + i_hv * R
    for t in range(0, eos - bos):
        tok = bos + t
        p = qkv + tok * stride_qkv
        b_q = tl.load(p + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p + H * K + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p + 2 * H * K + i_hv * V + o_v, mask=mask_v, other=0.0).to(tl.float32)
        x = tl.load(a + tok * stride_a + i_hv).to(tl.float32) + bias
        b_g = neg_a * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        b_beta = 1.0 / (1.0 + tl.exp(-tl.load(b + tok * stride_b + i_hv).to(tl.float32)))
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6) * scale
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        S = S * tl.exp(b_g)
        b_u = b_beta * (b_v - tl.sum(S * b_k[None, :], axis=1))
        S = S + b_u[:, None] * b_k[None, :]
        tl.store(out + (tok * HV + i_hv) * V + o_v,
                 tl.sum(S * b_q[None, :], axis=1).to(out.dtype.element_ty), mask=mask_v)
        slot = (pos + t) & (R - 1)
        tl.store(rec_u + slot * V + o_v, b_u.to(u.dtype.element_ty), mask=mask_v)
        if owns_k:
            tl.store(rec_k + slot * K + o_k, b_k.to(k.dtype.element_ty), mask=mask_k)
        if i_v == 0:
            tl.store(rec_g + slot, b_g)


@triton.jit
def _fold_kernel(state, u, k, g, plan, num_layers, stride_state_layer, stride_state,
                 stride_u_layer, stride_u, stride_k_layer, stride_k, stride_g_layer, stride_g,
                 HV: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                 R: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr):
    i_v, i_x, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    entry, layer = i_x // num_layers, (i_x % num_layers).to(tl.int64)
    src = tl.load(plan + entry * 5)
    dst = tl.load(plan + entry * 5 + 1)
    row = tl.load(plan + entry * 5 + 2)
    start = tl.load(plan + entry * 5 + 3)
    count = tl.load(plan + entry * 5 + 4)
    state = state + layer * stride_state_layer
    i_h = i_hv // (HV // H)
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, BK)
    mask_v, mask_k = o_v < V, o_k < K
    S = _replayed_state(state, u + layer * stride_u_layer, k + layer * stride_k_layer,
                        g + layer * stride_g_layer, src, row, start, count, i_hv, i_h, o_v, o_k,
                        mask_v, mask_k, stride_state, stride_u, stride_k, stride_g,
                        HV, H, K, V, R, BC)
    p_s = state + dst.to(tl.int64) * stride_state + i_hv * V * K
    tl.store(p_s + o_v[:, None] * K + o_k[None, :], S.to(state.dtype.element_ty),
             mask=mask_v[:, None] & mask_k[None, :])


@triton.jit
def _conv_kernel(x, weight, window, out, cu_seqlens, cursors, stride_x, stride_window,
                 D, W, KW: tl.constexpr, BKW: tl.constexpr, BD: tl.constexpr):
    i_n, i_d = tl.program_id(0), tl.program_id(1)
    row = tl.load(cursors + i_n * 3)
    if row < 0:
        return
    pos = tl.load(cursors + i_n * 3 + 2)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    o_d = i_d * BD + tl.arange(0, BD)
    mask_d = o_d < D
    o_j = tl.arange(0, BKW)
    tap = o_j < KW
    w = tl.load(weight + o_d[:, None] * KW + o_j[None, :],
                mask=mask_d[:, None] & tap[None, :], other=0.0).to(tl.float32)
    ring = window + row.to(tl.int64) * stride_window
    for t in range(0, eos - bos):
        # Inputs before ``pos`` come from the ring; this call's own inputs from ``x``.
        p = pos + t - (KW - 1) + o_j
        own = p >= pos
        x_own = tl.load(x + (bos + p - pos)[None, :] * stride_x + o_d[:, None],
                        mask=mask_d[:, None] & (tap & own)[None, :], other=0.0)
        x_old = tl.load(ring + ((p + W) % W)[None, :] * D + o_d[:, None],
                        mask=mask_d[:, None] & (tap & ~own)[None, :], other=0.0)
        acc = tl.sum((x_own.to(tl.float32) + x_old.to(tl.float32)) * w, axis=1)
        tl.store(out + (bos + t) * D + o_d, (acc / (1.0 + tl.exp(-acc))).to(out.dtype.element_ty),
                 mask=mask_d)
        tl.store(ring + ((pos + t) % W) * D + o_d,
                 tl.load(x + (bos + t) * stride_x + o_d, mask=mask_d), mask=mask_d)


_BV = 32


def gdn_replay(
    qkv: torch.Tensor,        # [tokens, 2*H*K + HV*V] post-conv q|k|v, activation dtype
    a: torch.Tensor,          # [tokens, HV] raw decay input
    b: torch.Tensor,          # [tokens, HV] raw beta input
    A_log: torch.Tensor,      # [HV] fp32
    dt_bias: torch.Tensor,    # [HV] fp32
    state: torch.Tensor,      # [slots, HV, V, K] checkpoints (read only)
    u: torch.Tensor,          # [rows, HV, R, V] update-vector records
    k: torch.Tensor,          # [rows, H, R, K] normalized-key records
    g: torch.Tensor,          # [rows, HV, R] fp32 log-decay records
    cu_seqlens: torch.Tensor,  # [n+1] int32 input offsets per sequence
    slots: torch.Tensor,      # [n] int32 checkpoint slot per sequence
    cursors: torch.Tensor,    # [n, 3] int32 (record row, checkpoint position b, first input position p)
    scale: float,
) -> torch.Tensor:
    """Gated delta rule over each sequence's inputs, starting from its checkpoint plus the
    records ``[b, p)``. Returns ``[tokens, HV, V]`` and writes the records of positions
    ``[p, p + T)``. A negative record row marks padding: zero output, nothing written."""
    HV, R, V = u.shape[1:]
    H, K = k.shape[1], k.shape[3]
    out = qkv.new_zeros(qkv.shape[0], HV, V)
    grid = (triton.cdiv(V, _BV), slots.shape[0], HV)
    _replay_kernel[grid](
        qkv, a, b, A_log, dt_bias, out, state, u, k, g, cu_seqlens, slots, cursors, scale,
        qkv.stride(0), a.stride(0), b.stride(0), state.stride(0), u.stride(0), k.stride(0),
        g.stride(0), HV=HV, H=H, K=K, V=V, R=R, BK=triton.next_power_of_2(K), BV=_BV,
        BC=max(16, R), num_warps=4,
    )
    return out


def gdn_replay_fold(
    state: torch.Tensor,  # [layers, slots, HV, V, K]
    u: torch.Tensor,      # [layers, rows, HV, R, V]
    k: torch.Tensor,      # [layers, rows, H, R, K]
    g: torch.Tensor,      # [layers, rows, HV, R]
    plan: torch.Tensor,   # [n, 5] int32 (source slot, destination slot, record row, b, count)
) -> None:
    """For every layer, write into the destination slot the state after the ``count`` records
    following the source checkpoint at position ``b``. Destination == source folds records
    into the checkpoint in place; records are never modified."""
    num_layers, _, HV, R, V = u.shape
    H, K = k.shape[2], k.shape[4]
    grid = (triton.cdiv(V, _BV), plan.shape[0] * num_layers, HV)
    _fold_kernel[grid](
        state, u, k, g, plan, num_layers, state.stride(0), state.stride(1), u.stride(0),
        u.stride(1), k.stride(0), k.stride(1), g.stride(0), g.stride(1), HV=HV, H=H, K=K, V=V,
        R=R, BK=triton.next_power_of_2(K), BV=_BV, BC=max(16, R), num_warps=4,
    )


def gdn_replay_conv(
    x: torch.Tensor,           # [tokens, D] raw conv inputs
    weight: torch.Tensor,      # [D, KW] depthwise conv weight
    window: torch.Tensor,      # [rows, W, D] raw conv inputs by absolute position, p % W
    cu_seqlens: torch.Tensor,  # [n+1] int32
    cursors: torch.Tensor,     # [n, 3] int32, as in gdn_replay (the b column is unused)
) -> torch.Tensor:
    """Causal depthwise conv + silu for inputs at positions ``[p, p + T)`` whose earlier
    ``KW - 1`` inputs are read from the window, which then receives this call's inputs.
    Needs ``W >= KW - 1 + T``. The target conv state is never touched."""
    D, KW = weight.shape
    out = x.new_zeros(x.shape[0], D)
    BD = 256
    _conv_kernel[(cu_seqlens.shape[0] - 1, triton.cdiv(D, BD))](
        x, weight, window, out, cu_seqlens, cursors, x.stride(0), window.stride(0), D,
        window.shape[1], KW=KW, BKW=triton.next_power_of_2(KW), BD=BD, num_warps=4,
    )
    return out


__all__ = ["gdn_replay", "gdn_replay_conv", "gdn_replay_fold"]
