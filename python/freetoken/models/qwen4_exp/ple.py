"""Per-Layer Embedding (PLE) for Qwen3.8-Flash-Next: hashed n-gram features injected at one
GDN layer. HF reference: ``Qwen4ExpTextNGramEmbedding`` and ``Qwen4ExpTextPLELayer``. Per token::

    E = table[hash(ngram)]                       # 16 heads (8 x 2-gram, 8 x 3-gram) x 160 -> 2560
    K = norm_key(key_proj(E)).view(hc, hidden)   # V = value_proj(E) [hidden]
    Q = norm_query(R).view(hc, hidden)
    u = <K_i, Q_i> / sqrt(hidden)                # per stream
    U = sigmoid(sign(u) * sqrt(max(|u|, 1e-6))) * V
    D = U + silu(conv1d(norm_conv(U)))           # depthwise, kernel 4, dilation ngram_size
    R += D                                       # before the attention hyper-connection mix

The table is the FP8 n-gram store pinned in host memory (``PinnedUVATable``); rows are
gathered over UVA by a Triton kernel at the layer's entry. Each request's conv history and
its last ``ngram_size-1`` token ids ride its linear-state slot (``config.slot_states``) and
advance exactly once per forward, when the layer runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP
from freetoken.models.quant_linear import make_replicated

from .config import PLE_CONV_STATE, PLE_NGRAM_STATE
from .hc import GroupedPlusOneRMSNorm

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

    from .config import Qwen4ExpArgs

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


class ZeroTable:
    """Dummy-weight stand-in: every lookup reads zeros (dummy checkpoints ship no table)."""

    def __init__(self, head_dim: int) -> None:
        self.head_dim = head_dim

    def lookup(self, row_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (row_ids.shape[0], row_ids.shape[1] * self.head_dim),
            dtype=torch.bfloat16, device=row_ids.device,
        )


class PinnedUVATable:
    """PLE table left in pinned host memory; rows are gathered over UVA into a bf16 staging
    buffer. ``weight`` is the filled and pinned ``HostBank.tensor`` (``[num_rows, head_dim]``
    fp8-e4m3); ``scale`` is the checkpoint's scalar ``weight_scale``. A captured graph keeps
    one staging buffer per row count for good; eager lookups share one growable buffer."""

    def __init__(self, weight: torch.Tensor, scale: float, device: torch.device) -> None:
        from freetoken.kernel.pinned import device_ptr

        assert weight.dtype == torch.float8_e4m3fn and weight.is_contiguous(), weight.dtype
        self.weight = weight  # keeps the pinned bank alive
        self.scale = float(scale)
        self.num_rows, self.head_dim = weight.shape
        self._device = device
        self._table_ptr = device_ptr(weight)
        self._staging: torch.Tensor | None = None
        self._graph_staging: dict[int, torch.Tensor] = {}

    def _stage(self, rows: int) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            buf = self._graph_staging.get(rows)
            if buf is None:
                buf = torch.empty((rows, self.head_dim), dtype=torch.bfloat16, device=self._device)
                self._graph_staging[rows] = buf
            return buf
        buf = self._staging
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty((rows, self.head_dim), dtype=torch.bfloat16, device=self._device)
            self._staging = buf
        return buf[:rows]

    def lookup(self, row_ids: torch.Tensor) -> torch.Tensor:
        """``[T, heads] int64`` global row ids -> ``[T, heads * head_dim]`` bf16 (a staging view)."""
        from freetoken.kernel.triton.ple import ple_gather_rows

        rows = ple_gather_rows(
            self._table_ptr, self.num_rows, self.head_dim, row_ids.reshape(-1),
            self._stage(row_ids.numel()), self.scale, True,
        )
        return rows.view(row_ids.shape[0], -1)


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def derive_ngram_hash_constants(
    *, vocab_size: int, ngram_size: int, num_ngram_heads: int, ngram_vocab_size_base: int,
    ple_layer_index: int, seed: int = 1234,
) -> Tuple[List[int], List[int], List[int]]:
    """(multipliers, per-head vocab sizes, per-head offsets) the way HF derives them at init.
    The checkpoint ships them as int64 tensors; this serves the dummy-weight path."""
    half_bound = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
    base_seed = seed + _PLE_LAYER_PRIME * ple_layer_index
    multipliers = [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(ngram_size)
    ]
    sizes, offsets, total = [], [], 0
    for head in range(num_ngram_heads):
        size = _nth_prime_after(ngram_vocab_size_base - 1, ple_layer_index * num_ngram_heads + head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return multipliers, sizes, offsets


@dataclass
class PLEMetadata:
    """This forward's PLE inputs, one ragged view over the batch's rows (decode rows first)::

      input_ids      [T] int device, this forward's tokens in request order
      cu_seqlens     [B+1] int64 device, query indptr
      seq_lens       host per-request token counts (the ragged conv needs them on the host)
      ngram_context  [B, ngram_size-1] int64 device, the tokens before each request's first
                     row, from the ``ple_ngram_ctx`` slot state; eos for fresh rows
      state_slots    [B] int64 device, linear-state slot per request
      fresh_slots    [B] bool device or None: the request starts a new sequence
      is_decode      one token per request, read off the persistent decode view
    """

    input_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    seq_lens: Sequence[int]
    ngram_context: torch.Tensor
    state_slots: torch.Tensor
    fresh_slots: torch.Tensor | None
    is_decode: bool


def build_ple_metadata(batch: Batch, context_pool: torch.Tensor, eos: int) -> PLEMetadata:
    """Derive the PLE view from the batch's GDN metadata (``FLAMetadata.decode/prefill``), so
    the two states always index the same slots. The decode-only view is pure device
    arithmetic over the persistent decode buffers (capture-safe)."""
    fla = batch.fla_metadata
    if fla.verify is not None:
        raise NotImplementedError("qwen4_exp does not serve speculative verify")
    device = batch.input_ids.device
    if fla.prefill is None:
        slots = fla.decode.cache_indices.long()
        bs = slots.numel()
        return PLEMetadata(
            input_ids=batch.input_ids,
            cu_seqlens=torch.arange(bs + 1, dtype=torch.int64, device=device),
            seq_lens=(1,) * bs,
            ngram_context=context_pool.index_select(0, slots).long(),
            state_slots=slots,
            fresh_slots=None,
            is_decode=True,
        )
    prefill = fla.prefill
    cu = prefill.cu_seqlens.long()
    slots = prefill.cache_indices.long()
    fresh = ~prefill.has_initial_state
    lens = [r.extend_len for r in batch.prefill_reqs]
    if fla.decode is not None:
        bs = batch.decode_size
        cu = torch.cat([torch.arange(bs + 1, dtype=torch.int64, device=device), cu[1:] + bs])
        slots = torch.cat([fla.decode.cache_indices.long(), slots])
        fresh = torch.cat([torch.zeros(bs, dtype=torch.bool, device=device), fresh])
        lens = [1] * bs + lens
    context = context_pool.index_select(0, slots).long()
    context = torch.where(fresh.unsqueeze(1), context.new_full((), eos), context)
    return PLEMetadata(
        input_ids=batch.input_ids,
        cu_seqlens=cu,
        seq_lens=tuple(lens),
        ngram_context=context,
        state_slots=slots,
        fresh_slots=fresh,
        is_decode=False,
    )


def commit_ngram_context(meta: PLEMetadata, context_pool: torch.Tensor,
                         track: tuple[torch.Tensor, torch.Tensor] | None) -> None:
    """Roll each request's ``ple_ngram_ctx`` past this forward's tokens, and write the
    window ending at each tracked chunk boundary (batch row, snapshot slot) to its slot."""
    ids = meta.input_ids.long()
    ctx_len = meta.ngram_context.shape[1]
    steps = torch.arange(ctx_len, device=ids.device)
    if meta.is_decode:
        nxt = torch.cat([meta.ngram_context[:, 1:], ids.view(-1, 1)], dim=1)
    else:
        cu = meta.cu_seqlens
        cand = cu[1:].unsqueeze(1) - ctx_len + steps
        # a request shorter than the window keeps the old context's tail: token j of the new
        # window sits at old column extend_len + j when it predates this forward
        old = meta.ngram_context.gather(
            1, ((cu[1:] - cu[:-1]).unsqueeze(1) + steps).clamp_(max=ctx_len - 1)
        )
        nxt = torch.where(cand >= cu[:-1].unsqueeze(1), ids[cand.clamp_min(0)], old)
    context_pool.index_copy_(0, meta.state_slots, nxt.to(context_pool.dtype))
    if track is not None:
        rows, dst = track
        win = ids[rows.unsqueeze(1) - ctx_len + steps]
        context_pool.index_copy_(0, dst, win.to(context_pool.dtype))


class NGramEmbedding(BaseOP):
    """Hashed n-gram lookup: splitmix64 mix of the last n token ids -> per-head prime vocab ->
    table rows. Weight keys: ``layer_multipliers`` [ngram_size], ``ngram_heads_vocab_sizes`` and
    ``ngram_heads_offsets`` [num_ngram_heads], int64. The table itself is attached by
    ``attach_table`` (``load_host_tables``), never a state-dict entry."""

    def __init__(self, args: Qwen4ExpArgs) -> None:
        self.ngram_size = args.ngram_size
        self.heads_per_ngram = args.heads_per_ngram
        self.eos_token_id = args.ngram_boundary_token_id
        self.layer_multipliers = torch.empty(args.ngram_size, dtype=torch.int64)
        self.ngram_heads_vocab_sizes = torch.empty(args.num_ngram_heads, dtype=torch.int64)
        self.ngram_heads_offsets = torch.empty(args.num_ngram_heads, dtype=torch.int64)
        self._table = None

    def attach_table(self, table) -> None:
        self._table = table

    @property
    def table(self):
        assert self._table is not None, "PLE table was never attached"
        return self._table

    def _window(self, meta: PLEMetadata):
        """The hash window as ``(packed [B, W], select)``; ``select`` picks this forward's rows."""
        ids = meta.input_ids.long()
        ctx_len = self.ngram_size - 1
        if meta.is_decode:
            # a window of exactly ngram_size columns holds every shift the hash can reach
            return torch.cat([meta.ngram_context, ids.view(-1, 1)], dim=1), lambda t: t[:, -1]
        num_reqs = len(meta.seq_lens)
        cu = meta.cu_seqlens
        flat_pos = torch.arange(ids.numel(), device=ids.device)
        req = (torch.searchsorted(cu, flat_pos, right=True) - 1).clamp_(max=num_reqs - 1)
        col = flat_pos - cu[req] + ctx_len
        packed = ids.new_full((num_reqs, ctx_len + max(meta.seq_lens)), self.eos_token_id)
        packed[:, :ctx_len] = meta.ngram_context
        packed[req, col] = ids
        return packed, lambda t: t[req, col]

    def _shift_ignore_eos(self, packed: torch.Tensor) -> List[torch.Tensor]:
        """``out[s][b, p]`` = the token ``s`` places left of ``p``, or eos past a boundary."""
        num_reqs, width = packed.shape
        pos = torch.arange(width, device=packed.device)
        eos_pos = torch.where(packed == self.eos_token_id, pos, -1)
        prev_eos = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat([eos_pos.new_full((num_reqs, 1), -1), prev_eos[:, :-1]], dim=1)
        in_segment = pos.unsqueeze(0) - prev_eos - 1
        shifted = [packed]
        for shift in range(1, self.ngram_size):
            src = pos - shift
            gathered = packed.gather(1, src.clamp_min(0).unsqueeze(0).expand(num_reqs, -1))
            valid = (src.unsqueeze(0) >= 0) & (in_segment >= shift)
            shifted.append(torch.where(valid, gathered, packed.new_full((), self.eos_token_id)))
        return shifted

    def row_ids(self, meta: PLEMetadata) -> torch.Tensor:
        """Global table row per (token, hash head): ``[T, num_ngram_heads]`` int64."""
        packed, select = self._window(meta)
        tokens = [select(s) for s in self._shift_ignore_eos(packed)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, tokens[position] * self.layer_multipliers[position])
            head_ids = torch.remainder(mixed.unsqueeze(-1), self.ngram_heads_vocab_sizes[start:end])
            blocks.append(head_ids + self.ngram_heads_offsets[start:end])
        return torch.cat(blocks, dim=-1)

    def forward(self, meta: PLEMetadata) -> torch.Tensor:
        return self.table.lookup(self.row_ids(meta))


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[width, 1, kernel]`` (key ``conv1d.weight``)."""

    def __init__(self, width: int, kernel: int) -> None:
        self.weight = torch.empty(width, 1, kernel)


class PLELayer(BaseOP):
    """``forward(R, batch) -> D [T, hc_count*hidden]``; the caller adds ``D`` to ``R``.

    The conv history slab is ``pool.slot_state("ple_conv", layer_id)``: ``[num_slots,
    hc_count*hidden, (kernel-1)*ngram_size]`` in the model dtype, the last conv-input columns
    per request, oldest first. Weight keys (prefix stripped): ``key_proj.weight``,
    ``value_proj.weight``, ``norm_key/norm_query/norm_conv.weight`` (zero-centered, RAW),
    ``conv1d.weight``, plus the three ``ple_embedding`` int64 hash buffers.
    """

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        args = config.qwen4_args
        self.args = args
        self.layer_id = layer_id
        self.ple_index = args.ple_layer_ids.index(layer_id)
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        self.dilation = args.ple_conv_dilation
        self.state_len = args.ple_conv_state_len
        width = args.stream_width
        self.ple_embedding = NGramEmbedding(args)
        self.key_proj = make_replicated(config, args.ple_embed_dim, width)
        self.value_proj = make_replicated(config, args.ple_embed_dim, args.hidden_size)
        self.norm_key = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.norm_query = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.norm_conv = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.conv1d = _DepthwiseConv1d(width, args.ple_conv_kernel_size)
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        # the track snapshot gathers the last state_len conv inputs before a xCHUNK boundary
        assert self.state_len <= CHUNK_SIZE, f"PLE conv history {self.state_len} exceeds {CHUNK_SIZE}"

    def forward(self, R: torch.Tensor, batch: Batch) -> torch.Tensor:
        pool = get_global_ctx().linear_state_pool
        context_pool = pool.slot_state(PLE_NGRAM_STATE)
        states = pool.slot_state(PLE_CONV_STATE, self.layer_id)
        prefill = batch.fla_metadata.prefill
        if prefill is not None:
            prefill.keep_start(context_pool, states)
        meta = build_ple_metadata(batch, context_pool, self.args.ngram_boundary_token_id)
        embeddings = self.ple_embedding.forward(meta).to(R.dtype)
        key = self.norm_key.forward(self.key_proj.forward(embeddings))
        value = self.value_proj.forward(embeddings)
        query = self.norm_query.forward(R)
        shape = (-1, self.hc_count, self.hidden_size)
        gate = (key.view(shape) * query.view(shape)).sum(-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated = (gate * value.unsqueeze(-2)).flatten(-2)
        x = self.norm_conv.forward(gated)
        track = None
        if prefill is not None and prefill.track_boundary_row is not None:
            # The boundary row indexes the prefill rows, which follow the decode rows here.
            track = (prefill.track_boundary_row + batch.decode_size, prefill.track_dst)
            self._write_track_snapshot(states, x, *track)
        out = gated + self._short_conv(x, meta, states)
        commit_ngram_context(meta, context_pool, track)
        return out

    def _write_track_snapshot(self, states: torch.Tensor, x: torch.Tensor, rows: torch.Tensor,
                              dst: torch.Tensor) -> None:
        """Conv history at the GDN track boundary into the same snapshot slot, so a prefix hit
        restores PLE and GDN state together. Track slots never alias the live slots this
        forward advances, so the two writes are order-independent."""
        src = rows.unsqueeze(1) + torch.arange(-self.state_len, 0, device=x.device)
        states.index_copy_(0, dst, x[src].transpose(-1, -2).contiguous().to(states.dtype))

    def _read_state(self, meta: PLEMetadata, states: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        state = states.index_select(0, meta.state_slots).to(dtype)
        if meta.fresh_slots is not None:
            state = torch.where(meta.fresh_slots.view(-1, 1, 1), torch.zeros_like(state), state)
        return state

    def _short_conv(self, x: torch.Tensor, meta: PLEMetadata, states: torch.Tensor) -> torch.Tensor:
        """silu of the dilated depthwise conv over [state | x]; rolls the per-request state."""
        if meta.is_decode:
            return self._decode_conv(x, meta, states)
        return self._prefill_conv(x, meta, states)

    def _decode_conv(self, x: torch.Tensor, meta: PLEMetadata, states: torch.Tensor) -> torch.Tensor:
        """Batched tap read: taps t-9, t-6, t-3 come off the state slab, tap t from this token."""
        state = self._read_state(meta, states, x.dtype)
        column = x.unsqueeze(-1)
        window = torch.cat([state[..., :: self.dilation], column], dim=-1).float()
        out = (window * self.conv1d.weight.squeeze(1).float()).sum(-1)
        states.index_copy_(0, meta.state_slots, torch.cat([state[..., 1:], column], dim=-1).to(states.dtype))
        return F.silu(out.to(x.dtype))

    def _prefill_conv(self, x: torch.Tensor, meta: PLEMetadata, states: torch.Tensor) -> torch.Tensor:
        """One conv over every request packed as ``[state_0 | chunk_0 | state_1 | chunk_1 | ...]``;
        the blocks abut exactly, so each output window stays inside its own request."""
        lens = list(meta.seq_lens)
        num_reqs, width = len(lens), x.shape[1]
        out_index, state_index, next_state_index = self._prefill_indices(lens, x.device)
        state = self._read_state(meta, states, x.dtype)
        history = x.new_empty(width, x.shape[0] + num_reqs * self.state_len)
        history.index_copy_(1, state_index, state.permute(1, 0, 2).reshape(width, -1))
        history.index_copy_(1, out_index + self.state_len, x.transpose(0, 1).contiguous())
        out = F.conv1d(history.unsqueeze(0), self.conv1d.weight, groups=width, dilation=self.dilation).squeeze(0)
        new_state = history.index_select(1, next_state_index).view(width, num_reqs, self.state_len)
        states.index_copy_(0, meta.state_slots, new_state.permute(1, 0, 2).to(states.dtype).contiguous())
        return F.silu(out.index_select(1, out_index).transpose(0, 1))

    def _prefill_indices(self, lens: List[int], device: torch.device):
        """Columns of the packed history: this forward's outputs, the state block, the next state block."""
        state_len = self.state_len
        counts = torch.tensor(lens, dtype=torch.int64)
        cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        pad = torch.arange(len(lens), dtype=torch.int64) * state_len
        base = cu[:-1] + pad
        out_index = torch.arange(int(cu[-1])) + torch.repeat_interleave(pad, counts)
        span = torch.arange(state_len, dtype=torch.int64)
        packed = torch.cat([
            out_index,
            (base.unsqueeze(1) + span).reshape(-1),
            ((base + counts).unsqueeze(1) + span).reshape(-1),
        ]).pin_memory().to(device, non_blocking=True)
        n_out, n_state = out_index.numel(), len(lens) * state_len
        return packed[:n_out], packed[n_out : n_out + n_state], packed[n_out + n_state :]


__all__ = [
    "NGramEmbedding",
    "PLELayer",
    "PLEMetadata",
    "PinnedUVATable",
    "ZeroTable",
    "build_ple_metadata",
    "derive_ngram_hash_constants",
]
