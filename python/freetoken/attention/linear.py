from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class FLAPathMetadata:
    """Metadata for one homogeneous GDN execution path.

    Fields:
      cu_seqlens          query indptr; decode = arange(bs+1) (1 token/request), prefill =
                          cumsum of extend_len.
      cache_indices       per-request recurrent/conv state slot (= Req.table_idx). int32.
      has_initial_state   prefill only: whether each request continues a cached prefix
                          (cached_len > 0). None for decode (state always present).
      fresh_state_indices prefill only: the state-pool slots whose sequence is fresh
                          (cached_len == 0) and must be zeroed before the chunk kernel
                          reads them in place. None if there are none / for decode.
    """

    cu_seqlens: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor | None = None
    fresh_state_indices: torch.Tensor | None = None

    # Public snapshots are written during prefill, before overlapped decode advances live
    # state. A short final chunk retains its initial state; longer chunks retain a boundary.
    track_dst: torch.Tensor | None = None        # [nt] int64 dst pool slot per tracked req
    track_h_row: torch.Tensor | None = None      # [nt] int64 row into h (boh_i + aligned//CHUNK)
    track_conv_src: torch.Tensor | None = None   # [nt, kernel-1] int64 conv-input token positions
    track_start_dst: torch.Tensor | None = None  # [ns] int64 snapshot slots
    track_start_src: torch.Tensor | None = None  # [ns] int64 live slots before prefill
    # [nt] int64 forward-local row of each track boundary: states with their own left
    # context (qwen4_exp PLE) derive their snapshot windows from it.
    track_boundary_row: torch.Tensor | None = None

    # --- ReplaySSM: [n] int32 record row per request (-1 = padding); None runs the
    # recurrent-state kernels. Positions come from the batch and checkpoint positions from the
    # GPU. Speculative rows (draft and verify) convolve through the replay window and never
    # fold or touch the target conv state.
    rows: torch.Tensor | None = None
    speculative: bool = False

    def keep_start(self, *states: torch.Tensor) -> None:
        """Copy the live rows of ``states`` (``[slots, ...]``) to this extend's start capture
        slots. Each owner calls it when its own layer runs: in a layered wave a later tile of
        the same request is embedded before the earlier tile reaches this layer."""
        if self.track_start_dst is not None:
            for t in states:
                t.index_copy_(0, self.track_start_dst, t.index_select(0, self.track_start_src))


@dataclass
class FLAVerifyStep:
    """One verified position of a speculative round, one row per request (fixed shape so a
    CUDA graph can replay it): the token row to read, the row to write, the slot holding the
    state before this position, and the decode view whose cache_indices receive the state
    after it. A request with no token at this position reads its last row and writes the
    dummy row and the pool's padding slot."""

    rows: torch.Tensor      # [bs] int64 rows into the ragged verify batch
    write: torch.Tensor     # [bs] int64 output rows (dummy row = token count)
    prev: torch.Tensor      # [bs] int64 slot per row to start from
    path: FLAPathMetadata   # cache_indices [bs] int32 destination slots, cu_seqlens arange


@dataclass
class FLAMetadata:
    """Per-forward GDN metadata, built once and shared by every GDN layer.

    Decode and prefill use different kernels. A mixed forward therefore carries one
    metadata view for each sub-batch instead of forcing its one-token decode sequences
    through the chunked prefill path. A ReplaySSM verify batch is a single ``decode`` view
    whose sequences are the verify windows.
    """

    decode: FLAPathMetadata | None = None
    prefill: FLAPathMetadata | None = None
    verify: list[FLAVerifyStep] | None = None


def build_fla_metadata(batch: "Batch", device: torch.device) -> FLAMetadata:
    """Build decode/prefill GDN views for one forward.

    Host-built fields use pinned staging and non-blocking H2D, matching the input and
    attention-metadata path. Decode-only CUDA graphs install their persistent decode view
    directly in ``GraphCaptureBuffer.set_batch`` instead.
    """
    pin = {"device": "cpu", "pin_memory": device.type == "cuda"}

    # GDN state slot per request: the hybrid-radix live slot (decoupled from table_idx) when
    # allocated, else table_idx (naive / force-naive GDN models keep the old keying).
    from freetoken.core import get_global_ctx

    def gdn_slot(r):
        return r.linear_slot_idx if r.linear_slot_idx is not None else r.table_idx

    replay = get_global_ctx().linear_state_pool.replay
    to = lambda xs: torch.tensor(xs, dtype=torch.int32, **pin).to(device, non_blocking=True)

    def build_decode(reqs, cache_indices=None, speculative=False):
        cu_seqlens = torch.arange(len(reqs) + 1, dtype=torch.int32, device=device)
        if cache_indices is None:
            cache_indices = to([gdn_slot(r) for r in reqs])
        rows = to(replay.record_rows(reqs)) if replay is not None else None
        return FLAPathMetadata(cu_seqlens=cu_seqlens, cache_indices=cache_indices,
                               rows=rows, speculative=speculative)

    def build_prefill(reqs):
        lens = [r.extend_len for r in reqs]
        cu_host = torch.tensor([0, *lens], dtype=torch.int64, **pin).cumsum_(0)
        idx_host = torch.tensor([gdn_slot(r) for r in reqs], dtype=torch.int32, **pin)
        has_init_host = torch.tensor(
            [r.cached_len > 0 for r in reqs], dtype=torch.bool, **pin
        )
        fresh = [gdn_slot(r) for r in reqs if r.cached_len == 0]
        fresh_host = torch.tensor(fresh, dtype=torch.int64, **pin) if fresh else None
        track = _build_track_metadata(reqs, cu_host, device, pin)
        return FLAPathMetadata(
            cu_seqlens=cu_host.to(device, non_blocking=True),
            cache_indices=idx_host.to(device, non_blocking=True),
            has_initial_state=has_init_host.to(device, non_blocking=True),
            fresh_state_indices=(
                fresh_host.to(device, non_blocking=True) if fresh_host is not None else None
            ),
            **track,
        )

    if batch.is_decode_only:
        # the scheduler stages linear_table_idx from gdn_slot (decode), reused as-is here
        assert batch.linear_table_idx is not None
        return FLAMetadata(decode=build_decode(
            batch.padded_reqs, batch.linear_table_idx, batch.draft_experts is not None))
    if replay is not None and batch.is_speculative_verify:
        # Every verify window reads its request's target records and rewrites the draft tail.
        reqs = batch.reqs
        cu_seqlens = to([0, *(r.extend_len for r in reqs)]).cumsum_(0)
        return FLAMetadata(decode=FLAPathMetadata(
            cu_seqlens=cu_seqlens, cache_indices=to([gdn_slot(r) for r in reqs]),
            rows=to(replay.record_rows(reqs)), speculative=True))

    decode = build_decode(batch.decode_reqs) if batch.has_decode else None
    prefill = build_prefill(batch.prefill_reqs) if batch.has_prefill else None
    verify = None
    if batch.speculative_states is not None:
        reqs = batch.prefill_reqs
        tokens = sum(r.extend_len for r in reqs)
        rows, write, prev, dst = verify_layout(
            reqs, batch.speculative_states, get_global_ctx().linear_state_pool.padding_slot,
            tokens, max(r.extend_len for r in reqs), pin)
        arange = torch.arange(len(reqs) + 1, dtype=torch.int32, device=device)
        verify = [FLAVerifyStep(
            rows=rows[j].to(device, non_blocking=True), write=write[j].to(device, non_blocking=True),
            prev=prev[j].to(device, non_blocking=True),
            path=FLAPathMetadata(cu_seqlens=arange, cache_indices=dst[j].to(device, non_blocking=True)),
        ) for j in range(rows.shape[0])]
    return FLAMetadata(
        decode=decode,
        prefill=prefill,
        verify=verify,
    )


def verify_layout(reqs, states, dummy_slot, dummy_row, positions, pin):
    """Host [positions, bs] index tensors for the fixed-shape verify: position j of request i
    verifies its (j+1)-th token. Before position 0 the state is the live slot, before j>0 the
    scratch slot written at j-1; positions past a request's tokens are inert."""
    offsets, total = [], 0
    for r in reqs:
        offsets.append(total)
        total += r.extend_len
    rows, write, prev, dst = ([[0] * len(reqs) for _ in range(positions)] for _ in range(4))
    for i, r in enumerate(reqs):
        live, scratch = states[i]
        for j in range(positions):
            active = j < r.extend_len
            rows[j][i] = offsets[i] + (j if active else r.extend_len - 1)
            write[j][i] = offsets[i] + j if active else dummy_row
            prev[j][i] = (live if j == 0 else scratch[j - 1]) if active else dummy_slot
            dst[j][i] = scratch[j] if active else dummy_slot
    return (torch.tensor(rows, dtype=torch.int64, **pin), torch.tensor(write, dtype=torch.int64, **pin),
            torch.tensor(prev, dtype=torch.int64, **pin), torch.tensor(dst, dtype=torch.int32, **pin))


def _build_track_metadata(reqs, cu_host, device, pin):
    """Freeze each slotted capture whose target this extend passes.

    Inside one extend the chunked recurrence has a state every CHUNK_SIZE tokens from its
    start (``h``) and the initial state at the start, so a capture takes the deepest of those
    at or before its target and records where it landed (``pos``).
    """
    names = ("track_dst", "track_h_row", "track_conv_src", "track_start_dst",
             "track_start_src", "track_boundary_row")
    if not any(c.slot is not None for r in reqs for c in r.state_captures):
        return dict.fromkeys(names)
    from freetoken.core import get_global_ctx
    from freetoken.kernel.fla.chunk import CHUNK_SIZE
    from freetoken.kernel.fla.index import prepare_chunk_offsets

    km1 = get_global_ctx().linear_state_pool.conv_states.shape[-1]  # conv_kernel_dim - 1
    boh = prepare_chunk_offsets(cu_host, CHUNK_SIZE).tolist()
    dst, h_row, conv_src, boundary = [], [], [], []
    start_dst, start_src = [], []
    for i, r in enumerate(reqs):
        for capture in r.state_captures:
            if capture.slot is None or not r.cached_len <= capture.target < r.device_len:
                continue
            c = (capture.target - r.cached_len) // CHUNK_SIZE
            capture.pos = r.cached_len + c * CHUNK_SIZE
            if capture.pos <= r.cache_handle.cached_len:
                capture.pos = None  # the reused prefix already ends here
            elif c == 0:
                start_dst.append(capture.slot)
                start_src.append(r.linear_slot_idx if r.linear_slot_idx is not None else r.table_idx)
            else:
                off = int(cu_host[i])
                dst.append(capture.slot)
                h_row.append(boh[i] + c)
                conv_src.append([off + c * CHUNK_SIZE - km1 + j for j in range(km1)])
                boundary.append(off + c * CHUNK_SIZE)
    to = lambda xs: torch.tensor(xs, dtype=torch.int64, **pin).to(device, non_blocking=True) if xs else None
    return dict(zip(names, (to(xs) for xs in (dst, h_row, conv_src, start_dst, start_src, boundary))))


__all__ = ["FLAMetadata", "FLAPathMetadata", "build_fla_metadata"]


class FLASpeculativeGraphs:
    """Stable state-index views for draft and position-by-position verify replay."""

    def __init__(self, pool, max_batch, query_width, device):
        self.pool, self.query_width, self.device = pool, query_width, device
        shape = (query_width, max_batch)
        self.rows, self.write, self.prev = (
            torch.zeros(shape, dtype=torch.int64, device=device) for _ in range(3))
        self.dst = torch.zeros(shape, dtype=torch.int32, device=device)
        self.cu = torch.arange(max_batch + 1, dtype=torch.int32, device=device)
        self.draft_slots = torch.zeros(max_batch, dtype=torch.int32, device=device)
        # The verify rows' request offsets and live slots (what the model's other declared
        # states read), and each row's record of them: the scratch slot, padding into the sink.
        self.req_cu = torch.zeros(max_batch + 1, dtype=torch.int64, device=device)
        self.live = torch.zeros(max_batch, dtype=torch.int32, device=device)
        self.records = torch.zeros(max_batch * query_width, dtype=torch.int64, device=device)

    def prepare_capture(self, batch, lengths, tokens):
        bs = batch.size
        if not batch.is_speculative_verify:
            self.draft_slots[:bs].fill_(self.pool.padding_slot)
            batch.fla_metadata = FLAMetadata(decode=FLAPathMetadata(
                cu_seqlens=self.cu[:bs + 1], cache_indices=self.draft_slots[:bs]))
            return
        offsets, offset = [], 0
        for length in lengths:
            offsets.append(offset)
            offset += length
        self.rows[:, :bs] = torch.tensor(offsets, device=self.device)
        self.write[:, :bs] = tokens
        self.prev[:, :bs] = self.pool.padding_slot
        self.dst[:, :bs] = self.pool.padding_slot
        self.req_cu[: bs + 1] = torch.tensor([*offsets, tokens], device=self.device)
        self.live[:bs] = self.pool.padding_slot
        self.records.fill_(self.pool.padding_slot)
        batch.fla_metadata = FLAMetadata(
            prefill=FLAPathMetadata(cu_seqlens=self.req_cu[: bs + 1], cache_indices=self.live[:bs]),
            verify=[FLAVerifyStep(
                rows=self.rows[j, :bs], write=self.write[j, :bs], prev=self.prev[j, :bs],
                path=FLAPathMetadata(cu_seqlens=self.cu[:bs + 1], cache_indices=self.dst[j, :bs]))
                for j in range(self.query_width)])
        batch.verify_records = (self.records[:tokens],)

    def prepare_replay(self, batch, physical_tokens):
        bs = batch.size
        if not batch.is_speculative_verify:
            self.draft_slots[:bs].copy_(batch.linear_table_idx, non_blocking=True)
            return
        host = verify_layout(batch.prefill_reqs, batch.speculative_states, self.pool.padding_slot,
                             physical_tokens, self.query_width, {"device": "cpu", "pin_memory": True})
        for buffer, tensor in zip((self.rows, self.write, self.prev, self.dst), host, strict=True):
            buffer[:, :bs].copy_(tensor, non_blocking=True)
        prefill = batch.fla_metadata.prefill
        self.req_cu[: bs + 1].copy_(prefill.cu_seqlens)
        self.live[:bs].copy_(prefill.cache_indices)
        if batch.verify_records is not None:
            real = batch.verify_records[0].numel()
            self.records[:real].copy_(batch.verify_records[0])
            self.records[real:].fill_(self.pool.padding_slot)


class ReplaySpeculativeGraphs:
    """Stable ReplaySSM views for draft steps and ragged verify windows, on the replay
    component's fixed graph buffers."""

    def __init__(self, pool):
        replay = pool.replay
        self.pool = pool
        self.cu, self.slots, self.rows = replay.graph_cu, replay.graph_slots, replay.graph_rows
        # Each verify row's (record row, input) of the other declared states; padding rows
        # write record row 0's last input, a sink no round reads.
        inputs = next(iter(replay.verify_states.values())).shape[2] if replay.verify_states else 1
        self.sink = inputs - 1
        self.record_rows = torch.zeros(self.rows.numel() * self.sink, dtype=torch.int64,
                                       device=self.rows.device)
        self.record_inputs = torch.full_like(self.record_rows, self.sink)

    def prepare_capture(self, batch, lengths, tokens):
        bs = batch.size
        self.slots[:bs].fill_(self.pool.padding_slot)
        self.rows[:bs].fill_(-1)  # capture never touches records
        self.cu[: bs + 1] = torch.tensor([0, *lengths], device=self.cu.device).cumsum(0)
        batch.fla_metadata = FLAMetadata(decode=FLAPathMetadata(
            cu_seqlens=self.cu[: bs + 1], cache_indices=self.slots[:bs],
            rows=self.rows[:bs], speculative=True))
        if self.pool.replay.verify_states:
            batch.verify_records = (self.record_rows[:tokens], self.record_inputs[:tokens])

    def prepare_replay(self, batch, physical_tokens):
        bs, source = batch.size, batch.fla_metadata.decode
        self.cu[: bs + 1].copy_(source.cu_seqlens)
        self.slots[:bs].copy_(source.cache_indices)
        self.rows[:bs].copy_(source.rows)
        if batch.verify_records is not None:
            rows, inputs = batch.verify_records
            self.record_rows[: rows.numel()].copy_(rows)
            self.record_inputs[: rows.numel()].copy_(inputs)
            self.record_rows[rows.numel():].zero_()
            self.record_inputs[rows.numel():].fill_(self.sink)
