"""QSA compressed-block sparse KV pool: paged GQA K/V + the compressed index-key slab.

Qwen3.8-Flash-Next scores whole ``index_ratio``-token groups, so the indexer slab holds ONE
compressed key row per group at row ``slot // index_ratio``. ``page_size % index_ratio == 0``
keeps a group inside one page, so the compressed rows are a 1/ratio shadow of the K/V pages
and follow page sharing and eviction for free. The raw keys of a request's open group live
on its linear-state slot (``models/qwen4_exp/config.py`` declares that slot state), not here.

K/V are stored in the compute dtype or, with the ``int8`` codec (``freetoken.quant.kv``), as
int8 plus one BF16 scale per token and KV head for K and for V. The compressed index keys
stay in the compute dtype under both codecs.

Rows from ``cmp_scratch_base`` on are per-request-slot sinks: a token whose group does not
close in its forward scatters there, so the compress kernel never needs a negative index.
The slab is amortized into the per-token KV price; the scratch rows are the fixed term.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

import torch
from freetoken.quant.kv import quantize_kv_int8

from .mha_pool import MHAKVCache

_INDEX_DTYPE_BYTES = 2  # spec_kv_bytes_per_token budgets the slab in the 2-byte compute dtype
_SCALE_DTYPE = torch.bfloat16


class QSAKVCache(MHAKVCache):
    kv_codecs: ClassVar[tuple[str, ...]] = ("bf16", "int8")

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        num_index_layers: int,
        index_ratio: int,
        num_req_slots: int,
        layer_ids: Sequence[int] | None = None,
        kv_dtype: str = "bf16",
    ) -> None:
        if index_ratio < 1 or page_size % index_ratio:
            raise ValueError(
                f"QSA needs page_size ({page_size}) divisible by index_ratio ({index_ratio})"
            )
        assert dtype.itemsize == _INDEX_DTYPE_BYTES, (
            f"QSA index slab budgets 2 bytes/token (spec_kv_bytes_per_token); got {dtype}"
        )
        if kv_dtype not in self.kv_codecs:
            raise ValueError(f"QSA K/V codec must be one of {self.kv_codecs}, got {kv_dtype!r}")
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._index_ratio = index_ratio
        self._num_req_slots = num_req_slots
        self._index_dtype = dtype
        self._page_size = page_size
        self._int8 = kv_dtype == "int8"
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=torch.int8 if self._int8 else dtype,
            device=device,
            layer_ids=layer_ids,
        )
        self._alloc_side_buffers(num_pages)

    def _alloc_side_buffers(self, num_pages: int) -> None:
        # Index slab zero-initialized: the score kernel reads whole rows of blocks unmasked and
        # relies on never-written tail rows dotting to a finite 0; it clamps the visible blocks
        # to kvlen // index_ratio, so stale rows are never selected.
        self._cmp_scratch_base = num_pages * self._page_size // self._index_ratio
        self._cmp_k_buffer = torch.zeros(
            self._num_index_layers,
            self._cmp_scratch_base + self._num_req_slots,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )
        # [K|V, layer, page, token, head] scales of the int8 codec; None under the bf16 codec.
        self._scale_buffer = (
            torch.empty(self._kv_buffer.shape[:-1], dtype=_SCALE_DTYPE, device=self._device)
            if self._int8 else None
        )

    def rebuild(self, num_pages: int) -> None:
        self._cmp_k_buffer = self._scale_buffer = None
        super().rebuild(num_pages)
        try:
            self._alloc_side_buffers(num_pages)
        except Exception:
            # a pool with a grown K/V slab and no index slab would mis-serve silently
            self._kv_buffer = self._k_buffer = self._v_buffer = None
            raise

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from freetoken.attention import AttnType
        from freetoken.utils import div_even

        from .base import spec_kv_bytes_per_token

        per_token = fixed = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            if spec.attn_type is not AttnType.QSA:
                per_token += spec_kv_bytes_per_token(spec, config)
                continue
            heads = div_even(spec.num_kv_heads, config.tp_info.size, allow_replicate=True)
            if getattr(config, "kv_dtype", "bf16") == "int8":
                kv = 2 * heads * (spec.head_dim + _SCALE_DTYPE.itemsize)
            else:
                kv = 2 * heads * spec.head_dim * config.dtype.itemsize
            row = spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES
            per_token += kv * spec.num_layers + row // spec.index_ratio
            fixed += (config.max_running_req + 1) * row
        return per_token * config.page_size, fixed, config.page_size, 0

    def unit_bytes(self) -> tuple[int, int]:
        kv, swa = super().unit_bytes()
        tokens = int(self._kv_buffer.shape[2]) * int(self._kv_buffer.shape[3])
        slab = (
            self._num_index_layers
            * self._cmp_scratch_base
            * self._index_head_dim
            * self._index_dtype.itemsize
        )
        if self._scale_buffer is not None:
            kv += int(self._scale_buffer.numel() * self._scale_buffer.element_size()) // tokens
        return kv + slab // tokens, swa

    def paged_views(self) -> list[torch.Tensor]:
        """K/V pages, the INT8 scales, and each page's compressed index rows (scratch rows
        excluded): a page copied without its index rows would score as all-zero blocks."""
        views = super().paged_views()
        if self._scale_buffer is not None:
            views += [self._scale_buffer[:, layer].movedim(1, 0)
                      for layer in range(self._scale_buffer.shape[1])]
        rows = self._page_size // self._index_ratio
        views += [self._cmp_k_buffer[slot, : self._cmp_scratch_base].view(-1, rows, self._index_head_dim)
                  for slot in range(self._num_index_layers)]
        return views

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        if not self._int8:
            return super().store_kv(k, v, out_loc, layer_id)
        dense = self._dense(layer_id)
        heads = self._kv_buffer.shape[4]
        rows = out_loc.to(torch.int64)
        for which, x in ((0, k), (1, v)):
            q, scale = quantize_kv_int8(x.view(-1, heads, self._kv_buffer.shape[5]))
            self._kv_buffer[which, dense].view(-1, heads, self._kv_buffer.shape[5]).index_copy_(0, rows, q)
            self._scale_buffer[which, dense].view(-1, heads).index_copy_(0, rows, scale)

    def k_scale(self, layer_id: int) -> torch.Tensor | None:
        """INT8 codec: ``[pages, page_size, heads]`` K scales of ``layer_id``; None under bf16."""
        return None if self._scale_buffer is None else self._scale_buffer[0, self._dense(layer_id)]

    def v_scale(self, layer_id: int) -> torch.Tensor | None:
        return None if self._scale_buffer is None else self._scale_buffer[1, self._dense(layer_id)]

    def cmp_k_cache(self, slot: int) -> torch.Tensor:
        """Compressed index keys of one sparse layer (sparse-layer order): ``[rows, dim]``."""
        return self._cmp_k_buffer[slot]

    @property
    def cmp_scratch_base(self) -> int:
        """First scratch row; row ``cmp_scratch_base + table_idx`` sinks a non-closing token."""
        return self._cmp_scratch_base

    @property
    def index_dtype(self) -> torch.dtype:
        return self._index_dtype

    @property
    def index_ratio(self) -> int:
        return self._index_ratio

    @property
    def index_head_dim(self) -> int:
        return self._index_head_dim


__all__ = ["QSAKVCache"]
