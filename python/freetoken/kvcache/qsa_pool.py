"""QSA compressed-block sparse KV pool: paged GQA K/V + the compressed index-key slab.

Qwen3.8-Flash-Next scores whole ``index_ratio``-token groups, so the indexer slab holds ONE
compressed key row per group at row ``slot // index_ratio``. ``page_size % index_ratio == 0``
keeps a group inside one page, so the compressed rows are a 1/ratio shadow of the K/V pages
and follow page sharing and eviction for free. The raw keys of a request's open group live
on its linear-state slot (``models/qwen4_exp/config.py`` declares that slot state), not here.

Rows from ``cmp_scratch_base`` on are per-request-slot sinks: a token whose group does not
close in its forward scatters there, so the compress kernel never needs a negative index.
The slab is amortized into the per-token KV price; the scratch rows are the fixed term.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .mha_pool import MHAKVCache

_INDEX_DTYPE_BYTES = 2  # spec_kv_bytes_per_token budgets the slab in the 2-byte compute dtype


class QSAKVCache(MHAKVCache):
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
    ) -> None:
        if index_ratio < 1 or page_size % index_ratio:
            raise ValueError(
                f"QSA needs page_size ({page_size}) divisible by index_ratio ({index_ratio})"
            )
        assert dtype.itemsize == _INDEX_DTYPE_BYTES, (
            f"QSA index slab budgets 2 bytes/token (spec_kv_bytes_per_token); got {dtype}"
        )
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._index_ratio = index_ratio
        self._num_req_slots = num_req_slots
        self._index_dtype = dtype
        self._page_size = page_size
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            layer_ids=layer_ids,
        )
        self._alloc_index_slab(num_pages)

    def _alloc_index_slab(self, num_pages: int) -> None:
        # Zero-initialized: the score kernel reads whole rows of blocks unmasked and relies
        # on never-written tail rows dotting to a finite 0; it clamps the visible blocks to
        # kvlen // index_ratio, so stale rows are never selected.
        self._cmp_scratch_base = num_pages * self._page_size // self._index_ratio
        self._cmp_k_buffer = torch.zeros(
            self._num_index_layers,
            self._cmp_scratch_base + self._num_req_slots,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )

    def rebuild(self, num_pages: int) -> None:
        self._cmp_k_buffer = None
        super().rebuild(num_pages)
        try:
            self._alloc_index_slab(num_pages)
        except Exception:
            # a pool with a grown K/V slab and no index slab would mis-serve silently
            self._kv_buffer = self._k_buffer = self._v_buffer = None
            raise

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from freetoken.attention import AttnType

        from .base import spec_kv_bytes_per_token

        per_token = fixed = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            per_token += spec_kv_bytes_per_token(spec, config)
            if spec.attn_type is AttnType.QSA:
                row = spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES
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
        return kv + slab // tokens, swa

    def cmp_k_cache(self, slot: int) -> torch.Tensor:
        """Compressed index keys of one sparse layer (sparse-layer order): ``[rows, dim]``."""
        return self._cmp_k_buffer[slot]

    @property
    def cmp_scratch_base(self) -> int:
        """First scratch row; row ``cmp_scratch_base + table_idx`` sinks a non-closing token."""
        return self._cmp_scratch_base

    @property
    def index_ratio(self) -> int:
        return self._index_ratio

    @property
    def index_head_dim(self) -> int:
        return self._index_head_dim


__all__ = ["QSAKVCache"]
