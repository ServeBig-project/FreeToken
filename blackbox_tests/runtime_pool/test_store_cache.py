"""store_cache on compact vs slot-major (token rows interleaved across layers) KV caches."""

import pytest
import torch

from freetoken.kernel import store_cache
from rp_common import BF16, Pool, gen, ints, randn

KV_HEADS, HEAD_DIM = 2, 256
ROW = KV_HEADS * HEAD_DIM


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inner", [(ROW,), (KV_HEADS, HEAD_DIM)], ids=["2d", "3d"])
@pytest.mark.parametrize("layers,layer", [(4, 1), (10, 9)])
def test_store_cache_strided_equals_compact(idx_dtype, inner, layers, layer):
    g = gen(41)
    tokens = 64
    kp = Pool(g, tokens, layers, layer, inner, BF16)
    vp = Pool(g, tokens, layers, layer, inner, BF16)
    loc = [tokens - 1, 0, 17, 5, 40, 41]
    k, v = randn(g, len(loc), ROW, dtype=BF16), randn(g, len(loc), ROW, dtype=BF16)
    idx = ints(loc, idx_dtype)

    store_cache(kp.compact, vp.compact, idx, k, v)
    store_cache(kp.strided, vp.strided, idx, k, v)

    for pool, src in ((kp, k), (vp, v)):
        pool.assert_same(loc)
        pool.assert_untouched(loc)
        assert torch.equal(pool.strided[loc].reshape(len(loc), ROW), src)


def test_store_cache_non_contiguous_source_rows():
    g = gen(42)
    tokens, layers, layer = 32, 4, 2
    kp = Pool(g, tokens, layers, layer, (ROW,), BF16)
    vp = Pool(g, tokens, layers, layer, (ROW,), BF16)
    kv = randn(g, 3, 2, ROW, dtype=BF16)  # fused [n, k|v, row] projection output
    loc = [9, 31, 0]
    store_cache(kp.strided, vp.strided, ints(loc), kv[:, 0], kv[:, 1])
    assert torch.equal(kp.strided[loc], kv[:, 0]) and torch.equal(vp.strided[loc], kv[:, 1])
    kp.assert_untouched(loc)
    vp.assert_untouched(loc)
