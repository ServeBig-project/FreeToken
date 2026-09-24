import torch

from freetoken.moe import offload_kernels
from freetoken.moe.offload_cache import OffloadMoeCache


def _cache() -> OffloadMoeCache:
    cache = object.__new__(OffloadMoeCache)
    cache.collect_decode_freq = False
    cache.decode_freq = [torch.zeros(16, dtype=torch.long)]
    return cache


def test_ensure_experts_keeps_small_query_behavior(monkeypatch) -> None:
    calls = []
    copies = []
    expert_ids = torch.arange(1024).reshape(256, 4)
    cache = _cache()

    monkeypatch.setattr(
        offload_kernels,
        "ensure_experts",
        lambda actual_cache, layer_id, chunk: calls.append(
            (actual_cache, layer_id, chunk)
        ),
    )
    monkeypatch.setattr(
        OffloadMoeCache, "copy_missing", lambda actual_cache: copies.append(actual_cache)
    )

    cache.ensure_experts(0, expert_ids)

    assert calls == [(cache, 0, expert_ids)]
    assert copies == []
    assert cache._pending_src_layer == 0
