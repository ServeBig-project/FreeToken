import torch

from freetoken.moe import offload_kernels
from freetoken.moe.offload_cache import OffloadMoeCache


def _cache(*, collect_decode_freq: bool = False, num_experts: int = 16) -> OffloadMoeCache:
    cache = object.__new__(OffloadMoeCache)
    cache.collect_decode_freq = collect_decode_freq
    cache.decode_freq = [torch.zeros(num_experts, dtype=torch.long)]
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


def test_ensure_experts_chunks_large_query_and_copies_in_order(monkeypatch) -> None:
    events = []
    raw_ids = (torch.arange(2050) % 16).reshape(410, 5)
    expert_ids = raw_ids.clone()
    cache = _cache(collect_decode_freq=True)

    def fake_ensure(actual_cache, layer_id, chunk) -> None:
        events.append(("ensure", layer_id, chunk.clone()))
        chunk.add_(100)

    def fake_copy(actual_cache) -> None:
        events.append(("copy", actual_cache._pending_src_layer))

    monkeypatch.setattr(offload_kernels, "ensure_experts", fake_ensure)
    monkeypatch.setattr(OffloadMoeCache, "copy_missing", fake_copy)

    cache.ensure_experts(0, expert_ids)

    assert [event[0] for event in events] == [
        "ensure",
        "copy",
        "ensure",
        "copy",
        "ensure",
        "copy",
    ]
    chunks = [event[2] for event in events if event[0] == "ensure"]
    assert [chunk.numel() for chunk in chunks] == [1024, 1024, 2]
    assert torch.equal(torch.cat(chunks), raw_ids.view(-1))
    assert torch.equal(expert_ids, raw_ids + 100)
    assert torch.equal(cache.decode_freq[0], torch.bincount(raw_ids.view(-1), minlength=16))
    assert cache._pending_src_layer == 0
