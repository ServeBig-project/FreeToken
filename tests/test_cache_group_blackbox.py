"""Black-box acceptance; live HTTP tests require FT_CACHE_GROUP_TEST_URL."""

import json
import os
import urllib.request
import uuid

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache
from freetoken.kvcache.radix_cache import RadixPrefixCache
from freetoken.kvcache.swa_radix_cache import SWARadixCache
from freetoken.message import BaseBackendMsg, BaseTokenizerMsg, TokenizeMsg, UserMsg
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq


DEVICE = torch.device("cpu")


def tensor_range(start, count):
    return torch.arange(start, start + count, dtype=torch.int32)


def group_arg(group):
    return {} if group is None else {"cache_group": group}


def assert_indices(actual, expected):
    assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("group", [None, "", "alice"])
def test_radix_group_reuse_and_isolation(page_size, group):
    cache = RadixPrefixCache(DEVICE, page_size=page_size)
    ids = tensor_range(1, 5 * page_size + 3)
    original_ids = ids.clone()
    first = tensor_range(100, len(ids))
    second = tensor_range(200, len(ids))
    length = len(ids) // page_size * page_size
    kwargs = group_arg(group)

    assert cache.insert_prefix(ids, first, **kwargs).cached_len == 0
    cold = cache.match_prefix(ids, cache_group="bob").cuda_handle
    assert cold.cached_len == 0
    assert cold.get_matched_indices().numel() == 0
    assert cache.insert_prefix(ids, second, cache_group="bob").cached_len == 0
    assert cache.insert_prefix(ids, tensor_range(500, len(ids)), **kwargs).cached_len == length

    for actual_group, expected in [(group, first), ("bob", second)]:
        match = cache.match_prefix(ids, **group_arg(actual_group)).cuda_handle
        assert match.cached_len == length
        assert_indices(match.get_matched_indices(), expected[:length])
    if group in (None, ""):
        assert cache.match_prefix(ids).cuda_handle.cached_len == length
        assert cache.match_prefix(ids, cache_group="").cuda_handle.cached_len == length
    assert_indices(ids, original_ids)
    assert_indices(first, tensor_range(100, len(ids)))


@pytest.mark.parametrize("page_size", [1, 4])
def test_radix_partial_prefix_split_stays_in_group(page_size):
    cache = RadixPrefixCache(DEVICE, page_size=page_size)
    ids = tensor_range(1, 4 * page_size)
    branch = torch.cat((ids[: 2 * page_size], tensor_range(50, 2 * page_size)))
    first = tensor_range(100, len(ids))
    second = tensor_range(200, len(ids))
    other_group = tensor_range(300, len(ids))
    cache.insert_prefix(ids, first, cache_group="alice")

    assert cache.match_prefix(branch, cache_group="alice").cuda_handle.cached_len == 2 * page_size
    assert cache.match_prefix(branch, cache_group="bob").cuda_handle.cached_len == 0
    assert cache.insert_prefix(branch, second, cache_group="alice").cached_len == 2 * page_size
    assert cache.insert_prefix(branch, other_group, cache_group="bob").cached_len == 0
    expected = torch.cat((first[: 2 * page_size], second[2 * page_size :]))
    assert_indices(cache.match_prefix(branch, cache_group="alice").cuda_handle.get_matched_indices(), expected)
    assert_indices(cache.match_prefix(ids, cache_group="alice").cuda_handle.get_matched_indices(), first)
    assert_indices(cache.match_prefix(branch, cache_group="bob").cuda_handle.get_matched_indices(), other_group)


def test_radix_global_budget_and_locked_handle():
    cache = RadixPrefixCache(DEVICE, page_size=1)
    ids = tensor_range(1, 8)
    first, second = tensor_range(100, 8), tensor_range(200, 8)
    cache.insert_prefix(ids, first, cache_group="alice")
    cache.insert_prefix(ids, second, cache_group="bob")
    assert cache.size_info.evictable_size == 16
    assert cache.size_info.total_size == 16
    handle = cache.match_prefix(ids, cache_group="alice").cuda_handle
    cache.lock_handle(handle)
    assert cache.size_info.protected_size == 8
    assert cache.size_info.evictable_size == 8

    assert set(cache.evict(8).tolist()) == set(second.tolist())
    assert cache.match_prefix(ids, cache_group="alice").cuda_handle.cached_len == 8
    assert cache.match_prefix(ids, cache_group="bob").cuda_handle.cached_len == 0
    cache.lock_handle(handle, unlock=True)
    assert set(cache.evict(8).tolist()) == set(first.tolist())
    assert cache.size_info.total_size == 0


def test_radix_lru_is_shared_across_groups():
    cache = RadixPrefixCache(DEVICE, page_size=1)
    ids = tensor_range(1, 8)
    first, second = tensor_range(100, 8), tensor_range(200, 8)
    cache.insert_prefix(ids, first, cache_group="z-old")
    cache.insert_prefix(ids, second, cache_group="a-new")
    assert set(cache.evict(8).tolist()) == set(first.tolist())
    assert cache.match_prefix(ids, cache_group="z-old").cuda_handle.cached_len == 0
    assert_indices(cache.match_prefix(ids, cache_group="a-new").cuda_handle.get_matched_indices(), second)


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("group", [None, "", "alice"])
def test_hybrid_group_reuse_and_snapshot_isolation(page_size, group):
    cache = HybridRadixCache(DEVICE, page_size=page_size)
    ids = tensor_range(1, 128)
    original_ids = ids.clone()
    first, second = tensor_range(100, 128), tensor_range(300, 128)
    kwargs = group_arg(group)

    assert cache.insert(ids, first, 11, **kwargs) == (0, False)
    cold = cache.match_prefix(ids, cache_group="bob")
    assert cold.cached_len == 0
    assert cold.kv_indices.numel() == 0
    assert cache.insert(ids, second, 22, cache_group="bob") == (0, False)
    assert cache.insert(ids, tensor_range(500, 128), 33, **kwargs) == (128, True)
    for actual_group, expected, snapshot in [(group, first, 11), ("bob", second, 22)]:
        match = cache.match_prefix(ids, **group_arg(actual_group))
        assert match.cached_len == 128
        assert match.mamba_value == snapshot
        assert_indices(match.kv_indices, expected)
    if group in (None, ""):
        assert cache.match_prefix(ids).mamba_value == 11
        assert cache.match_prefix(ids, cache_group="").mamba_value == 11
    assert_indices(ids, original_ids)


@pytest.mark.parametrize("page_size", [1, 64])
def test_hybrid_matches_only_own_valid_snapshot_boundary(page_size):
    cache = HybridRadixCache(DEVICE, page_size=page_size)
    ids = tensor_range(1, 128)
    first, second = tensor_range(100, 128), tensor_range(300, 128)
    assert cache.insert(ids[:64], first[:64], 11, cache_group="alice") == (0, False)
    assert cache.insert(ids, first, 12, cache_group="alice") == (64, False)
    assert cache.insert(ids, second, 22, cache_group="bob") == (0, False)

    match = cache.match_prefix(ids[:96], cache_group="alice")
    assert match.cached_len == 64
    assert match.mamba_value == 11
    assert_indices(match.kv_indices, first[:64])
    assert cache.match_prefix(ids[:96], cache_group="bob").cached_len == 0
    assert cache.match_prefix(ids, cache_group="alice").mamba_value == 12
    assert cache.match_prefix(ids, cache_group="bob").mamba_value == 22


def test_hybrid_global_budget_and_locked_snapshot():
    cache = HybridRadixCache(DEVICE, page_size=1)
    ids = tensor_range(1, 64)
    first, second = tensor_range(100, 64), tensor_range(200, 64)
    cache.insert(ids, first, 11, cache_group="alice")
    cache.insert(ids, second, 22, cache_group="bob")
    assert cache.full_evictable_size == 128
    assert cache.mamba_evictable_size == 2
    locked = cache.match_prefix(ids, cache_group="alice")
    cache.inc_lock(locked.node)
    assert cache.full_evictable_size == 64
    assert cache.mamba_evictable_size == 1

    evicted = cache.evict_full(128)
    assert set(evicted.kv_indices.tolist()) == set(second.tolist())
    assert {int(slot) for slot in evicted.mamba_slots} == {22}
    assert cache.match_prefix(ids, cache_group="alice").mamba_value == 11
    assert cache.match_prefix(ids, cache_group="bob").cached_len == 0
    cache.dec_lock(locked.node)
    assert set(cache.evict_full(64).kv_indices.tolist()) == set(first.tolist())
    assert cache.full_evictable_size == 0
    assert cache.mamba_evictable_size == 0


def test_hybrid_snapshot_eviction_is_global_and_does_not_cross_groups():
    cache = HybridRadixCache(DEVICE, page_size=1)
    ids = tensor_range(1, 64)
    first, second = tensor_range(100, 64), tensor_range(200, 64)
    cache.insert(ids, first, 11, cache_group="z-old")
    cache.insert(ids, second, 22, cache_group="a-new")
    evicted = cache.evict_mamba(1)
    assert {int(slot) for slot in evicted.mamba_slots} == {11}
    assert cache.match_prefix(ids, cache_group="z-old").cached_len == 0
    remaining = cache.match_prefix(ids, cache_group="a-new")
    assert remaining.mamba_value == 22
    assert_indices(remaining.kv_indices, second)


@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("group", [None, "", "alice"])
def test_swa_group_reuse_and_isolation(page_size, group):
    cache = SWARadixCache(DEVICE, page_size=page_size, sliding_window_size=16)
    ids = tensor_range(1, 64)
    original_ids = ids.clone()
    first, second = tensor_range(100, 64), tensor_range(200, 64)
    kwargs = group_arg(group)
    assert cache.insert(ids, first, **kwargs)[0] == 0
    cold = cache.match_prefix(ids, cache_group="bob")
    assert cold.cached_len == 0
    assert cold.kv_indices.numel() == 0
    assert cache.insert(ids, second, cache_group="bob")[0] == 0
    assert cache.insert(ids, tensor_range(500, 64), **kwargs)[0] == 64
    for actual_group, expected in [(group, first), ("bob", second)]:
        match = cache.match_prefix(ids, **group_arg(actual_group))
        assert match.cached_len == 64
        assert_indices(match.kv_indices, expected)
    if group in (None, ""):
        assert cache.match_prefix(ids).cached_len == 64
        assert cache.match_prefix(ids, cache_group="").cached_len == 64
    assert_indices(ids, original_ids)


@pytest.mark.parametrize("page_size", [1, 4])
def test_swa_trimming_is_limited_to_its_group(page_size):
    cache = SWARadixCache(DEVICE, page_size=page_size, sliding_window_size=16)
    ids = tensor_range(1, 64)
    branch = torch.cat((ids[:32], tensor_range(1000, 32)))
    first, second = tensor_range(100, 64), tensor_range(300, 64)
    for group, indices in [("alice", first), ("bob", second)]:
        cache.insert(ids, indices, cache_group=group)
        cache.insert(branch, indices + 1000, cache_group=group)
        assert cache.match_prefix(ids[:32], cache_group=group).cached_len == 32

    cache.trim_head_swa(ids, keep_from=32, cache_group="alice")
    assert cache.match_prefix(ids[:32], cache_group="alice").cached_len == 0
    assert cache.match_prefix(ids[:32], cache_group="bob").cached_len == 32
    assert_indices(cache.match_prefix(ids, cache_group="alice").kv_indices, first)
    assert_indices(cache.match_prefix(ids, cache_group="bob").kv_indices, second)


def test_swa_global_eviction_respects_locked_handle():
    cache = SWARadixCache(DEVICE, page_size=1, sliding_window_size=16)
    ids = tensor_range(1, 64)
    first, second = tensor_range(100, 64), tensor_range(200, 64)
    cache.insert(ids, first, cache_group="alice")
    cache.insert(ids, second, cache_group="bob")
    locked = cache.match_prefix(ids, cache_group="alice")
    window_handle = cache.inc_lock(locked.node)
    evicted = cache.evict_full(128)
    assert set(evicted.kv_indices.tolist()) == set(second.tolist())
    assert cache.match_prefix(ids, cache_group="alice").cached_len == 64
    assert cache.match_prefix(ids, cache_group="bob").cached_len == 0
    cache.dec_lock(locked.node, window_handle)
    assert set(cache.evict_full(64).kv_indices.tolist()) == set(first.tolist())


def test_swa_lru_is_shared_across_groups():
    cache = SWARadixCache(DEVICE, page_size=1, sliding_window_size=16)
    ids = tensor_range(1, 64)
    first, second = tensor_range(100, 64), tensor_range(200, 64)
    cache.insert(ids, first, cache_group="z-old")
    cache.insert(ids, second, cache_group="a-new")
    assert set(cache.evict_full(64).kv_indices.tolist()) == set(first.tolist())
    assert cache.match_prefix(ids, cache_group="z-old").cached_len == 0
    assert_indices(cache.match_prefix(ids, cache_group="a-new").kv_indices, second)


@pytest.mark.parametrize("group", [None, "", "alice", "用户甲"])
def test_tokenizer_message_preserves_group_on_round_trip(group):
    msg = TokenizeMsg(uid=1, text="same prompt", sampling_params=SamplingParams(), **group_arg(group))
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert decoded.cache_group == (group or "")
    assert decoded.text == "same prompt"
    assert decoded.uid == 1


@pytest.mark.parametrize("group", [None, "", "alice", "用户甲"])
def test_backend_message_preserves_group_on_round_trip(group):
    ids = tensor_range(1, 8)
    msg = UserMsg(uid=1, input_ids=ids, sampling_params=SamplingParams(), **group_arg(group))
    decoded = BaseBackendMsg.decoder(msg.encoder())
    assert decoded.cache_group == (group or "")
    assert_indices(decoded.input_ids, ids)
    assert decoded.uid == 1


@pytest.mark.parametrize("group", [None, "", "alice"])
def test_pending_and_scheduled_request_expose_group(group):
    ids = tensor_range(1, 8)
    kwargs = group_arg(group)
    params = SamplingParams()
    pending = PendingReq(uid=1, input_ids=ids, sampling_params=params, **kwargs)
    cache = RadixPrefixCache(DEVICE, page_size=1)
    handle = cache.match_prefix(ids, **kwargs).cuda_handle
    req = Req(input_ids=ids, table_idx=0, cached_len=0, output_len=1, uid=1,
              sampling_params=params, cache_handle=handle, **kwargs)
    assert pending.cache_group == (group or "")
    assert req.cache_group == (group or "")
    assert_indices(pending.input_ids, ids)
    assert_indices(req.input_ids, ids)


@pytest.mark.parametrize("group", [None, "", "alice"])
def test_prefill_chunks_preserve_group(group):
    page_table = torch.zeros((3, 128), dtype=torch.int32)
    cache = CacheManager(num_pages=256, page_size=1, page_table=page_table, type="radix")
    tables = TableManager(max_running_reqs=2, page_table=page_table)
    manager = PrefillManager(cache, tables, DecodeManager(page_size=1))
    ids = tensor_range(1, 12)
    msg = UserMsg(uid=1, input_ids=ids, sampling_params=SamplingParams(max_tokens=1),
                  **group_arg(group))
    manager.add_one_req(msg)
    assert manager.pending_list[0].cache_group == (group or "")

    for chunk in range(3):
        batch = manager.schedule_next_batch(prefill_budget=4)
        assert batch is not None
        assert len(batch.reqs) == 1
        req = batch.reqs[0]
        assert req.cache_group == (group or "")
        if chunk < 2:
            req.commit_prefill_kv()
            assert manager.pending_list[0].cache_group == (group or "")
    assert_indices(ids, tensor_range(1, 12))


def http_json(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base_url + path, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        assert response.status == 200
        return json.load(response)


@pytest.fixture(scope="module")
def http_cache_server():
    base_url = os.environ.get("FT_CACHE_GROUP_TEST_URL", "").rstrip("/")
    if not base_url:
        pytest.skip("set FT_CACHE_GROUP_TEST_URL to test a running model server")
    assert http_json(base_url, "/health")["status"] == "ok"
    model = http_json(base_url, "/v1/models")["data"][0]
    return base_url, model


def online_prompt():
    return (f"{uuid.uuid4().hex}\n"
            + "The orchard has red apples and green pears. " * 60
            + "\nContinue the description in one short sentence:\n")


def assert_http_usage(response, expected_prompt_tokens=None):
    usage = response["usage"]
    assert 128 <= usage["prompt_tokens"] <= 1000
    if expected_prompt_tokens is not None:
        assert usage["prompt_tokens"] == expected_prompt_tokens
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == usage["prompt_tokens"] + 4
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    assert 0 <= cached <= usage["prompt_tokens"]
    return usage["prompt_tokens"], cached


def test_http_completions_isolate_groups_and_keep_default_alias(http_cache_server):
    from transformers import AutoTokenizer

    base_url, model = http_cache_server
    prompt = online_prompt()
    tokenizer = AutoTokenizer.from_pretrained(model["root"], local_files_only=True)
    original_prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    payload = {"model": model["id"], "prompt": prompt, "max_tokens": 4,
               "temperature": 0, "ignore_eos": True}
    run_id = uuid.uuid4().hex
    groups = [f"{run_id}-a", f"{run_id}-a", f"{run_id}-b", f"{run_id}-b", None, "", None]
    prompt_tokens = original_prompt_tokens
    baseline_text = None
    cached_tokens = []
    for group in groups:
        response = http_json(base_url, "/v1/completions", payload | group_arg(group))
        prompt_tokens, cached = assert_http_usage(response, prompt_tokens)
        cached_tokens.append(cached)
        text = response["choices"][0]["text"]
        assert isinstance(text, str) and text
        if baseline_text is None:
            baseline_text = text
        assert text == baseline_text
    assert [cached_tokens[index] for index in (0, 2, 4)] == [0, 0, 0]
    assert all(cached_tokens[index] > 0 for index in (1, 3, 5, 6))
    assert payload["prompt"] == prompt
    print(f"completions: prompt_tokens={prompt_tokens}; cached_tokens={cached_tokens}; "
          f"completion_tokens=4; text={baseline_text!r}")


def test_http_chat_accepts_and_preserves_cache_group(http_cache_server):
    base_url, model = http_cache_server
    payload = {"model": model["id"], "messages": [{"role": "user", "content": online_prompt()}],
               "max_tokens": 4, "temperature": 0, "ignore_eos": True}
    run_id = uuid.uuid4().hex
    prompt_tokens = None
    baseline_message = None
    cached_tokens = []
    for group in (f"{run_id}-chat-a", f"{run_id}-chat-a", f"{run_id}-chat-b"):
        response = http_json(base_url, "/v1/chat/completions", payload | {"cache_group": group})
        prompt_tokens, cached = assert_http_usage(response, prompt_tokens)
        cached_tokens.append(cached)
        message = response["choices"][0]["message"]
        assert message["role"] == "assistant"
        if baseline_message is None:
            baseline_message = message
        assert message == baseline_message
    assert cached_tokens[0] == cached_tokens[2] == 0
    assert cached_tokens[1] > 0
    print(f"chat: prompt_tokens={prompt_tokens}; cached_tokens={cached_tokens}; completion_tokens=4")
