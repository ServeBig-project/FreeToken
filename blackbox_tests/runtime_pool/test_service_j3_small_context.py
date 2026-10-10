"""A small explicit context in shared mode (round J item 1): the service starts with
--max-seq-len-override 64, serves requests within 64 tokens, and refuses longer ones with the
public length error.

    --runtime-cache-gib 0.5 --max-seq-len-override 64
"""
import os

import pytest

from service_common import COMMON, assert_length, enum_prompt, service, tok
from service_scenarios import shared_stats_contract

NAME = "j3_small_context"
ARGS = COMMON + ["--runtime-cache-gib", "0.5", "--max-seq-len-override", "64", "--enable-cache-report"]


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("RP_GPU_OK"):
        pytest.skip("set RP_GPU_OK=1 once a GPU is free")
    with service(NAME, ARGS) as s:
        yield s


def test_ready_publishes_the_small_context(svc):
    rt = svc.rt()
    assert 1 <= rt["context_tokens"] <= 64, rt
    shared_stats_contract(svc)


def test_requests_within_64_tokens_complete(svc):
    prompt = enum_prompt(100, 6)  # ~24 tokens
    assert tok().n(prompt) + 32 <= 64
    for _ in range(3):
        assert_length(svc.c.complete(prompt, 32), 32)


def test_longer_prompt_gets_the_public_length_error(svc):
    code, j = svc.c.generate(tok().filler(120, seed=5), 8, ignore_eos=True, timeout=300)
    assert 400 <= code < 500 and (j.get("error") or {}).get("code") == "context_length_exceeded", (code, j)
    assert_length(svc.c.complete(enum_prompt(200, 6), 16), 16)
