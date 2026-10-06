"""Other batching policies on Qwen3-30B with an ample CPU tier: cold/shared restore and cancellation."""

import pytest

from conftest import assert_clean
from harness import QWEN3, Server
from scenarios import Checks, run_matrix

SIZE = ["--num-tokens", "3072", "--max-running-requests", "4", "--prefix-cache-host-gib", "2"]
POLICIES = {
    "mixed": (["--batching-policy", "mixed"], True),
    "layered": (["--batching-policy", "layered"], False),
    "layered-pipeline": (["--batching-policy", "layered-pipeline"], False),
    "joint": (["--batching-policy", "joint", "--attention-backend", "triton"], True),
}


@pytest.mark.parametrize("policy", list(POLICIES))
def test_policy(policy):
    flags, strict = POLICIES[policy]
    label = f"q3_batching_{policy}"
    s = Server(label, QWEN3 + SIZE + flags)
    try:
        c = Checks(label, strict_text=strict)
        run_matrix(s, c, sentences=30, pressure_count=5, gdn=False, policy="baseline", parts="core")
        assert s.alive()
    finally:
        s.close()
    assert_clean(c)
