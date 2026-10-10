"""Shared-mode startup errors (round J items 2 and 5).

joint batching is refused before ready, naming joint and suggesting layered-pipeline or legacy.
A prefill limit below one page is refused before ready, naming the page; it is reachable only
where the configuration accepts a page size above 1, otherwise the case is reported unreachable.
"""
import os

import pytest

from service_common import COMMON, Server

gpu = pytest.mark.skipif(not os.environ.get("RP_GPU_OK"), reason="set RP_GPU_OK=1 once a GPU is free")


def _startup(name, args):
    s = Server(name, args).start()
    try:
        state, detail = s.wait_ready()
        return state, s.log_text()
    finally:
        s.stop()


@gpu
def test_joint_batching_is_refused_with_alternatives():
    state, log = _startup("j6_joint", COMMON + ["--runtime-cache-gib", "0.5", "--batching-policy", "joint"])
    tail = log[-6000:].lower()
    assert state == "error", f"joint started in shared mode ({state})"
    assert "joint" in tail and "layered-pipeline" in tail and "legacy" in tail, log[-2000:]


@gpu
def test_prefill_limit_below_one_page():
    state, log = _startup("j6_subpage_prefill", COMMON + ["--runtime-cache-gib", "0.5", "--page-size", "16",
                                                          "--max-extend-length", "8"])
    tail = log[-6000:].lower()
    assert state == "error", f"a prefill limit of 8 tokens with 16-token pages started ({state})"
    if not any(w in tail for w in ("extend", "prefill")):
        pytest.skip("unreachable: this model/attention configuration refuses page size 16 itself: "
                    + log[-400:].replace("\n", " "))
    assert "page" in tail, log[-2000:]
