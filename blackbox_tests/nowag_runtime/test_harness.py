"""Lightweight checks of the independent runner; these are not candidate acceptance."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import harness as H
from test_service import measure


def test_stream_usage_is_separate_from_content_chunks(monkeypatch):
    class Response:
        def __enter__(self):
            return iter([
                b'data: {"choices":[{"text":""}]}\n',
                b'data: {"choices":[{"text":"one two three"}]}\n',
                b'data: {"choices":[],"usage":{"completion_tokens":3}}\n',
                b'data: [DONE]\n',
            ])

        def __exit__(self, *args):
            pass

    times = iter([10.0, 11.0, 13.0])
    monkeypatch.setattr(H.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(H.urllib.request, "urlopen", lambda *a, **k: Response())
    result = H.stream("http://localhost", "/v1/completions", {})
    assert result["done"] and result["text"] == "one two three"
    assert result["ttft"] == 1 and result["seconds"] == 3
    assert result["usage"]["completion_tokens"] == 3


def test_performance_uses_reported_tokens_not_event_count():
    class Service:
        def greedy(self, *args):
            return ["warm"]

        def stream(self, *args, **kwargs):
            assert kwargs["stream_options"] == {"include_usage": True}
            return {"done": True, "ttft": 1.0, "seconds": 3.0,
                    "usage": {"completion_tokens": 9}, "chunks": [{}, {}]}

    result = measure(Service())
    assert result["timings"]["decode_tps"] == pytest.approx(4.0)
    assert [r["completion_tokens"] for r in result["requests"]] == [9] * len(H.PROMPTS)


def test_greedy_requests_pin_top_p_and_leave_sampling_alone():
    body = H.completion_body("p", 4, {})
    assert body["temperature"] == 0 and body["top_p"] == 1
    assert H.completion_body("p", 4, {"top_p": 0.5})["top_p"] == 0.5
    sampled = H.completion_body("p", 4, {"temperature": 0.8})
    assert sampled["temperature"] == 0.8 and "top_p" not in sampled
    streamed = H.completion_body("p", 4, {"stream": True, "cache_group": "g"})
    assert streamed["top_p"] == 1 and streamed["stream"] and streamed["cache_group"] == "g"


def test_stop_group_waits_for_children_that_outlive_the_leader():
    import subprocess
    proc = subprocess.Popen(["sh", "-c", "sleep 30 & exit 0"], start_new_session=True)
    proc.wait()
    assert H.group_alive(proc.pid)              # the child outlived the leader
    H.stop_group(proc, term_timeout=5, kill_timeout=5)
    assert not H.group_alive(proc.pid)


def test_common_prefix_rule():
    words = " ".join(f"w{i}" for i in range(32))
    late = " ".join(f"w{i}" for i in range(13)) + " x" * 19
    H.common_prefix([words], [late])                       # diverges after 13 words: allowed
    with pytest.raises(AssertionError):
        H.common_prefix([words], ["w0 w1 w2 x" + words[11:]])
    H.common_prefix(["w0 w1"], ["w0 w1"])                  # stopped early, identical
    with pytest.raises(AssertionError):
        H.common_prefix(["w0 w1"], ["w0 w1 w2"])           # shorter than 8 must match fully
    with pytest.raises(AssertionError):
        H.common_prefix([words], [""])
