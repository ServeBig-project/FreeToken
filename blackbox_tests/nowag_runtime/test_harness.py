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
