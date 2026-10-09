"""OpenAI-compatible HTTP helpers (greedy unless stated)."""
import http.client
import json
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

from . import env


class Client:
    def __init__(self, url):
        self.url = url
        self.model = requests.get(url + "/v1/models", timeout=30).json()["data"][0]["id"]

    def get(self, path):
        r = requests.get(self.url + path, timeout=60)
        r.raise_for_status()
        return r.json()

    def post(self, path, body, timeout=None):
        return requests.post(self.url + path, json=body, timeout=timeout or env.REQ_TIMEOUT)

    def complete(self, prompt, max_tokens, group=None, **kw):
        body = dict(model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=0, **kw)
        if group is not None:
            body["cache_group"] = group
        t0 = time.time()
        r = self.post("/v1/completions", body)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:1500]}"
        j = r.json()
        c = j["choices"][0]
        return dict(text=c["text"], finish=c.get("finish_reason"), usage=j.get("usage") or {},
                    logprobs=c.get("logprobs"), seconds=time.time() - t0)

    def chat(self, messages, max_tokens, **kw):
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=0, **kw)
        r = self.post("/v1/chat/completions", body)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:1500]}"
        j = r.json()
        c = j["choices"][0]
        m = c["message"]
        return dict(text=m.get("content") or "", reasoning=m.get("reasoning_content") or "",
                    finish=c.get("finish_reason"), usage=j.get("usage") or {})

    def stream(self, prompt, max_tokens, group=None, stop_after_chunks=None, **kw):
        body = dict(model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=0, stream=True,
                    stream_options={"include_usage": True}, **kw)
        if group is not None:
            body["cache_group"] = group
        r = requests.post(self.url + "/v1/completions", json=body, stream=True, timeout=env.REQ_TIMEOUT)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:1500]}"
        text, chunks, usage, finish, done = "", 0, None, None, False
        try:
            for line in r.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                j = json.loads(data)
                usage = j.get("usage") or usage
                for c in j.get("choices", []):
                    if c.get("text"):
                        text += c["text"]
                        chunks += 1
                    finish = c.get("finish_reason") or finish
                if stop_after_chunks is not None and chunks >= stop_after_chunks:
                    break
        finally:
            r.close()
        return dict(text=text, chunks=chunks, usage=usage, finish=finish, done=done)

    def abort_after(self, prompt, max_tokens, seconds, group=None):
        """Client disconnect `seconds` after submitting a streaming request (e.g. during prefill)."""
        u = urlparse(self.url)
        body = dict(model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=0, stream=True)
        if group is not None:
            body["cache_group"] = group
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=60)
        conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
        time.sleep(seconds)
        conn.sock.close()
        conn.close()


def parallel(jobs, stagger_s=0.0):
    """Run [(fn, args, kwargs)] concurrently with optional staggered arrival; results in order."""
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
        futs = []
        for i, (fn, a, k) in enumerate(jobs):
            if i and stagger_s:
                time.sleep(stagger_s)
            futs.append(ex.submit(fn, *a, **k))
        return [f.result(timeout=env.REQ_TIMEOUT) for f in futs]


def cached(r):
    return ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
