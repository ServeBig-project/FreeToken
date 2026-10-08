"""OpenAI-compatible HTTP client helpers and prompt construction."""
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from . import env

_WORDS = ("river mountain lantern copper violet harbor engine meadow silver cobalt orchard thunder "
          "pencil canyon marble falcon garden window ladder ember quartz willow tunnel beacon saddle "
          "glacier compass velvet anchor prairie cinder ribbon harvest summit lagoon timber").split()


class Tok:
    def __init__(self, path):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(path)

    def n(self, text):
        return len(self.t.encode(text, add_special_tokens=False))

    def filler(self, n_tokens, seed):
        """Deterministic numbered prose of about n_tokens tokens, distinct per seed."""
        rng = random.Random(seed)
        parts, text = [f"Document {seed}."], ""
        i = 0
        while True:
            i += 1
            parts.append(f" Note {i}: the {rng.choice(_WORDS)} near the {rng.choice(_WORDS)} "
                         f"holds {rng.randint(10, 999)} {rng.choice(_WORDS)}s.")
            cand = "".join(parts)
            if self.n(cand) > n_tokens:
                return text or cand
            text = cand


def count_prompt(start=1):
    return "Continue the list of numbers, one per line:\n" + "".join(f"{i}\n" for i in range(start, start + 12))


class Client:
    def __init__(self, url):
        self.url = url
        self.model = requests.get(url + "/v1/models", timeout=30).json()["data"][0]["id"]

    def get(self, path):
        r = requests.get(self.url + path, timeout=60)
        r.raise_for_status()
        return r.json()

    def stats(self):
        return self.get("/v1/stats")

    def cache_status(self):
        return self.get("/v1/cache/status")

    def post(self, path, body, timeout=None):
        return requests.post(self.url + path, json=body, timeout=timeout or env.REQ_TIMEOUT)

    def complete(self, prompt, max_tokens, ignore_eos=True, **kw):
        body = dict(model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=0,
                    ignore_eos=ignore_eos, **kw)
        r = self.post("/v1/completions", body)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:2000]}"
        j = r.json()
        c = j["choices"][0]
        return dict(text=c["text"], finish=c.get("finish_reason"), usage=j.get("usage", {}), raw=j)

    def chat(self, content, max_tokens, ignore_eos=False, **kw):
        body = dict(model=self.model, messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens, temperature=0, ignore_eos=ignore_eos, **kw)
        r = self.post("/v1/chat/completions", body)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:2000]}"
        j = r.json()
        c = j["choices"][0]
        m = c["message"]
        return dict(text=(m.get("content") or ""), reasoning=m.get("reasoning_content") or "",
                    finish=c.get("finish_reason"), usage=j.get("usage", {}), raw=j)

    def stream(self, prompt, max_tokens, stop_after_chunks=None, stop_after_s=None, **kw):
        """SSE completion. Optionally disconnect early (client cancel). Returns text, chunks, usage."""
        body = dict(model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=0,
                    ignore_eos=True, stream=True, stream_options={"include_usage": True}, **kw)
        t0 = time.time()
        r = requests.post(self.url + "/v1/completions", json=body, stream=True, timeout=env.REQ_TIMEOUT)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:2000]}"
        text, chunks, usage, finish, done = "", 0, None, None, False
        try:
            for line in r.iter_lines():
                if stop_after_s is not None and time.time() - t0 > stop_after_s:
                    break
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                j = json.loads(data)
                if j.get("usage"):
                    usage = j["usage"]
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

    def parallel(self, jobs, stagger_s=0.0):
        """Run [(fn, args, kwargs)] concurrently, optionally staggering arrival. Returns results in order."""
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
            futs = []
            for i, (fn, a, k) in enumerate(jobs):
                if i and stagger_s:
                    time.sleep(stagger_s)
                futs.append(ex.submit(fn, *a, **k))
            return [f.result(timeout=env.REQ_TIMEOUT) for f in futs]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def record(name, obj):
    """Append an observation (numerical differences, stats snapshots) to the results directory."""
    os.makedirs(env.RESULTS, exist_ok=True)
    with _lock, open(os.path.join(env.RESULTS, "observations.jsonl"), "a") as f:
        f.write(json.dumps({"case": name, **obj}, ensure_ascii=False, default=str) + "\n")


def dump(name, obj):
    os.makedirs(env.RESULTS, exist_ok=True)
    with open(os.path.join(env.RESULTS, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)


_lock = threading.Lock()
