"""Public-interface helpers: launch a server, send HTTP requests, read cache status."""

import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("FT_PYTHON", "/home/nengneng/miniconda3/envs/freetoken-dev/bin/python")
GPU = os.environ.get("FT_GPU", "GPU-847e9c75-56a9-1090-4f4f-7d70a71792dd")
BASE_SOURCE = os.environ.get("FT_BASE_SOURCE")
LOG_DIR = Path(os.environ.get("FT_LOG_DIR", "/tmp/cold_prefix_blackbox_logs"))
RESULTS = Path(os.environ.get("FT_RESULTS", str(LOG_DIR / "results.jsonl")))

QWEN3 = ["--model-path", "/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B",
         "--moe-backend", "offload", "--moe-cache-size", "1024"]
QWEN36 = ["--model-path", "/data1/lmcache_kv/models/Qwen3.6-35B-A3B",
          "--moe-backend", "offload", "--moe-cache-size", "1536"]
DFLASH = ["--speculative-num-steps", "8", "--speculative-draft-model-path",
          "/data2/servebig-envs/dflash_models/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/"
          "f181eece646affea2c38b2765f1aaa01a9734ccd",
          "--speculative-draft-experts", "3", "--enable-gdn-replayssm",
          "--gdn-state-budget-bytes", "3000000000"]
DSV4 = ["--model-path", "/data2/lmcache_kv/models/DeepSeek-V4-Flash-0731",
        "--moe-backend", "offload", "--moe-cache-size", "640"]


def gptoss():
    root = Path("/data2/lmcache_kv/hf-cache/models--openai--gpt-oss-20b/snapshots")
    return ["--model-path", str(sorted(root.iterdir())[0])]


def record(name, **fields):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as handle:
        handle.write(json.dumps({"check": name, "time": time.strftime("%H:%M:%S"), **fields}) + "\n")


class Server:
    def __init__(self, label, args, port=31731, source=None, timeout=900):
        self.label, self.port = label, port
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"port {port} already serving; one server at a time")
        self.url = f"http://127.0.0.1:{port}"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{label}.log"
        env = {**os.environ, "PYTHONPATH": str(source or WORKTREE / "python")}
        cmd = [PYTHON, "-m", "freetoken", "--gpu", GPU, "--port", str(port),
               "--enable-cache-report", *args]
        self.cmd = cmd
        self.log = self.log_path.open("w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self._wait(timeout)

    def _wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server {self.label} exited {self.proc.returncode}; log {self.log_path}")
            try:
                if get(self.url, "/v1/cache/status", timeout=5)["body"].get("state") == "serving":
                    return
            except Exception:
                pass
            time.sleep(3)
        self.close()
        raise RuntimeError(f"server {self.label} not serving after {timeout}s")

    def close(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(30)
        self.log.close()

    def alive(self):
        return self.proc.poll() is None

    def status(self):
        return get(self.url, "/v1/cache/status")["body"]

    def pc(self):
        return self.status().get("prefix_cache")

    def complete(self, prompt, max_tokens=16, group=None, **extra):
        body = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, **extra}
        if group is not None:
            body["cache_group"] = group
        return post(self.url, "/v1/completions", body)

    def stream(self, prompt, max_tokens=16, group=None, cancel_after=None, **extra):
        body = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                "stream": True, **extra}
        if group is not None:
            body["cache_group"] = group
        return stream(self.url, "/v1/completions", body, cancel_after)

    def parallel(self, calls):
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = [pool.submit(fn) for fn in calls]
            return [f.result() for f in futures]

    def wait_idle(self, timeout=120):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.status()
            if idle(last):
                return last
            time.sleep(0.5)
        raise AssertionError(f"server not idle after {timeout}s: {json.dumps(last)[:800]}")


def idle(status):
    for key in ("running_requests", "pending_requests", "waiting_requests", "num_running", "num_waiting"):
        if status.get(key):
            return False
    return True


def _call(url, method, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read().decode()
        status = response.status
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"raw": raw}
    return {"status": status, "body": payload, "seconds": time.monotonic() - started}


def get(url, path, timeout=30):
    return _call(url, "GET", path, timeout=timeout)


def post(url, path, body, timeout=600):
    return _call(url, "POST", path, body, timeout)


def stream(url, path, body, cancel_after=None, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks, done, started, first = [], False, time.monotonic(), None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            chunks.append(json.loads(data))
            if first is None:
                first = time.monotonic() - started
            if cancel_after is not None and len(chunks) >= cancel_after:
                break
    text = "".join(c["choices"][0].get("text", "") for c in chunks if c.get("choices"))
    return {"chunks": chunks, "done": done, "text": text, "ttft": first,
            "seconds": time.monotonic() - started}


def text(resp):
    assert resp["status"] == 200, resp
    return resp["body"]["choices"][0]["text"]


def cached(resp):
    usage = resp["body"].get("usage") or {}
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens")


def prompt_tokens(resp):
    return resp["body"]["usage"]["prompt_tokens"]


def delta(after, before, keys=("gpu_reused_tokens", "host_reused_tokens", "recomputed_tokens",
                               "h2d_bytes", "d2h_bytes", "h2d_batches", "d2h_batches",
                               "checkpoint_created", "checkpoint_pruned", "checkpoint_evicted",
                               "checkpoint_deduplicated")):
    return {k: after[k] - before[k] for k in keys}


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


TOPICS = ["volcanoes", "medieval castles", "deep sea fish", "the history of tea", "orbital mechanics",
          "bread baking", "glaciers", "railway signalling", "honeybees", "ancient Rome",
          "jazz improvisation", "coral reefs", "printing presses", "desert ecology", "lighthouses",
          "chess openings", "monsoons", "paper making", "telescopes", "salmon migration"]


def document(seed, sentences=60):
    """Deterministic filler text; roughly 12-14 tokens per sentence."""
    topic = TOPICS[seed % len(TOPICS)]
    lines = []
    for i in range(sentences):
        lines.append(f"Note {seed}-{i}: an observation about {topic} number {i * 7 + seed} "
                     f"records value {(i * 37 + seed * 11) % 997} for later review.")
    return " ".join(lines)


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        uuid, used = [x.strip() for x in line.split(",")]
        if uuid == GPU:
            return int(used)
    raise AssertionError(f"GPU {GPU} not listed")


def chat(s, content, max_tokens=16, group=None):
    body = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0}
    if group is not None:
        body["cache_group"] = group
    return post(s.url, "/v1/chat/completions", body)
