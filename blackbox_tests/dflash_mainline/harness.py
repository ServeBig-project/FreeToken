"""Public-interface helpers: launch one `python -m freetoken` service, talk HTTP, read stats/status."""

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PYTHON = os.environ.get("FT_PYTHON", "/home/nengneng/miniconda3/envs/freetoken-dev/bin/python")
SOURCE = os.environ.get("FT_SOURCE", "/home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees/dflash-mainline/python")
GPU = os.environ.get("FT_GPU", "GPU-b8a2a927-a7dd-4a70-5fca-aa2f74a142cd")
LOG_DIR = Path(os.environ.get("FT_LOG_DIR", "/tmp/dflash_mainline_blackbox"))
PORT = int(os.environ.get("FT_PORT", "31781"))

NVFP4 = "/data1/lmcache_kv/models/Qwen3.6-35B-A3B-NVFP4"
BF16 = "/data1/lmcache_kv/models/Qwen3.6-35B-A3B"
QWEN3 = "/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B"
DRAFTER = ("/data2/servebig-envs/dflash_models/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/"
           "f181eece646affea2c38b2765f1aaa01a9734ccd")


def qwen36(path=NVFP4, moe=2048, tokens=65536, running=4, graph=4, seq=32768, gdn=2000000000,
           policy="legacy", backend="offload"):
    return ["--model-path", path, "--moe-backend", backend, "--moe-cache-size", str(moe),
            "--batching-policy", policy, "--num-tokens", str(tokens), "--max-running-requests", str(running),
            "--attention-backend", "fi", "--cuda-graph-max-bs", str(graph),
            "--max-seq-len-override", str(seq), "--enable-gdn-replayssm",
            "--gdn-state-budget-bytes", str(gdn)]


def dflash(steps=8, *extra):
    return ["--speculative-num-steps", str(steps), "--speculative-draft-model-path", DRAFTER, *extra]


class Server:
    def __init__(self, label, args, timeout=900):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", PORT)) == 0:
                raise RuntimeError(f"port {PORT} already serving")
        self.label, self.url = label, f"http://127.0.0.1:{PORT}"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{label}.log"
        self.cmd = [PYTHON, "-m", "freetoken", "--gpu", GPU, "--port", str(PORT),
                    "--enable-cache-report", *args]
        self.log = self.log_path.open("w")
        self.proc = subprocess.Popen(self.cmd, env={**os.environ, "PYTHONPATH": SOURCE},
                                     stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.log.close()
                raise StartupError(f"{label} exited {self.proc.returncode}", self.log_path.read_text()[-4000:])
            try:
                if get(self.url, "/v1/cache/status", timeout=5)["body"].get("state") == "serving":
                    return
            except Exception:
                pass
            time.sleep(3)
        self.close()
        raise RuntimeError(f"{label} not serving after {timeout}s")

    def close(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(30)
        if not self.log.closed:
            self.log.close()

    def alive(self):
        return self.proc.poll() is None

    def status(self):
        return get(self.url, "/v1/cache/status")["body"]

    def stats(self):
        return get(self.url, "/v1/stats")["body"]

    def complete(self, prompt, max_tokens=16, group=None, timeout=420, **extra):
        body = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, **extra}
        if group is not None:
            body["cache_group"] = group
        return post(self.url, "/v1/completions", body, timeout)

    def post_chat(self, content, max_tokens):
        body = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        return post(self.url, "/v1/chat/completions", body)

    def stream(self, prompt, max_tokens=16, group=None, cancel_after=None, **extra):
        body = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                "stream": True, **extra}
        if group is not None:
            body["cache_group"] = group
        return stream(self.url, "/v1/completions", body, cancel_after)

    def parallel(self, calls):
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            return [f.result() for f in [pool.submit(fn) for fn in calls]]

    def rebuild(self, body, timeout=600):
        return post(self.url, "/v1/cache/rebuild", body, timeout)

    def wait_idle(self, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            stats = self.stats()
            if not stats["requests"]["active"]:
                return stats
            time.sleep(0.3)
        raise AssertionError(f"not idle after {timeout}s: {json.dumps(stats['requests'])}")

    def measured(self, call):
        """Run one call alone and return (result, public speculative counter delta)."""
        from workloads import spec_delta
        self.wait_idle()
        before = self.stats()
        result = call()
        after = self.wait_idle()
        return result, spec_delta(after, before)

    @property
    def outwave(self):
        """True when every drafted round is outside prefill waves (legacy, or layered outwave)."""
        return self.flag("--speculative-phase", "outwave") == "outwave"

    def flag(self, name, default=None):
        return self.cmd[self.cmd.index(name) + 1] if name in self.cmd else default


class StartupError(RuntimeError):
    def __init__(self, message, log_tail):
        super().__init__(message)
        self.log_tail = log_tail


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
        raw, status = response.read().decode(), response.status
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"raw": raw}
    return {"status": status, "body": payload, "seconds": time.monotonic() - started}


def get(url, path, timeout=30):
    return _call(url, "GET", path, timeout=timeout)


def post(url, path, body, timeout=900):
    return _call(url, "POST", path, body, timeout)


def stream(url, path, body, cancel_after=None, timeout=900):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks, done = [], False
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
            if cancel_after is not None and len(chunks) >= cancel_after:
                break
    text = "".join(c["choices"][0].get("text", "") for c in chunks if c.get("choices"))
    return {"chunks": chunks, "done": done, "text": text}


def text(resp):
    assert resp["status"] == 200, json.dumps(resp)[:600]
    return resp["body"]["choices"][0]["text"]


def usage(resp):
    return resp["body"].get("usage") or {}


def cached(resp):
    return (usage(resp).get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def document(seed, sentences=60):
    """Deterministic, non-repeating filler; roughly 26 Qwen3.6 tokens per sentence."""
    topics = ["volcanoes", "medieval castles", "deep sea fish", "tea", "orbital mechanics", "bread",
              "glaciers", "railway signals", "honeybees", "ancient Rome", "jazz", "coral reefs"]
    topic = topics[seed % len(topics)]
    return " ".join(f"Note {seed}-{i}: an observation about {topic} number {i * 7 + seed} records "
                    f"value {(i * 37 + seed * 11) % 997} for later review." for i in range(sentences))


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        uuid, used = [x.strip() for x in line.split(",")]
        if uuid == GPU:
            return int(used)
    raise AssertionError(f"GPU {GPU} not listed")


def process_gpu_mib(pid):
    """GPU memory of one process tree root (service is a session leader; workers are its children)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    children = set(subprocess.run(["pgrep", "-s", str(pid)], capture_output=True, text=True).stdout.split())
    total = 0
    for line in out.splitlines():
        p, uuid, used = [x.strip() for x in line.split(",")]
        if uuid == GPU and (p in children or p == str(pid)):
            total += int(used)
    return total
