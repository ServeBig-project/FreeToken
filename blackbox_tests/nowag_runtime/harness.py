"""Public CLI/HTTP helpers: ft serve / ft checkpoint, completions, cache status.

The candidate is selected only through NOWAG_SOURCE (its python/ dir, put on PYTHONPATH).
"""

import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PYTHON = os.environ.get("NOWAG_PYTHON", "/home/nengneng/miniconda3/envs/freetoken-dev/bin/python")
SOURCE = os.environ.get("NOWAG_SOURCE")
BASELINE_SOURCE = os.environ.get("NOWAG_BASELINE_SOURCE")
LOG_DIR = Path(os.environ.get("NOWAG_LOG_DIR", "/tmp/nowag_blackbox_logs"))
PORT = int(os.environ.get("NOWAG_PORT", "31791"))
FT = "import sys; from freetoken.cli import main; sys.argv = ['ft', *sys.argv[1:]]; sys.exit(main())"


def env(source=None):
    src = source or SOURCE
    if not src:
        raise RuntimeError("NOWAG_SOURCE (candidate python/ dir) is required for CLI/HTTP tests")
    return {**os.environ, "PYTHONPATH": src}


def ft(args, timeout=7200, source=None, label="ft"):
    """Run `ft <args>`; returns CompletedProcess with the combined log also saved to LOG_DIR."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([PYTHON, "-c", FT, *map(str, args)], env=env(source), timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (LOG_DIR / f"{label}.log").write_text(proc.stdout)
    return proc


class StartupFailed(RuntimeError):
    def __init__(self, label, code, log):
        super().__init__(f"{label} exited {code} before serving:\n{log[-3000:]}")
        self.code, self.log = code, log


class Server:
    def __init__(self, label, args, gpu, timeout=1800, source=None, port=PORT):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"port {port} busy; one server at a time")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.label, self.url = label, f"http://127.0.0.1:{port}"
        self.log_path = LOG_DIR / f"{label}.log"
        self.cmd = [PYTHON, "-c", FT, "serve", "--gpu", gpu, "--port", str(port),
                    "--enable-cache-report", *map(str, args)]
        self._log = self.log_path.open("w")
        self._log.write(" ".join(self.cmd) + "\n")
        self._log.flush()
        self.proc = subprocess.Popen(self.cmd, env=env(source), stdout=self._log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self._wait(timeout)

    def _wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self._log.close()
                raise StartupFailed(self.label, self.proc.returncode, self.log_path.read_text())
            try:
                if get(self.url, "/v1/cache/status", 5)["body"].get("state") == "serving":
                    return
            except Exception:
                pass
            time.sleep(3)
        self.close()
        raise RuntimeError(f"{self.label} not serving after {timeout}s; log {self.log_path}")

    def close(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(90)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(30)
        if not self._log.closed:
            self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def alive(self):
        return self.proc.poll() is None

    def status(self):
        return get(self.url, "/v1/cache/status")["body"]

    def complete(self, prompt, max_tokens=32, **extra):
        return post(self.url, "/v1/completions", completion_body(prompt, max_tokens, extra))

    def stream(self, prompt, max_tokens=32, cancel_after=None, **extra):
        body = completion_body(prompt, max_tokens, dict(extra, stream=True))
        return stream(self.url, "/v1/completions", body, cancel_after)

    def greedy(self, prompts, max_tokens=32):
        return [text(self.complete(p, max_tokens)) for p in prompts]

    def parallel(self, calls):
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            return [f.result() for f in [pool.submit(c) for c in calls]]


def completion_body(prompt, max_tokens, extra):
    """Greedy unless the caller asks for a temperature: temperature 0 alone still inherits the
    server's default top_p (0.95), which is not the public greedy path, so greedy requests also
    send top_p=1. A caller's explicit top_p, and any sampling request, are left as given."""
    body = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, **extra}
    if body.setdefault("temperature", 0) == 0:
        body.setdefault("top_p", 1)
    return body


def expect_rejected(label, args, gpu, timeout=1800):
    """The configuration must be refused before the server reports ready (contract §6)."""
    try:
        server = Server(label, args, gpu, timeout=timeout)
    except StartupFailed as failure:
        assert failure.code != 0
        return failure.log
    server.close()
    raise AssertionError(f"{label}: invalid configuration reached serving")


def _call(url, method, path, body=None, timeout=900):
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
            if first is None and any(c.get("text") for c in chunks[-1].get("choices", [])):
                first = time.monotonic() - started
            if cancel_after is not None and len(chunks) >= cancel_after:
                break
    usage = next((c["usage"] for c in reversed(chunks) if c.get("usage")), None)
    return {"chunks": chunks, "done": done, "ttft": first, "seconds": time.monotonic() - started,
            "usage": usage,
            "text": "".join(c["choices"][0].get("text", "") for c in chunks if c.get("choices"))}


def text(resp):
    assert resp["status"] == 200, resp
    return resp["body"]["choices"][0]["text"]


def experts(status):
    return status["geometry"]["experts"]


# ------------------------------------------------------------------ frozen output protocol
# Frozen 2026-10-09 before any candidate output was seen.
# * Same weights, same device(s), same execution configuration (rename, status reads, FTW dir
#   renamed): greedy text must be identical.
# * Same compressed weights, different execution path (fused/offload/cpu/hybrid, cache size,
#   batching policy, TP1/TP2, SD on/off, native/FTW): kernels may break near-ties
#   differently, so per prompt either identical text, or both outputs pass the task check and
#   neither loops; at least half the prompts must be identical.
# * Task check: the expected answer appears in the greedy continuation; a run passes when at
#   least 3 of the 4 prompts do. Loop check: no word 4-gram repeats more than 3 times.

PROMPTS = [("The capital of France is", "Paris"),
           ("1, 2, 3, 4, 5, 6,", "7"),
           ("The chemical symbol for gold is", "Au"),
           ("Water freezes at a temperature of 0 degrees", "Celsius")]


def loops(s):
    words = re.findall(r"\S+", s)
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return any(grams.count(g) > 3 for g in set(grams))


def task_ok(outputs):
    hits = sum(ans.lower() in out.lower() for (_, ans), out in zip(PROMPTS, outputs))
    return hits >= 3 and not any(loops(o) for o in outputs)


def same_execution(a, b, label=""):
    assert a == b, f"{label}: identical configuration gave different text\n{a}\n{b}"


def cross_path(a, b, label="", task=True):
    same = [x == y for x, y in zip(a, b)]
    if task:
        assert task_ok(a) and task_ok(b), f"{label}: task check failed\n{a}\n{b}"
    assert sum(same) * 2 >= len(same), f"{label}: only {sum(same)}/{len(same)} identical\n{a}\n{b}"


def run_prompts(server, max_tokens=32):
    return server.greedy([p for p, _ in PROMPTS], max_tokens)
