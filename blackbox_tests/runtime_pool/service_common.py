"""Service-level black-box helpers for the shared runtime pool (contract sections 2-6).

Only the public surface is used: the CLI, OpenAI-style generation, /v1/cache/status, /v1/stats
and /v1/cache/rebuild. Paths and parameters follow docs/runtime-pool-acceptance-config.md and
can be overridden through RP_* environment variables. One server runs at a time on the test GPU.

    CUDA_VISIBLE_DEVICES=<uuid> python -m pytest blackbox_tests/runtime_pool/test_service_*.py
"""

import http.client
import json
import os
import random
import re
import signal
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREES = "/home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees"
PY = os.environ.get("RP_PYTHON", "/home/nengneng/miniconda3/envs/freetoken-dev/bin/python")
IMPL = os.environ.get("RP_IMPL", f"{WORKTREES}/runtime-pool/python")
BASELINE = os.environ.get("RP_BASELINE", f"{WORKTREES}/runtime-pool-base/python")  # main@732f1ee
GPU = os.environ.get("RP_GPU", "GPU-b8a2a927-a7dd-4a70-5fca-aa2f74a142cd")
MODEL = os.environ.get("RP_MODEL", "/data1/lmcache_kv/models/Qwen3.6-35B-A3B-NVFP4")
DFLASH = os.environ.get("RP_DFLASH", "/data2/servebig-envs/dflash_models/models--z-lab--Qwen3.6-35B-A3B-DFlash/"
                        "snapshots/f181eece646affea2c38b2765f1aaa01a9734ccd")
RESULTS = os.environ.get("RP_RESULTS", os.path.join(HERE, "_results"))
STARTUP_TIMEOUT = float(os.environ.get("RP_STARTUP_TIMEOUT", "900"))
CPUS = os.environ.get("RP_CPUS", "8-15,24-31")  # the cpuset the acceptance configuration assigns
PORTS = range(31920, 31960, 2)  # the server also takes port + 1
COMMON = ["--model-path", MODEL, "--moe-backend", "offload", "--moe-cache-size", "2048",
          "--attention-backend", "fi"]
DFLASH_ARGS = ["--speculative-num-steps", "4", "--speculative-draft-model-path", DFLASH]
GIB = 1 << 30
MODEL_CONTEXT = 262144
PARITY = os.path.join(RESULTS, "e_dflash_parity.json")  # written by the DFlash module, read by the baseline


# ---------------------------------------------------------------- results

def dump(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)


_record_lock = threading.Lock()


def record(case, **obj):
    """Append one observation (numbers to report, not to gate) to observations.jsonl."""
    os.makedirs(RESULTS, exist_ok=True)
    with _record_lock, open(os.path.join(RESULTS, "observations.jsonl"), "a") as f:
        f.write(json.dumps({"case": case, "at": time.strftime("%H:%M:%S"), **obj},
                           ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------- server process

def gpu_pids():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return [line.split(",")[1].strip() for line in out.splitlines() if GPU in line]


def other_acceptance():
    """Containers of the NoWAG acceptance session, which shares this GPU."""
    out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
    return [n for n in out.split() if n.startswith("nowag-acceptance-")]


def wait_gpu_free(timeout=float(os.environ.get("RP_GPU_WAIT", "14400"))):
    """GPU1 is time-shared: wait (never preempt) until it has no compute process and no
    NoWAG acceptance container is running."""
    end = time.monotonic() + timeout
    while gpu_pids() or other_acceptance():
        if time.monotonic() > end:
            raise RuntimeError(f"GPU {GPU} busy: {gpu_pids()} {other_acceptance()}")
        time.sleep(15)


def _free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def free_port():
    for p in PORTS:
        if _free(p) and _free(p + 1):
            return p
    raise RuntimeError("no free port in 31920-31960")


class Server:
    """One `python -m freetoken` process. `gpu=False` hides every GPU (startup-conflict cases)."""

    def __init__(self, name, args, impl=IMPL, gpu=True):
        self.name, self.impl, self.gpu = name, impl, gpu
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.args = ["--host", "127.0.0.1", "--port", str(self.port), *args]
        if gpu:
            self.args += ["--gpu", GPU]
        os.makedirs(RESULTS, exist_ok=True)
        self.log_path = os.path.join(RESULTS, f"{name}.log")
        self.proc = None

    def start(self):
        if self.gpu:
            wait_gpu_free()
        env = dict(os.environ, PYTHONPATH=self.impl, CUDA_VISIBLE_DEVICES=GPU if self.gpu else "")
        cmd = ["taskset", "-c", CPUS, PY, "-m", "freetoken", *self.args]
        self.log = open(self.log_path, "w")
        self.log.write("CMD: PYTHONPATH=%s %s\n" % (self.impl, " ".join(cmd)))
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        return self

    def wait_ready(self, timeout=STARTUP_TIMEOUT):
        """('ok', status) once /v1/cache/status says serving; ('error', detail) when the process
        exits or /health reports an error; ('timeout', detail) otherwise."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                return "error", f"exit code {self.proc.returncode}"
            try:
                h = requests.get(self.url + "/health", timeout=5).json()
                if h.get("status") == "error":
                    return "error", json.dumps(h)
                st = requests.get(self.url + "/v1/cache/status", timeout=5).json()
                if st.get("state") == "serving":
                    return "ok", st
            except (requests.RequestException, ValueError):
                pass
            time.sleep(3)
        return "timeout", f"not serving after {timeout}s"

    def log_text(self):
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(30)
        if self.proc:
            self.log.close()
        if self.gpu:
            end = time.monotonic() + 120
            while gpu_pids() and time.monotonic() < end:
                time.sleep(3)


def expect_startup_error(name, args, gpu, timeout=STARTUP_TIMEOUT):
    """The configuration must fail before ready (contract section 2). Returns the process log."""
    s = Server(name, args, gpu=gpu).start()
    try:
        state, detail = s.wait_ready(timeout)
        log = s.log_text()
    finally:
        s.stop()
    assert state == "error", f"[{name}] expected a startup error, got {state}: {detail}\n{log[-3000:]}"
    return log


# ---------------------------------------------------------------- HTTP client

class Client:
    def __init__(self, url):
        self.url = url
        self.host, self.port = url.split("//")[1].split(":")
        self.port = int(self.port)
        self.model = requests.get(url + "/v1/models", timeout=30).json()["data"][0]["id"]

    def get(self, path):
        r = requests.get(self.url + path, timeout=60)
        r.raise_for_status()
        return r.json()

    def status(self):
        return self.get("/v1/cache/status")

    def stats(self):
        return self.get("/v1/stats")

    def runtime(self):
        return self.status()["prefix_cache"]["runtime"]

    def geometry(self):
        return self.status()["geometry"]

    def post(self, path, body, timeout=1800):
        r = requests.post(self.url + path, json=body, timeout=timeout)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"text": r.text}

    def body(self, prompt, max_tokens, **kw):
        return dict(model=self.model, prompt=prompt, max_tokens=max_tokens, **{"temperature": 0, **kw})

    def generate(self, prompt, max_tokens, timeout=1800, **kw):
        """Non-streaming completion; returns (http status, body) so error bodies stay visible."""
        return self.post("/v1/completions", self.body(prompt, max_tokens, **kw), timeout)

    def complete(self, prompt, max_tokens, ignore_eos=True, **kw):
        code, j = self.generate(prompt, max_tokens, ignore_eos=ignore_eos, **kw)
        assert code == 200, f"HTTP {code}: {json.dumps(j)[:1500]}"
        c = j["choices"][0]
        return dict(text=c["text"], finish=c.get("finish_reason"), usage=j.get("usage") or {}, raw=j)

    def chat(self, content, max_tokens, **kw):
        body = dict(model=self.model, messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens, temperature=0, **kw)
        code, j = self.post("/v1/chat/completions", body)
        assert code == 200, f"HTTP {code}: {json.dumps(j)[:1500]}"
        c = j["choices"][0]
        return dict(text=c["message"].get("content") or "", finish=c.get("finish_reason"),
                    usage=j.get("usage") or {}, raw=j)

    def stream(self, prompt, max_tokens, **kw):
        return Stream(self, self.body(prompt, max_tokens, **kw))

    def rebuild(self, body, timeout=900):
        return self.post("/v1/cache/rebuild", {"mode": "if_idle", "timeout": timeout, **body}, timeout + 60)

    def wait_idle(self, timeout=300):
        end = time.monotonic() + timeout
        while True:
            active = self.stats()["requests"]["active"]
            if active == 0:
                return
            assert time.monotonic() < end, f"{active} requests still active after {timeout}s"
            time.sleep(0.5)


class Stream:
    """One SSE completion on its own connection. `run()` blocks (call it from a thread);
    `cancel()` from any thread disconnects the client, which is the public cancel."""

    def __init__(self, client, body, timeout=1800):
        self.client, self.timeout = client, timeout
        self.body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        self.conn = self.sock = None
        self.text, self.chunks, self.usage, self.finish = "", 0, None, None
        self.done = self.cancelled = False
        self.error = self.http_status = None
        self.started = self.ended = self.first_s = None
        self.max_gap_s = 0.0

    def run(self):
        self.conn = http.client.HTTPConnection(self.client.host, self.client.port, timeout=self.timeout)
        self.started = last = time.monotonic()
        try:
            self.conn.connect()
            self.sock = self.conn.sock  # the connection forgets it once the response owns the stream
            self.conn.request("POST", "/v1/completions", json.dumps(self.body),
                              {"Content-Type": "application/json"})
            resp = self.conn.getresponse()
            self.http_status = resp.status
            if resp.status != 200:
                body = json.loads(resp.read())
                self.error = body.get("error", body)
                return self
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    self.done = True
                    break
                j = json.loads(data)
                if "error" in j:
                    self.error = j["error"]
                    continue
                if j.get("usage"):
                    self.usage = j["usage"]
                now = time.monotonic()
                for c in j.get("choices", []):
                    if c.get("text"):
                        self.text += c["text"]
                        self.chunks += 1
                        if self.first_s is None:
                            self.first_s = now - self.started
                        else:
                            self.max_gap_s = max(self.max_gap_s, now - last)
                        last = now
                    if c.get("finish_reason"):
                        self.finish = c["finish_reason"]
        except (OSError, http.client.HTTPException):
            if not self.cancelled:
                raise
        finally:
            self.ended = time.monotonic()
            self.conn.close()
        return self

    def cancel(self):
        self.cancelled = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass

    def summary(self):
        return dict(chunks=self.chunks, finish=self.finish, usage=self.usage, done=self.done,
                    cancelled=self.cancelled, error=self.error, first_s=self.first_s,
                    max_gap_s=round(self.max_gap_s, 2),
                    seconds=None if self.ended is None else round(self.ended - self.started, 1))


def start_streams(streams, stagger_s=0.0):
    ex = ThreadPoolExecutor(max_workers=len(streams))
    futures = []
    for i, s in enumerate(streams):
        if i and stagger_s:
            time.sleep(stagger_s)
        futures.append(ex.submit(s.run))
    return ex, futures


def wait_streams(ex, futures, timeout):
    end = time.monotonic() + timeout
    try:
        for f in futures:
            f.result(timeout=max(1.0, end - time.monotonic()))
    except TimeoutError:
        raise AssertionError(f"{sum(not f.done() for f in futures)} streams still running after {timeout}s")
    finally:
        ex.shutdown(wait=False)


def run_streams(streams, timeout, stagger_s=0.0):
    wait_streams(*start_streams(streams, stagger_s), timeout)
    return streams


def overlap(streams):
    """How many streams produced their first token before the earliest stream finished:
    a lower bound on the number of requests generating at the same time."""
    first_end = min(s.ended for s in streams)
    return sum(1 for s in streams if s.first_s is not None and s.started + s.first_s < first_end)


# ---------------------------------------------------------------- runtime status views

def components(rt):
    c = rt.get("components") or {}
    if isinstance(c, list):
        c = {x["name"]: x for x in c}
    return c


class Watch:
    """Background sampler of prefix_cache.runtime: per-component maxima of held/used bytes,
    the highest total held, and every sample where held exceeds the budget (sections 4, 5)."""

    def __init__(self, client, interval=0.25):
        self.client, self.interval = client, interval
        self.samples, self.violations, self.max_total = 0, [], 0
        self.max_component = {}
        self.series = []  # per sample: {component: held_bytes}, for comparisons at one instant
        self.last = self.failure = None
        self._stop = threading.Event()

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=15)
        if self.failure is not None and exc[0] is None:
            raise AssertionError(f"runtime status could not be read: {self.failure!r}; last={self.last}")

    def _loop(self):
        while not self._stop.is_set():
            try:
                rt = self.client.runtime()
            except requests.RequestException:
                self._stop.wait(self.interval)
                continue
            try:
                self._sample(rt)
            except (KeyError, TypeError) as e:
                self.failure = e
                return
            self._stop.wait(self.interval)

    def _sample(self, rt):
        self.samples += 1
        self.last = rt
        self.max_total = max(self.max_total, rt["held_bytes"])
        if not (rt["used_bytes"] <= rt["held_bytes"] <= rt["budget_bytes"]
                and 0 <= rt["evictable_bytes"] <= rt["held_bytes"]):
            self.violations.append({k: rt[k] for k in ("used_bytes", "held_bytes", "budget_bytes", "evictable_bytes")})
        self.series.append({name: c.get("held_bytes") or 0 for name, c in components(rt).items()})
        for name, c in components(rt).items():
            m = self.max_component.setdefault(name, {"held_bytes": 0, "used_bytes": 0})
            for k in m:
                m[k] = max(m[k], c.get(k) or 0)

    def report(self):
        return dict(samples=self.samples, violations=self.violations[:5], max_total_held=self.max_total,
                    max_component=self.max_component, last=self.last)


PAUSE_KEYS = ("paused", "restored", "recompute", "recomputed_tokens", "paused_ms",
              "short_decode", "short_prefill", "compactions", "map_count", "unmap_count")


def counter_delta(before, after):
    return {k: (after.get(k) or 0) - (before.get(k) or 0) for k in PAUSE_KEYS}


def cached_tokens(usage):
    return ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)


# ---------------------------------------------------------------- /v1/stats views

def _at(obj, path):
    for k in path.split("."):
        if not isinstance(obj, dict) or k not in obj:
            raise KeyError(path)
        obj = obj[k]
    return obj


def _flat_sum(v):
    if isinstance(v, dict):
        return sum(_flat_sum(x) for x in v.values())
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0
    return v


SD_PATHS = {
    "drafted": ["speculative.draft_tokens", "speculative.drafted_tokens"],
    "accepted": ["speculative.accepted_draft_tokens", "speculative.accepted_tokens"],
    "rounds": ["speculative.verify_rounds", "speculative.verify_steps"],
}


def sd_num(stats, name):
    """A cumulative speculative counter; a missing one fails with the speculative object shown."""
    for p in SD_PATHS[name]:
        try:
            return _flat_sum(_at(stats, p))
        except KeyError:
            continue
    raise AssertionError(f"/v1/stats has no '{name}' at {SD_PATHS[name]}: {stats.get('speculative')}")


def sd_enabled(stats):
    return bool((stats.get("speculative") or {}).get("enabled"))


def graph_replays(stats):
    g = stats.get("cuda_graph") or {}
    return sum(_flat_sum(g.get(k)) for k in ("target_decode", "draft", "verify", "verify_range",
                                             "replays", "replay_count"))


# ---------------------------------------------------------------- prompts

ITEM = re.compile(r"item(\d+)")


def enum_prompt(start, items=40, preamble=""):
    """`preamble` then `item<start> item<start+1> ...`; the model is expected to continue it."""
    return preamble + " ".join(f"item{start + i}" for i in range(items))


def enum_check(text, first):
    """The item numbers in `text` must run first, first+1, ... with no gap or repeat; only the
    final number may be cut by the token limit. Returns (ok, count, detail)."""
    nums = [int(n) for n in ITEM.findall(text)]
    if not nums:
        return False, 0, f"no items in {text[:80]!r}"
    for i, n in enumerate(nums[:-1]):
        if n != first + i:
            return False, i, f"expected item{first + i} at index {i}, got item{n}"
    want = first + len(nums) - 1
    if nums[-1] != want and not str(want).startswith(str(nums[-1])):
        return False, len(nums) - 1, f"expected item{want} (or its cut prefix) last, got item{nums[-1]}"
    return True, len(nums), "ok"


def assert_enum(text, first, max_tokens):
    """Continuity plus a plausible count: an item is about five tokens, so demand half of that."""
    ok, count, detail = enum_check(text, first)
    assert ok, f"{detail}; stray={ITEM.sub('', text).strip()[:80]!r}"
    assert count >= max_tokens // 10, f"only {count} items in {max_tokens} tokens: {text[:200]!r}"
    return count


def assert_length(result, n):
    """ignore_eos + max_tokens n: finish 'length' and exactly n generated tokens."""
    usage = result["usage"]
    assert usage.get("completion_tokens") == n, f"completion_tokens {usage} != {n}"
    assert result["finish"] == "length", result["finish"]


_WORDS = ("river mountain lantern copper violet harbor engine meadow silver cobalt orchard thunder "
          "pencil canyon marble falcon garden window ladder ember quartz willow tunnel beacon saddle "
          "glacier compass velvet anchor prairie cinder ribbon harvest summit lagoon timber").split()


class Tok:
    """The checkpoint's public tokenizer, for sizing prompts in tokens."""

    def __init__(self, path=MODEL):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(path)

    def n(self, text):
        return len(self.t.encode(text, add_special_tokens=False))

    def filler(self, n_tokens, seed):
        """Deterministic numbered prose of at most n_tokens tokens (within ~20 of it), distinct per seed."""
        rng = random.Random(seed)
        parts = [f"Document {seed}."]
        i = 0
        while True:
            for _ in range(16):
                i += 1
                parts.append(f" Note {i}: the {rng.choice(_WORDS)} near the {rng.choice(_WORDS)} "
                             f"holds {rng.randint(10, 999)} {rng.choice(_WORDS)}s.")
            if self.n("".join(parts)) > n_tokens:
                break
        while self.n("".join(parts)) > n_tokens:
            parts.pop()
        return "".join(parts)


_tok = None


def tok():
    global _tok
    if _tok is None:
        _tok = Tok()
    return _tok


# ---------------------------------------------------------------- one service per test module

@dataclass
class Svc:
    name: str
    server: Server
    client: Client

    @property
    def c(self):
        return self.client

    def rt(self):
        return self.client.runtime()


@contextmanager
def service(name, args, impl=IMPL):
    s = Server(name, args, impl).start()
    state, detail = s.wait_ready()
    if state != "ok":
        tail = s.log_text()[-4000:]
        s.stop()
        raise AssertionError(f"[{name}] startup {state}: {detail}\n--- log tail ---\n{tail}")
    c = Client(s.url)
    dump(f"{name}_ready_status", c.status())
    dump(f"{name}_ready_stats", c.stats())
    try:
        yield Svc(name, s, c)
    finally:
        try:
            dump(f"{name}_final_status", c.status())
            dump(f"{name}_final_stats", c.stats())
        except Exception:
            pass
        s.stop()
