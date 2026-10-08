"""Launch one `freetoken.cli serve` process on the test GPU and wait for readiness."""
import json
import os
import signal
import socket
import subprocess
import time

import requests

from . import env


def gpu_busy():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return env.GPU in out


def wait_gpu_free(timeout=3600):
    end = time.time() + timeout
    while gpu_busy():
        if time.time() > end:
            raise RuntimeError(f"GPU {env.GPU} still has compute processes after {timeout}s")
        time.sleep(10)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_env():
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=env.GPU, PYTHONPATH=os.path.join(env.IMPL, "python"),
             OMP_NUM_THREADS="1", FREETOKEN_BENCHBW_PATH=env.BENCHBW)
    return e


class Server:
    def __init__(self, name, args):
        self.name, self.args = name, list(args)
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        os.makedirs(env.RESULTS, exist_ok=True)
        self.log_path = os.path.join(env.RESULTS, f"{name}.log")
        self.proc = None

    def start(self):
        wait_gpu_free()
        cmd = ["taskset", "-c", env.CPUS, env.PY, "-m", "freetoken.cli", "serve",
               "--host", "127.0.0.1", "--port", str(self.port), *self.args]
        self.log = open(self.log_path, "w")
        self.log.write("CMD: " + " ".join(cmd) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=launch_env(),
                                     start_new_session=True, cwd=env.RESULTS)
        return self

    def wait_ready(self):
        """Return ('ok', health) when ready, ('error', detail) on startup failure."""
        end = time.time() + env.STARTUP_TIMEOUT
        while time.time() < end:
            if self.proc.poll() is not None:
                return "error", f"exit code {self.proc.returncode}"
            try:
                h = requests.get(self.url + "/health", timeout=5).json()
                if h.get("status") == "ok":
                    return "ok", h
                if h.get("status") == "error":
                    return "error", json.dumps(h)
            except (requests.RequestException, ValueError):
                pass
            time.sleep(3)
        return "timeout", f"not ready after {env.STARTUP_TIMEOUT}s"

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
        end = time.time() + 120
        while gpu_busy() and time.time() < end:
            time.sleep(3)


def start_ready(name, args):
    s = Server(name, args).start()
    state, detail = s.wait_ready()
    if state != "ok":
        tail = s.log_text()[-4000:]
        s.stop()
        raise AssertionError(f"[{name}] startup {state}: {detail}\n--- log tail ---\n{tail}")
    return s


def expect_startup_error(name, args):
    """Explicit unsupported configurations must fail before ready (section 2/3). Returns the log."""
    s = Server(name, args).start()
    try:
        state, detail = s.wait_ready()
        log = s.log_text()
    finally:
        s.stop()
    assert state == "error", f"[{name}] expected startup error, got {state}: {detail}\n{log[-3000:]}"
    return log
