"""Start one server (candidate `freetoken.cli serve` or the upstream reference `ft serve`) and wait."""
import os
import signal
import socket
import subprocess
import time

import requests

from . import env


def free_port():
    """An HTTP port whose +1 (torch distributed port) is also free."""
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p + 1))
                return p
            except OSError:
                pass


class Server:
    def __init__(self, name, args, reference=False):
        self.name, self.args, self.reference = name, list(args), reference
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        os.makedirs(env.RESULTS, exist_ok=True)
        self.log_path = os.path.join(env.RESULTS, f"{name}.log")
        self.proc = None

    def start(self):
        wait_gpu_free()
        e = dict(os.environ)
        if self.reference:
            e.pop("PYTHONPATH", None)
            cmd = [env.REF_FT, "serve"]
        else:
            e["PYTHONPATH"] = os.path.join(env.IMPL, "python")
            cmd = [env.PY, "-m", "freetoken.cli", "serve"]
        cmd += ["--host", "127.0.0.1", "--port", str(self.port), *self.args]
        self.log = open(self.log_path, "w")
        self.log.write("CMD: " + " ".join(cmd) + "\n")
        self.log.flush()
        self.t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=e,
                                     start_new_session=True, cwd=env.RESULTS)
        return self

    def wait_ready(self):
        """('ok', seconds) when /health is ok, ('error', detail) if it exits or reports an error."""
        end = time.time() + env.STARTUP_TIMEOUT
        while time.time() < end:
            if self.proc.poll() is not None:
                return "error", f"exit code {self.proc.returncode}"
            try:
                h = requests.get(self.url + "/health", timeout=5).json()
                if h.get("status") == "ok":
                    return "ok", time.time() - self.t0
                if h.get("status") == "error":
                    return "error", str(h)
            except (requests.RequestException, ValueError):
                pass
            time.sleep(3)
        return "timeout", f"not ready after {env.STARTUP_TIMEOUT}s"

    def log_text(self):
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.alive():
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(90)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(60)
        if self.proc:
            self.log.close()
        wait_gpu_free()


def start_ready(name, args, reference=False):
    s = Server(name, args, reference).start()
    state, detail = s.wait_ready()
    if state != "ok":
        tail = s.log_text()[-5000:]
        s.stop()
        raise AssertionError(f"[{name}] startup {state}: {detail}\n--- log tail ---\n{tail}")
    s.load_seconds = detail
    return s


def expect_startup_error(name, args):
    """Unsupported or broken inputs must fail before ready. Returns (state, seconds, log)."""
    s = Server(name, args).start()
    try:
        state, detail = s.wait_ready()
        secs = time.time() - s.t0
        log = s.log_text()
    finally:
        s.stop()
    return state, secs, log


def wait_gpu_free(limit_mib=2000, timeout=300):
    """The previous server's memory is released some time after its process group exits."""
    end = time.time() + timeout
    while (gpu_used_mib() or 0) > limit_mib:
        if time.time() > end:
            raise RuntimeError(f"GPU still holds {gpu_used_mib()} MiB after {timeout}s")
        time.sleep(3)


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None
