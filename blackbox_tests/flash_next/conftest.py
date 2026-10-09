"""Flash-Next round-1 black-box suite (contract: flash-next-public-contract.md, I1 + I2 scope).

Each test_<x>_*.py module is one server launch serving many checks. Run inside the GPU2 container:

    run-on-gpu2.sh "cd <blackbox worktree> && <conda python> -m pytest \
        -c blackbox_tests/flash_next/pytest.ini blackbox_tests/flash_next -k 'test_a_'"

Records (outputs, usage, stats snapshots) go to FT_RESULTS_DIR/<session>.json; test_z compares sessions.
"""
import json
import os
from dataclasses import dataclass, field

import pytest

from . import env, tasks
from .client import Client
from .server import Server, gpu_used_mib, start_ready

_TOK = {}


def tok():
    if "t" not in _TOK:
        _TOK["t"] = tasks.Tok()
    return _TOK["t"]


@dataclass
class Session:
    name: str
    cfg: dict
    server: Server
    c: Client
    tok: tasks.Tok
    rec: dict = field(default_factory=dict)

    @property
    def reference(self):
        return self.cfg.get("reference", False)


def save(name, rec):
    os.makedirs(env.RESULTS, exist_ok=True)
    with open(os.path.join(env.RESULTS, f"{name}.json"), "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False, default=str)


def load(name):
    p = os.path.join(env.RESULTS, f"{name}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def se(request):
    cfg = request.module.SESSION
    name = cfg["name"]
    s = start_ready(name, cfg["args"], cfg.get("reference", False))
    sess = Session(name, cfg, s, Client(s.url), tok())
    sess.rec.update(config={k: v for k, v in cfg.items()}, load_seconds=s.load_seconds,
                    gpu_mib_ready=gpu_used_mib(), stats_ready=sess.c.get("/v1/stats"),
                    cache_ready=sess.c.get("/v1/cache/status"))
    save(name, sess.rec)
    try:
        yield sess
        sess.rec.update(gpu_mib_end=gpu_used_mib(), stats_end=sess.c.get("/v1/stats"),
                        cache_end=sess.c.get("/v1/cache/status"))
    finally:
        save(name, sess.rec)
        s.stop()
