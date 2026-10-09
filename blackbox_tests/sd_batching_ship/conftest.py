import os
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sd_batching_ship import env  # noqa: E402
from sd_batching_ship.client import Client, Tok, dump  # noqa: E402
from sd_batching_ship.server import Server, start_ready  # noqa: E402


@dataclass
class Session:
    name: str
    server: Server
    c: Client
    tok: Tok
    steps: int


@pytest.fixture(scope="module")
def srv(request):
    """One server per test module: SESSION = (name, model key, extra CLI args, expected max steps)."""
    name, model, extra, steps = request.module.SESSION
    attach = os.environ.get("FT_ATTACH_URL")  # reuse an already-ready server of this SESSION
    s = Server(name, []) if attach else start_ready(name, env.model_args(model) + extra)
    if attach:
        s.url = attach
    try:
        c = Client(s.url)
        dump(f"{name}_ready_stats", c.stats())
        dump(f"{name}_ready_cache", c.cache_status())
        yield Session(name, s, c, Tok(env.tokenizer_path(model)), steps)
        dump(f"{name}_final_stats", c.stats())
    finally:
        if not attach:
            s.stop()
