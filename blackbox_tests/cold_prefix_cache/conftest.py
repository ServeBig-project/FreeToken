import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from harness import Server  # noqa: E402


def server_fixture(label, args, **kwargs):
    @pytest.fixture  # function scope: one server on the GPU at a time
    def server():
        s = Server(label, args, **kwargs)
        try:
            yield s
        finally:
            s.close()
    return server


def assert_clean(c):
    assert not c.failures, "\n".join(str(f)[:600] for f in c.failures)
