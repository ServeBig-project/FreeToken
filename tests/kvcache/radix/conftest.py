"""A deterministic clock for every test here: LRU order assertions must not depend on the
resolution of the wall clock the tree may stamp nodes with."""
from __future__ import annotations

import itertools
import time

import pytest


@pytest.fixture(autouse=True)
def det_clock(monkeypatch):
    monkeypatch.setattr(time, "monotonic_ns", itertools.count(1_000_000_001).__next__)
