"""Shared test fixtures.

`cached_batch` memoises `simulator.generate.build_batch` to a gitignored
`.cache/` dir keyed by (n, seed). The generator's continuous cash-flow
back-simulation is the slow part of the suite; the batches are deterministic
given the seed, so caching them is safe and cuts a full run by ~2x.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import pytest
import yaml

# service/app.py reads this once at import time to decide whether to start its
# background auto-tick loop. Set before anything imports it (whichever test
# module happens first) so no stray background task ever touches a test's
# temp SQLite file after that test's fixtures have torn down.
os.environ.setdefault("RECOUP_TICK_INTERVAL_SECONDS", "0")

from simulator.generate import build_batch

_CACHE = Path(__file__).resolve().parent.parent / ".cache"
PRIORS = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "priors.yaml").read_text())


def _cached_build(n: int, seed: int):
    _CACHE.mkdir(exist_ok=True)
    prior_hash = hashlib.sha256(repr(sorted(PRIORS.items())).encode()).hexdigest()[:12]
    fp = _CACHE / f"batch_{n}_{seed}_{prior_hash}.pkl"
    if fp.exists():
        with fp.open("rb") as f:
            return pickle.load(f)
    out = build_batch(n, seed, PRIORS)
    with fp.open("wb") as f:
        pickle.dump(out, f)
    return out


@pytest.fixture(scope="session")
def cached_batch():
    return _cached_build
