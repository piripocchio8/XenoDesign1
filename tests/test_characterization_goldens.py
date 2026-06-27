"""S0 characterization goldens: pin current CPU dispatch behavior before the S1 refactor.

Each test runs dispatch.run_design with a fixed seed through a deterministic CPU fake stack
(no GPU, no network) and asserts the report equals a committed golden JSON. Regenerate the
goldens (e.g. after an intentional behavior change) with:  XENO_REGOLD=1 pytest -k golden
The S1 parity tests reuse the SAME fakes + goldens, so any greedy-routing divergence shows up
here as a mismatched key.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from xenodesign import dispatch
from xenodesign.config import resolve_config

_GOLDEN_DIR = Path(__file__).parent / "golden"


class _FakePred:
    """Deterministic Prediction stand-in: the attributes loop/objective/referee read."""
    coords = np.zeros((3, 3))
    iptm = 0.5
    token_index = np.array([1, 1, 1])
    plddt = np.array([80.0, 80.0, 80.0])


def _drop_runspecific(report: dict) -> dict:
    """Strip keys that are inherently run-specific (not behavior)."""
    out = dict(report)
    for k in ("wall_time_s", "out_dir", "constraint_path"):
        out.pop(k, None)
    return out


def _load_or_regold(name: str, report: dict) -> dict:
    """Compare ``report`` against tests/golden/<name>.json, or (re)write it under XENO_REGOLD=1.

    Returns the golden dict to compare against. JSON round-trips the report so numpy scalars /
    tuples are normalized the same way on both sides (the on-disk golden is the source of truth).
    """
    path = _GOLDEN_DIR / f"{name}.json"
    normalized = json.loads(json.dumps(_drop_runspecific(report), default=str))
    if os.environ.get("XENO_REGOLD") == "1":
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalized, indent=2, sort_keys=True))
    if not path.exists():
        raise AssertionError(
            f"golden {path} missing; regenerate with XENO_REGOLD=1 pytest -k golden")
    return json.loads(path.read_text())


def test_golden_helper_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setenv("XENO_REGOLD", "1")
    got = _load_or_regold("_helper_selftest", {"a": 1, "wall_time_s": 9.9})
    assert got == {"a": 1}            # run-specific key dropped, value round-tripped
