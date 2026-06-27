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


import xenodesign.classes.alpha as alpha_mod

_ALPHA_SEED = "ACDEFGHIKLMNPQRSTVWYG"   # fixed 21-mer (ends in the Gly anchor)
_ALPHA_TARGET = "GSHMKVLITGGAGFIGSHLVDRL"


def _alpha_fakes(monkeypatch):
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakePred(), lambda *a, **k: _FakePred()))
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "protein", "name": "target",
                                       "sequence": _ALPHA_TARGET, "chirality": "L"}], None, None))
    monkeypatch.setattr("xenodesign.seed.reflect_binder_in_complex_from_cif",
                        lambda *a, **k: np.zeros((3, 3)))
    monkeypatch.setattr(alpha_mod.Alpha, "seed",
                        lambda self, cfg, target_seq: __import__(
                            "xenodesign.classes.base", fromlist=["SeedSpec"]
                        ).SeedSpec(one_letter=_ALPHA_SEED))
    # Deterministic seq-update: re-emit the seed every iteration (no MPNN/GPU).
    monkeypatch.setattr(alpha_mod, "make_alpha_seq_update_fn",
                        lambda wrapper, **k: (lambda pred: _ALPHA_SEED))


def test_golden_alpha_greedy(tmp_path, monkeypatch):
    _alpha_fakes(monkeypatch)
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 2, "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False, "loop.backend": "ligandmpnn"})
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("alpha_greedy", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden
