"""S1 parity gate: the greedy path routed through SequenceUpdate (XENO_SEQ_STAGE=1)
reproduces the S0 goldens byte-for-byte.

Reuses the fake stack + _load_or_regold helper from test_characterization_goldens so there
is exactly ONE source of truth for the fakes and ONE set of committed goldens. If a future
change makes the stage diverge from the legacy path, these tests will fail — that is the
intent. Regolding is WRONG here; instead, investigate and restore byte-parity (or update both
the S0 goldens AND the stage goldens together with a documented intentional change).
"""
from __future__ import annotations

import json

import pytest

from xenodesign import dispatch
from xenodesign.config import resolve_config

# Re-use the S0 fakes and the load-or-regold helper — do NOT duplicate.
from tests.test_characterization_goldens import (
    _alpha_fakes,
    _nonalpha_fakes,
    _cyclic_metal_fakes,
    _cyclic_metal_coord_residues,
    _drop_runspecific,
    _load_or_regold,
)


# ---------------------------------------------------------------------------
# S1.8: alpha greedy parity
# ---------------------------------------------------------------------------

def test_stage_parity_alpha_greedy(tmp_path, monkeypatch):
    """Alpha greedy with XENO_SEQ_STAGE=1 must reproduce the committed alpha_greedy.json golden."""
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    _alpha_fakes(monkeypatch)
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 2, "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False, "loop.backend": "ligandmpnn"})
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("alpha_greedy", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden, \
        "Stage ON diverges from S0 alpha_greedy golden — stage introduced a regression"


# ---------------------------------------------------------------------------
# S1.8: non_alpha greedy parity
# ---------------------------------------------------------------------------

def test_stage_parity_nonalpha_greedy(tmp_path, monkeypatch):
    """non_alpha greedy with XENO_SEQ_STAGE=1 must reproduce the committed nonalpha_greedy.json golden."""
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    _nonalpha_fakes(monkeypatch)
    cfg = resolve_config("non_alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 2, "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False, "loop.backend": "ligandmpnn"})
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("nonalpha_greedy", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden, \
        "Stage ON diverges from S0 nonalpha_greedy golden — stage introduced a regression"


# ---------------------------------------------------------------------------
# S1.8: cyclic-metal greedy parity
# ---------------------------------------------------------------------------

def test_stage_parity_cyclic_metal_greedy(tmp_path, monkeypatch):
    """cyclic-metal greedy with XENO_SEQ_STAGE=1 must reproduce the committed cyclic_metal_greedy.json golden."""
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    _cyclic_metal_fakes(monkeypatch, tmp_path)
    cfg = resolve_config(
        "cyclic", target_type="metal", out_dir=str(tmp_path),
        cli_overrides={"loop.iters": 2, "use_pepmlm": False, "use_pll": False,
                       "restraints_on": False, "loop.backend": "ligandmpnn",
                       "mixed_chirality": "none",
                       "restraint.params": {"coord_residues": _cyclic_metal_coord_residues()}})
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("cyclic_metal_greedy", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden, \
        "Stage ON diverges from S0 cyclic_metal_greedy golden — stage introduced a regression"
