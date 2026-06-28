"""S3a characterization golden for the ABC-METAL path. Runs the cyclic+metal+ABC-A dispatch path
through the deterministic CPU fake stack and pins the CURRENT (degenerate: closure-only fitness,
position-only frozen, bespoke result dict, no MetalHawk gate) baseline. S3a.5 routes through the
uniform restraints/gates/assembly behind XENO_SEQ_STAGE and REGOLDS to the corrected behaviour
(coordination rows + identity/chirality frozen + panel-selected dict), diff documented in the commit.

Regenerate with: XENO_REGOLD=1 PYTHONPATH=$PWD python -m pytest -k golden_metal -q
"""
from __future__ import annotations

import json

from xenodesign import dispatch
from xenodesign.config import resolve_config

from tests.test_characterization_goldens import _FakePred, _drop_runspecific, _load_or_regold
from tests.test_characterization_goldens_search import _abc_fakes, _echo_mpnn  # reuse S2 ABC fakes


# 4 coordinators: His 6/12/18/24, L/D/L/D (the 6UFA S2-symmetric tetrahedron), each with a liganding
# atom so the metal_coordination_rows native covalent @atom path is exercised.
_COORD = [
    (6, "H", "HIS", "L", "ND1"), (12, "H", "HIS", "D", "ND1"),
    (18, "H", "HIS", "L", "ND1"), (24, "H", "HIS", "D", "ND1"),
]


def test_golden_metal_abc(tmp_path, monkeypatch):
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")   # S3a.5c: the coordinated+gated+panel ABC-metal is the contract
    _abc_fakes(monkeypatch)
    cfg = resolve_config(
        "cyclic", target_type="metal", out_dir=str(tmp_path),
        cli_overrides={"use_pepmlm": False, "use_pll": False, "restraints_on": True,
                       "mixed_chirality": "A", "abc.cycles": 2, "abc.colony_size": 3,
                       "abc.scout_limit": 2, "abc.chai_eval_budget": 12,
                       "restraint.params": {"coord_residues": _COORD}})
    # restraints_on=True + metal: target_entities gates on the chai dist-restraint patch. Stub the
    # gate open on CPU (no chai import in the fake stack).
    monkeypatch.setattr("xenodesign.targets._metal_patch_verified", lambda: True, raising=True)
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("abc_metal", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden
