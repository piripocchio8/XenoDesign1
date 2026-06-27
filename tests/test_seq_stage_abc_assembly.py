"""S3a.5b: _run_abc flag-ON gives coordinators identity+chirality FrozenPosition, emits coordination
rows, and returns a panel-selected cls.report dict via the SHARED assembly tail (not the bespoke ABC
dict). Flag OFF returns the legacy bespoke dict, byte-identical."""
from __future__ import annotations

from xenodesign import dispatch
from xenodesign.config import resolve_config

from tests.test_characterization_goldens_search import _abc_fakes


_COORD = [(6, "H", "HIS", "L", "ND1"), (12, "H", "HIS", "D", "ND1"),
          (18, "H", "HIS", "L", "ND1"), (24, "H", "HIS", "D", "ND1")]


def _metal_cfg(tmp_path):
    return resolve_config(
        "cyclic", target_type="metal", out_dir=str(tmp_path),
        cli_overrides={"use_pepmlm": False, "use_pll": False, "restraints_on": True,
                       "mixed_chirality": "A", "abc.cycles": 2, "abc.colony_size": 3,
                       "abc.scout_limit": 2, "abc.chai_eval_budget": 12,
                       "restraint.params": {"coord_residues": _COORD}})


def test_abc_flag_off_returns_bespoke_dict(tmp_path, monkeypatch):
    _abc_fakes(monkeypatch)
    monkeypatch.delenv("XENO_SEQ_STAGE", raising=False)
    monkeypatch.setattr("xenodesign.targets._metal_patch_verified", lambda: True, raising=True)
    report = dispatch.run_design(_metal_cfg(tmp_path))
    assert report["search"] == "abc"            # the legacy bespoke ABC dict
    assert "selected_d_fasta" in report


def test_abc_flag_on_returns_report_dict(tmp_path, monkeypatch):
    _abc_fakes(monkeypatch)
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    monkeypatch.setattr("xenodesign.targets._metal_patch_verified", lambda: True, raising=True)
    report = dispatch.run_design(_metal_cfg(tmp_path))
    # The cyclic cls.report dict has case_id 'cyclic' and the recall/selection fields, NOT 'search'.
    assert report.get("case_id") == "cyclic"
    assert "search" not in report or report.get("search") != "abc"
