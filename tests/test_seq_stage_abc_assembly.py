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


def test_abc_metal_eval_predicts_under_coordination(tmp_path, monkeypatch):
    """Pin the headline: flag-ON ABC-metal feeds a COORDINATION restraint to every eval (not the
    closure-only legacy). Captured via a fitness that records its constraint file content."""
    seen = {}

    def _fake_make_fitness(backend, **k):
        rows = k.get("coord_restraint_rows")
        seen["coord_rows"] = rows
        def fitness(sequence, chirality_pattern):
            fitness.last_structure = None
            return float(len(set(sequence)))
        fitness.last_structure = None
        return fitness

    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    _abc_fakes(monkeypatch)
    # Override the fitness with our capturing version AFTER _abc_fakes so it wins.
    monkeypatch.setattr("xenodesign.abc.fitness.make_abc_fitness", _fake_make_fitness)
    monkeypatch.setattr("xenodesign.dispatch.make_abc_fitness", _fake_make_fitness, raising=False)
    monkeypatch.setattr("xenodesign.targets._metal_patch_verified", lambda: True, raising=True)
    dispatch.run_design(_metal_cfg(tmp_path))
    assert seen["coord_rows"] is not None and any("H6@ND1" in r for r in seen["coord_rows"])


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
