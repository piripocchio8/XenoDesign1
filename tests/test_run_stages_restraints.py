"""S3a.2: build_run_restraints emits coordination + closure rows for the metal case (the rows ABC
currently lacks), reusing the @atom covalent grammar. Returns None when restraints are off."""
from __future__ import annotations

from xenodesign.config import resolve_config
from xenodesign.run_stages import build_run_restraints


_COORD = [(6, "H", "HIS", "L", "ND1"), (12, "H", "HIS", "D", "ND1")]


def test_build_run_restraints_metal_has_coordination_and_closure(tmp_path):
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False,
                                        "restraint.params": {"coord_residues": _COORD}})
    path = build_run_restraints(cfg, out_dir=tmp_path)
    assert path is not None
    text = path.read_text()
    # Native covalent @atom coordination rows (8cyo form): binder 'H6@ND1' <-> metal '@ZN'.
    assert "H6@ND1" in text and "@ZN" in text and "covalent" in text
    # Head-to-tail closure row present (a metal macrocycle closes by default).
    assert "head_to_tail" in text or "cyclic_closure" in text


def test_build_run_restraints_off_is_none(tmp_path):
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path),
                         cli_overrides={"restraints_on": False,
                                        "use_pepmlm": False,
                                        "restraint.params": {"coord_residues": _COORD}})
    assert build_run_restraints(cfg, out_dir=tmp_path) is None
