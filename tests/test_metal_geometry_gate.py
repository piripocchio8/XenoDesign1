"""CPU tests for the MetalHawk metal-coordination geometry gate adapter.

Pure-CPU: exercises sphere-PDB construction, the (class_index, entropy) -> perplexity
threshold decision, stdout parsing, and the best-effort guards -- all WITHOUT MetalHawk
installed. A single live-MetalHawk smoke test is skipped unless its env + repo are
present (so the baseline CPU suite stays green on machines without MetalHawk).
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from xenodesign.eval.metal_geometry_gate import (
    DEFAULT_PERPLEXITY_THRESH,
    GEOMETRY_CLASSES,
    GateResult,
    build_sphere_pdb,
    decision_from_prediction,
    metal_geometry_gate,
    _parse_cif_atoms,
    _parse_metalhawk_stdout,
    _pdb_atom_line,
    _select_metal,
)
from xenodesign.config import GateConfig, resolve_config


# --- A tiny synthetic CIF: a Zn (chain A) with His-NE2 + Asp-OD2 + carbons. ----------
SYNTHETIC_CIF = """\
data_model
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
HETATM 1 Zn ZN LIG A 0.000 0.000 0.000 A
ATOM 2 N NE2 HIS B 2.030 0.000 0.000 B
ATOM 3 O OD2 ASP B 0.000 1.900 0.000 B
ATOM 4 N NE2 HIS B 0.000 0.000 2.100 B
ATOM 5 C CE1 HIS B 2.900 0.500 0.000 B
ATOM 6 C CB ALA B 12.000 12.000 12.000 B
#
"""


# --------------------------------------------------------------------------------------
# CIF parsing + metal selection
# --------------------------------------------------------------------------------------

def test_parse_cif_atoms_reads_loop():
    atoms = _parse_cif_atoms(SYNTHETIC_CIF)
    assert len(atoms) == 6
    assert atoms[0]["type_symbol"] == "Zn"
    assert atoms[1]["label_atom_id"] == "NE2"


def test_select_metal_by_element():
    atoms = _parse_cif_atoms(SYNTHETIC_CIF)
    m = _select_metal(atoms, "ZN")
    assert m is not None and m["type_symbol"] == "Zn"


def test_select_metal_auto_detects():
    atoms = _parse_cif_atoms(SYNTHETIC_CIF)
    m = _select_metal(atoms, None)
    assert m is not None and m["type_symbol"].upper() == "ZN"


def test_select_metal_returns_none_when_absent():
    atoms = _parse_cif_atoms(SYNTHETIC_CIF)
    assert _select_metal(atoms, "FE") is None


# --------------------------------------------------------------------------------------
# Sphere-PDB construction
# --------------------------------------------------------------------------------------

def test_build_sphere_pdb_columns_and_translation():
    pdb = build_sphere_pdb(SYNTHETIC_CIF, metal_element="ZN", radius=3.5)
    assert pdb is not None
    lines = [l for l in pdb.splitlines() if l.startswith("HETATM")]
    # Metal first, at the origin; element in strict cols 77-78; >=4 atoms within 3.5 A.
    assert lines[0][76:78] == "ZN"
    assert lines[0][30:38].strip() == "0.000"
    assert len(lines) >= 4
    # Donor element parsing (N/O) lands in cols 77-78.
    elems = {l[76:78].strip() for l in lines}
    assert "N" in elems and "O" in elems
    # The far ALA-CB (>10 A) is excluded by the 3.5 A radius.
    assert all("ALA" not in l for l in lines)


def test_build_sphere_pdb_none_without_metal():
    no_metal = SYNTHETIC_CIF.replace("Zn ZN LIG A 0.000 0.000 0.000",
                                     "C  CA LIG A 0.000 0.000 0.000")
    assert build_sphere_pdb(no_metal, metal_element="ZN") is None


def test_build_sphere_pdb_none_when_no_donors():
    # Metal alone (radius too small to catch any neighbour).
    assert build_sphere_pdb(SYNTHETIC_CIF, metal_element="ZN", radius=0.5) is None


def test_pdb_atom_line_strict_columns():
    line = _pdb_atom_line(1, name="ZN", resn="LIG", element="ZN",
                          x=0.0, y=0.0, z=0.0)
    assert line.startswith("HETATM")
    assert line[76:78] == "ZN"             # element cols 77-78
    assert line[30:38] == "%8.3f" % 0.0    # x cols 31-38


# --------------------------------------------------------------------------------------
# Decision logic: (class_index, entropy) -> perplexity threshold
# --------------------------------------------------------------------------------------

def test_decision_low_entropy_passes():
    # entropy ~0 -> perplexity ~1.0 -> clean geometry -> PASS.
    r = decision_from_prediction(class_index=2, entropy=0.0476)
    assert r.geometry == "TET"
    assert r.ok is True
    assert math.isclose(r.perplexity, math.exp(0.0476), rel_tol=1e-9)
    assert r.passed is True


def test_decision_high_entropy_fails():
    # Maximally confused 7-class entropy ln(7) -> perplexity 7 -> FAIL at default thresh.
    r = decision_from_prediction(class_index=6, entropy=math.log(7))
    assert r.geometry == "OCT"
    assert math.isclose(r.perplexity, 7.0, rel_tol=1e-9)
    assert r.passed is False


def test_decision_threshold_is_inclusive():
    thresh = 1.5
    entropy = math.log(thresh)  # perplexity == thresh exactly
    r = decision_from_prediction(class_index=0, entropy=entropy, threshold=thresh)
    assert math.isclose(r.perplexity, thresh, rel_tol=1e-12)
    assert r.passed is True  # perplexity <= threshold


def test_decision_custom_threshold_flips_result():
    entropy = math.log(1.4)  # perplexity 1.4
    assert decision_from_prediction(0, entropy, threshold=1.5).passed is True
    assert decision_from_prediction(0, entropy, threshold=1.2).passed is False


def test_decision_bad_class_index_is_not_ok():
    r = decision_from_prediction(class_index=99, entropy=0.0)
    assert r.geometry is None
    assert r.ok is False


def test_geometry_classes_order():
    assert GEOMETRY_CLASSES == ("LIN", "TRI", "TET", "SPL", "SQP", "TBP", "OCT")


# --------------------------------------------------------------------------------------
# MetalHawk stdout parsing
# --------------------------------------------------------------------------------------

def test_parse_metalhawk_stdout_picks_json_line():
    out = "some noise\n{\"class_index\": 2, \"entropy\": 0.0476}\n"
    cls, ent = _parse_metalhawk_stdout(out)
    assert cls == 2 and math.isclose(ent, 0.0476)


def test_parse_metalhawk_stdout_raises_without_json():
    with pytest.raises(ValueError):
        _parse_metalhawk_stdout("no json here\n")


# --------------------------------------------------------------------------------------
# Best-effort guards: the gate must NEVER raise.
# --------------------------------------------------------------------------------------

def test_gate_passthrough_when_metalhawk_dir_missing(tmp_path, monkeypatch):
    # Ensure neither the arg nor the env var resolves -> deterministic pass-through.
    monkeypatch.delenv("METALHAWK_DIR", raising=False)
    cif = tmp_path / "x.cif"
    cif.write_text(SYNTHETIC_CIF)
    r = metal_geometry_gate(cif, metal_element="ZN", metalhawk_dir=None,
                            metalhawk_env="metalhawk")
    assert isinstance(r, GateResult)
    assert r.ok is False and r.passed is True  # pass-through, no veto
    assert r.error is not None


def test_gate_passthrough_when_no_metal(tmp_path, monkeypatch):
    # Point at a real dir so we get past the dir check, then fail on "no metal".
    cif = tmp_path / "x.cif"
    cif.write_text(SYNTHETIC_CIF.replace("Zn ZN", "C  CA"))
    r = metal_geometry_gate(cif, metal_element="ZN", metalhawk_dir=str(tmp_path))
    assert r.ok is False and r.passed is True
    assert "sphere" in (r.error or "").lower()


def test_gate_passthrough_on_runner_failure(tmp_path):
    # Real metal + a metalhawk_dir that exists but has no model -> subprocess fails,
    # gate still returns a pass-through (never raises).
    cif = tmp_path / "x.cif"
    cif.write_text(SYNTHETIC_CIF)
    r = metal_geometry_gate(cif, metal_element="ZN", metalhawk_dir=str(tmp_path),
                            metalhawk_env=None)  # env=None -> plain `python` runner
    assert isinstance(r, GateResult)
    assert r.ok is False and r.passed is True


# --------------------------------------------------------------------------------------
# Config wiring: off by default, tunable via dotted CLI override.
# --------------------------------------------------------------------------------------

def test_metal_geometry_gate_off_by_default():
    assert GateConfig().metal_geometry is False
    assert GateConfig().metal_perplexity_thresh == 1.5


def test_metal_geometry_gate_enable_via_override():
    cfg = resolve_config("cyclic", target_type="metal",
                         cli_overrides={"gates.metal_geometry": True,
                                        "gates.metal_perplexity_thresh": 1.2})
    assert cfg.gates.metal_geometry is True
    assert cfg.gates.metal_perplexity_thresh == 1.2


# --------------------------------------------------------------------------------------
# Live MetalHawk smoke test (skipped unless its env + repo are installed).
# --------------------------------------------------------------------------------------

def _metalhawk_available() -> bool:
    mh_dir = os.environ.get("METALHAWK_DIR", str(Path.home() / "tools" / "MetalHawk"))
    if not (Path(mh_dir) / "models" / "HPO_CSD_CSD_CV.model").is_file():
        return False
    if shutil.which("micromamba") is None:
        return False
    try:
        r = subprocess.run(
            ["micromamba", "run", "-n", "metalhawk", "python", "-c", "import sklearn"],
            capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _metalhawk_available(),
                    reason="MetalHawk env/repo not installed")
def test_live_metalhawk_on_synthetic_zn(tmp_path):
    mh_dir = os.environ.get("METALHAWK_DIR", str(Path.home() / "tools" / "MetalHawk"))
    cif = tmp_path / "zn.cif"
    cif.write_text(SYNTHETIC_CIF)
    r = metal_geometry_gate(cif, metal_element="ZN", metalhawk_dir=mh_dir,
                            metalhawk_env="metalhawk")
    assert r.ok is True
    assert r.geometry in GEOMETRY_CLASSES
    assert r.entropy is not None and r.perplexity is not None
    assert math.isclose(r.perplexity, math.exp(r.entropy), rel_tol=1e-6)
