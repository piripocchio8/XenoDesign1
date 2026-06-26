import numpy as np
import pytest
from xenodesign.chirality import (
    L_REFERENCE_SIGN,
    expected_sign,
    is_chirality_violation,
    chirality_violation_fraction,
)
from tests.conftest import IDEAL_L_ALA

REFL_Z = np.diag([1.0, 1.0, -1.0])


def _mirror(d):
    return {k: v @ REFL_Z for k, v in d.items()}


def test_reference_sign_is_plus_or_minus_one():
    assert L_REFERENCE_SIGN in (1, -1)


def test_ideal_L_is_not_a_violation_when_labeled_L():
    a = IDEAL_L_ALA
    assert is_chirality_violation(a["N"], a["CA"], a["C"], a["CB"], "L") is False


def test_ideal_L_is_a_violation_when_labeled_D():
    a = IDEAL_L_ALA
    assert is_chirality_violation(a["N"], a["CA"], a["C"], a["CB"], "D") is True


def test_mirrored_L_is_not_a_violation_when_labeled_D():
    a = _mirror(IDEAL_L_ALA)
    assert is_chirality_violation(a["N"], a["CA"], a["C"], a["CB"], "D") is False


def test_near_planar_center_is_not_penalized():
    # CB coplanar with N/CA/C -> |volume| ~ 0 -> not a violation regardless of label.
    n = np.array([-1.0, 1.0, 0.0]); ca = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0]); cb = np.array([0.0, -1.0, 0.0])
    assert is_chirality_violation(n, ca, c, cb, "D", epsilon=0.02) is False


def test_expected_sign_opposite_for_L_and_D():
    assert expected_sign("L") == -expected_sign("D")


def test_violation_fraction_counts_only_real_violations():
    a = IDEAL_L_ALA
    # 3 residues labeled L (all correct), 1 labeled D (a real violation).
    residues = [a, a, a, a]
    labels = ["L", "L", "L", "D"]
    frac = chirality_violation_fraction(residues, labels)
    assert frac == pytest.approx(0.25)


from xenodesign.chirality import backbone_torsions, phi_psi_violation


def test_phi_psi_violation_within_tolerance():
    assert phi_psi_violation(100.0, -40.0, 90.0, -45.0, tol_deg=25.0) is False


def test_phi_psi_violation_outside_tolerance():
    assert phi_psi_violation(100.0, -40.0, 60.0, -45.0, tol_deg=25.0) is True


def test_phi_psi_violation_wraps_around_180():
    # 179 vs -179 are 2 degrees apart, not 358.
    assert phi_psi_violation(179.0, 0.0, -179.0, 0.0, tol_deg=25.0) is False


def test_backbone_torsions_recovers_known_phi():
    # Place C(i-1), N(i), CA(i), C(i) to make phi(i) = +90 (see geometry test).
    res0 = {"N": [0, 2, 0], "CA": [0, 3, 0], "C": [0, 1, 0]}      # C(0) at (0,1,0)
    res1 = {"N": [0, 0, 0], "CA": [1, 0, 0], "C": [1, 0, 1]}      # +90 construction
    res2 = {"N": [2, 0, 1], "CA": [2, 1, 1], "C": [2, 2, 1]}
    phi, psi = backbone_torsions([res0, res1, res2])
    assert phi[1] == pytest.approx(90.0, abs=1e-6)
    # First residue has no phi, last residue has no psi -> NaN sentinels.
    assert np.isnan(phi[0])
    assert np.isnan(psi[-1])
