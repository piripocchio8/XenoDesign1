"""CPU tests for the cyclic intramolecular-objective geometry helpers (no-target mode, T-none).

Pure numpy: synthetic coordinates exercise each geometry term with no GPU/CIF. Covers the
amide-omega planarity score (planar trans vs twisted), the valence-angle term (ideal vs
distorted), and the combined backbone-geometry score normaliser. The CIF/chirality/pLDDT
terms that need a parsed structure are exercised through the cyclic objective test with a
synthetic in-memory ``Prediction`` (see test_classes_cyclic_notarget.py).
"""
import numpy as np
import pytest

from xenodesign.geometry import (
    valence_angle,
    amide_omega_score,
    angle_deviation_score,
)


# ── valence_angle ────────────────────────────────────────────────────────────

def test_valence_angle_right_angle():
    # b at origin; a along +x, c along +y -> 90 deg.
    assert valence_angle([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(90.0, abs=1e-6)


def test_valence_angle_straight():
    assert valence_angle([1, 0, 0], [0, 0, 0], [-1, 0, 0]) == pytest.approx(180.0, abs=1e-6)


def test_valence_angle_tetrahedral():
    a = [1.0, 1.0, 1.0]
    c = [1.0, -1.0, -1.0]
    assert valence_angle(a, [0, 0, 0], c) == pytest.approx(109.4712, abs=1e-3)


# ── amide_omega_score (trans peptide bond, omega ~ +/-180) ───────────────────

def _omega_atoms(omega_deg):
    """CA_i, C_i, N_j, CA_j placed so the C-N dihedral about C-N equals omega_deg.

    Build the standard 4-point dihedral frame: bond axis along x (C->N), CA_i in the xy
    plane on the negative side, CA_j rotated by omega about the axis.
    """
    c = np.array([0.0, 0.0, 0.0])
    n = np.array([1.33, 0.0, 0.0])           # C->N peptide bond ~1.33 A
    ca_i = c + np.array([-0.75, 0.9, 0.0])   # CA_i off the axis (defines reference plane)
    th = np.radians(omega_deg)
    # CA_j placed off N by a vector at angle omega about the C->N axis (x).
    ca_j = n + np.array([0.75, 0.9 * np.cos(th), 0.9 * np.sin(th)])
    return ca_i, c, n, ca_j


def test_amide_omega_score_planar_trans_is_one():
    ca_i, c, n, ca_j = _omega_atoms(180.0)
    assert amide_omega_score(ca_i, c, n, ca_j) == pytest.approx(1.0, abs=1e-6)


def test_amide_omega_score_cis_also_planar():
    # cis (omega ~ 0) is planar too; the term scores planarity, not trans-ness.
    ca_i, c, n, ca_j = _omega_atoms(0.0)
    assert amide_omega_score(ca_i, c, n, ca_j) == pytest.approx(1.0, abs=1e-6)


def test_amide_omega_score_twisted_is_low():
    ca_i, c, n, ca_j = _omega_atoms(90.0)   # maximally non-planar
    assert amide_omega_score(ca_i, c, n, ca_j) == pytest.approx(0.0, abs=1e-6)


def test_amide_omega_score_monotonic():
    s10 = amide_omega_score(*_omega_atoms(170.0))
    s40 = amide_omega_score(*_omega_atoms(140.0))
    assert 1.0 >= s10 > s40 >= 0.0


# ── angle_deviation_score (normalised valence-angle sanity) ──────────────────

def test_angle_deviation_score_ideal_is_one():
    angles = [111.0, 110.0, 121.0]
    ideals = [111.0, 110.0, 121.0]
    assert angle_deviation_score(angles, ideals, tol_deg=10.0) == pytest.approx(1.0)


def test_angle_deviation_score_distorted_is_low():
    angles = [90.0, 90.0, 150.0]    # ~20-30 deg off each
    ideals = [111.0, 110.0, 121.0]
    s = angle_deviation_score(angles, ideals, tol_deg=10.0)
    assert 0.0 <= s < 0.3


def test_angle_deviation_score_clamps_at_zero():
    angles = [0.0]
    ideals = [120.0]
    assert angle_deviation_score(angles, ideals, tol_deg=10.0) == pytest.approx(0.0)


def test_angle_deviation_score_empty_is_one():
    # No angles to judge -> nothing wrong -> perfect (neutral) score.
    assert angle_deviation_score([], [], tol_deg=10.0) == pytest.approx(1.0)
