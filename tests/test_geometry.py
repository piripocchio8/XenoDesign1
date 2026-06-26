import numpy as np
import pytest
from xenodesign.geometry import dihedral, signed_chiral_volume, kabsch_rmsd
from tests.conftest import IDEAL_L_ALA


def test_dihedral_plus_90():
    # Classic +90 degree dihedral construction.
    p0 = [0, 1, 0]
    p1 = [0, 0, 0]
    p2 = [1, 0, 0]
    p3 = [1, 0, 1]
    assert dihedral(p0, p1, p2, p3) == pytest.approx(90.0, abs=1e-6)


def test_dihedral_sign_flips_under_reflection():
    p0 = np.array([0, 1, 0]); p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0]); p3 = np.array([1, 0, 1])
    refl = np.diag([1, 1, -1])  # reflect z
    d = dihedral(p0, p1, p2, p3)
    dr = dihedral(p0 @ refl, p1 @ refl, p2 @ refl, p3 @ refl)
    assert dr == pytest.approx(-d, abs=1e-6)


def test_signed_chiral_volume_nonzero_for_ideal_L():
    v = signed_chiral_volume(
        IDEAL_L_ALA["N"], IDEAL_L_ALA["CA"], IDEAL_L_ALA["C"], IDEAL_L_ALA["CB"]
    )
    assert abs(v) > 1.0


def test_signed_chiral_volume_negates_under_reflection():
    refl = np.diag([1.0, 1.0, -1.0])
    a = IDEAL_L_ALA
    v = signed_chiral_volume(a["N"], a["CA"], a["C"], a["CB"])
    vr = signed_chiral_volume(
        a["N"] @ refl, a["CA"] @ refl, a["C"] @ refl, a["CB"] @ refl
    )
    assert vr == pytest.approx(-v, abs=1e-9)


def test_kabsch_rmsd_zero_for_identical():
    x = np.random.RandomState(0).rand(10, 3)
    assert kabsch_rmsd(x, x.copy()) == pytest.approx(0.0, abs=1e-9)


def test_kabsch_rmsd_invariant_to_rotation_translation():
    rng = np.random.RandomState(1)
    x = rng.rand(12, 3)
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    y = x @ rot.T + np.array([5.0, -2.0, 1.0])
    assert kabsch_rmsd(x, y) == pytest.approx(0.0, abs=1e-9)
