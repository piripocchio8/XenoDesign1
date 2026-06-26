import numpy as np
import pytest
from xenodesign.mirror import (
    L_TO_D,
    D_TO_L,
    reflect_coords,
    remap_chirality,
    mirror_residue_codes,
)


def test_l_to_d_has_nineteen_entries_and_is_invertible():
    assert len(L_TO_D) == 19  # 19 chiral standard AAs (GLY is achiral)
    for l, d in L_TO_D.items():
        assert D_TO_L[d] == l


def test_reflect_twice_is_identity():
    rng = np.random.RandomState(0)
    x = rng.rand(8, 3)
    assert np.allclose(reflect_coords(reflect_coords(x)), x)


def test_reflect_negates_chosen_axis_only():
    x = np.array([[1.0, 2.0, 3.0]])
    r = reflect_coords(x, axis=2)
    assert np.allclose(r, [[1.0, 2.0, -3.0]])


def test_reflect_does_not_mutate_input():
    x = np.array([[1.0, 2.0, 3.0]])
    _ = reflect_coords(x, axis=0)
    assert np.allclose(x, [[1.0, 2.0, 3.0]])


def test_remap_chirality_round_trip():
    assert remap_chirality("ALA") == "DAL"
    assert remap_chirality("DAL") == "ALA"


def test_remap_chirality_passthrough_for_achiral():
    assert remap_chirality("GLY") == "GLY"


def test_mirror_residue_codes_list():
    assert mirror_residue_codes(["ALA", "GLY", "SER"]) == ["DAL", "GLY", "DSN"]


from xenodesign.mirror import mirror_discrepancy


def test_mirror_discrepancy_zero_for_exact_mirror():
    rng = np.random.RandomState(2)
    a = rng.rand(15, 3)
    b = reflect_coords(a, axis=0)  # b is the exact mirror of a
    assert mirror_discrepancy(a, b, axis=0) == pytest.approx(0.0, abs=1e-9)


def test_mirror_discrepancy_invariant_to_rigid_move_of_twin():
    rng = np.random.RandomState(3)
    a = rng.rand(15, 3)
    b = reflect_coords(a, axis=0)
    theta = 0.5
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    b_moved = b @ rot.T + np.array([3.0, 1.0, -2.0])
    assert mirror_discrepancy(a, b_moved, axis=0) == pytest.approx(0.0, abs=1e-9)


def test_mirror_discrepancy_positive_for_nonmirror():
    rng = np.random.RandomState(4)
    a = rng.rand(15, 3)
    b = rng.rand(15, 3)
    assert mirror_discrepancy(a, b, axis=0) > 0.1
