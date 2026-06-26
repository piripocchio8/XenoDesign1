# tests/test_secondary_structure.py
"""CPU tests for the forward SS-content + contact-count geometry helpers (lane A).

No DSSP dependency: helix fraction is a CA-pseudo-torsion heuristic. Synthetic coordinates:
an ideal alpha-helix (rise 1.5 Angstrom/res, ~100 deg/res rotation, radius 2.3) scores high
helix fraction; an extended/straight strand scores low. Contact counts are distance-cutoff
counts split intra- vs inter-chain by a chain-index array."""
import numpy as np
import pytest
from xenodesign.secondary_structure import (
    helix_fraction, ca_pseudo_torsions, count_contacts,
)


def _ideal_helix_ca(n, rise=1.5, turn_deg=100.0, radius=2.3):
    out = []
    for i in range(n):
        ang = np.deg2rad(turn_deg) * i
        out.append([radius * np.cos(ang), radius * np.sin(ang), rise * i])
    return np.array(out, dtype=float)


def _extended_ca(n, step=3.8):
    return np.array([[step * i, 0.0, 0.0] for i in range(n)], dtype=float)


def test_helix_fraction_high_for_ideal_helix():
    ca = _ideal_helix_ca(20)
    assert helix_fraction(ca) > 0.8


def test_helix_fraction_low_for_extended():
    ca = _extended_ca(20)
    assert helix_fraction(ca) < 0.2


def test_helix_fraction_high_for_LEFT_handed_D_helix():
    # D-peptides form LEFT-handed helices (mean CA pseudo-torsion ~ -50 deg). The detector must
    # be handedness-agnostic — a left-handed helix (turn_deg < 0) must also score high. Before the
    # 2026-06-16 |torsion| fix this returned ~0.0, silently disabling the SS-bias term for all-D
    # designs (the real binders are ~100% helical at -52 deg).
    ca_left = _ideal_helix_ca(20, turn_deg=-100.0)
    assert helix_fraction(ca_left) > 0.8
    # right- and left-handed ideal helices score the SAME (mirror images).
    assert helix_fraction(ca_left) == pytest.approx(helix_fraction(_ideal_helix_ca(20)))


def test_helix_fraction_in_unit_range():
    for ca in (_ideal_helix_ca(12), _extended_ca(12), _ideal_helix_ca(5)):
        f = helix_fraction(ca)
        assert 0.0 <= f <= 1.0


def test_helix_fraction_too_short_returns_zero():
    assert helix_fraction(_ideal_helix_ca(3)) == 0.0


def test_ca_pseudo_torsions_count():
    ca = _ideal_helix_ca(10)
    tors = ca_pseudo_torsions(ca)
    assert tors.shape == (10 - 3,)


def test_count_contacts_intra_inter_split():
    coords = np.array([
        [0.0, 0.0, 0.0],   # chain 0
        [4.0, 0.0, 0.0],   # chain 0  (close to res 0)
        [8.0, 0.0, 0.0],   # chain 0  (close to res 1)
        [9.0, 0.0, 0.0],   # chain 1  (close to res 2 -> inter contact)
        [40.0, 0.0, 0.0],  # chain 1  (isolated)
        [44.0, 0.0, 0.0],  # chain 1  (close to res 4)
    ])
    chain_index = np.array([0, 0, 0, 1, 1, 1])
    out = count_contacts(coords, chain_index, cutoff=5.0, min_seqsep=1)
    # intra pairs <5A, >=1 apart: chain0 (0,1)=4A,(1,2)=4A -> 2 ; chain1 (4,5)=4A -> 1 ; total 3
    assert out["intra"] == 3
    # inter pair: res2 @ x=8 and res3 @ x=9 = 1A -> 1
    assert out["inter"] == 1
    assert out["intra"] >= 0 and out["inter"] >= 0
