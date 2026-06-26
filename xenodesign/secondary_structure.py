# xenodesign/secondary_structure.py
"""Forward secondary-structure-content + contact-count geometry (lane A_scorer).

Pure-numpy, DSSP-free helpers computed from predicted CA coordinates, used to build the
OPTIONAL, default-weight-0.0 SS / contact terms in scorer.design_score (and the per-case
anti-alpha / helix-reward SS-bias term). No torch / chai / GPU; CPU-unit-tested.

Helix detection uses a CA pseudo-torsion heuristic: an ideal alpha-helix has a characteristic
CA pseudo-dihedral of magnitude ~50 deg and a CA(i)..CA(i+3) distance near 5.4 Angstrom. A
residue is counted 'helical' when BOTH gates hold; helix_fraction is the fraction of scorable
residues that pass. This is a coarse forward proxy (not DSSP), sufficient for steering selection
toward / away from helix.

HANDEDNESS-AGNOSTIC (fix 2026-06-16): the gate is on the ABSOLUTE pseudo-dihedral |tors|~50, so it
detects BOTH a right-handed L-helix (~+50 deg) AND a left-handed D-helix (~-50 deg). The earlier
+50-only gate read 0.0 for every D-peptide (D-helices sit near -52 deg), which silently made the
per-case SS-bias selection term (P1c) inert for the all-D alpha binder — the designs were in fact
100% helical (mean CA pseudo-torsion -52 deg).
"""
from __future__ import annotations

import numpy as np

from xenodesign.geometry import dihedral

_HELIX_TORSION_CENTER = 50.0   # deg, ideal CA pseudo-dihedral
_HELIX_TORSION_TOL = 35.0      # deg, +/- window
_HELIX_I3_DIST = 5.4           # Angstrom, ideal CA(i)..CA(i+3) separation
_HELIX_I3_TOL = 1.6            # Angstrom, +/- window


def ca_pseudo_torsions(ca: np.ndarray) -> np.ndarray:
    """CA pseudo-dihedral angles (deg) over consecutive CA quadruples.

    Returns an array of length len(ca) - 3 (empty if fewer than 4 CA atoms).
    """
    ca = np.asarray(ca, dtype=float)
    n = ca.shape[0]
    if n < 4:
        return np.zeros((0,), dtype=float)
    return np.array(
        [dihedral(ca[i], ca[i + 1], ca[i + 2], ca[i + 3]) for i in range(n - 3)],
        dtype=float,
    )


def helix_fraction(ca: np.ndarray) -> float:
    """Fraction of residues in an alpha-helical CA conformation, in [0, 1].

    A quadruple i..i+3 is helical when its pseudo-dihedral is within _HELIX_TORSION_TOL of
    _HELIX_TORSION_CENTER AND the CA(i)..CA(i+3) distance is within _HELIX_I3_TOL of
    _HELIX_I3_DIST. Returns 0.0 for chains too short to score.
    """
    ca = np.asarray(ca, dtype=float)
    n = ca.shape[0]
    if n < 4:
        return 0.0
    tors = ca_pseudo_torsions(ca)
    i3 = np.linalg.norm(ca[3:] - ca[:-3], axis=1)   # CA(i)..CA(i+3), length n-3
    # |tors| so both right-handed L (~+50) and left-handed D (~-50) helices are detected.
    tors_ok = np.abs(np.abs(tors) - _HELIX_TORSION_CENTER) <= _HELIX_TORSION_TOL
    dist_ok = np.abs(i3 - _HELIX_I3_DIST) <= _HELIX_I3_TOL
    helical = tors_ok & dist_ok
    return float(helical.sum()) / float(len(helical))


def count_contacts(
    coords: np.ndarray,
    chain_index: np.ndarray,
    cutoff: float = 8.0,
    min_seqsep: int = 1,
) -> dict:
    """Count residue-residue contacts (CA within cutoff), split intra- vs inter-chain.

    Args:
        coords: (L, 3) CA coordinates.
        chain_index: (L,) integer chain id per residue.
        cutoff: contact distance threshold (Angstrom).
        min_seqsep: minimum |i - j| index separation for INTRA-chain pairs (excludes trivial
            sequence neighbours when > 1). Inter-chain pairs are always counted regardless of
            sequence separation.

    Returns:
        {"intra": int, "inter": int} — unique unordered pairs in contact.
    """
    coords = np.asarray(coords, dtype=float)
    chain_index = np.asarray(chain_index)
    L = coords.shape[0]
    intra = inter = 0
    for i in range(L):
        for j in range(i + 1, L):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d >= cutoff:
                continue
            if chain_index[i] == chain_index[j]:
                if (j - i) >= min_seqsep:
                    intra += 1
            else:
                inter += 1
    return {"intra": intra, "inter": inter}
