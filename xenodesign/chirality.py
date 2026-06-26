"""Chirality metrics: per-residue chiral-volume sign and violation fraction.

Convention is derived (not hard-coded) from an ideal L-alanine frame so the sign
convention always matches `geometry.signed_chiral_volume`.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from xenodesign.geometry import signed_chiral_volume, dihedral

_IDEAL_L = {
    "N": np.array([-0.525, 1.363, 0.000]),
    "CA": np.array([0.000, 0.000, 0.000]),
    "C": np.array([1.526, 0.000, 0.000]),
    "CB": np.array([-0.529, -0.774, -1.205]),
}

L_REFERENCE_SIGN: int = 1 if signed_chiral_volume(
    _IDEAL_L["N"], _IDEAL_L["CA"], _IDEAL_L["C"], _IDEAL_L["CB"]
) > 0 else -1


def expected_sign(chirality_label: str) -> int:
    """Expected chiral-volume sign for the given chirality label ('L' or 'D')."""
    label = chirality_label.upper()
    if label == "L":
        return L_REFERENCE_SIGN
    if label == "D":
        return -L_REFERENCE_SIGN
    raise ValueError(f"chirality_label must be 'L' or 'D', got {chirality_label!r}")


def is_chirality_violation(n, ca, c, cb, chirality_label: str, epsilon: float = 0.02) -> bool:
    """A stereocenter is a violation iff its sign is wrong AND |volume| > epsilon.

    Near-planar centers (|volume| <= epsilon) are not penalized (spec §3).
    """
    v = signed_chiral_volume(n, ca, c, cb)
    if abs(v) <= epsilon:
        return False
    sign = 1 if v > 0 else -1
    return sign != expected_sign(chirality_label)


def chirality_violation_fraction(
    residues: Sequence[Mapping[str, np.ndarray]],
    labels: Sequence[str],
    epsilon: float = 0.02,
) -> float:
    """Fraction of stereocenters whose chirality sign is wrong.

    Each residue is a mapping with keys 'N','CA','C','CB'. Residues lacking 'CB'
    (e.g. glycine) are skipped (not stereocenters).
    """
    if len(residues) != len(labels):
        raise ValueError("residues and labels must have equal length")
    total = 0
    violations = 0
    for res, label in zip(residues, labels):
        if "CB" not in res or res["CB"] is None:
            continue
        total += 1
        if is_chirality_violation(res["N"], res["CA"], res["C"], res["CB"], label, epsilon):
            violations += 1
    if total == 0:
        return 0.0
    return violations / total


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees (handles wrap)."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def phi_psi_violation(phi, psi, ref_phi, ref_psi, tol_deg: float = 25.0) -> bool:
    """True if (phi, psi) deviates from the reference by more than tol_deg on either axis."""
    return (
        _angular_diff_deg(phi, ref_phi) > tol_deg
        or _angular_diff_deg(psi, ref_psi) > tol_deg
    )


def backbone_torsions(residues: Sequence[Mapping[str, object]]):
    """Compute (phi, psi) arrays for a chain of residues with 'N','CA','C' coords.

    phi[i] = dihedral(C[i-1], N[i], CA[i], C[i]); phi[0] = NaN.
    psi[i] = dihedral(N[i], CA[i], C[i], N[i+1]); psi[-1] = NaN.
    """
    n = len(residues)
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    for i in range(n):
        if i > 0:
            phi[i] = dihedral(
                residues[i - 1]["C"], residues[i]["N"], residues[i]["CA"], residues[i]["C"]
            )
        if i < n - 1:
            psi[i] = dihedral(
                residues[i]["N"], residues[i]["CA"], residues[i]["C"], residues[i + 1]["N"]
            )
    return phi, psi
