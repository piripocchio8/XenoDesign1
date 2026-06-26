"""Pure-numpy geometric primitives: dihedral angles, signed chiral volume, Kabsch RMSD."""
from __future__ import annotations

import numpy as np


def dihedral(p0, p1, p2, p3) -> float:
    """Signed dihedral angle (degrees) about the p1-p2 bond, range (-180, 180]."""
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def signed_chiral_volume(n, ca, c, cb) -> float:
    """Signed volume of the (N-CA, C-CA, CB-CA) frame. Sign encodes L/D chirality."""
    n, ca, c, cb = (np.asarray(p, dtype=float) for p in (n, ca, c, cb))
    v1 = n - ca
    v2 = c - ca
    v3 = cb - ca
    return float(np.dot(np.cross(v1, v2), v3))


def valence_angle(a, b, c) -> float:
    """Angle (degrees) at vertex ``b`` of the triangle a-b-c, range [0, 180]."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom <= 0:
        return 0.0
    cos = float(np.dot(v1, v2) / denom)
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def amide_omega_score(ca_prev, c_prev, n_next, ca_next) -> float:
    """PLANARITY of the (CA_i, C_i, N_j, CA_j) peptide bond, normalised to [0, 1].

    The peptide-bond dihedral omega = CA_i-C_i-N_j-CA_j is ~180 deg (trans) or ~0 deg (cis)
    when the amide plane is FLAT, and ~+/-90 deg when maximally twisted. This term scores
    planarity (closeness to 0 or 180), NOT trans-ness: ``1 - |sin(omega)|`` is 1 at omega in
    {0, 180} and 0 at omega = +/-90. Used for the head-to-tail closure bond C(L)-N(1) and is
    geometry-only (cheap, CPU-pure). Most natural amide bonds are trans, so a flat closure here
    is the desired signal.
    """
    omega = dihedral(ca_prev, c_prev, n_next, ca_next)
    return float(1.0 - abs(np.sin(np.radians(omega))))


def angle_deviation_score(angles, ideals, tol_deg: float = 10.0) -> float:
    """Normalised backbone valence-angle sanity in [0, 1] from per-angle deviations.

    For each (angle, ideal) pair, the per-angle penalty is ``min(1, |angle-ideal|/tol_deg)``;
    the score is ``1 - mean(penalty)``. A deviation of ``tol_deg`` zeroes that angle's credit,
    so the term is at full credit when every backbone valence angle sits within ``tol_deg`` of
    ideal and decays linearly past it (clamped to [0, 1]). An EMPTY input is a neutral 1.0
    (nothing to fault). Pure / CPU-testable; the caller supplies measured + ideal angles.
    """
    angles = list(angles)
    ideals = list(ideals)
    if len(angles) != len(ideals):
        raise ValueError("angles and ideals must have equal length")
    if not angles:
        return 1.0
    if tol_deg <= 0:
        raise ValueError("tol_deg must be > 0")
    penalties = [min(1.0, abs(a - i) / tol_deg) for a, i in zip(angles, ideals)]
    return float(max(0.0, 1.0 - sum(penalties) / len(penalties)))


def kabsch_rmsd(a, b) -> float:
    """RMSD between point sets a and b after optimal rigid superposition (Kabsch)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    h = ac.T @ bc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    a_aligned = ac @ rot.T
    diff = a_aligned - bc
    return float(np.sqrt((diff * diff).sum() / a.shape[0]))
