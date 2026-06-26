"""Mirror operations for heterochiral design.

L<->D three-letter code map mirrors chai_lab's D_partners
(chai_lab/data/parsing/structure/sequence.py). GLY is achiral (no D form).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from xenodesign.geometry import kabsch_rmsd

# Standard L three-letter code -> Chai D three-letter code.
L_TO_D = {
    "ALA": "DAL", "ARG": "DAR", "ASP": "DAS", "CYS": "DCY", "GLU": "DGL",
    "GLN": "DGN", "HIS": "DHI", "ILE": "DIL", "LEU": "DLE", "LYS": "DLY",
    "PHE": "DPN", "PRO": "DPR", "ASN": "DSG", "SER": "DSN", "THR": "DTH",
    "TRP": "DTR", "TYR": "DTY", "VAL": "DVA", "MET": "MED",
}
D_TO_L = {d: l for l, d in L_TO_D.items()}


def reflect_coords(coords, axis: int = 0) -> np.ndarray:
    """Return a reflected copy of coords with the chosen axis negated."""
    out = np.array(coords, dtype=float, copy=True)
    out[..., axis] *= -1.0
    return out


def remap_chirality(code: str) -> str:
    """Map an L three-letter code to its D partner (or vice versa); pass through achiral."""
    c = code.upper()
    if c in L_TO_D:
        return L_TO_D[c]
    if c in D_TO_L:
        return D_TO_L[c]
    return c


def mirror_residue_codes(codes: Sequence[str]) -> list[str]:
    """Remap a sequence of three-letter codes to their chiral partners."""
    return [remap_chirality(c) for c in codes]


def mirror_discrepancy(coords_a, coords_b, axis: int = 0) -> float:
    """Kabsch RMSD between coords_a and the reflected coords_b.

    Zero when B is the exact mirror image of A (up to rigid motion). Used as the
    forward-only mirror self-consistency score for Tier-1 seed selection (spec §2.2).
    """
    return kabsch_rmsd(coords_a, reflect_coords(coords_b, axis=axis))
