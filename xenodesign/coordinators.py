"""Declarative metal-coordinator / scaffold-residue parsing (DECLARATIVE flags).

A coordinator declaration specifies, IN ADVANCE, which residues coordinate a metal (or
otherwise scaffold a design) — identity, position, AND chirality — so the coordinating
residues are NOT hardcoded per binder class. One token = identity+position; the identity
FORM encodes chirality:

  * a 1-letter code (e.g. ``H``)  -> an L-residue (chirality 'L'); three_letter is the
    canonical L 3-letter code (e.g. ``HIS``) looked up from ``io_spec.AA1_TO_AA3``.
  * a CCD code (e.g. ``DHI``)     -> a D-residue (chirality 'D'); three_letter is that
    D-CCD code, and one_letter is its L parent's 1-letter code (``H``).

This generalizes beyond His/Zn to ANY donor (Cys/Asp/Glu; N/O/S) and any metal: the parser
only knows residue identity + chirality, never a specific metal or geometry. The parsed list
drives BOTH (a) the seed's OPT-IN fixed positions (place that residue, that chirality, at
that position) and (b) the metal-coordination restraint rows (coordinator<->metal contacts).

CPU-only / pure: validates codes against the L/D maps in ``xenodesign.mirror`` /
``xenodesign.io_spec`` and rejects anything unknown; it never imports torch/chai.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from xenodesign.io_spec import AA1_TO_AA3, AA3_TO_AA1
from xenodesign.mirror import D_TO_L


@dataclass(frozen=True)
class CoordResidue:
    """One declared coordinating/scaffold residue.

    pos:         1-based position in the from-scratch binder.
    one_letter:  the residue's canonical (L-parent) 1-letter code (e.g. 'H' for both H and DHI).
    three_letter: the canonical L 3-letter code for an L token (e.g. 'HIS'), or the D-CCD code
                 for a D token (e.g. 'DHI'). None is never produced by the parser (always set),
                 but the field is Optional so callers can construct a bare residue if needed.
    chirality:   'L' (1-letter token) or 'D' (CCD token).
    atom:        the liganding atom name (e.g. 'ND1', 'SG'), from an explicit ``@<atom>`` suffix
                 or the per-element default (His->ND1, Cys->SG, Met->SD, Asp->OD1, Glu->OE1,
                 Lys->NZ); None for any residue without a sensible default and no explicit atom.
    """
    pos: int
    one_letter: str
    three_letter: str | None
    chirality: str
    atom: str | None = None


# Default liganding atom per coordinating residue (keyed by L-parent 1-letter code).
DEFAULT_LIGAND_ATOM = {
    "H": "ND1",   # His
    "C": "SG",    # Cys
    "M": "SD",    # Met
    "D": "OD1",   # Asp
    "E": "OE1",   # Glu
    "K": "NZ",    # Lys
}


def _split_atom_suffix(token: str) -> tuple[str, str | None]:
    """Split an optional ``@<atom>`` suffix off a coordinator token.

    Returns (token_without_atom, atom_or_None). The atom must be a non-empty run of atom-name
    characters (letters/digits/'); raises ValueError on an empty or malformed atom.
    """
    t = token.strip()
    if "@" not in t:
        return t, None
    base, _, atom = t.partition("@")
    atom = atom.strip()
    if not atom:
        raise ValueError(f"coordinator token {token!r} has an empty @atom suffix")
    if not all(ch.isalnum() or ch == "'" for ch in atom):
        raise ValueError(f"coordinator token {token!r} has an invalid atom name {atom!r}")
    return base.strip(), atom.upper()


def _split_identity_position(token: str) -> tuple[str, int]:
    """Split a coordinator token into (identity, 1-based position).

    The position is the trailing run of digits; the identity is everything before it.
    Raises ValueError if there is no trailing position or the identity is empty.
    """
    t = token.strip()
    if not t:
        raise ValueError("empty coordinator token")
    i = len(t)
    while i > 0 and t[i - 1].isdigit():
        i -= 1
    identity, digits = t[:i], t[i:]
    if not digits:
        raise ValueError(f"coordinator token {token!r} has no trailing position (e.g. 'H6', 'DHI12')")
    if not identity:
        raise ValueError(f"coordinator token {token!r} has no residue identity")
    pos = int(digits)
    if pos < 1:
        raise ValueError(f"coordinator position must be >= 1, got {pos} in {token!r}")
    return identity, pos


def parse_coord_token(token: str) -> CoordResidue:
    """Parse ONE coordinator token (e.g. 'H6' or 'DHI12') into a CoordResidue.

    The identity form disambiguates chirality:
      * length-1 identity -> an L-residue 1-letter code (validated against AA1_TO_AA3).
      * longer identity   -> a D-CCD code (validated against mirror.D_TO_L; the D->L->1-letter
        parent is recorded as one_letter).

    An optional ``@<atom>`` suffix (e.g. ``H6@ND1``, ``DHI12@ND1``, ``C7@SG``) names the
    liganding atom; when omitted it defaults per residue element (see DEFAULT_LIGAND_ATOM).

    Raises ValueError on an unknown 1-letter code, an unknown CCD code, or a malformed token.
    """
    base, atom = _split_atom_suffix(token)
    identity, pos = _split_identity_position(base)
    ident = identity.upper()
    if len(ident) == 1:
        # L-residue, 1-letter code.
        if ident not in AA1_TO_AA3:
            raise ValueError(
                f"unknown 1-letter residue code {identity!r} in coordinator token {token!r}")
        one_letter = ident
        three_letter, chirality = AA1_TO_AA3[ident], "L"
    else:
        # D-residue, CCD code (must be a known chai D partner).
        if ident not in D_TO_L:
            raise ValueError(
                f"unknown D-CCD code {identity!r} in coordinator token {token!r}; "
                f"known D codes: {sorted(D_TO_L)}")
        l_three = D_TO_L[ident]
        one_letter, three_letter, chirality = AA3_TO_AA1[l_three], ident, "D"
    if atom is None:
        atom = DEFAULT_LIGAND_ATOM.get(one_letter)
    return CoordResidue(pos=pos, one_letter=one_letter, three_letter=three_letter,
                        chirality=chirality, atom=atom)


def parse_coord_residues(spec: str) -> List[CoordResidue]:
    """Parse a comma-separated coordinator declaration into a list of CoordResidue.

    e.g. ``parse_coord_residues("H6,DHI12,H18,DHI24")`` ->
        [CoordResidue(6,'H','HIS','L'), CoordResidue(12,'H','DHI','D'),
         CoordResidue(18,'H','HIS','L'), CoordResidue(24,'H','DHI','D')]

    Empty / whitespace-only input -> []. Duplicate positions are rejected (a position can
    carry only one coordinator). Raises ValueError on any malformed/unknown token.
    """
    if spec is None:
        return []
    tokens = [t for t in (s.strip() for s in spec.split(",")) if t]
    out: List[CoordResidue] = []
    seen: set[int] = set()
    for tok in tokens:
        cr = parse_coord_token(tok)
        if cr.pos in seen:
            raise ValueError(f"duplicate coordinator position {cr.pos} in {spec!r}")
        seen.add(cr.pos)
        out.append(cr)
    return out


def fixed_chirality_map(coords: List[CoordResidue]) -> dict:
    """{1-based pos: 'D'|'L'} for the declared coordinators (the seed's fixed_chirality)."""
    return {c.pos: c.chirality for c in coords}
