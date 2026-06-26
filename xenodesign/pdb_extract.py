"""Extract Chai inputs + chirality labels from experimental structures (real-PDB benchmark).

Pure helpers (chirality_label(s), codes_to_entity_sequence, has_d_residue) are CPU-tested.
CIF parsing (gemmi) and RCSB download are gated (network/gpu) and run on the user's machine,
because this build environment's network policy reaches GitHub only (RCSB/MONDE-T are blocked).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from xenodesign.io_spec import AA1_TO_AA3
from xenodesign.mirror import D_TO_L

AA3_TO_AA1 = {three: one for one, three in AA1_TO_AA3.items()}
_CANONICAL_3 = set(AA1_TO_AA3.values())


def chirality_label(code: str) -> str:
    """Classify a residue three-letter code as 'L', 'D', or 'ncAA' for the gate."""
    c = code.upper()
    if c in D_TO_L:
        return "D"
    if c in _CANONICAL_3:
        return "L"
    return "ncAA"


def chirality_labels(codes: Sequence[str]) -> list[str]:
    return [chirality_label(c) for c in codes]


def codes_to_entity_sequence(codes: Sequence[str]) -> str:
    """Three-letter residue codes -> a Chai sequence.

    Standard L amino acids become one-letter; everything else (D residues, ncAA) becomes a
    parenthesized CCD block (e.g. '(DAL)', '(SEP)') — chai_lab 0.2.1 only accepts `()` for
    modified residues. The result is written verbatim into the Chai FASTA.
    """
    out = []
    for code in codes:
        cc = code.upper()
        out.append(AA3_TO_AA1[cc] if cc in _CANONICAL_3 else f"({cc})")
    return "".join(out)


def has_d_residue(codes: Sequence[str]) -> bool:
    """True if any residue is a standard D-amino acid (chai D_partners)."""
    return any(c.upper() in D_TO_L for c in codes)


def fetch_cif(pdb_id: str, dest_dir) -> Path:  # pragma: no cover (network)
    """Download <pdb_id>.cif from RCSB into dest_dir (cached). Needs open network."""
    import urllib.request

    dest = Path(dest_dir) / f"{pdb_id}.cif"
    if not dest.exists():
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_id}.cif", dest
        )
    return dest


def chains_in_cif(cif_path) -> list[str]:  # pragma: no cover (gemmi)
    """List chain names in the first model of a CIF."""
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    for model in st:
        return [chain.name for chain in model]
    return []


def parse_cif_chain(cif_path, chain_name: str):  # pragma: no cover (gemmi)
    """Return (codes, backbone) for one chain: codes=list[str] (residue names),
    backbone=list of {'N','CA','C'[,'CB']} numpy arrays. First model only."""
    import gemmi
    import numpy as np

    st = gemmi.read_structure(str(cif_path))
    codes, backbone = [], []
    for model in st:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                atoms = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res}
                if not {"N", "CA", "C"} <= atoms.keys():
                    continue
                rec = {k: atoms[k] for k in ("N", "CA", "C")}
                if "CB" in atoms:
                    rec["CB"] = atoms["CB"]
                codes.append(res.name)
                backbone.append(rec)
        break
    return codes, backbone
