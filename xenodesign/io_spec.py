"""Build Chai-1 input FASTA, including all-D peptide entities.

Chai FASTA: header `>protein|<name>` then the sequence on the next line. Modified /
D residues are written inline in PARENTHESES, e.g. `(DAL)` — chai_lab 0.2.1's
`constituents_of_modified_fasta` only allows `()` (not `[]`) for modified residues
(verified against ameg/chai-1 and the gradio chai_runner). NOTE: chai needs >=1 canonical
residue per chain to tokenize — a fully-NCAA (e.g. glycine-free all-D) chain is rejected.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from xenodesign.mirror import L_TO_D, D_TO_L

AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


def to_d_fasta(seq_one_letter: str) -> str:
    """Convert a one-letter L sequence to a Chai all-D parenthesized sequence.

    Glycine (achiral) stays a single 'G'. Raises KeyError (with the offending letter
    and its 1-based position) on unknown letters.
    """
    out = []
    for i, aa in enumerate(seq_one_letter):
        try:
            three = AA1_TO_AA3[aa]
        except KeyError:
            raise KeyError(f"unknown amino-acid letter {aa!r} at position {i + 1}") from None
        if three == "GLY":
            out.append("G")
        else:
            out.append(f"({L_TO_D[three]})")
    return "".join(out)


def _is_fully_ncaa(seq: str) -> bool:
    """True if every residue in `seq` is a parenthesized CCD block (no bare canonical AA).

    Chai-1 needs >=1 canonical residue per chain to tokenize structure, so a fully-NCAA
    chain (e.g. a glycine-free all-D peptide) is rejected by chai. Mirrors the gradio
    chai_runner guard.
    """
    i = 0
    seen_any = False
    while i < len(seq):
        seen_any = True
        if seq[i] == "(":
            j = seq.find(")", i)
            if j == -1:
                return False  # malformed; not "fully ncaa" for this purpose
            i = j + 1
        else:
            return False  # a bare single-letter (canonical) residue
    return seen_any


def build_fasta(entities: Sequence[Mapping[str, object]]) -> str:
    """Build a Chai FASTA string from a list of entity specs.

    Each entity: {'type': 'protein', 'name': str, 'sequence': str, 'chirality': 'L'|'D'}.
    Only protein entities with chirality 'D' are converted to parenthesized D sequences.

    Raises ValueError if a protein chain ends up fully non-canonical (e.g. a glycine-free
    all-D peptide): chai-1 cannot tokenize it. Add a canonical residue (a glycine) or use
    the mirror-into-L path for such designs.
    """
    lines = []
    for e in entities:
        etype = e["type"]
        name = e["name"]
        # Ligand entities (e.g. the Zn metal cation) carry SMILES/CCD, not a residue sequence.
        # Chai writes them as `>ligand|name=<name>` + the SMILES (or `>ligand|ccd=<CCD>` line).
        # Matches scripts.classes.cyclic.build_cyclic_input_fasta's hand-written Zn entity.
        if etype == "ligand":
            smiles = e.get("smiles")
            ccd = e.get("ccd")
            metal_ccd = e.get("metal_ccd")
            if metal_ccd:
                # METAL-(b): emit the metal as `>ligand|name=<CODE>` with the CCD code on the
                # sequence line — the form chai_patches._patch_ligand_ccd_feeding recognizes
                # (both the entity name AND the sequence are the CCD code), so the metal tokenizes
                # via the cached conformer (residue=<CODE>, atom=<CODE>) and atom-aware coordination
                # restraints resolve. Falls back to SMILES below when no CCD code is requested.
                code = str(metal_ccd).upper()
                lines.append(f">ligand|name={code}")
                lines.append(code)
            elif smiles:
                lines.append(f">ligand|name={name}")
                lines.append(str(smiles))
            elif ccd:
                lines.append(f">ligand|ccd={ccd}")
            else:
                raise ValueError(
                    f"ligand entity {name!r} needs a 'smiles', 'ccd', or 'metal_ccd' value")
            continue
        seq = e["sequence"]
        chirality = str(e.get("chirality", "L")).upper()
        if etype == "protein" and chirality == "D":
            seq = to_d_fasta(seq)
        if etype == "protein" and _is_fully_ncaa(seq):
            raise ValueError(
                f"protein chain {name!r} is fully non-canonical ({seq!r}); chai-1 needs "
                f">=1 canonical residue (e.g. a glycine) per chain to tokenize. Add a "
                f"canonical residue or use the mirror-into-L path."
            )
        lines.append(f">{etype}|{name}")
        lines.append(seq)
    return "\n".join(lines) + "\n"


AA3_TO_AA1 = {three: one for one, three in AA1_TO_AA3.items()}

# Modified residues Chai/CIF may emit that map onto a standard L residue.
MODIFIED_TO_STANDARD = {"MSE": "MET"}  # selenomethionine -> methionine


def d_fasta_to_one_letter(d_fasta: str) -> str:
    """Inverse of `to_d_fasta`: parse a Chai D/standard sequence into one-letter L codes.

    Parenthesized CCD blocks ((DAL)) map D->L->one-letter; bare single letters pass through.
    Modified residues (e.g. (MSE)) are normalized to their standard parent (MET).
    """
    out = []
    i = 0
    while i < len(d_fasta):
        ch = d_fasta[i]
        if ch == "(":
            j = d_fasta.index(")", i)
            code = d_fasta[i + 1:j].upper()
            three = D_TO_L.get(code, code)
            three = MODIFIED_TO_STANDARD.get(three, three)
            out.append(AA3_TO_AA1[three])
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# Residues whose D-CCD partners are tokenized by chai (DSN/DSG/DPR) and so are natural
# cyclic pre-mutation targets to swap to an achiral Gly without changing the ring size.
_CYCLIC_GLY_SWAP_PREFERENCE = ('S', 'N', 'P')  # Ser, Asn, Pro (one-letter, L)


def glycine_satisfy_guard(seq_one_letter: str, cyclic: bool = False) -> str:
    """Ensure a one-letter L sequence stays Chai-tokenizable after `to_d_fasta` (#8).

    Chai needs >=1 canonical residue per chain to tokenize; an all-D chain becomes fully
    parenthesized (fully-NCAA) under `to_d_fasta` and is rejected by `build_fasta`. Glycine is
    achiral (`to_d_fasta` keeps it a bare 'G'), so a single Gly satisfies the guard.

    - If the sequence already contains a glycine ('G'), it is already safe -> returned unchanged
      (idempotent).
    - Non-cyclic: append a trailing 'G' (extends the chain by one; fine for a linear binder).
    - Cyclic: do NOT grow the ring -> mutate the FIRST D-Ser/D-Asn/D-Pro-able position (S/N/P)
      to 'G' in place; if none is present, fall back to appending a trailing 'G'.

    Args:
        seq_one_letter: the binder sequence as one-letter L codes (pre-`to_d_fasta`).
        cyclic: True for a head-to-tail macrocycle (preserve ring size by in-place mutation).

    Returns:
        A one-letter L sequence guaranteed to contain >=1 'G' (so the post-`to_d_fasta`
        chain is not fully-NCAA). Compose as `to_d_fasta(glycine_satisfy_guard(seq, ...))`.
    """
    if 'G' in seq_one_letter:
        return seq_one_letter  # already canonical-safe
    if cyclic:
        for swap in _CYCLIC_GLY_SWAP_PREFERENCE:
            idx = seq_one_letter.find(swap)
            if idx != -1:
                return seq_one_letter[:idx] + 'G' + seq_one_letter[idx + 1:]
        # no S/N/P to mutate -> fall back to appending (documented in the docstring).
    return seq_one_letter + 'G'
