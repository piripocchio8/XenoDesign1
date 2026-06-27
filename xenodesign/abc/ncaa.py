"""ABC Variant-B ncAA palette + lightweight CCD-token validation (track #2).

Variant B may propose NON-CANONICAL amino acids by their CCD 3-letter code, emitted in the
FASTA as a ``(XXX)`` parenthesized block (the same modified-residue contract as ``(DXX)`` D
residues). The palette is CONFIG-DRIVEN and CONSERVATIVE: an empty palette turns ncAA OFF
(the default — existing behaviour unchanged), and a non-empty palette opts in.

``validate_palette`` is a CPU-only gate (NO GPU / no Chai tokenizer call): it keeps only
well-formed 3-letter codes that resolve in the repo's known-ncAA table
(``ncaa_proxy.CONFORMATIONAL_PROXY``), so a typo or an un-resolvable code can never reach the
move-set. Codes that survive are guaranteed to have a canonical proxy (used by MPNN
conditioning) and a known parent chemistry.
"""
from __future__ import annotations

from xenodesign.ncaa_proxy import proxy_for

# Conservative default Variant-B palette. Each is a small, plausibly-tokenizable ncAA with a
# clean canonical proxy: AIB (2-aminoisobutyric, helix-stabilising), ORN (ornithine, Lys-like),
# NLE (norleucine, Met/Leu-like), HYP (4-hydroxyproline, turn-stabilising). All resolve in
# CONFORMATIONAL_PROXY (the validation table), so this set is its own validated fixpoint.
DEFAULT_NCAA_PALETTE = ("AIB", "ORN", "NLE", "HYP")


def _is_wellformed(code: str) -> bool:
    """True iff ``code`` is a 3-letter alphanumeric CCD-style token (the ``(XXX)`` slot)."""
    return len(code) == 3 and code.isalnum()


def validate_palette(palette) -> list[str]:
    """Return the validated, upper-cased, de-duplicated subset of ``palette``.

    A code survives iff it is a well-formed 3-letter token AND resolvable in the known-ncAA
    table (``proxy_for`` returns a canonical parent). Order is preserved; duplicates collapse.
    CPU-only — no Chai/tokenizer import.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in palette:
        code = str(raw).strip().upper()
        if code in seen:
            continue
        if _is_wellformed(code) and proxy_for(code) is not None:
            out.append(code)
            seen.add(code)
    return out
