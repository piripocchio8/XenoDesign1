"""ABC Variant-B ncAA palette + lightweight CCD-token validation (track #2 + --ncaa_dict).

Variant B may propose NON-CANONICAL amino acids by their CCD 3-letter code, emitted in the
FASTA as a ``(XXX)`` parenthesized block (the same modified-residue contract as ``(DXX)`` D
residues). The palette SCOPE is config-driven via ``--ncaa_dict {d_only,d_common,all}``,
derived from the MONDE-T catalog (``xenodesign.mondet``):

  - ``d_only``  -> the canonical D-amino-acid CCD set (mixed-chirality, no exotic ncAA);
  - ``d_common``-> D-canonical + top-``ncaa_top_x`` MONDE-T ncAA by entity_count;
  - ``all``     -> D-canonical + ALL MONDE-T ncAA (no cap; count logged by the caller).

``validate_palette`` is a CPU-only gate (NO GPU / no Chai tokenizer call): it keeps only
well-formed 3-letter codes that RESOLVE, either in the repo's curated proxy table
(``ncaa_proxy.CONFORMATIONAL_PROXY``) OR in the MONDE-T catalog via their canonical
``parent`` (so the larger palettes are not rejected). Codes that survive are guaranteed to
have a canonical parent chemistry for MPNN conditioning.
"""
from __future__ import annotations

from typing import Optional

from xenodesign import mondet
from xenodesign.ncaa_proxy import proxy_for as _proxy_table_lookup

# Canonical D-amino-acid CCD codes (the 19 chiral standard residues; Gly is achiral, no D form).
# Derived from mirror.L_TO_D so it can never drift from the project's L<->D mapping, but pinned
# as an explicit tuple for readability/provenance. (Verified against MONDE-T parents.)
from xenodesign.mirror import L_TO_D as _L_TO_D

D_CANONICAL = tuple(_L_TO_D[k] for k in (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
))

# Conservative legacy Variant-B palette, kept as a FALLBACK only. The canonical palette is now
# DERIVED from --ncaa_dict (build_palette). AIB/ORN/NLE/HYP all resolve in CONFORMATIONAL_PROXY.
DEFAULT_NCAA_PALETTE = ("AIB", "ORN", "NLE", "HYP")

VALID_NCAA_DICT = ("d_only", "d_common", "all")


def _is_wellformed(code: str) -> bool:
    """True iff ``code`` is a 3-letter alphanumeric CCD-style token (the ``(XXX)`` slot)."""
    return len(code) == 3 and code.isalnum()


def _resolves(code: str, csv_path=None) -> bool:
    """True iff ``code`` has a known canonical parent (curated proxy OR MONDE-T parent)."""
    if _proxy_table_lookup(code) is not None:
        return True
    return mondet.mondet_parent(code, csv_path=csv_path) is not None


def proxy_for(code: str, csv_path=None) -> Optional[str]:
    """Canonical 3-letter proxy for a code: curated table first, else MONDE-T parent.

    Extends ``ncaa_proxy.proxy_for`` so MONDE-T codes (the d_common/all palettes) get a parent
    for MPNN conditioning even when they are not in the small curated CONFORMATIONAL_PROXY map.
    Returns None when the code is unknown to both.
    """
    hit = _proxy_table_lookup(code)
    if hit is not None:
        return hit
    return mondet.mondet_parent(code, csv_path=csv_path)


def validate_palette(palette, csv_path=None) -> list[str]:
    """Return the validated, upper-cased, de-duplicated subset of ``palette``.

    A code survives iff it is a well-formed 3-letter token AND resolvable to a canonical parent
    (curated proxy table OR MONDE-T ``parent``). Order is preserved; duplicates collapse.
    CPU-only — no Chai/tokenizer import.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in palette:
        code = str(raw).strip().upper()
        if code in seen:
            continue
        if _is_wellformed(code) and _resolves(code, csv_path=csv_path):
            out.append(code)
            seen.add(code)
    return out


def build_palette(ncaa_dict: str, ncaa_top_x: int = 20, csv_path=None) -> list[str]:
    """Build the effective Variant-B palette for an ``--ncaa_dict`` scope.

      - ``d_only``  -> the D-canonical set.
      - ``d_common``-> D-canonical + top-``ncaa_top_x`` MONDE-T ncAA by entity_count.
      - ``all``     -> D-canonical + ALL MONDE-T ncAA (no cap).

    The result is de-duplicated, order-stable (D-canonical first, then MONDE-T by frequency),
    and validated. CPU-only.
    """
    if ncaa_dict not in VALID_NCAA_DICT:
        raise ValueError(
            f"ncaa_dict must be one of {VALID_NCAA_DICT}, got {ncaa_dict!r}")
    codes: list[str] = list(D_CANONICAL)
    if ncaa_dict == "d_common":
        codes += mondet.top_ncaa(ncaa_top_x, csv_path=csv_path)
    elif ncaa_dict == "all":
        codes += [c for c, _p, _n in mondet.load_mondet(csv_path)]
    palette = validate_palette(codes, csv_path=csv_path)
    if ncaa_dict == "all":
        # 'all' = NO cap (ncaa_top_x ignored); the whole MONDE-T tail is in play. Log the size
        # so a run's provenance records how large the move-set actually was.
        import logging
        logging.getLogger(__name__).info(
            "ncaa_dict='all': palette uncapped at %d validated codes", len(palette))
    return palette
