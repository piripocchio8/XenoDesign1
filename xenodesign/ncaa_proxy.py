"""ncAA -> canonical-L proxy by conformational (Ramachandran) similarity (spec §2.8).

First version is a CURATED map grounded in MONDE-T findings (Waldherr et al., 2025): the
proxy is the canonical L amino acid whose backbone (phi/psi) propensity best matches the
ncAA. A data-driven refinement (phi/psi histogram overlap + Tanimoto) is future work.
Used only for inverse-folding conditioning; the real ncAA is restored for the Chai input.
"""
from __future__ import annotations

from typing import Optional

# code -> canonical three-letter proxy. Grounded in MONDE-T / known chemistry.
CONFORMATIONAL_PROXY = {
    "MSE": "MET",  # selenomethionine: identical Ramachandran distribution to MET
    "AIB": "ALA",  # alpha-aminoisobutyric acid: alpha-helical (L/D symmetric)
    "SEC": "CYS",  # selenocysteine
    "PYL": "LYS",  # pyrrolysine
    "HYP": "PRO",  # 4-hydroxyproline
    "SEP": "SER",  # phosphoserine
    "TPO": "THR",  # phosphothreonine
    "PTR": "TYR",  # phosphotyrosine
    "MLY": "LYS",  # methyllysine
    "M3L": "LYS",  # trimethyllysine
    "CSO": "CYS",  # S-hydroxycysteine
    "CSD": "CYS",  # S-cysteinesulfinic acid
    "NLE": "LEU",  # norleucine
    "ABA": "ALA",  # alpha-aminobutyric acid
    "ORN": "LYS",  # ornithine
    # NB: standard D residues are NOT listed here — their chirality is handled by the
    # mirror-into-L step (mirror.D_TO_L); the proxy is only for genuine ncAA side chains.
}


def proxy_for(ncaa_code: str) -> Optional[str]:
    """Canonical three-letter proxy for an ncAA code, or None if unknown (-> fixed context)."""
    return CONFORMATIONAL_PROXY.get(ncaa_code.upper())
