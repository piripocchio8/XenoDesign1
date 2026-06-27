"""CPU tests for the ABC ncAA palette + its lightweight CCD-token validation (track #2).

The Variant-B ncAA palette is a CONSERVATIVE, CONFIG-DRIVEN list of CCD 3-letter codes
the identity search may propose (emitted as ``(XXX)`` in the FASTA). ``validate_palette``
is a CPU-only gate: it keeps only well-formed 3-letter codes that resolve in the repo's
known-ncAA table (``ncaa_proxy.CONFORMATIONAL_PROXY``); everything else is dropped.
"""
from __future__ import annotations

from xenodesign.abc.ncaa import DEFAULT_NCAA_PALETTE, validate_palette


def test_default_palette_is_the_validated_conservative_set():
    # The conservative palette chosen for Variant B: AIB, ORN, NLE, HYP (all resolvable).
    assert DEFAULT_NCAA_PALETTE == ("AIB", "ORN", "NLE", "HYP")
    # And it is its own validated fixpoint (no member is dropped).
    assert validate_palette(DEFAULT_NCAA_PALETTE) == list(DEFAULT_NCAA_PALETTE)


def test_validate_drops_unknown_and_malformed_codes():
    # ZZZ is well-formed but not in the known table; "AI" / "TOOLONG" / "" are malformed.
    survived = validate_palette(["AIB", "ZZZ", "AI", "TOOLONG", ""])
    assert survived == ["AIB"]


def test_validate_is_case_insensitive_and_uppercases():
    assert validate_palette(["aib", "hyp"]) == ["AIB", "HYP"]


def test_validate_dedupes_preserving_order():
    assert validate_palette(["AIB", "AIB", "ORN"]) == ["AIB", "ORN"]


def test_validate_empty_is_empty():
    assert validate_palette([]) == []


def test_validate_rejects_caps_and_chromophores():
    # m4: terminal caps (ACE/NH2 -> parent 'CAPS') and multi-residue chromophores
    # (CRO/CR2/NRQ -> parent 'CHROMOPHORE') resolve in MONDE-T but are NOT real amino
    # acids; they must NOT leak into the 'all'/'d_common' palettes.
    assert validate_palette(["ACE", "NH2", "CRO", "CR2", "NRQ"]) == []


def test_validate_accepts_normal_ncaa_with_canonical_aa_parent():
    # m4: a normal ncAA whose resolved parent IS one of the 20 standard AAs survives.
    # MSE -> MET, AIB -> a standard AA (Gly), both accepted.
    assert validate_palette(["MSE"]) == ["MSE"]
    assert validate_palette(["AIB"]) == ["AIB"]


def test_validate_still_accepts_d_canonicals():
    # m4: the D-canonical CCD codes (parent = the standard L AA, e.g. DAL->ALA) stay valid.
    from xenodesign.abc.ncaa import D_CANONICAL

    assert validate_palette(list(D_CANONICAL)) == list(D_CANONICAL)
