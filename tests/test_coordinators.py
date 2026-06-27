"""Tests for the declarative coordinator parser (xenodesign.coordinators)."""
import pytest

from xenodesign.coordinators import (
    CoordResidue,
    fixed_chirality_map,
    parse_coord_residues,
    parse_coord_token,
)


# ── L vs D disambiguation by code FORM ──────────────────────────────────────────

def test_single_letter_is_L_residue():
    cr = parse_coord_token("H6")
    assert cr == CoordResidue(pos=6, one_letter="H", three_letter="HIS", chirality="L",
                              atom="ND1")


def test_ccd_code_is_D_residue():
    cr = parse_coord_token("DHI12")
    # D-CCD -> chirality D, three_letter is the CCD code, one_letter the L parent.
    assert cr == CoordResidue(pos=12, one_letter="H", three_letter="DHI", chirality="D",
                              atom="ND1")


def test_d_cys_and_d_asp_donors_generalize():
    # generalizes beyond His: D-Cys (S donor), D-Asp (O donor).
    assert parse_coord_token("DCY7") == CoordResidue(7, "C", "DCY", "D", "SG")
    assert parse_coord_token("DAS9") == CoordResidue(9, "D", "DAS", "D", "OD1")


def test_l_donors_generalize():
    assert parse_coord_token("C3") == CoordResidue(3, "C", "CYS", "L", "SG")
    assert parse_coord_token("E18") == CoordResidue(18, "E", "GLU", "L", "OE1")


# ── positions ───────────────────────────────────────────────────────────────────

def test_multi_digit_position():
    assert parse_coord_token("H124").pos == 124


def test_list_parse_and_order_preserved():
    crs = parse_coord_residues("H6,DHI12,H18,DHI24")
    assert [c.pos for c in crs] == [6, 12, 18, 24]
    assert [c.chirality for c in crs] == ["L", "D", "L", "D"]
    assert [c.one_letter for c in crs] == ["H", "H", "H", "H"]
    assert [c.three_letter for c in crs] == ["HIS", "DHI", "HIS", "DHI"]


def test_whitespace_tolerant():
    crs = parse_coord_residues(" H6 , DHI12 ")
    assert [c.pos for c in crs] == [6, 12]


def test_empty_and_none_yield_empty_list():
    assert parse_coord_residues("") == []
    assert parse_coord_residues("   ") == []
    assert parse_coord_residues(None) == []


def test_fixed_chirality_map():
    crs = parse_coord_residues("H6,DHI12,H18,DHI24")
    assert fixed_chirality_map(crs) == {6: "L", 12: "D", 18: "L", 24: "D"}


# ── bad input rejected ────────────────────────────────────────────────────────────

def test_unknown_1letter_rejected():
    with pytest.raises(ValueError, match="unknown 1-letter"):
        parse_coord_token("Z6")


def test_unknown_ccd_rejected():
    with pytest.raises(ValueError, match="unknown D-CCD"):
        parse_coord_token("XYZ6")


def test_no_position_rejected():
    with pytest.raises(ValueError, match="no trailing position"):
        parse_coord_token("H")


def test_zero_position_rejected():
    with pytest.raises(ValueError, match=">= 1"):
        parse_coord_token("H0")


def test_duplicate_position_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_coord_residues("H6,DHI6")


def test_l_aspartate_not_confused_with_d_prefix():
    # 'D6' is the 1-letter L-Asp code, NOT a D-residue marker.
    assert parse_coord_token("D6") == CoordResidue(6, "D", "ASP", "L", "OD1")


# ── @atom liganding-atom suffix (atom-specific metal coordination) ─────────────────

def test_explicit_atom_suffix_L():
    cr = parse_coord_token("H6@ND1")
    assert cr == CoordResidue(pos=6, one_letter="H", three_letter="HIS",
                              chirality="L", atom="ND1")


def test_explicit_atom_suffix_D():
    cr = parse_coord_token("DHI12@ND1")
    assert cr == CoordResidue(pos=12, one_letter="H", three_letter="DHI",
                              chirality="D", atom="ND1")


def test_explicit_atom_suffix_cys_sg():
    assert parse_coord_token("C7@SG") == CoordResidue(7, "C", "CYS", "L", "SG")


def test_atom_defaults_per_element_when_omitted():
    # His->ND1, Cys->SG, Met->SD, Asp->OD1, Glu->OE1, Lys->NZ.
    assert parse_coord_token("H6").atom == "ND1"
    assert parse_coord_token("DHI12").atom == "ND1"
    assert parse_coord_token("C7").atom == "SG"
    assert parse_coord_token("M5").atom == "SD"
    assert parse_coord_token("D10").atom == "OD1"
    assert parse_coord_token("E18").atom == "OE1"
    assert parse_coord_token("K3").atom == "NZ"


def test_atom_none_for_non_default_residue():
    # A residue with no sensible default donor atom -> atom None when omitted.
    assert parse_coord_token("A4").atom is None


def test_empty_atom_token_rejected():
    with pytest.raises(ValueError, match="atom"):
        parse_coord_token("H6@")


def test_atom_with_invalid_chars_rejected():
    with pytest.raises(ValueError, match="atom"):
        parse_coord_token("H6@N D1")
