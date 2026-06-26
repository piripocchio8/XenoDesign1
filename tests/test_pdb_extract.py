import pytest
from xenodesign.pdb_extract import (
    chirality_label,
    chirality_labels,
    codes_to_entity_sequence,
    has_d_residue,
)


def test_chirality_label_buckets():
    assert chirality_label("ALA") == "L"
    assert chirality_label("DAL") == "D"
    assert chirality_label("SEP") == "ncAA"
    assert chirality_label("dal") == "D"  # case-insensitive


def test_chirality_labels_list():
    assert chirality_labels(["ALA", "DAL", "SEP"]) == ["L", "D", "ncAA"]


def test_codes_to_entity_sequence_mixed():
    # standard L -> one letter; D and ncAA -> parenthesized CCD, verbatim for Chai.
    assert codes_to_entity_sequence(["ALA", "DAL", "SEP", "GLY"]) == "A(DAL)(SEP)G"


def test_codes_to_entity_sequence_all_standard():
    assert codes_to_entity_sequence(["ALA", "CYS", "SER"]) == "ACS"


def test_has_d_residue():
    assert has_d_residue(["ALA", "DAL", "SER"]) is True
    assert has_d_residue(["ALA", "SER", "AIB"]) is False  # AIB is ncAA, not a D_partner
