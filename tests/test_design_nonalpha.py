"""CPU tests for the 9DXX non-alpha D-knottin:HA driver (P3b).

Pure-CPU: exercise the ICK cystine-knot scaffold helpers, the binder seed Cys placement, the
disulfide COVALENT row emission, and the HA-target FASTA parsing. The GPU path
(run_nonalpha_design) is not imported here (it loads chai/torch).
"""
from __future__ import annotations

import pytest

from scripts.design_nonalpha import (
    build_binder_seed,
    build_nonalpha_disulfide_rows,
    ick_disulfide_pairs,
    knottin_cys_positions,
    load_ha_entities,
    place_cys,
)


# ── ICK cystine-knot scaffold ──────────────────────────────────────────────────────

def test_knottin_cys_positions_six_distinct_in_range():
    pos = knottin_cys_positions(31)
    assert len(pos) == 6
    assert len(set(pos)) == 6                 # all distinct
    assert all(1 <= p <= 31 for p in pos)
    assert pos == sorted(pos)                 # ordered I..VI


def test_knottin_cys_positions_rejects_too_short():
    with pytest.raises(ValueError):
        knottin_cys_positions(5)


def test_ick_disulfide_pairs_connectivity():
    pairs = ick_disulfide_pairs([3, 8, 13, 18, 23, 28])
    # ICK: I-IV, II-V, III-VI
    assert pairs == [(3, 18), (8, 23), (13, 28)]


def test_ick_disulfide_pairs_requires_six():
    with pytest.raises(ValueError):
        ick_disulfide_pairs([1, 2, 3])


# ── Binder seed Cys placement ───────────────────────────────────────────────────────

def test_place_cys_overwrites_positions():
    out = place_cys("AAAAAAAA", [2, 5])      # C at 1-based positions 2 and 5
    assert out == "ACAACAAA"
    assert out[1] == "C" and out[4] == "C"


def test_build_binder_seed_length_and_cys():
    pos = knottin_cys_positions(31)
    seed = build_binder_seed(31, pos, rng_seed=1)
    assert len(seed) == 31
    for p in pos:
        assert seed[p - 1] == "C"
    # the random fill avoids extra Cys, so the ONLY Cys are the 6 scaffold ones
    assert seed.count("C") == 6
    # >=1 glycine anchor so chai can tokenize the all-D chain (the bug that crashed the run)
    assert "G" in seed


def test_build_binder_seed_explicit_rejects_wrong_length():
    with pytest.raises(ValueError):
        build_binder_seed(31, [1, 2, 3, 4, 5, 6], seed_seq="TOOSHORT")


def test_build_binder_seed_deterministic():
    pos = knottin_cys_positions(31)
    a = build_binder_seed(31, pos, rng_seed=7)
    b = build_binder_seed(31, pos, rng_seed=7)
    assert a == b


# ── Disulfide COVALENT rows on the binder chain ─────────────────────────────────────

def test_build_nonalpha_disulfide_rows_three_covalent_sg_sg():
    rows = build_nonalpha_disulfide_rows([3, 8, 13, 18, 23, 28], binder_chain="C")
    assert len(rows) == 3
    assert all(",covalent," in r and "@SG" in r for r in rows)
    # first ICK bond I-IV = (3, 18) on chain C
    cols = rows[0].split(",")
    assert cols[0] == "C" and cols[1] == "C3@SG"
    assert cols[2] == "C" and cols[3] == "C18@SG"


# ── HA target FASTA parsing (the MSA-keyed sequences) ───────────────────────────────

def test_load_ha_entities_parses_two_L_chains(tmp_path):
    fasta = tmp_path / "ha.fasta"
    fasta.write_text(">protein|HA1\nPGDTIC\n>protein|HA2\nGLFGAI\n")
    ents = load_ha_entities(fasta)
    assert [e["name"] for e in ents] == ["HA1", "HA2"]
    assert all(e["chirality"] == "L" and e["type"] == "protein" for e in ents)
    assert ents[0]["sequence"] == "PGDTIC" and ents[1]["sequence"] == "GLFGAI"
