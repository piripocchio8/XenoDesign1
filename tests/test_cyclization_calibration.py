"""CPU tests for the cyclization-calibration wiring (pure parts; GPU body is smoke-tested
in tests/gpu). Covers the panel construction, the termini one-letter parse, the
head-to-tail closure restraint emission (general — NO Zn/coordination), and the per-term
objective's no-CIF fallback.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import xenodesign.classes.base  # noqa: F401  (import-order guard)
import scripts.run_cyclization_calibration as R
from xenodesign.abc.calibration import intramolecular_per_term_fn


def test_build_lane_pos_is_real_mixed_chirality():
    cases = R.build_lane("pos")
    ids = {c["id"] for c in cases}
    assert ids == {"POS-6", "POS-24"}
    assert all(c["is_good"] for c in cases)
    by = {c["id"]: c for c in cases}
    assert by["POS-6"]["d_fasta"] == "EP(DPR)KP(DPR)"          # EPpKPp
    # 5 D blocks per 12-mer (DGN,DGL,DLY,DLE,DHI; the two AIB are ALA proxies) x2 = 10.
    assert by["POS-24"]["d_fasta"].count("(") == 10
    assert by["POS-24"]["length"] == 24


def test_build_lane_neg_is_full_L_no_d_blocks():
    cases = R.build_lane("neg")
    ids = {c["id"] for c in cases}
    assert ids == {"NEG-6", "NEG-24"}
    assert not any(c["is_good"] for c in cases)
    for c in cases:
        assert "(" not in c["d_fasta"]                         # full-L, no D-CCD blocks
        assert "G" not in c["d_fasta"]                         # every position a stereocenter
        assert len(c["d_fasta"]) == c["length"]


def test_neg_is_deterministic():
    assert R.build_lane("neg") == R.build_lane("neg")


def test_build_lane_rejects_unknown():
    with pytest.raises(ValueError):
        R.build_lane("bogus")


def test_termini_one_letters_handles_d_blocks():
    # POS-6 termini: E (L) ... (DPR) -> P parent; POS-24: K ... (DHI) -> H parent.
    assert R._termini_one_letters("EP(DPR)KP(DPR)") == ("E", 6, "P")
    n, length, c = R._termini_one_letters(R.POS24_FASTA)
    assert (n, length, c) == ("K", 24, "H")


def test_closure_restraint_is_only_head_to_tail_covalent(tmp_path):
    p = R.write_closure_restraint(tmp_path / "c.restraints", "EP(DPR)KP(DPR)")
    text = p.read_text().strip().splitlines()
    assert len(text) == 2                                       # header + exactly ONE row
    row = text[1]
    # head-to-tail: C-term carbonyl C (pos 6) -> N-term amide N (pos 1), intra-chain A, covalent
    assert row.startswith("A,P6@C,A,E1@N,covalent")
    assert "contact" not in p.read_text()                      # NO coordination restraint
    assert "pocket" not in p.read_text()
    assert "ZN" not in p.read_text() and "Zn" not in p.read_text()


def test_per_term_fn_no_cif_fallback_is_ptm_only():
    fn = intramolecular_per_term_fn("A")

    class _P:  # prediction with NO _cif_path
        ptm = 0.5

    out = fn(_P())
    assert out["ptm"] == pytest.approx(0.5)
    assert out["mainchain_plddt"] is None and out["geometry"] is None
    assert out["cn_distance"] is None and out["closed"] is False
    # objective is the ptm-weighted fallback (matches intramolecular_objective_fn).
    from xenodesign.classes.cyclic import INTRAMOLECULAR_WEIGHTS
    assert out["objective"] == pytest.approx(INTRAMOLECULAR_WEIGHTS["ptm"] * 0.5)
