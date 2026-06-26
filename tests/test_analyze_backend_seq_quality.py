"""CPU unit tests for scripts/analyze_backend_seq_quality.py (synthetic seqs, known answers)."""
import json
import math

import pytest

from scripts.analyze_backend_seq_quality import (
    composition_stats,
    heptad_quality,
    score_sequence,
    analyze_run,
    aggregate,
    _to_dpeptide,
)


# ── composition (Ala%, Gly%, Ala+Gly) ────────────────────────────────────────

def test_composition_known_fractions():
    # 10 residues: 4 Ala, 2 Gly, 4 Leu -> A%=0.4, G%=0.2, A+G=0.6
    stats = composition_stats("AAAAGGLLLL", ala_gly_floor=0.40)
    assert stats["len"] == 10
    assert stats["ala_frac"] == pytest.approx(0.4)
    assert stats["gly_frac"] == pytest.approx(0.2)
    assert stats["ala_gly_frac"] == pytest.approx(0.6)
    assert stats["ala_gly_over_floor"] is True  # 0.6 > 0.40


def test_composition_under_floor_not_flagged():
    # 3/10 Ala, 0 Gly -> A+G = 0.3 < 0.40
    stats = composition_stats("AAALLLLEEE", ala_gly_floor=0.40)
    assert stats["ala_gly_frac"] == pytest.approx(0.3)
    assert stats["ala_gly_over_floor"] is False


def test_composition_empty():
    stats = composition_stats("", ala_gly_floor=0.40)
    assert stats == {"len": 0, "ala_frac": 0.0, "gly_frac": 0.0,
                     "ala_gly_frac": 0.0, "ala_gly_over_floor": False}


# ── normalized Shannon entropy (reuses scorer.sequence_complexity) ────────────

def test_entropy_homopolymer_is_zero():
    rec = score_sequence("AAAAAAAA", ala_gly_floor=0.40)
    assert rec["norm_entropy"] == 0.0


def test_entropy_two_equal_letters_known_value():
    # 4 A + 4 L: H = 1 bit; normalized = 1 / log2(20)
    rec = score_sequence("AAAALLLL", ala_gly_floor=0.40)
    assert rec["norm_entropy"] == pytest.approx(1.0 / math.log2(20), abs=1e-4)


# ── heptad register quality ───────────────────────────────────────────────────

def test_heptad_perfect_amphipathic_core():
    # 14 residues = exactly 2 heptads. Place hydrophobic L at a(0),d(3),a(7),d(10);
    # everything else polar E. Core (a/d) -> all hydrophobic; surface -> none.
    s = list("EEEEEEEEEEEEEE")
    for i in (0, 3, 7, 10):  # heptad a and d positions for start='a'
        s[i] = "L"
    seq = "".join(s)
    h = heptad_quality(seq, heptad_start="a")
    assert h["n_core"] == 4 and h["n_surface"] == 10
    assert h["core_hydrophobic_frac"] == pytest.approx(1.0)
    assert h["surface_hydrophobic_frac"] == pytest.approx(0.0)
    assert h["heptad_match_fraction"] == pytest.approx(1.0)
    assert h["heptad_amphipathy"] == pytest.approx(1.0)


def test_heptad_polar_helix_has_no_core_seam():
    h = heptad_quality("EEEEEEEEEEEEEE", heptad_start="a")
    assert h["core_hydrophobic_frac"] == 0.0
    assert h["heptad_match_fraction"] == 0.0


def test_heptad_polyala_low_amphipathy():
    # poly-Ala: A is hydrophobic, so BOTH core and surface are saturated -> amphipathy ~ 0
    h = heptad_quality("A" * 14, heptad_start="a")
    assert h["core_hydrophobic_frac"] == pytest.approx(1.0)
    assert h["surface_hydrophobic_frac"] == pytest.approx(1.0)
    assert h["heptad_amphipathy"] == pytest.approx(0.0)


def test_heptad_start_phase_shifts_core():
    # shifting heptad_start re-assigns which indices are core; the contrast should respond.
    seq = "L" + "E" * 13  # one hydrophobic at index 0
    core_when_a = heptad_quality(seq, heptad_start="a")["core_hydrophobic_frac"]
    core_when_b = heptad_quality(seq, heptad_start="b")["core_hydrophobic_frac"]
    # index 0 is 'a' (core) for start='a' but 'b' (surface) for start='b'
    assert core_when_a > core_when_b


# ── D-peptide lowercase reporting (Gly stays G) ───────────────────────────────

def test_dpeptide_lowercase_gly_uppercase():
    assert _to_dpeptide("SLLNRTFARKGIEELIEEKLV", "D") == "sllnrtfarkGieelieeklv"


def test_l_chirality_stays_uppercase():
    assert _to_dpeptide("SLLNRTFARKGIEELIEEKLV", "L") == "SLLNRTFARKGIEELIEEKLV"


# ── end-to-end on a synthetic alpha_result.json ───────────────────────────────

def test_analyze_run_end_to_end(tmp_path):
    result = {
        "case_id": "synthetic_alpha",
        "selected_iter": 2,
        "selected_l_seq": "EEELLEEELLEEELLEEELLE",  # diverse, low A+G
        "selected_iptm": 0.71,
        "selected_chirality": 0.0,
        "trajectory": [
            {"iter": 0, "l_seq": "AAAAAAAAAAAAAAAAAAAAA", "iptm": 0.40,
             "chirality": 0.5},                       # poly-Ala, NOT clean
            {"iter": 1, "l_seq": "EELLEELLEELLEELLEELLE", "iptm": 0.60,
             "chirality": 0.05},                      # clean
            {"iter": 2, "l_seq": "EEELLEEELLEEELLEEELLE", "iptm": 0.71,
             "chirality": 0.0},                       # clean (winner seq)
        ],
    }
    rj = tmp_path / "alpha_result.json"
    rj.write_text(json.dumps(result))

    run = analyze_run(rj, backend="chai", ala_gly_floor=0.40, chir_max=0.10)
    assert run["backend"] == "chai"
    assert run["case_id"] == "synthetic_alpha"
    assert run["n_iters"] == 3
    # 2 of 3 steps have chirality <= 0.10
    assert run["chirality_clean_rate"] == pytest.approx(2 / 3)
    # winner reported lowercase (all-D)
    assert run["winner"]["seq"] == "eeelleeelleeelleeelle"
    assert run["winner"]["iptm"] == pytest.approx(0.71)
    assert run["winner"]["ala_gly_over_floor"] is False
    # iter 0 is the poly-Ala blob -> flagged over floor + zero entropy
    it0 = run["per_iter"][0]
    assert it0["ala_frac"] == pytest.approx(1.0)
    assert it0["ala_gly_over_floor"] is True
    assert it0["norm_entropy"] == 0.0

    agg = aggregate([run])
    assert agg["n_runs"] == 1 and agg["n_winners"] == 1
    # aggregate rounds to 4 dp for readable output (0.6667), so use a 4-dp tolerance
    assert agg["mean_chirality_clean_rate"] == pytest.approx(2 / 3, abs=1e-4)
