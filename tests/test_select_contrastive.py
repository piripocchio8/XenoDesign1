"""CPU-only tests for scripts/select_contrastive.py — pure replicate-averaging +
contrastive-margin + noise-aware population-select logic (T01 / ADR-019 / ADR-021).

The GPU predictor (predict_batch/predict_complex) and the per-dir scorer
(contrastive_rank._score_dir) are MOCKED: we feed canned per-(design,kind,rep)
scores and assert on the pure margin/selection math.  No GPU, torch, chai, or
freesasa required.  Run with:
    PYTHONPATH=$PWD python -m pytest tests/test_select_contrastive.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.select_contrastive import (
    binder_shifts,
    select_by_margin,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _scores(spec: dict, shifts=(3, 4, 7)) -> dict:
    """Build a per-(design,kind,rep) score dict from a compact spec.

    spec = {design: {"real": [obj_rep0, obj_rep1, ...],
                     "shift3": [...], "shift4": [...], "shift7": [...]}}
    obj values may be floats (iptm mirrored to the same value) or
    (obj, iptm) tuples.
    """
    out: dict = {}
    for design, kinds in spec.items():
        for kind, reps in kinds.items():
            for r, v in enumerate(reps):
                if isinstance(v, (tuple, list)):
                    obj, iptm = v
                else:
                    obj, iptm = v, v
                out[(design, kind, r)] = {"obj": obj, "iptm": iptm}
    return out


# ── binder_shifts (shifts applied to the binder) ──────────────────────────────

class TestBinderShifts:

    def test_default_shifts_3_4_7(self):
        b = "abcdefghij"
        shifted = binder_shifts(b)
        assert set(shifted.keys()) == {3, 4, 7}

    def test_circular_shift_of_binder(self):
        b = "abcdefghij"
        shifted = binder_shifts(b, shifts=(3,))
        # circular shift by 3: b[3:] + b[:3]
        assert shifted[3] == "defghijabc"

    def test_shift_does_not_change_length(self):
        b = "acdefgaiae"  # a D-binder (lowercase, no W/Y placeholders)
        for s, seq in binder_shifts(b, shifts=(3, 4, 7)).items():
            assert len(seq) == len(b)

    def test_shifts_applied_to_binder_not_target(self):
        """Each shift is a distinct rotation of the *binder* sequence."""
        b = "abcdefghij"
        shifted = binder_shifts(b, shifts=(3, 4, 7))
        assert shifted[3] == b[3:] + b[:3]
        assert shifted[4] == b[4:] + b[:4]
        assert shifted[7] == b[7:] + b[:7]
        # all three differ from the real binder and from each other
        seqs = {b, *shifted.values()}
        assert len(seqs) == 4


# ── select_by_margin: margin = mean(real) − max_shift mean(shift) ─────────────

class TestMargin:

    def test_margin_is_mean_real_minus_max_shift_mean(self):
        # real mean = 0.80 ; shift means: 0.50, 0.70, 0.60 -> worst (max) = 0.70
        scores = _scores({
            "d0": {
                "real": [0.80, 0.80, 0.80],
                "shift3": [0.50, 0.50, 0.50],
                "shift4": [0.70, 0.70, 0.70],
                "shift7": [0.60, 0.60, 0.60],
            }
        })
        rows = select_by_margin(scores, reps=3)
        r = rows[0]
        assert r["real_obj_mean"] == pytest.approx(0.80)
        assert r["worst_shift"] == "shift4"
        assert r["worst_shift_obj_mean"] == pytest.approx(0.70)
        assert r["obj_margin"] == pytest.approx(0.80 - 0.70)

    def test_replicate_mean_over_reps(self):
        scores = _scores({
            "d0": {
                "real": [0.6, 0.8, 1.0],     # mean 0.8
                "shift3": [0.4, 0.5, 0.6],   # mean 0.5
                "shift4": [0.1, 0.2, 0.3],   # mean 0.2
                "shift7": [0.0, 0.1, 0.2],   # mean 0.1
            }
        })
        rows = select_by_margin(scores, reps=3)
        r = rows[0]
        assert r["real_obj_mean"] == pytest.approx(0.8)
        assert r["worst_shift_obj_mean"] == pytest.approx(0.5)
        assert r["obj_margin"] == pytest.approx(0.3)


class TestStd:

    def test_std_computed_over_reps(self):
        # population std of [0.6, 0.8, 1.0] = 0.16329...
        import statistics
        expect = statistics.pstdev([0.6, 0.8, 1.0])
        scores = _scores({
            "d0": {
                "real": [0.6, 0.8, 1.0],
                "shift3": [0.5, 0.5, 0.5],
                "shift4": [0.5, 0.5, 0.5],
                "shift7": [0.5, 0.5, 0.5],
            }
        })
        rows = select_by_margin(scores, reps=3)
        assert rows[0]["real_obj_std"] == pytest.approx(expect)

    def test_margin_std_propagated_from_real_and_worst_shift(self):
        import math
        import statistics
        real = [0.6, 0.8, 1.0]
        worst = [0.30, 0.50, 0.70]   # this is the max-mean shift (mean 0.50)
        other = [0.0, 0.0, 0.0]
        scores = _scores({
            "d0": {
                "real": real,
                "shift3": worst,
                "shift4": other,
                "shift7": other,
            }
        })
        rows = select_by_margin(scores, reps=3)
        r = rows[0]
        exp = math.sqrt(statistics.pstdev(real) ** 2 + statistics.pstdev(worst) ** 2)
        assert r["obj_margin_std"] == pytest.approx(exp)


# ── noise-aware selection: margin − k·std > 0 (NOT bare sign test) ────────────

class TestNoiseAwareSelection:

    def test_clean_margin_selected(self):
        # margin 0.30, std tiny -> selected
        scores = _scores({
            "clean": {
                "real": [0.80, 0.81, 0.79],
                "shift3": [0.50, 0.50, 0.50],
                "shift4": [0.49, 0.50, 0.51],
                "shift7": [0.48, 0.49, 0.50],
            }
        })
        rows = select_by_margin(scores, reps=3, k=1.0)
        assert rows[0]["selected"] is True

    def test_margin_within_noise_not_selected(self):
        # positive but tiny margin, large per-rep spread -> margin < std -> NOT selected
        scores = _scores({
            "noisy": {
                "real": [0.55, 0.85, 0.40],     # mean 0.60, big spread
                "shift3": [0.40, 0.80, 0.30],   # mean 0.50, big spread
                "shift4": [0.0, 0.0, 0.0],
                "shift7": [0.0, 0.0, 0.0],
            }
        })
        rows = select_by_margin(scores, reps=3, k=1.0)
        r = rows[0]
        assert r["obj_margin"] > 0                  # bare sign test would PASS
        assert r["obj_margin"] < r["obj_margin_std"]  # but within noise
        assert r["selected"] is False               # noise-aware: rejected

    def test_negative_margin_not_selected(self):
        scores = _scores({
            "bad": {
                "real": [0.40, 0.40, 0.40],
                "shift3": [0.70, 0.70, 0.70],   # binds shifted BETTER -> non-specific
                "shift4": [0.50, 0.50, 0.50],
                "shift7": [0.50, 0.50, 0.50],
            }
        })
        rows = select_by_margin(scores, reps=3, k=1.0)
        assert rows[0]["obj_margin"] < 0
        assert rows[0]["selected"] is False

    def test_k_zero_is_bare_sign_test(self):
        """With k=0 the threshold collapses to the bare sign test (margin > 0)."""
        scores = _scores({
            "noisy": {
                "real": [0.55, 0.85, 0.40],
                "shift3": [0.40, 0.80, 0.30],
                "shift4": [0.0, 0.0, 0.0],
                "shift7": [0.0, 0.0, 0.0],
            }
        })
        rows = select_by_margin(scores, reps=3, k=0.0)
        # margin is positive, so with k=0 it IS selected (this is the behaviour we improve on)
        assert rows[0]["obj_margin"] > 0
        assert rows[0]["selected"] is True

    def test_higher_k_is_stricter(self):
        scores = _scores({
            "borderline": {
                "real": [0.70, 0.80, 0.90],     # mean 0.80, pstd ~0.0816
                "shift3": [0.60, 0.70, 0.80],   # mean 0.70, pstd ~0.0816
                "shift4": [0.0, 0.0, 0.0],
                "shift7": [0.0, 0.0, 0.0],
            }
        })
        # margin 0.10 ; margin_std = sqrt(0.0816^2 + 0.0816^2) ~ 0.1155
        lenient = select_by_margin(scores, reps=3, k=0.5)
        strict = select_by_margin(scores, reps=3, k=1.0)
        assert lenient[0]["selected"] is True   # 0.10 - 0.5*0.1155 > 0
        assert strict[0]["selected"] is False   # 0.10 - 1.0*0.1155 < 0


# ── ranking & pool-level behaviour ────────────────────────────────────────────

class TestRanking:

    def test_pool_ranked_by_margin_descending(self):
        scores = _scores({
            "lo": {"real": [0.60] * 3, "shift3": [0.55] * 3,
                   "shift4": [0.50] * 3, "shift7": [0.50] * 3},   # margin 0.05
            "hi": {"real": [0.90] * 3, "shift3": [0.50] * 3,
                   "shift4": [0.50] * 3, "shift7": [0.50] * 3},   # margin 0.40
            "mid": {"real": [0.80] * 3, "shift3": [0.60] * 3,
                    "shift4": [0.60] * 3, "shift7": [0.60] * 3},  # margin 0.20
        })
        rows = select_by_margin(scores, reps=3)
        order = [r["design"] for r in rows]
        assert order == ["hi", "mid", "lo"]
        margins = [r["obj_margin"] for r in rows]
        assert margins == sorted(margins, reverse=True)

    def test_iptm_margin_reported_secondary(self):
        scores = _scores({
            "d0": {
                "real": [(0.80, 0.60)] * 3,
                "shift3": [(0.50, 0.40)] * 3,
                "shift4": [(0.50, 0.30)] * 3,
                "shift7": [(0.50, 0.20)] * 3,
            }
        })
        rows = select_by_margin(scores, reps=3)
        r = rows[0]
        # obj margin uses obj axis; iptm margin is reported alongside
        assert r["obj_margin"] == pytest.approx(0.30)
        assert r["real_iptm_mean"] == pytest.approx(0.60)
        # iptm worst shift = max over shifts of iptm mean = 0.40
        assert r["iptm_margin"] == pytest.approx(0.60 - 0.40)

    def test_iptm_axis_selector(self):
        """When axis='iptm', the margin/selection is computed on the iptm field."""
        scores = _scores({
            "d0": {
                "real": [(0.40, 0.90)] * 3,     # obj would give NEGATIVE margin...
                "shift3": [(0.80, 0.50)] * 3,   # ...but iptm gives +0.40
                "shift4": [(0.80, 0.50)] * 3,
                "shift7": [(0.80, 0.50)] * 3,
            }
        })
        rows = select_by_margin(scores, reps=3, axis="iptm")
        r = rows[0]
        assert r["obj_margin"] == pytest.approx(0.90 - 0.50)  # margin now on iptm
        assert r["selected"] is True

    def test_reps_two_still_computes(self):
        scores = _scores({
            "d0": {
                "real": [0.80, 0.82],
                "shift3": [0.50, 0.52],
                "shift4": [0.40, 0.42],
                "shift7": [0.30, 0.32],
            }
        })
        rows = select_by_margin(scores, reps=2)
        assert rows[0]["reps"] == 2
        assert rows[0]["obj_margin"] == pytest.approx(0.81 - 0.51)


# ── incompleteness handling (flag, don't crash) ───────────────────────────────

class TestIncomplete:

    def test_missing_real_design_skipped(self):
        scores = _scores({
            "d0": {
                "shift3": [0.5, 0.5, 0.5],
                "shift4": [0.4, 0.4, 0.4],
                "shift7": [0.3, 0.3, 0.3],
            }
        })
        rows = select_by_margin(scores, reps=3)
        assert rows == []

    def test_no_shifts_design_skipped(self):
        scores = _scores({"d0": {"real": [0.8, 0.8, 0.8]}})
        rows = select_by_margin(scores, reps=3)
        assert rows == []

    def test_fewer_reps_than_expected_flagged_not_crash(self):
        # only 2 of 3 reps present for the real kind
        scores = {
            ("d0", "real", 0): {"obj": 0.80, "iptm": 0.80},
            ("d0", "real", 1): {"obj": 0.80, "iptm": 0.80},
            ("d0", "shift3", 0): {"obj": 0.50, "iptm": 0.50},
            ("d0", "shift3", 1): {"obj": 0.50, "iptm": 0.50},
            ("d0", "shift3", 2): {"obj": 0.50, "iptm": 0.50},
            ("d0", "shift4", 0): {"obj": 0.40, "iptm": 0.40},
            ("d0", "shift4", 1): {"obj": 0.40, "iptm": 0.40},
            ("d0", "shift4", 2): {"obj": 0.40, "iptm": 0.40},
            ("d0", "shift7", 0): {"obj": 0.30, "iptm": 0.30},
            ("d0", "shift7", 1): {"obj": 0.30, "iptm": 0.30},
            ("d0", "shift7", 2): {"obj": 0.30, "iptm": 0.30},
        }
        rows = select_by_margin(scores, reps=3)
        # still scored, but flagged incomplete and averaged over available reps
        assert len(rows) == 1
        assert rows[0]["incomplete"] is True
        assert rows[0]["real_obj_mean"] == pytest.approx(0.80)


class TestRepItemsPaths:
    """Regression: _rep_items must strip the FULL (multi-level) out_root, not double the path."""

    def test_strips_full_out_root_no_doubling(self):
        from scripts.select_contrastive import _rep_items
        base = [{"out_dir": "a/b/c/GT/real", "name": "GT__real"}]
        out = _rep_items(base, "a/b/c", 0)
        assert out[0]["out_dir"] == "a/b/c/rep0/GT/real"  # not a/b/c/rep0/a/b/c/GT/real

    def test_rep_index_and_kind_preserved(self):
        from scripts.select_contrastive import _rep_items
        base = [{"out_dir": "root/d0/shift3", "name": "d0__shift3"}]
        out = _rep_items(base, "root", 2)
        assert out[0]["out_dir"] == "root/rep2/d0/shift3"
