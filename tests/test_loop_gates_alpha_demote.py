"""S3a.3a: alpha_demote_gated_accept rejects an over-helical non_alpha candidate (knottins are
anti-alpha) and accepts a low-helix one. Mirrors chirality_gated_accept's panel/score_fn shape.

Tests drive the gate with the REAL object shape: a JudgePanel with a score_fn that returns a
RefereeScore with helix_fraction set, so a regression to reading a nonexistent prediction field
would yield helix=None → always-accept and would FAIL these tests."""
from __future__ import annotations

import pytest
from xenodesign.loop import LoopState, LoopStep, alpha_demote_gated_accept
from xenodesign.judges.panel import JudgePanel, RefereeScore


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_panel(helix_fraction):
    """Build a JudgePanel whose score_fn returns a RefereeScore with helix_fraction.

    This is the REAL shape the loop passes: a JudgePanel score_fn receives a LoopStep and
    returns a RefereeScore — helix_fraction lives on RefereeScore, NOT on prediction.
    """
    def _score_fn(step: LoopStep) -> RefereeScore:
        # In production the alpha score_fn computes helix from the CIF via
        # _binder_helix_fraction; in tests we inject it via the LoopStep.score field
        # as a stand-in (a simple convention for these unit tests only).
        return RefereeScore(
            chirality_violation=0.0,
            iptm=0.5,
            helix_fraction=helix_fraction,
        )
    return JudgePanel(score_fn=_score_fn)


def _step():
    """Minimal LoopStep — prediction has NO helix_fraction (mirrors real Prediction)."""
    class _Pred:
        iptm = 0.5
        # deliberately NO helix_fraction attribute, as on the real Prediction object
    return LoopStep(
        state=LoopState(d_fasta="", coords=None),
        prediction=_Pred(),
        score=0.0,
    )


# ── gate construction ─────────────────────────────────────────────────────────

def test_alpha_demote_requires_panel_with_score_fn():
    """alpha_demote_gated_accept must raise if panel has no score_fn (mirrors chirality gate)."""
    panel_no_fn = JudgePanel()  # no score_fn
    with pytest.raises(ValueError, match="score_fn"):
        alpha_demote_gated_accept(panel_no_fn, max_helix_frac=0.5)


# ── accept / reject logic ────────────────────────────────────────────────────

def test_alpha_demote_rejects_helical():
    """Over-helical candidate (helix > threshold) is rejected."""
    panel = _make_panel(helix_fraction=0.9)
    gate = alpha_demote_gated_accept(panel, max_helix_frac=0.5)
    assert gate(_step(), _step()) is False      # 0.9 > 0.5 → reject


def test_alpha_demote_accepts_low_helix():
    """Low-helix candidate (helix < threshold) is accepted."""
    panel = _make_panel(helix_fraction=0.2)
    gate = alpha_demote_gated_accept(panel, max_helix_frac=0.5)
    assert gate(_step(), _step()) is True       # 0.2 ≤ 0.5 → accept


def test_alpha_demote_boundary_at_threshold_accepted():
    """Candidate at exactly max_helix_frac (not strictly greater) is accepted."""
    panel = _make_panel(helix_fraction=0.5)
    gate = alpha_demote_gated_accept(panel, max_helix_frac=0.5)
    assert gate(_step(), _step()) is True       # 0.5 is NOT > 0.5 → accept


def test_alpha_demote_boundary_just_above_threshold_rejected():
    """Candidate just above max_helix_frac is rejected."""
    panel = _make_panel(helix_fraction=0.501)
    gate = alpha_demote_gated_accept(panel, max_helix_frac=0.5)
    assert gate(_step(), _step()) is False      # 0.501 > 0.5 → reject


def test_alpha_demote_missing_helix_accepts():
    """None helix_fraction (unreadable CIF) never silently kills a trajectory — accept."""
    panel = _make_panel(helix_fraction=None)
    gate = alpha_demote_gated_accept(panel, max_helix_frac=0.5)
    assert gate(_step(), _step()) is True


def test_alpha_demote_regression_reads_referee_not_prediction():
    """Regression guard: if helix_fraction were read from prediction (where it doesn't exist),
    the gate would always see None and always accept — this test catches that no-op.

    The panel score_fn returns helix=0.9 (should reject). If the gate reads from prediction
    instead (which has no helix_fraction), it sees None → returns True (no-op). A regression
    to reading prediction would make this test FAIL (True != False)."""
    panel = _make_panel(helix_fraction=0.9)
    gate = alpha_demote_gated_accept(panel, max_helix_frac=0.5)
    # prediction has no helix_fraction; gate MUST read from RefereeScore via panel._score_fn
    assert gate(_step(), _step()) is False
