"""ABC T2 — calibration helpers (pure CPU): spearman, summarize, K* selection.

The GATING build step in pure form. Given per-(step, sequence) objective values,
compute Spearman rank correlation vs the full-200 reference and the good-vs-bad
separation margin, then pick the smallest step count that clears both thresholds.
No GPU here (the real predict + GPU run is T3).
"""
import pytest

from xenodesign.abc.calibration import (
    calibrate_fast_oracle,
    spearman,
    select_k_star,
    summarize_calibration,
)


# ── spearman ────────────────────────────────────────────────────────────────

def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    # Tied ranks must not blow up; constant vector -> correlation defined as 0.0.
    assert spearman([1, 1, 1], [1, 2, 3]) == pytest.approx(0.0)
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)


def test_spearman_monotone_nonlinear_is_one():
    # Rank correlation cares about order, not linearity.
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)


# ── summarize_calibration ─────────────────────────────────────────────────────

def test_summarize_reports_corr_and_margin():
    # objective_by_step[step] = {seq_id: value}; full reference = 200.
    good, bad = "good", "bad"
    objective_by_step = {
        50:  {good: 0.80, bad: 0.40},
        200: {good: 0.90, bad: 0.30},
    }
    labels = {good: True, bad: False}     # True = known-good
    summ = summarize_calibration(objective_by_step, labels=labels, reference_step=200)
    assert summ[50]["rank_corr"] == pytest.approx(1.0)      # ranks good>bad like 200
    assert summ[50]["margin"] == pytest.approx(0.40)        # good-bad separation at this step


def test_summarize_margin_is_mean_separation_multi_member():
    good_a, good_b, bad_a, bad_b = "ga", "gb", "ba", "bb"
    objective_by_step = {
        50:  {good_a: 0.9, good_b: 0.7, bad_a: 0.3, bad_b: 0.1},
        200: {good_a: 0.9, good_b: 0.8, bad_a: 0.2, bad_b: 0.1},
    }
    labels = {good_a: True, good_b: True, bad_a: False, bad_b: False}
    summ = summarize_calibration(objective_by_step, labels=labels, reference_step=200)
    # margin = mean(good) - mean(bad) = 0.8 - 0.2 = 0.6 at step 50.
    assert summ[50]["margin"] == pytest.approx(0.6)
    assert summ[50]["rank_corr"] == pytest.approx(1.0)


# ── select_k_star (plan API: per-step corr + threshold) ───────────────────────

def test_select_k_star_picks_smallest_passing_step():
    per_step = {10: 0.4, 25: 0.92, 50: 0.95, 100: 0.99, 200: 1.0}
    assert select_k_star(per_step, threshold=0.9) == 25


def test_select_k_star_none_when_no_step_passes():
    per_step = {10: 0.2, 25: 0.3, 50: 0.5}
    assert select_k_star(per_step, threshold=0.9) is None   # spec §7: STOP/rethink


# ── select_k_star (prompt API: summary table + min_rank_corr + min_margin) ────

def test_select_k_star_from_summary_requires_both_corr_and_margin():
    # A step can clear rank-corr but FAIL the margin gate (good/bad collapse) -> rejected.
    summary = {
        10:  {"rank_corr": 0.95, "margin": 0.02},   # good corr, BAD margin -> reject
        25:  {"rank_corr": 0.95, "margin": 0.30},   # both pass -> K*
        50:  {"rank_corr": 0.99, "margin": 0.40},
        200: {"rank_corr": 1.00, "margin": 0.45},
    }
    assert select_k_star(summary, min_rank_corr=0.9, min_margin=0.1) == 25


def test_select_k_star_from_summary_none_when_margin_never_clears():
    summary = {
        10:  {"rank_corr": 0.95, "margin": 0.02},
        25:  {"rank_corr": 0.95, "margin": 0.03},
        50:  {"rank_corr": 0.99, "margin": 0.05},
    }
    assert select_k_star(summary, min_rank_corr=0.9, min_margin=0.1) is None


def test_select_k_star_from_summary_negative_margin_rejected():
    # good < bad (ranking inverted at low steps) -> margin negative -> reject even if |corr| high.
    summary = {
        10:  {"rank_corr": 0.95, "margin": -0.10},
        25:  {"rank_corr": 0.95, "margin": 0.20},
    }
    assert select_k_star(summary, min_rank_corr=0.9, min_margin=0.1) == 25


# ── monotone passes / shuffled fails (the spec's headline contract) ───────────

def test_monotone_objective_table_yields_a_kstar():
    # Each step ranks the panel identically to full-200 (monotone) and keeps good>bad.
    panel = {"g1": True, "g2": True, "b1": False, "b2": False}
    ref = {"g1": 0.9, "g2": 0.7, "b1": 0.4, "b2": 0.2}
    objective_by_step = {}
    for step, scale in ((10, 0.5), (25, 0.7), (50, 0.85), (100, 0.95), (200, 1.0)):
        # scale compresses values but preserves ORDER -> rank_corr stays 1.0, margin shrinks with scale.
        objective_by_step[step] = {k: v * scale for k, v in ref.items()}
    summary = summarize_calibration(objective_by_step, labels=panel, reference_step=200)
    k = select_k_star(summary, min_rank_corr=0.9, min_margin=0.1)
    assert k is not None and k <= 200
    assert summary[k]["rank_corr"] >= 0.9 and summary[k]["margin"] >= 0.1


def test_shuffled_low_step_objective_fails_kstar():
    # At step 10 the objective is scrambled (anti-correlated) -> must NOT be chosen.
    panel = {"g1": True, "g2": True, "b1": False, "b2": False}
    ref = {"g1": 0.9, "g2": 0.7, "b1": 0.4, "b2": 0.2}
    objective_by_step = {
        10:  {"g1": 0.2, "g2": 0.4, "b1": 0.7, "b2": 0.9},   # inverted!
        25:  dict(ref),
        200: dict(ref),
    }
    summary = summarize_calibration(objective_by_step, labels=panel, reference_step=200)
    assert summary[10]["rank_corr"] < 0.9          # shuffled fails the rank gate
    k = select_k_star(summary, min_rank_corr=0.9, min_margin=0.1)
    assert k == 25                                  # smallest step that actually ranks


# ── calibrate_fast_oracle orchestration (fake predict/objective, no GPU) ──────

def test_calibrate_fast_oracle_orchestrates_pure_helpers():
    # Fake predictor returns a per-step scale; fake objective is identity * scale.
    cases = [
        {"id": "g", "is_good": True, "base": 0.9},
        {"id": "b", "is_good": False, "base": 0.3},
    ]
    scale = {10: 0.5, 25: 0.8, 200: 1.0}

    def predict_fn(case, step):
        return {"v": case["base"] * scale[step]}

    def objective_fn(pred):
        return pred["v"]

    res = calibrate_fast_oracle(
        cases, steps=[10, 25, 200], predict_fn=predict_fn,
        objective_fn=objective_fn, reference_step=200,
        min_rank_corr=0.9, min_margin=0.1,
    )
    assert res["k_star"] == 10                    # monotone scaling preserves the ranking
    assert res["summary"][10]["rank_corr"] == pytest.approx(1.0)
    assert res["summary"][10]["margin"] > 0.1


def test_calibrate_fast_oracle_scores_inf_on_predict_error():
    cases = [{"id": "g", "is_good": True}, {"id": "b", "is_good": False}]

    def boom(case, step):
        raise RuntimeError("chai blew up")

    res = calibrate_fast_oracle(
        cases, steps=[200], predict_fn=boom, objective_fn=lambda p: 1.0,
        reference_step=200,
    )
    assert res["objective_by_step"][200]["g"] == float("-inf")  # graceful, never crashes


# ── GPU smoke (the real low-step Chai path; container only) ───────────────────

@pytest.mark.gpu
def test_calibrate_gpu_smoke_two_steps_one_pair(tmp_path):
    """Real chai at {10,200} on GOOD vs one BAD; both steps return finite corr/margin.

    Cheap gate that the GPU wiring (predict -> CIF -> intramolecular objective ->
    summarize) is sound. The scientific K* comes from the full panel run (the
    results doc), not this smoke.
    """
    from tests.gpu.conftest import require_chai, require_cuda
    require_cuda()
    require_chai()

    import xenodesign.classes.base  # noqa: F401  (import-order guard)
    from pathlib import Path

    import scripts.run_diffstep_calibration as runner
    from xenodesign.abc.calibration import (
        chai_predict_fn,
        intramolecular_objective_fn,
    )

    cif = Path("XenoDesign1_local_ref/6UFA.cif")
    if not cif.exists():
        pytest.skip("6UFA deposit CIF not present")
    panel = runner.build_panel(cif, n_bad=1)
    for c in panel:
        c["out_root"] = str(tmp_path / "unrestr")
        c["restrained"] = False
        c["constraint_path"] = None

    predict_fn = chai_predict_fn(device="cuda:0", seed=0)
    objective_fn = intramolecular_objective_fn(chain_name="A")
    obj = {10: {}, 200: {}}
    for case in panel:
        for step in (10, 200):
            pred = predict_fn(case, step)
            obj[step][case["id"]] = float(objective_fn(pred))

    labels = {c["id"]: bool(c["is_good"]) for c in panel}
    summ = summarize_calibration(obj, labels=labels, reference_step=200)
    import math
    for step in (10, 200):
        assert math.isfinite(summ[step]["rank_corr"])
        assert math.isfinite(summ[step]["margin"])
