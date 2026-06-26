# tests/test_score_controls.py
"""CPU tests for the pure pre-registered verdict math in scripts/score_controls.py.

score_to_verdict(rows, margins) is the GPU-free core of the within-Chai controls
(#35): it consumes the per-case bundles (interface_iptm + ipae_mean per label) and
applies WIN_MARGINS to decide SPECIFIC. These tests pin every dimension of the
verdict — the ipTM gaps, the ipAE gaps, and the incomplete-controls guard — without
ever touching Chai/GPU.
"""
import pytest

from scripts.score_controls import score_to_verdict

# WIN_MARGINS-shaped margins (mirrors xenodesign.eval.controls.WIN_MARGINS).
MARGINS = {"iptm_gap": 0.08, "ipae_gap_A": 2.0, "off_target_gap": 0.15}


def _row(label, iptm, ipae, ok=True):
    return {"label": label, "ok": ok, "interface_iptm": iptm, "ipae_mean": ipae}


def _passing_rows():
    """A fully-passing synthetic set: design clearly beats both controls + reference.

    design ipTM 0.70 / ipAE 6.0; scrambles ~0.50 / ~9.0; off-targets ~0.45 / ~10.0;
    reference 0.44. Clears iptm_gap (0.08), off_target_gap (0.15) and ipae_gap_A (2.0).
    """
    return [
        _row("design", 0.70, 6.0),
        _row("scramble1", 0.50, 9.0),
        _row("scramble2", 0.48, 9.5),
        _row("reference", 0.44, 12.0),
        _row("offtgt:helixA", 0.45, 10.0),
        _row("offtgt:helixB", 0.40, 11.0),
    ]


def test_fully_passing_is_specific():
    v = score_to_verdict(_passing_rows(), MARGINS)
    assert v["SPECIFIC"] is True
    # Every dimension was evaluated and passed.
    assert v["checks"] == {
        "beats_scramble_by_margin": True,
        "specific_vs_offtarget": True,
        "beats_reference": True,
        "ipae_gap_vs_scramble": True,
        "ipae_gap_vs_offtarget": True,
    }
    assert v["reasons"] == []
    assert v["scramble_present"] and v["offtarget_present"]


def test_worst_decoy_is_reported():
    """The verdict reports the WORST off-target (max ipTM = hardest decoy) + its label."""
    v = score_to_verdict(_passing_rows(), MARGINS)
    # off-targets are 0.45 (helixA) and 0.40 (helixB); the worst (max) is helixA at 0.45.
    assert v["worst_offtarget_iptm"] == pytest.approx(0.45)
    assert v["worst_offtarget_label"] == "offtgt:helixA"
    # The mean is still reported as a secondary number.
    assert v["mean_offtarget_iptm"] == pytest.approx((0.45 + 0.40) / 2)


def test_high_offtarget_iptm_fails_specificity():
    """Off-target ipTM high -> specific_vs_offtarget False, SPECIFIC False."""
    rows = _passing_rows()
    # Push off-target ipTM up so design - worst_off < off_target_gap (0.15).
    rows = [r for r in rows if not r["label"].startswith("offtgt:")]
    rows += [_row("offtgt:helixA", 0.68, 10.0), _row("offtgt:helixB", 0.66, 11.0)]
    v = score_to_verdict(rows, MARGINS)
    assert v["checks"]["specific_vs_offtarget"] is False
    assert v["SPECIFIC"] is False


def test_worst_decoy_not_mean_drives_specificity():
    """A design that beats the MEAN off-target but NOT the worst decoy is NOT specific.

    design 0.70; off-targets {0.40, 0.40, 0.62}. mean = 0.473 so design - mean = 0.227 >= 0.15
    (would PASS on the mean), but worst = 0.62 so design - worst = 0.08 < 0.15 -> FAIL on worst.
    This pins that the honest check uses the hardest single decoy, not the mean.
    """
    rows = [r for r in _passing_rows() if not r["label"].startswith("offtgt:")]
    rows += [
        _row("offtgt:helixA", 0.40, 10.0),
        _row("offtgt:helixB", 0.40, 11.0),
        _row("offtgt:register_shift", 0.62, 10.5),   # the hardest decoy
    ]
    v = score_to_verdict(rows, MARGINS)
    # Mean would have passed; worst does not.
    mean_off = (0.40 + 0.40 + 0.62) / 3
    assert v["design_iptm"] - mean_off >= MARGINS["off_target_gap"]   # mean-based would pass
    assert v["worst_offtarget_iptm"] == pytest.approx(0.62)
    assert v["worst_offtarget_label"] == "offtgt:register_shift"
    assert v["checks"]["specific_vs_offtarget"] is False
    assert v["SPECIFIC"] is False


def test_ipae_gap_uses_most_confident_decoy():
    """The off-target ipAE gap uses the decoy with the LOWEST ipAE (most-confident decoy).

    design ipAE 6.0; off-target ipAEs {10.0, 11.0, 7.5}. The most-confident decoy is 7.5,
    and 7.5 - 6.0 = 1.5 < ipae_gap_A (2.0) -> ipae_gap_vs_offtarget FAILS, even though the
    MEAN off-target ipAE (9.5) would clear the gap.
    """
    rows = [r for r in _passing_rows() if not r["label"].startswith("offtgt:")]
    rows += [
        _row("offtgt:helixA", 0.40, 10.0),
        _row("offtgt:helixB", 0.42, 11.0),
        _row("offtgt:register_shift", 0.44, 7.5),    # most-confident (lowest ipAE) decoy
    ]
    v = score_to_verdict(rows, MARGINS)
    assert v["min_offtarget_ipae"] == pytest.approx(7.5)
    assert v["checks"]["ipae_gap_vs_offtarget"] is False
    assert v["SPECIFIC"] is False


def test_empty_scramble_channel_is_incomplete():
    """No scramble rows -> SPECIFIC False with an incomplete_controls reason."""
    rows = [r for r in _passing_rows() if not r["label"].startswith("scramble")]
    v = score_to_verdict(rows, MARGINS)
    assert v["SPECIFIC"] is False
    assert v["scramble_present"] is False
    assert any("incomplete_controls" in reason and "scramble" in reason
               for reason in v["reasons"])
    # The off-target channel is still present and was scored.
    assert v["offtarget_present"] is True


def test_errored_scramble_rows_count_as_missing_channel():
    """A scramble channel that only errored out (ok=False / None aggregate) is not a pass."""
    rows = [r for r in _passing_rows() if not r["label"].startswith("scramble")]
    rows += [_row("scramble1", None, None, ok=False)]
    v = score_to_verdict(rows, MARGINS)
    assert v["scramble_present"] is False
    assert v["SPECIFIC"] is False
    assert any("scramble" in reason for reason in v["reasons"])


def test_design_ipae_worse_than_scramble_fails_gap():
    """Design ipAE worse (HIGHER) than scramble -> ipae_gap_vs_scramble False."""
    rows = [r for r in _passing_rows() if r["label"] != "design"]
    # design ipAE 9.0 is NOT >= 2.0 lower than the best scramble ipAE (9.0).
    rows += [_row("design", 0.70, 9.0)]
    v = score_to_verdict(rows, MARGINS)
    assert v["checks"]["ipae_gap_vs_scramble"] is False
    assert v["SPECIFIC"] is False


def test_empty_offtarget_channel_is_incomplete():
    """No off-target rows -> SPECIFIC False with an incomplete_controls reason."""
    rows = [r for r in _passing_rows() if not r["label"].startswith("offtgt:")]
    v = score_to_verdict(rows, MARGINS)
    assert v["SPECIFIC"] is False
    assert v["offtarget_present"] is False
    assert any("incomplete_controls" in reason and "offtarget" in reason
               for reason in v["reasons"])
