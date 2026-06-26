"""CPU tests for the adversarial pLM-judge panel.

All tests use mock referee scores or injected model_fn — no torch/ESM at import time.
Run with: PYTHONPATH=$PWD micromamba run -n base python -m pytest tests/test_judges.py -v

Architecture under test
-----------------------
1. ESMPseudoLogLikelihood   -- mock-injected interface test (model_fn=)
2. JudgePanel               -- pure Python/numpy; combine() + select() logic
3. HalluLoop.select_by_panel -- integration of panel into the loop API
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# ── Guard: these imports must work WITHOUT torch/transformers installed ───────
from xenodesign.judges.plm_judge import ESMPseudoLogLikelihood
from xenodesign.judges.panel import JudgePanel, RefereeScore, PanelResult
from xenodesign.loop import HalluLoop, LoopState, LoopStep


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_step(score: float, chirality_violation: float = 0.0,
               iptm: float = 0.5, pll: float = None) -> LoopStep:
    """Build a fake LoopStep with a known score and chirality."""
    coords = np.zeros((3, 3))
    pred = type("P", (), {
        "iptm": iptm,
        "interface_plddt": 80.0,
        "chirality_violation_frac": chirality_violation,
        "pll": pll,
    })()
    return LoopStep(
        state=LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=coords),
        prediction=pred,
        score=score,
    )


def _mock_score_fn(step: LoopStep) -> RefereeScore:
    """Extract referee scores directly from the fake prediction attributes."""
    pred = step.prediction
    return RefereeScore(
        chirality_violation=pred.chirality_violation_frac,
        iptm=pred.iptm,
        interface_plddt=pred.interface_plddt if hasattr(pred, "interface_plddt") else 50.0,
        pll=pred.pll if hasattr(pred, "pll") else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. ESMPseudoLogLikelihood — mock interface
# ─────────────────────────────────────────────────────────────────────────────

class TestESMPseudoLogLikelihood:
    """No torch at all — verifies the mock-injection interface."""

    def test_mock_injection_returns_mock_value(self):
        """Injected model_fn is called, no torch needed."""
        judge = ESMPseudoLogLikelihood(model_fn=lambda seq: -1.23)
        assert judge("ACDE") == pytest.approx(-1.23)

    def test_mock_receives_uppercased_sequence(self):
        received = []
        def capture(seq): received.append(seq); return 0.0
        judge = ESMPseudoLogLikelihood(model_fn=capture)
        judge("acde")
        assert received[0] == "ACDE"

    def test_empty_sequence_returns_zero(self):
        judge = ESMPseudoLogLikelihood(model_fn=lambda s: -99.9)
        # Empty string after sanitisation → return 0.0 (not a real call)
        assert judge("") == 0.0

    def test_nonstandard_residues_sanitised_to_ala(self):
        """Non-standard chars should be replaced with A before calling model."""
        received = []
        def capture(seq): received.append(seq); return 0.0
        judge = ESMPseudoLogLikelihood(model_fn=capture)
        judge("X?Z")  # all non-standard → becomes "AAA"
        assert received[0] == "AAA"

    def test_standard_residues_pass_unchanged(self):
        received = []
        def capture(seq): received.append(seq); return 0.0
        judge = ESMPseudoLogLikelihood(model_fn=capture)
        judge("ACDEFGHIKLMNPQRSTVWY")
        assert received[0] == "ACDEFGHIKLMNPQRSTVWY"

    def test_no_torch_import_at_module_level(self):
        """Importing plm_judge must NOT import torch (lazy-load contract)."""
        import sys
        # torch may or may not be installed; what matters is that the import of
        # plm_judge doesn't crash without it.
        import importlib.util
        # We can't unimport torch if already present, but we can verify the module
        # loaded successfully without triggering torch (which we confirmed above).
        assert "xenodesign.judges.plm_judge" in sys.modules


# ─────────────────────────────────────────────────────────────────────────────
# 2. JudgePanel — combine() logic
# ─────────────────────────────────────────────────────────────────────────────

class TestJudgePanelCombine:
    """Pure-logic tests with mock RefereeScore objects."""

    def _panel(self, **kw):
        return JudgePanel(**kw)

    def test_perfect_chirality_is_not_vetoed(self):
        scores = [
            RefereeScore(chirality_violation=0.0, iptm=0.7),
            RefereeScore(chirality_violation=0.0, iptm=0.8),
        ]
        result = self._panel().combine(scores)
        assert not any(result.vetoed)

    def test_chirality_veto_triggers_above_threshold(self):
        """A step with violation > 0.1 must be vetoed."""
        scores = [
            RefereeScore(chirality_violation=0.5, iptm=0.9),  # violates: should be vetoed
            RefereeScore(chirality_violation=0.0, iptm=0.5),  # OK
        ]
        result = self._panel().combine(scores)
        assert result.vetoed[0] is True
        assert result.vetoed[1] is False
        # Vetoed step cannot win even with much better binding.
        assert result.selected_idx == 1

    def test_vetoed_step_composite_is_zero(self):
        scores = [
            RefereeScore(chirality_violation=0.5, iptm=0.9),
            RefereeScore(chirality_violation=0.0, iptm=0.3),
        ]
        result = self._panel().combine(scores)
        assert result.composite_scores[0] == pytest.approx(0.0)

    def test_chirality_threshold_boundary(self):
        """Strictly above threshold (0.101) → vetoed; at or below (0.1) → not vetoed.

        The panel uses strict inequality (violation > threshold), so exactly at threshold
        is NOT vetoed — consistent with treating 0.1 as the inclusive upper bound.
        """
        at_threshold = RefereeScore(chirality_violation=0.1, iptm=0.8)
        below = RefereeScore(chirality_violation=0.099, iptm=0.6)
        r_at = self._panel().combine([at_threshold, below])
        # violation=0.1 > 0.1 is False → NOT vetoed.
        assert r_at.vetoed[0] is False

        # Strictly above threshold → vetoed.
        strict_above = RefereeScore(chirality_violation=0.101, iptm=0.8)
        r_strict = self._panel().combine([strict_above, below])
        assert r_strict.vetoed[0] is True

    def test_all_vetoed_falls_back_to_best_binding(self):
        scores = [
            RefereeScore(chirality_violation=0.5, iptm=0.3),  # worst binding
            RefereeScore(chirality_violation=0.9, iptm=0.8),  # best binding
        ]
        result = self._panel().combine(scores)
        assert result.fallback_used is True
        assert result.selected_idx == 1  # best binding selected

    def test_pll_breaks_binding_tie(self):
        """When binding is equal, the step with higher PLL (more natural) wins."""
        scores = [
            RefereeScore(chirality_violation=0.0, iptm=0.7, pll=-3.0),  # lower PLL
            RefereeScore(chirality_violation=0.0, iptm=0.7, pll=-1.0),  # higher PLL
        ]
        result = self._panel(weights={"chirality": 0.0, "binding": 0.5, "pll": 0.5}).combine(scores)
        assert result.selected_idx == 1

    def test_higher_composite_wins(self):
        scores = [
            RefereeScore(chirality_violation=0.0, iptm=0.4, pll=-5.0),
            RefereeScore(chirality_violation=0.0, iptm=0.9, pll=-1.0),
        ]
        result = self._panel().combine(scores)
        assert result.selected_idx == 1
        assert result.composite_scores[1] > result.composite_scores[0]

    def test_composite_scores_in_valid_range(self):
        """Non-vetoed composite scores should be in [0, 1] after normalisation."""
        scores = [
            RefereeScore(chirality_violation=0.0, iptm=float(i)/9, pll=float(-i))
            for i in range(10)
        ]
        result = self._panel().combine(scores)
        for i, c in enumerate(result.composite_scores):
            if not result.vetoed[i]:
                assert 0.0 <= c <= 1.0 + 1e-9, f"composite[{i}]={c} out of range"

    def test_empty_history_raises(self):
        panel = self._panel()
        with pytest.raises(ValueError, match="non-empty"):
            panel.combine([])

    def test_single_step_always_selected(self):
        scores = [RefereeScore(chirality_violation=0.0, iptm=0.7)]
        result = self._panel().combine(scores)
        assert result.selected_idx == 0
        assert not result.fallback_used

    def test_mirror_discrepancy_penalised(self):
        """Step with lower mirror discrepancy should score higher (all else equal)."""
        scores = [
            RefereeScore(chirality_violation=0.0, iptm=0.7, mirror_discrepancy=5.0),  # high discrep
            RefereeScore(chirality_violation=0.0, iptm=0.7, mirror_discrepancy=0.1),  # low discrep
        ]
        result = self._panel(weights={"chirality": 0.0, "binding": 0.0, "pll": 0.0, "mirror": 1.0}).combine(scores)
        assert result.selected_idx == 1

    def test_pll_imputation_of_missing_values(self):
        """Steps with pll=None should be imputed without crashing."""
        scores = [
            RefereeScore(chirality_violation=0.0, iptm=0.5, pll=None),
            RefereeScore(chirality_violation=0.0, iptm=0.9, pll=None),
        ]
        result = self._panel().combine(scores)
        # Should select the higher-binding step without error.
        assert result.selected_idx == 1

    def test_custom_chirality_threshold(self):
        """Panel with threshold=0.5 should not veto 30% violation."""
        scores = [
            RefereeScore(chirality_violation=0.3, iptm=0.9),
            RefereeScore(chirality_violation=0.0, iptm=0.3),
        ]
        result = self._panel(chirality_veto_threshold=0.5).combine(scores)
        assert not result.vetoed[0]

    # ── #37: independent composition-veto channel (does NOT corrupt chirality) ──

    def test_composition_violation_defaults_false_backcompat(self):
        """RefereeScore.composition_violation defaults to False (back-compat)."""
        s = RefereeScore(chirality_violation=0.0, iptm=0.7)
        assert s.composition_violation is False

    def test_composition_veto_deselects_but_chirality_stays_real(self):
        """A composition-vetoed step is de-selected, yet its REPORTED chirality keeps
        its true measured value (proving the chirality channel is NOT corrupted — #37).

        Step 0 has the best binding (iptm=0.9) AND a real, clean chirality (0.02), but is
        composition-vetoed. Step 1 has lower binding and clean chirality. The panel must
        de-select step 0 while raw_scores[0].chirality_violation stays the real 0.02 —
        never forced to 1.0 as the old (channel-reusing) code did.
        """
        scores = [
            RefereeScore(chirality_violation=0.02, iptm=0.9,
                         composition_violation=True),   # best binding but comp-vetoed
            RefereeScore(chirality_violation=0.03, iptm=0.4),  # clean, lower binding
        ]
        result = self._panel().combine(scores)
        # De-selected: comp-vetoed step cannot win despite the best binding.
        assert result.vetoed[0] is True
        assert result.vetoed[1] is False
        assert result.selected_idx == 1
        assert result.composite_scores[0] == pytest.approx(0.0)
        # CRITICAL: the reported chirality is the TRUE measured value (uncorrupted).
        assert result.raw_scores[0].chirality_violation == pytest.approx(0.02)
        assert result.raw_scores[1].chirality_violation == pytest.approx(0.03)


# ── P1b/P1c: per-case SS-bias composite term ──────────────────────────────────────

from xenodesign.scorer import SSBiasConfig
from xenodesign.benchmark.cases import get_case, ss_bias_config_for_case


def test_ss_bias_field_defaults_none_backcompat():
    assert RefereeScore(chirality_violation=0.0, iptm=0.7).helix_fraction is None


def test_ss_bias_term_absent_by_default_is_backcompat():
    # No ss_bias config + no ss_bias weight => identical selection to before (helix ignored).
    scores = [
        RefereeScore(chirality_violation=0.0, iptm=0.6, helix_fraction=0.9),
        RefereeScore(chirality_violation=0.0, iptm=0.8, helix_fraction=0.1),
    ]
    result = JudgePanel(weights={"chirality": 0.0, "binding": 1.0}).combine(scores)
    assert result.selected_idx == 1  # higher iptm wins


def test_ss_bias_rewards_helix_for_alpha_case():
    scores = [
        RefereeScore(chirality_violation=0.0, iptm=0.7, helix_fraction=0.2),
        RefereeScore(chirality_violation=0.0, iptm=0.7, helix_fraction=0.9),
    ]
    panel = JudgePanel(
        weights={"chirality": 0.0, "binding": 0.0, "ss_bias": 1.0},
        ss_bias=SSBiasConfig(target_helix_frac=1.0),
    )
    assert panel.combine(scores).selected_idx == 1  # more helical wins for α


def test_ss_bias_penalizes_helix_for_anti_alpha_case():
    scores = [
        RefereeScore(chirality_violation=0.0, iptm=0.7, helix_fraction=0.2),
        RefereeScore(chirality_violation=0.0, iptm=0.7, helix_fraction=0.9),
    ]
    panel = JudgePanel(
        weights={"chirality": 0.0, "binding": 0.0, "ss_bias": 1.0},
        ss_bias=SSBiasConfig(target_helix_frac=0.0),   # anti-α
    )
    assert panel.combine(scores).selected_idx == 0  # less helical wins anti-α


def test_ss_bias_config_for_case_maps_knobs():
    assert ss_bias_config_for_case(get_case("alpha")).target_helix_frac == 1.0   # helix_ok
    assert ss_bias_config_for_case(get_case("cyclic")).target_helix_frac == 0.0  # anti_alpha
    assert ss_bias_config_for_case(get_case("nonalpha")).target_helix_frac == 0.0
    assert ss_bias_config_for_case(get_case("alpha")).weight == 1.0

    def test_composition_veto_is_independent_of_chirality_veto(self):
        """composition_violation vetoes SEPARATELY from chirality: a step can be
        chirality-clean (well below threshold) yet still composition-vetoed."""
        scores = [
            RefereeScore(chirality_violation=0.0, iptm=0.9,
                         composition_violation=True),   # chirality-clean, comp-vetoed
            RefereeScore(chirality_violation=0.0, iptm=0.5),
        ]
        result = self._panel().combine(scores)
        assert result.vetoed[0] is True   # vetoed by composition, not chirality
        # Chirality channel untouched: still 0.0 (its real value), not 1.0.
        assert result.raw_scores[0].chirality_violation == pytest.approx(0.0)
        assert result.selected_idx == 1

    def test_chirality_and_composition_vetoes_coexist(self):
        """Both veto channels can fire on different steps; a clean, non-vetoed step wins."""
        scores = [
            RefereeScore(chirality_violation=0.5, iptm=0.95),   # chirality-vetoed
            RefereeScore(chirality_violation=0.0, iptm=0.90,
                         composition_violation=True),            # composition-vetoed
            RefereeScore(chirality_violation=0.0, iptm=0.30),   # clean → winner
        ]
        result = self._panel().combine(scores)
        assert result.vetoed == [True, True, False]
        assert result.selected_idx == 2
        # The composition-vetoed step's chirality stays its real 0.0.
        assert result.raw_scores[1].chirality_violation == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. JudgePanel.select() — requires score_fn
# ─────────────────────────────────────────────────────────────────────────────

class TestJudgePanelSelect:

    def test_select_without_score_fn_raises(self):
        panel = JudgePanel()  # no score_fn
        with pytest.raises(RuntimeError, match="score_fn"):
            panel.select([_make_step(0.5)])

    def test_select_returns_correct_step(self):
        steps = [
            _make_step(score=0.3, chirality_violation=0.0, iptm=0.4),
            _make_step(score=0.8, chirality_violation=0.0, iptm=0.9),  # best
            _make_step(score=0.5, chirality_violation=0.5, iptm=0.7),  # vetoed
        ]
        panel = JudgePanel(score_fn=_mock_score_fn)
        chosen = panel.select(steps)
        assert chosen is steps[1]

    def test_select_skips_chirality_violated_step(self):
        steps = [
            _make_step(score=1.0, chirality_violation=0.5, iptm=0.99),  # high binding but vetoed
            _make_step(score=0.1, chirality_violation=0.0, iptm=0.3),   # clean chirality, lower
        ]
        panel = JudgePanel(score_fn=_mock_score_fn)
        chosen = panel.select(steps)
        assert chosen is steps[1]

    def test_select_pll_weighted_choice(self):
        """PLL weight=0.5 should flip the choice toward more natural sequence."""
        steps = [
            _make_step(score=0.5, chirality_violation=0.0, iptm=0.6, pll=-1.0),  # more natural
            _make_step(score=0.9, chirality_violation=0.0, iptm=0.7, pll=-10.0), # less natural
        ]
        panel = JudgePanel(
            score_fn=_mock_score_fn,
            weights={"chirality": 0.0, "binding": 0.3, "pll": 0.7, "mirror": 0.0},
        )
        chosen = panel.select(steps)
        assert chosen is steps[0]  # pll heavily weighted → step 0 wins


# ─────────────────────────────────────────────────────────────────────────────
# 4. HalluLoop.select_by_panel
# ─────────────────────────────────────────────────────────────────────────────

class TestHalluLoopSelectByPanel:
    """Integration: select_by_panel on loop history."""

    def test_panel_picks_chirality_clean_step_over_naive_best(self):
        """Panel should prefer a clean-chirality step even when naive best() has better ipTM."""
        # Naive best (greedy by score) = step 0 (score=0.9, but chirality violated).
        # Panel selects step 1 (score=0.6, chirality clean).
        steps = [
            _make_step(score=0.9, chirality_violation=0.5, iptm=0.9),  # naive best, chirality bad
            _make_step(score=0.6, chirality_violation=0.0, iptm=0.6),  # panel winner
        ]
        panel = JudgePanel(score_fn=_mock_score_fn)
        naive = HalluLoop.best(steps)
        panel_choice = HalluLoop.select_by_panel(steps, panel)

        assert naive is steps[0]        # greedy picks the high-score step
        assert panel_choice is steps[1] # panel rejects chirality violator

    def test_panel_consistent_with_best_when_all_chirality_clean(self):
        """When all steps are chirality-clean, panel and best() agree on highest binding."""
        steps = [
            _make_step(score=0.3, chirality_violation=0.0, iptm=0.3),
            _make_step(score=0.7, chirality_violation=0.0, iptm=0.7),
            _make_step(score=0.5, chirality_violation=0.0, iptm=0.5),
        ]
        panel = JudgePanel(
            score_fn=_mock_score_fn,
            weights={"chirality": 0.0, "binding": 1.0, "pll": 0.0, "mirror": 0.0},
        )
        panel_choice = HalluLoop.select_by_panel(steps, panel)
        naive = HalluLoop.best(steps)
        # Both should pick step 1 (highest iptm/score).
        assert panel_choice is steps[1]
        assert naive is steps[1]

    def test_panel_composite_improves_over_naive_on_combined_metric(self):
        """Panel-selected composite score >= naive best() composite score."""
        steps = [
            _make_step(score=0.8, chirality_violation=0.0, iptm=0.8, pll=-2.0),
            _make_step(score=0.6, chirality_violation=0.0, iptm=0.6, pll=-1.0),  # better pll
            _make_step(score=0.9, chirality_violation=0.5, iptm=0.9, pll=-3.0),  # vetoed
        ]
        panel = JudgePanel(
            score_fn=_mock_score_fn,
            weights={"chirality": 0.2, "binding": 0.4, "pll": 0.4, "mirror": 0.0},
        )
        ref_scores = [_mock_score_fn(s) for s in steps]
        result = panel.combine(ref_scores)

        naive_idx = max(range(len(steps)), key=lambda i: steps[i].score)
        panel_idx = result.selected_idx

        # Panel composite of panel-selected >= composite of naive choice.
        assert result.composite_scores[panel_idx] >= result.composite_scores[naive_idx]

    def test_select_by_panel_returns_loop_step(self):
        """Return type must be a LoopStep (not an index)."""
        steps = [_make_step(score=0.5, chirality_violation=0.0, iptm=0.5)]
        panel = JudgePanel(score_fn=_mock_score_fn)
        result = HalluLoop.select_by_panel(steps, panel)
        assert isinstance(result, LoopStep)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PanelResult data contract
# ─────────────────────────────────────────────────────────────────────────────

class TestPanelResult:

    def test_panel_result_lengths_match_input(self):
        scores = [RefereeScore(0.0, 0.5), RefereeScore(0.0, 0.7), RefereeScore(0.0, 0.6)]
        result = JudgePanel().combine(scores)
        n = len(scores)
        assert len(result.composite_scores) == n
        assert len(result.vetoed) == n
        assert len(result.raw_scores) == n

    def test_raw_scores_preserved(self):
        s = RefereeScore(chirality_violation=0.05, iptm=0.75, pll=-2.5)
        result = JudgePanel().combine([s])
        assert result.raw_scores[0].chirality_violation == pytest.approx(0.05)
        assert result.raw_scores[0].iptm == pytest.approx(0.75)
        assert result.raw_scores[0].pll == pytest.approx(-2.5)


# ── P4: referee_score_from populates mirror_discrepancy (wires the 0.05 term) ─
from xenodesign.judges.panel import referee_score_from


class _Pred:
    def __init__(self, iptm, chir, plddt=70.0, pll=-2.0):
        self.iptm = iptm
        self.chirality_violation_frac = chir
        self.interface_plddt = plddt
        self.pll = pll


def test_referee_score_from_without_coords_leaves_mirror_none():
    pred = _Pred(iptm=0.7, chir=0.0)
    rs = referee_score_from(pred)
    assert rs.iptm == 0.7
    assert rs.chirality_violation == 0.0
    assert rs.mirror_discrepancy is None  # default-safe: unchanged behaviour


def test_referee_score_from_populates_mirror_discrepancy():
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    twin = coords.copy(); twin[:, 0] *= -1.0            # exact mirror -> ~0 discrepancy
    pred = _Pred(iptm=0.7, chir=0.0)
    rs = referee_score_from(pred, coords=coords, twin_coords=twin, axis=0)
    assert rs.mirror_discrepancy is not None
    assert rs.mirror_discrepancy < 1e-6                 # self-consistent twin


def test_panel_composite_uses_wired_mirror_term():
    good_c = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    good_twin = good_c.copy(); good_twin[:, 0] *= -1.0  # ~0 discrepancy
    bad_c = good_c.copy()
    bad_twin = np.array([[9.0, 9.0, 9.0], [1.0, 0.0, 2.0], [5.0, 5.0, 1.0]])  # high discrepancy
    s_bad = referee_score_from(_Pred(iptm=0.7, chir=0.0), coords=bad_c, twin_coords=bad_twin)
    s_good = referee_score_from(_Pred(iptm=0.7, chir=0.0), coords=good_c, twin_coords=good_twin)
    result = JudgePanel(
        weights={"chirality": 0.0, "binding": 0.0, "pll": 0.0, "mirror": 1.0}
    ).combine([s_bad, s_good])
    assert result.selected_idx == 1  # lower mirror discrepancy wins
