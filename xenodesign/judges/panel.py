"""Adversarial judge panel: composite scoring + hard veto for design selection.

Design
------
The panel encodes the adversarial game as a three-layer check:

  Layer 1 — Hard veto (chirality referee).
      Any step whose D-chirality violation fraction exceeds ``chirality_veto_threshold``
      (default 0.1, i.e. 10% of stereocenters wrong) is completely excluded from
      selection — it doesn't matter how good the binding is.  This is the primary
      defence against LigandMPNN's L-bias.

  Layer 2 — Weighted composite score (normalised in [0,1]).
      Among non-vetoed steps, the panel computes:
        composite = w_chirality * chirality_score
                  + w_binding  * binding_score
                  + w_pll      * pll_score
                  + w_mirror   * mirror_score   (optional)

      where each score is normalised to [0,1] across the history.

  Layer 3 — ``select(history)`` returns the non-vetoed step with the highest
      composite.  Falls back to ``HalluLoop.best()`` (ipTM-greedy) if all steps
      are vetoed (safety net — should not happen with a well-seeded loop).

Normalisation
-------------
Scores are min-max normalised across the *entire* history so that no single referee
dominates due to scale.  PLL values are negative (log-probs), so higher (less negative)
= better.  All scores are oriented so that *higher composite = better design*.

CPU purity
----------
The panel itself is pure Python/numpy — no torch import at module level.  PLL values
are passed in as floats (pre-computed by ``ESMPseudoLogLikelihood``); the panel just
combines them.  This means the panel logic is fully unit-testable without GPU.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RefereeScore:
    """Per-step referee scores for one loop iteration.

    All fields are floats in arbitrary scale.  The panel normalises them before
    compositing.  Missing scores (None) are imputed to the population mean.

    Attributes
    ----------
    chirality_violation:
        Fraction of stereocenters with wrong chirality sign, in [0, 1].
        Lower is better (0 = perfectly D-correct).
    iptm:
        Inter-chain ipTM confidence, in [0, 1].  Higher is better.
    interface_plddt:
        Mean per-residue pLDDT on the binder chain (proxy for interface pLDDT),
        in [0, 100].  Higher is better.  Optional.
    pll:
        ESM-2 pseudo-log-likelihood (masked-marginal, nats/residue).  Negative float;
        higher (less negative) = more natural sequence.  Optional.
    mirror_discrepancy:
        Kabsch RMSD between predicted structure and its mirror image (lower = more
        self-consistent as a D-peptide).  Optional.
    composition_violation:
        Independent hard-veto flag for low-complexity / degenerate sequences (e.g. poly-Ala).
        True ⇒ the step is excluded from selection, SEPARATELY from the chirality veto.
        This is its OWN channel precisely so that ``chirality_violation`` always reports the
        TRUE measured chirality and is never overwritten to encode a composition reject
        (the #37 de-conflation).  Defaults to False (back-compat: existing callers that never
        set it behave exactly as before).
    helix_fraction:
        Forward CA-geometry helix fraction of the design, in [0, 1] (P1c; from
        ``secondary_structure.helix_fraction``). Feeds the OPTIONAL per-case SS-bias composite
        term (see ``JudgePanel(ss_bias=...)``): rewarded toward helix for the α case, away from
        helix for the anti-α cyclic / knottin cases. ``None`` (default) ⇒ imputed to the
        population mean, and with no ``ss_bias`` config the term is absent entirely (back-compat).
    """
    chirality_violation: float
    iptm: float
    interface_plddt: float = 50.0   # neutral default
    pll: Optional[float] = None
    mirror_discrepancy: Optional[float] = None
    composition_violation: bool = False
    helix_fraction: Optional[float] = None


@dataclass
class PanelResult:
    """Full panel output: per-step verdicts + the selected step index."""

    selected_idx: int
    """Index into ``history`` of the panel-chosen step."""

    composite_scores: List[float]
    """Normalised composite score for each step (0.0 if vetoed)."""

    vetoed: List[bool]
    """True for steps excluded by EITHER hard-veto channel (chirality OR composition)."""

    raw_scores: List[RefereeScore]
    """Raw referee scores as supplied to the panel (for logging / analysis)."""

    fallback_used: bool = False
    """True when all steps were vetoed and the greedy best() was used as fallback."""

    soft_composite_scores: List[float] = field(default_factory=list)
    """Weighted composite for each step WITHOUT the veto-zeroing — the SAME normalised
    composite as ``composite_scores`` but keeping each (even vetoed) step's real value.

    Here chirality is a HEAVY SOFT penalty (the ``chirality`` weight on ``1 - violation``),
    never a hard veto, so a vetoed step still gets a comparable score reflecting how "dirty"
    it is. Beam ADVANCEMENT ranks by this (the least-dirty children carry forward, mirroring
    the greedy loop reaching clean basins through dirty intermediates); WINNER SELECTION still
    uses ``composite_scores`` + ``vetoed`` so a vetoed step can never be chosen. Defaults to
    an empty list only for back-compat with any hand-constructed PanelResult; ``combine()``
    always populates it."""


class JudgePanel:
    """Multi-referee adversarial judge panel.

    Parameters
    ----------
    weights:
        Dict of ``{"chirality": w, "binding": w, "pll": w, "mirror": w}``.
        Missing keys default to ``_DEFAULT_WEIGHTS``.  Weights need not sum to 1.
    chirality_veto_threshold:
        Chirality violation fraction above which a step is hard-vetoed.
        Default: 0.1 (spec §3 gate criterion).
    score_fn:
        Optional callable ``(step) -> RefereeScore`` to extract referee scores from
        a ``LoopStep`` object.  When provided, ``select(history)`` automatically
        computes scores from the history steps.
        When ``None``, caller must pass pre-computed ``List[RefereeScore]`` to
        ``combine(scores)``.
    """

    _DEFAULT_WEIGHTS = {
        "chirality": 0.35,
        "binding":   0.40,
        "pll":       0.20,
        "mirror":    0.05,
    }

    def __init__(
        self,
        weights: Optional[dict] = None,
        chirality_veto_threshold: float = 0.1,
        score_fn: Optional[Callable] = None,
        ss_bias=None,
    ):
        self._weights = {**self._DEFAULT_WEIGHTS, **(weights or {})}
        self._veto_threshold = chirality_veto_threshold
        self._score_fn = score_fn
        # P1c per-case SS-bias: an scorer.SSBiasConfig supplying target_helix_frac (direction).
        # The composite MAGNITUDE comes from weights['ss_bias'] (like every other referee); the
        # config's own .weight is unused here (it is for the standalone ss_bias_score key). When
        # None, the SS term is absent (back-compat: callers that never pass it are unchanged).
        self._ss_bias = ss_bias

    # ── public API ──────────────────────────────────────────────────────────────

    def combine(
        self,
        scores: Sequence[RefereeScore],
    ) -> PanelResult:
        """Compute composite scores and select the best non-vetoed step.

        Parameters
        ----------
        scores:
            One ``RefereeScore`` per loop step (in order).

        Returns
        -------
        PanelResult
            Full panel verdict including per-step composites and the selected index.
        """
        n = len(scores)
        if n == 0:
            raise ValueError("scores must be non-empty")

        # ── 1. Hard veto (two INDEPENDENT channels) ─────────────────────────────
        # (a) chirality: violation fraction above threshold (the L-bias defence).
        # (b) composition: a separate boolean flag for low-complexity / degenerate seqs.
        # They are OR-combined for selection but tracked separately so neither channel
        # corrupts the other's reported value (#37 de-conflation: chirality_violation must
        # always stay the TRUE measured chirality, never a stand-in for a composition reject).
        chir_vetoed = [s.chirality_violation > self._veto_threshold for s in scores]
        comp_vetoed = [bool(s.composition_violation) for s in scores]
        vetoed = [c or k for c, k in zip(chir_vetoed, comp_vetoed)]
        logger.debug(
            "Panel: %d/%d steps vetoed (chirality=%d, composition=%d)",
            sum(vetoed), n, sum(chir_vetoed), sum(comp_vetoed),
        )

        # ── 2. Raw score vectors (oriented so higher = better everywhere) ───────
        chirality_raw = np.array([1.0 - s.chirality_violation for s in scores])
        binding_raw   = np.array([
            0.5 * s.iptm + 0.5 * (s.interface_plddt / 100.0)
            for s in scores
        ])

        # PLL: impute missing with column mean (before normalisation).
        pll_vals = [s.pll if s.pll is not None else None for s in scores]
        pll_raw = _impute_and_array(pll_vals, default=0.0)

        mirror_vals = [s.mirror_discrepancy for s in scores]
        # Mirror discrepancy: lower = better → invert.
        mirror_discrepancy_raw = _impute_and_array(mirror_vals, default=0.0)
        mirror_raw = -mirror_discrepancy_raw   # now higher = better (less discrepancy)

        # SS-bias (P1c): proximity of the design's helix fraction to the per-case target
        # (1 = on target, 0 = opposite). Direction is the case's target_helix_frac (α -> 1.0
        # reward helix; anti-α cyclic/knottin -> 0.0 penalise helix); missing helix imputed to
        # the population mean so an unscored step is neutral, not falsely on-target. Absent
        # (no ss_bias config) -> a zero vector so the term drops out entirely.
        if self._ss_bias is not None:
            target = float(self._ss_bias.target_helix_frac)
            helix_arr = _impute_and_array([s.helix_fraction for s in scores], default=target)
            ss_raw = 1.0 - np.abs(helix_arr - target)
        else:
            ss_raw = np.zeros(n)

        # ── 3. Min-max normalise each referee to [0, 1] ─────────────────────────
        c_norm  = _minmax_norm(chirality_raw)
        b_norm  = _minmax_norm(binding_raw)
        pll_norm = _minmax_norm(pll_raw)
        m_norm  = _minmax_norm(mirror_raw)
        ss_norm = _minmax_norm(ss_raw) if self._ss_bias is not None else ss_raw

        # ── 4. Weighted composite ────────────────────────────────────────────────
        w = self._weights
        composite = (
            w.get("chirality", 0.0) * c_norm
            + w.get("binding",   0.0) * b_norm
            + w.get("pll",       0.0) * pll_norm
            + w.get("mirror",    0.0) * m_norm
            + w.get("ss_bias",   0.0) * ss_norm
        )
        # Zero out vetoed steps so they can never be selected.
        composite_final = np.where(vetoed, 0.0, composite)
        # SOFT composite keeps every step's real value (chirality stays a heavy soft penalty,
        # never a hard veto) — beam ADVANCEMENT ranks by this so an all-vetoed cycle still
        # yields the least-dirty children to carry forward.
        soft_composite = composite

        # ── 5. Select ────────────────────────────────────────────────────────────
        fallback = False
        if all(vetoed):
            # Safety net: all steps chirality-vetoed — fall back to best binding.
            # NOTE: composite_scores[selected_idx] will be 0.0 in this path (all steps were
            # zeroed out by the veto mask above). Callers MUST check PanelResult.fallback_used
            # to distinguish "legitimately zero composite" from "all-vetoed fallback" and should
            # treat the selected_idx as a best-binding rescue, not a panel-approved design.
            logger.warning(
                "All %d steps vetoed by chirality! Falling back to best binding score "
                "(composite=0.0 — check fallback_used before using this result).", n
            )
            selected_idx = int(np.argmax(binding_raw))
            fallback = True
        else:
            selected_idx = int(np.argmax(composite_final))

        # Logging summary.
        logger.debug(
            "Panel select: idx=%d  composite=%.4f  "
            "chirality_viol=%.3f  iptm=%.4f  pll=%s",
            selected_idx,
            composite_final[selected_idx],
            scores[selected_idx].chirality_violation,
            scores[selected_idx].iptm,
            f"{scores[selected_idx].pll:.3f}" if scores[selected_idx].pll is not None else "N/A",
        )

        return PanelResult(
            selected_idx=selected_idx,
            composite_scores=composite_final.tolist(),
            vetoed=list(vetoed),
            raw_scores=list(scores),
            fallback_used=fallback,
            soft_composite_scores=soft_composite.tolist(),
        )

    def select(self, history) -> object:
        """Select the best non-vetoed step from a ``HalluLoop`` history.

        Requires that ``score_fn`` was provided at construction time.

        Parameters
        ----------
        history:
            List of ``LoopStep`` objects as returned by ``HalluLoop.run()``.

        Returns
        -------
        LoopStep
            The panel-selected step.
        """
        if self._score_fn is None:
            raise RuntimeError(
                "JudgePanel.select() requires score_fn to be set at construction time. "
                "Pass a callable (step) -> RefereeScore, or use combine() directly."
            )
        scores = [self._score_fn(step) for step in history]
        result = self.combine(scores)
        return history[result.selected_idx]


# ── helpers ──────────────────────────────────────────────────────────────────────

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1].  Returns 0.5 for flat arrays (all equal)."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def _impute_and_array(vals: list, default: float = 0.0) -> np.ndarray:
    """Convert a list of Optional[float] to numpy array, imputing missing with mean.

    If all values are None, uses ``default``.
    """
    present = [v for v in vals if v is not None]
    fill = float(np.mean(present)) if present else default
    return np.array([v if v is not None else fill for v in vals], dtype=float)


def referee_score_from(prediction, *, coords=None, twin_coords=None, axis: int = 0,
                        chirality_attr: str = "chirality_violation_frac") -> RefereeScore:
    """Build a RefereeScore from a prediction object, POPULATING mirror_discrepancy when
    both ``coords`` and ``twin_coords`` are supplied (spec §5; tracker #6).

    This is the wiring that makes the panel's 0.05 'mirror' composite term LIVE — previously
    every caller passed mirror_discrepancy=None (imputed away). When the predicted complex
    coords and its mirror-twin coords are available, the Tier-1 self-consistency discrepancy
    (mirror.mirror_discrepancy, Kabsch RMSD to the reflected twin) is attached so the panel
    rewards self-consistent D-designs.

    DEFAULT-SAFE: if either coords array is None, mirror_discrepancy stays None (unchanged
    behaviour — the term is imputed to the population mean as before).

    Args:
        prediction: object exposing .iptm, the chirality attr, optionally .interface_plddt
            and .pll.
        coords / twin_coords: (n_atoms, 3) predicted complex coords + its predicted mirror
            twin; both required to compute the discrepancy.
        axis: reflection axis (must match the pipeline default, 0 = x).
        chirality_attr: attribute name carrying the chirality-violation fraction.

    Returns:
        RefereeScore with mirror_discrepancy set iff both coord arrays were given.
    """
    md = None
    if coords is not None and twin_coords is not None:
        from xenodesign.mirror import mirror_discrepancy
        md = float(mirror_discrepancy(coords, twin_coords, axis=axis))
    return RefereeScore(
        chirality_violation=float(getattr(prediction, chirality_attr)),
        iptm=float(prediction.iptm),
        interface_plddt=float(getattr(prediction, "interface_plddt", 50.0)),
        pll=getattr(prediction, "pll", None),
        mirror_discrepancy=md,
    )
