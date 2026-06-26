"""Greedy HalluDesign loop on Chai (spec §2.4): truncated_refine -> sequence update ->
score -> accept (always). Backend, sequence-update, and scoring are injected so the control
flow is unit-testable; the real wiring uses ChaiBackend.truncated_refine + SequenceUpdater.

Per-iteration structure step is configurable via `refine_fn`:
  - Default (None): calls ``backend.truncated_refine(state, ref_time_steps, out_dir)`` —
    50-step structure-conditioned diffusion, fast (~50 s/iter) but may degrade D-chirality.
  - Custom callable: called as ``refine_fn(state, ref_time_steps, out_dir)``; return must be a
    Prediction-compatible object with ``.coords`` and ``.iptm`` attributes.  Pass a wrapper
    around ``ChaiBackend.predict`` for full 200-step prediction (slower, preserves chirality).

Optional acceptance gate (spec §2.4 ablation):
  - Default: accept-always (unchanged greedy behaviour; all existing tests still pass).
  - Pass ``accept_fn=chirality_gated_accept(panel, max_violation=0.1)`` to reject candidate
    steps whose chirality violation exceeds ``max_violation``.  On rejection the loop retains
    the current (clean) state and re-designs from it on the next iteration.  This is an
    OPTIONAL ablation; it does not change the science of refine/design/score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    d_fasta: str
    coords: np.ndarray


@dataclass
class LoopStep:
    state: LoopState
    prediction: object
    score: float


class HalluLoop:
    def __init__(
        self,
        backend,
        sequence_update_fn: Callable,
        score_fn: Callable,
        refine_fn: Optional[Callable] = None,
    ):
        """Initialise the greedy HalluDesign loop.

        Args:
            backend: Object exposing ``truncated_refine(state, ref_time_steps, out_dir)``.
                     Used as the default per-iteration structure step unless ``refine_fn``
                     is provided.
            sequence_update_fn: ``(prediction) -> str`` one-letter L sequence.
            score_fn: ``(prediction) -> float`` composite design score.
            refine_fn: Optional override for the per-iteration structure step.
                       Signature: ``(state: LoopState, ref_time_steps: int, out_dir) ->
                       Prediction``.  When ``None`` (default), falls back to
                       ``backend.truncated_refine`` — preserving the existing behaviour
                       byte-for-byte.  Pass a wrapper around ``ChaiBackend.predict`` to use
                       full 200-step prediction (better chirality, ~4× slower per iteration).
        """
        self._backend = backend
        self._sequence_update_fn = sequence_update_fn
        self._score_fn = score_fn
        self._refine_fn = refine_fn  # None → use backend.truncated_refine (default path)

    def run(
        self,
        init: LoopState,
        iterations: int,
        ref_time_steps: int,
        out_dir,
        accept_fn: Optional[Callable[["LoopStep", "LoopStep"], bool]] = None,
        schedule: Optional["AnnealSchedule"] = None,
        iterations_for_schedule: Optional[int] = None,
    ) -> list[LoopStep]:
        """Run the HalluDesign loop.

        Args:
            init: Initial loop state (D-fasta + seed coordinates).
            iterations: Number of refine→design→score iterations to run.
            ref_time_steps: Diffusion steps for truncated refinement (ignored when
                ``refine_fn`` is provided — e.g. ``_PredictBackendWrapper``).
            out_dir: Output directory root; per-iteration subdirs are created as
                ``{out_dir}/iter_{i:03d}/``.
            accept_fn: Optional acceptance gate.  Called as
                ``accept_fn(candidate_step, current_step) -> bool`` after each
                refine/score step.  When it returns ``False`` the candidate is
                **rejected**: the loop appends a placeholder step with the current
                (retained) state and the candidate prediction/score for logging,
                and continues re-designing from the same clean state.  When ``None``
                (default), every candidate is accepted (original greedy behaviour).

                Use :func:`chirality_gated_accept` to build a gate that rejects
                candidates with chirality violation above a threshold.
            schedule: Optional xenodesign.schedule.AnnealSchedule. When None
                (default) the scalar ref_time_steps is used for every iteration —
                byte-for-byte the original behaviour. When provided, the per-iteration
                ref_time_steps is resolved as schedule.ref_time_steps(i, N) where
                N = iterations_for_schedule or iterations; ref_time_steps is still the
                fallback/base. num_seqs / mpnn_temperature scheduling is consumed by the
                injected sequence_update_fn (lane C), not here — this loop only resolves
                the diffusion-truncation depth.
            iterations_for_schedule: Total iteration count to use as the schedule horizon
                (lets a resumed/segmented run anneal over the full campaign rather than this
                segment). Defaults to iterations when None.

        Returns:
            List of :class:`LoopStep` objects — one per iteration.  Rejected steps
            carry ``step.score`` from the candidate (for analysis) but ``step.state``
            from the *retained* current state.
        """
        from pathlib import Path as _Path
        history: list[LoopStep] = []
        current = init
        base_dir = _Path(out_dir)
        # Resolve the per-iteration structure callable once (avoids repeated attribute lookups).
        _step_fn = self._refine_fn if self._refine_fn is not None else self._backend.truncated_refine
        # Schedule horizon: None -> this segment's iteration count (default-OFF: a None
        # schedule means the scalar ref_time_steps is used unchanged below).
        _sched_N = iterations_for_schedule if iterations_for_schedule is not None else iterations
        for i in range(iterations):
            iter_dir = base_dir / f"iter_{i:03d}"
            rt = schedule.ref_time_steps(i, _sched_N) if schedule is not None else ref_time_steps
            pred = _step_fn(current, rt, iter_dir)
            new_seq = self._sequence_update_fn(pred)  # L one-letter -> caller maps to D-CCD
            score = self._score_fn(pred)
            candidate_state = LoopState(d_fasta=_to_d_fasta_safe(new_seq), coords=pred.coords)
            candidate_step = LoopStep(state=candidate_state, prediction=pred, score=score)

            if accept_fn is None:
                # Default: accept-always (original greedy behaviour, byte-for-byte).
                current = candidate_state
                history.append(candidate_step)
            else:
                # Build a minimal "current step" object for the accept_fn comparison.
                # We use the last accepted step from history, or a sentinel at iter 0.
                current_step = history[-1] if history else LoopStep(
                    state=current, prediction=None, score=-1.0
                )
                if accept_fn(candidate_step, current_step):
                    current = candidate_state
                    history.append(candidate_step)
                    logger.debug("iter %03d: accepted  (score=%.4f)", i, score)
                else:
                    # Rejected: keep current state; record candidate scores for analysis.
                    retained_step = LoopStep(state=current, prediction=pred, score=score)
                    history.append(retained_step)
                    logger.info(
                        "iter %03d: REJECTED by accept_fn (score=%.4f) — "
                        "retaining current state",
                        i, score,
                    )

        return history

    @staticmethod
    def best(history: list[LoopStep]) -> LoopStep:
        return max(history, key=lambda h: h.score)

    @staticmethod
    def select_by_panel(history: list[LoopStep], panel) -> LoopStep:
        """Select the best step using an adversarial JudgePanel.

        The panel applies hard chirality veto + weighted composite scoring across
        chirality, binding (ipTM / pLDDT), ESM-2 naturalness (PLL), and mirror
        self-consistency.  This replaces the naive greedy ``best()`` (ipTM-only)
        with an adversarial selection that simultaneously defends against chirality
        drift and pushes designs toward evolutionary-plausible sequences.

        Parameters
        ----------
        history:
            List of ``LoopStep`` objects returned by ``HalluLoop.run()``.
        panel:
            A ``xenodesign.judges.panel.JudgePanel`` instance configured with a
            ``score_fn`` (``score_fn=(step) -> RefereeScore``).

        Returns
        -------
        LoopStep
            The step chosen by the panel (chirality-clean + highest composite).
        """
        return panel.select(history)


def chirality_gated_accept(
    panel,
    max_violation: float = 0.1,
    min_composite_margin: float = 0.0,
) -> Callable[["LoopStep", "LoopStep"], bool]:
    """Factory for a chirality-aware acceptance gate (spec §2.4 optional ablation).

    Returns an ``accept_fn(candidate_step, current_step) -> bool`` that rejects a
    candidate iteration if:
      1. Its chirality violation fraction exceeds ``max_violation`` (hard gate), OR
      2. Its panel composite score is worse than the current step's composite by more
         than ``min_composite_margin`` (optional quality gate; default 0.0 = disabled).

    When the candidate is rejected the HalluLoop retains the current (clean) state and
    re-designs from it, using iteration stochasticity to try again.  This keeps the
    whole accepted trajectory chirality-clean, at the cost of some steps being wasted.

    Args:
        panel: A ``xenodesign.judges.panel.JudgePanel`` instance configured with a
               ``score_fn`` callable so it can score individual steps.  The gate
               calls ``panel.combine([candidate_referee, current_referee])`` to
               compare composite scores.
        max_violation: Chirality violation fraction above which the candidate is
               hard-rejected (default 0.1, matching the panel veto threshold).
        min_composite_margin: If > 0, additionally reject candidates whose panel
               composite is more than this margin *worse* than the current step's
               composite (useful to prevent binding from degrading; default 0.0).

    Returns:
        A callable ``(candidate_step, current_step) -> bool``.  Returns ``True``
        (accept) when the candidate passes both gates; ``False`` (reject) otherwise.

    Example::

        from xenodesign.judges.panel import JudgePanel, RefereeScore
        from xenodesign.loop import HalluLoop, chirality_gated_accept

        def my_score_fn(step):
            pred = step.prediction
            return RefereeScore(
                chirality_violation=pred.chirality_violation_frac,
                iptm=pred.iptm,
                interface_plddt=pred.interface_plddt,
            )

        panel = JudgePanel(score_fn=my_score_fn)
        gate  = chirality_gated_accept(panel, max_violation=0.1)
        history = loop.run(init, iterations=7, ref_time_steps=50,
                           out_dir=out_dir, accept_fn=gate)
    """
    if panel._score_fn is None:
        raise ValueError(
            "chirality_gated_accept requires a JudgePanel with score_fn set. "
            "Pass score_fn=(step)->RefereeScore when constructing JudgePanel."
        )

    def _gate(candidate_step: "LoopStep", current_step: "LoopStep") -> bool:
        # Score the candidate with the panel's score_fn.
        cand_ref = panel._score_fn(candidate_step)

        # Gate 1: hard chirality veto.
        if cand_ref.chirality_violation > max_violation:
            logger.info(
                "chirality_gated_accept: REJECT — chirality violation %.3f > %.3f",
                cand_ref.chirality_violation, max_violation,
            )
            return False

        # Gate 2: composite quality margin (optional; skipped when margin <= 0).
        if min_composite_margin > 0.0 and current_step.prediction is not None:
            curr_ref = panel._score_fn(current_step)
            result = panel.combine([cand_ref, curr_ref])
            cand_composite = result.composite_scores[0]
            curr_composite = result.composite_scores[1]
            if cand_composite < curr_composite - min_composite_margin:
                logger.info(
                    "chirality_gated_accept: REJECT — composite %.4f < current %.4f - margin %.4f",
                    cand_composite, curr_composite, min_composite_margin,
                )
                return False

        return True

    return _gate


def periodicity_gated_accept(
    heptad_thresh: float = 0.35,
) -> Callable[["LoopStep", "LoopStep"], bool]:
    """Factory for a DESIGN-TIME register-achievability acceptance gate.

    Returns an ``accept_fn(candidate_step, current_step) -> bool`` that REJECTS a candidate whose
    designed binder sequence is register-UNACHIEVABLE, i.e. so strongly heptad-periodic that a
    7-residue register shift reproduces the same hydrophobic interface face (so no score can
    prefer the native register over the shifted one — see ``scripts/seq_periodicity.py``). This
    is the sequence-only, reference-free counterpart of :func:`chirality_gated_accept`: where the
    chirality gate keeps the accepted path D-clean, this gate keeps it register-SCORABLE.

    The candidate's binder sequence is read from ``candidate_step.state.d_fasta`` (the D-CCD seq
    designed that iteration) and decoded to its one-letter L projection for the hydropathy
    autocorrelation — chirality does not change a side-chain's hydrophobicity, so the
    Kyte-Doolittle profile is chirality-blind (``seq_periodicity`` docstring). On rejection the
    HalluLoop retains the current state and re-designs from it next iteration (identical control
    flow to the chirality gate).

    Args:
        heptad_thresh: lag-7 hydropathy-autocorrelation at/above which (when lag-7 is the
            dominant peak) the helix is judged heptad-periodic and register is NOT achievable.
            Forwarded to ``seq_periodicity.compute``; default 0.35 (empirically confirmed bound).

    Returns:
        A callable ``(candidate_step, current_step) -> bool``. ``True`` (accept) when the
        candidate's binder sequence is register-achievable; ``False`` (reject) when it is
        heptad-periodic. An unreadable/empty candidate sequence is treated as register-achievable
        (accept) — this gate only ever rejects a sequence it can positively prove is periodic, so
        a parse failure never silently kills a trajectory.

    Example::

        from xenodesign.loop import HalluLoop, periodicity_gated_accept
        gate = periodicity_gated_accept(heptad_thresh=0.35)
        history = loop.run(init, iterations=30, ref_time_steps=50,
                           out_dir=out_dir, accept_fn=gate)
    """
    from scripts.seq_periodicity import compute as _periodicity_compute
    from xenodesign.io_spec import d_fasta_to_one_letter

    def _gate(candidate_step: "LoopStep", current_step: "LoopStep") -> bool:
        d_fasta = getattr(candidate_step.state, "d_fasta", "") or ""
        try:
            l_seq = d_fasta_to_one_letter(d_fasta) if d_fasta else ""
        except Exception:
            # ponytail: an undecodable D-CCD seq can't be proven periodic -> accept (never
            # silently kill a trajectory on a parse failure; the gate only rejects proven-periodic).
            return True
        if not l_seq:
            return True
        feats = _periodicity_compute(l_seq, heptad_thresh=heptad_thresh)
        if not feats["register_achievable"]:
            logger.info(
                "periodicity_gated_accept: REJECT — %s",
                feats["register_achievable_reason"],
            )
            return False
        return True

    return _gate


def greedy_iptm_accept(
    min_delta: float = 0.0,
) -> Callable[["LoopStep", "LoopStep"], bool]:
    """Factory for a strict greedy hill-climb acceptance gate (lane B, the accept-half of
    BoltzDesign's semi_greedy_steps).

    Returns an accept_fn(candidate_step, current_step) -> bool that accepts the candidate
    ONLY if it strictly improves on the current step's score by more than min_delta::

        accept  iff  candidate_step.score > current_step.score + min_delta

    Because the loop's score_fn already composes ipTM (and pLDDT / chirality via
    design_score), this turns the always-accept greedy loop into a zero-temperature ipTM /
    composite-score hill-climb: a candidate that does not improve is rejected and the loop
    retains the current (best-so-far) state, re-designing from it next iteration.

    This stays 'greedy' (no Metropolis / no uphill acceptance), so it is compatible with
    config._VALID_ACCEPT == {'greedy'} — no config change is needed.

    Args:
        min_delta: Minimum score improvement required to accept (default 0.0 = accept on any
            strict improvement; equal scores are rejected). Set > 0 to require a margin
            (suppresses dithering on noise).

    Returns:
        A callable (candidate_step, current_step) -> bool.

    Note:
        At iteration 0 the loop supplies a sentinel current_step with score == -1.0
        (see HalluLoop.run), so the first real candidate is always accepted.

    Example::

        from xenodesign.loop import HalluLoop, greedy_iptm_accept
        history = loop.run(init, iterations=30, ref_time_steps=50,
                           out_dir=out_dir, accept_fn=greedy_iptm_accept(min_delta=0.0))
    """

    def _gate(candidate_step: "LoopStep", current_step: "LoopStep") -> bool:
        improved = candidate_step.score > current_step.score + min_delta
        if not improved:
            logger.debug(
                "greedy_iptm_accept: REJECT — candidate score %.4f < current %.4f + delta %.4f",
                candidate_step.score, current_step.score, min_delta,
            )
        return bool(improved)

    return _gate


def compose_accept_fns(
    *gates: Optional[Callable[["LoopStep", "LoopStep"], bool]],
) -> Optional[Callable[["LoopStep", "LoopStep"], bool]]:
    """AND-compose zero or more accept gates into a single accept_fn (None-safe).

    ``None`` gates are dropped. With no surviving gates returns ``None`` (the loop's
    accept-always default). With exactly one, returns it unchanged (byte-for-byte). With two or
    more, returns a gate that accepts iff EVERY sub-gate accepts (short-circuit on the first
    reject) — used to stack the chirality gate and the periodicity gate for the "-dep" variants.
    """
    active = [g for g in gates if g is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _all(candidate_step: "LoopStep", current_step: "LoopStep") -> bool:
        return all(g(candidate_step, current_step) for g in active)

    return _all


def _to_d_fasta_safe(seq: str) -> str:
    """Encode designed one-letter L sequence as D-CCD if it isn't already encoded.

    D-CCD uses parentheses, e.g. ``(DAL)`` (see io_spec.to_d_fasta / ADR-004), so the
    guard checks for ``(`` — a string already in D-CCD form is passed through unchanged.
    """
    if "(" in seq:
        return seq
    from xenodesign.io_spec import to_d_fasta
    return to_d_fasta(seq)
