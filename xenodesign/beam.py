"""Beam + anneal search over the HalluDesign-on-Chai loop (WF-1 spec, ADR-018).

A pure-Python, GPU-free orchestrator that widens the greedy α loop into a beam search and
finishes with a short simulated-anneal polish. It NEVER imports torch / chai / GPU at module
load: every expensive operation (the Chai predict, the MPNN design, the CIF extraction, the
referee scoring) is INJECTED as a callable, exactly the seams the design_alpha driver already
owns. That keeps the control flow unit-testable on CPU and leaves loop.py / design_alpha.py
untouched (the driver in scripts/design_alpha_beam.py wires the real callables).

Design
------
Per cycle (``c`` in ``range(cycles)``), for each of the ``beam_width`` live states:

  1. EXPAND — call the design_fn (a ``MultiCandidate(top_k=m)`` over the context-aware
     LigandMPNN) ONCE on the parent's scored backbone -> ``m`` best L seqs (cheap, MPNN-only,
     NO Chai). Each becomes a child ``BeamState`` carrying ``parent_id``.
  2. DEDUP — drop a child whose L seq is already in the global ``seen`` set, or equal to its
     parent's L seq (no information gained). Saves the per-child Chai predict.
  3. PREDICT — run the predict-wrapper (one Chai predict per surviving child) -> scored CIF +
     ipTM / chirality / coords.
  4. PRUNE — score each child with the reused referee, run ``JudgePanel.combine`` over the
     cycle's child batch (binding is a RELATIVE min-max gradient within the batch, never an
     absolute ipTM cutoff), then split two ways: the next BEAM is the TOP-``beam_width`` by the
     SOFT composite (chirality a HEAVY SOFT penalty, NOT a hard veto) so the search ADVANCES
     THROUGH even an all-hard-vetoed cycle (mirroring the accept-always greedy loop reaching
     clean basins at iter2+); the WINNER POOL keeps ONLY the hard-veto-passing (clean) children.

A pool of ALL hard-veto-PASSING children across cycles is retained (vetoed intermediates advance
the beam but never enter the pool, so they can never be SELECTED). ``anneal_best`` then takes
the global top-``top_n`` by composite and runs a short ``HalluLoop`` with three cooling levers
(decreasing ref_time_steps + cooling MPNN temperature + greedy zero-T accept). The final pick is
a ``JudgePanel.combine`` over the union of anneal states.

Cost
----
``CostAccount`` counts Chai PREDICTS ONLY (MPNN forward passes and ``sequence_quality_key`` are
free). For ``B`` beam_width, ``m`` children_per_branch, ``C`` cycles, ``a`` anneal_steps,
``top_n`` anneal seeds::

    predicts = 1 (seed) + m (first cycle) + (C-1)*B*m + top_n*a   - dedup hits

Defaults (B=3, m=3, C=3, a=5, top_n=3) => ~1+3+12+15 = 31..37 predicts (vs 8 for a 7-iter
greedy run). ``CostAccount.summary()`` reports predicts + an est wall-clock + the ratio.

D-peptide reporting
-------------------
``BeamState`` stores the design chain's one-letter L sequence (mirror frame) in ``l_seq``; any
driver that surfaces the D-peptide reports it LOWERCASE with Gly as ``G`` (project convention).
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default beam knobs (the driver overrides these; --smoke shrinks them).
_DEFAULT_BEAM_WIDTH = 3
_DEFAULT_CHILDREN_PER_BRANCH = 3
_DEFAULT_CYCLES = 3

# Est. wall-clock per Chai predict (s) used only for CostAccount.summary()'s estimate.
_SECONDS_PER_PREDICT = 50.0
# Greedy baseline predict count for the ratio (1 seed + 7 iters, the 7-iter greedy run).
_GREEDY_BASELINE_PREDICTS = 8


@dataclass
class BeamState:
    """One node in the beam — a superset of ``loop.LoopState`` (d_fasta + coords) carrying the
    search bookkeeping the panel/pruner needs.

    Fields
    ------
    d_fasta / coords:
        The ``LoopState`` payload — D-CCD sequence + seed/scored coordinates.
    l_seq:
        Design-chain one-letter L sequence (mirror frame) — the dedup/identity key.
    cif_path:
        Path to this state's scored CIF (None until predicted).
    iptm / chirality:
        The predict-wrapper's measured inter-chain ipTM and D-chirality violation fraction.
    composite:
        The veto-zeroed panel composite assigned during ``prune`` (None until scored). Used to
        rank the CLEAN winner pool (0.0 for any vetoed step — but vetoed steps never enter the
        pool, so pool members carry their true composite).
    soft:
        The panel SOFT composite (un-vetoed; chirality as a heavy soft penalty) assigned during
        ``prune`` (None until scored). Used ONLY to rank BEAM ADVANCEMENT, so an all-vetoed cycle
        still carries the least-dirty children forward. Never used for winner selection.
    parent_id / id / cycle:
        Lineage bookkeeping (parent's id, this node's id, the cycle it was born in).
    """
    d_fasta: str
    coords: np.ndarray
    l_seq: str = ""
    cif_path: Optional[object] = None
    iptm: float = 0.0
    chirality: float = 0.0
    composite: Optional[float] = None
    soft: Optional[float] = None
    parent_id: Optional[int] = None
    id: int = 0
    cycle: int = 0


class CostAccount:
    """Counts Chai predicts only (the single expensive op). MPNN / sequence_quality_key are
    free and never counted. ``summary()`` reports the count + an estimated wall-clock + the
    ratio vs the 7-iter greedy baseline."""

    def __init__(self):
        self.predicts = 0
        self.dedup_hits = 0

    def add_predicts(self, n: int = 1) -> None:
        self.predicts += int(n)

    def add_dedup_hit(self, n: int = 1) -> None:
        self.dedup_hits += int(n)

    def summary(self) -> str:
        wall_s = self.predicts * _SECONDS_PER_PREDICT
        ratio = self.predicts / float(_GREEDY_BASELINE_PREDICTS)
        return (
            f"CostAccount: {self.predicts} Chai predicts "
            f"({self.dedup_hits} dedup hits saved) | "
            f"est wall ~{wall_s / 60.0:.1f} min "
            f"({ratio:.1f}x the 7-iter greedy baseline of "
            f"{_GREEDY_BASELINE_PREDICTS} predicts)"
        )


# ── Beam primitives ─────────────────────────────────────────────────────────────

_ID_COUNTER = itertools.count(1)


def _next_id() -> int:
    return next(_ID_COUNTER)


def expand_state(parent: BeamState, design_fn, extract_fn,
                 anchor_fn: Optional[Callable] = None,
                 next_id: Optional[Callable[[], int]] = None,
                 known_seq_fn: Optional[Callable] = None,
                 encode_fn: Optional[Callable] = None) -> list[BeamState]:
    """Expand one parent into its ``m`` children (m = the design_fn's top_k).

    Calls ``extract_fn(parent)`` -> the inverse-folding inputs dict (design backbone + context),
    then the design_fn ONCE (a ``MultiCandidate(top_k=m)``) to get the m best L seqs, best first.
    An optional ``anchor_fn`` is applied to each candidate L seq. NO Chai predict happens here.

    S2.1 (SequenceUpdate routing, flag-on dispatch path): when ``known_seq_fn`` is supplied, the
    parent's REAL evolving sequence (``known_seq_fn(parent)`` -> the prior L seq) is threaded into
    the design_fn as ``known_seq`` so free positions condition on real context instead of all-Ala
    (the beam.py:162 starvation bug). When ``encode_fn`` is supplied, each child's d_fasta is the
    stage's chirality-correct + canonical-anchored encode of the designed L seq, replacing the
    inline whole-chain ``to_d_fasta``. Both default ``None`` => byte-identical legacy behaviour
    (the standalone scripts/design_alpha_beam.py driver passes neither).
    """
    next_id = next_id or _next_id
    inputs = extract_fn(parent)
    design_backbone = inputs["design_backbone"]
    n = design_backbone.shape[0]
    fixed_mask = [False] * n
    if known_seq_fn is not None:
        known_seq = known_seq_fn(parent)
        candidates = design_fn(
            design_backbone, inputs["context_coords"], inputs["context_elements"],
            fixed_mask, 0.1, n, known_seq=known_seq,
        )
    else:
        candidates = design_fn(
            design_backbone, inputs["context_coords"], inputs["context_elements"],
            fixed_mask, 0.1, n,
        )
    if not candidates:
        return []
    if anchor_fn is not None:
        candidates = [anchor_fn(c) for c in candidates]

    from xenodesign.io_spec import to_d_fasta

    children: list[BeamState] = []
    for l_seq in candidates:
        d_fasta = encode_fn(l_seq) if encode_fn is not None else to_d_fasta(l_seq)
        children.append(BeamState(
            d_fasta=d_fasta, coords=parent.coords, l_seq=l_seq,
            cif_path=None, iptm=0.0, chirality=0.0, composite=None,
            parent_id=parent.id, id=next_id(), cycle=parent.cycle + 1,
        ))
    return children


def dedup(children: list[BeamState], seen: set, parent_l_seq: Optional[str] = None,
          cost: Optional[CostAccount] = None) -> list[BeamState]:
    """Drop a child whose ``l_seq`` is already in ``seen`` (global), equal to ``parent_l_seq``,
    or duplicated earlier in this batch. Mutates ``seen`` with every kept l_seq. Records dedup
    hits on ``cost`` (so the saved Chai predicts are visible in the summary)."""
    kept: list[BeamState] = []
    for c in children:
        if c.l_seq == parent_l_seq or c.l_seq in seen:
            if cost is not None:
                cost.add_dedup_hit()
            continue
        seen.add(c.l_seq)
        kept.append(c)
    return kept


def predict_children(children: list[BeamState], predict_fn, cost: CostAccount,
                     ref_time_steps: int = 50, out_dir="/tmp/beam") -> list[BeamState]:
    """Run ONE Chai predict per child (via the injected predict-wrapper) and attach the scored
    fields (coords / iptm / chirality / cif_path). Counts one predict per child on ``cost``."""
    from pathlib import Path as _Path

    base = _Path(out_dir)
    for c in children:
        child_dir = base / f"cycle_{c.cycle:03d}" / f"node_{c.id:05d}"
        pred = predict_fn(c, ref_time_steps, child_dir)
        cost.add_predicts(1)
        c.coords = pred.coords
        c.iptm = float(pred.iptm)
        c.chirality = float(getattr(pred, "chirality_violation_frac", 0.0))
        c.cif_path = getattr(predict_fn, "last_out_dir", None) or child_dir
        c.prediction = pred  # retained for the referee_fn (token_index / plddt)
    return children


def prune(children: list[BeamState], referee_fn, panel, keep_b: int):
    """Score the cycle's child batch with the reused referee + ``JudgePanel.combine``, then
    split two ways (the beam GENERALIZES the greedy loop — see ADR):

      * BEAM ADVANCEMENT — the TOP-``keep_b`` children ranked by the panel's SOFT composite
        (``result.soft_composite_scores``), in which chirality is a HEAVY SOFT penalty, NOT a
        hard veto. So even an ALL-hard-vetoed cycle yields a non-empty next beam (the least-dirty
        B children), letting the search advance through dirty intermediates to clean basins
        exactly the way the accept-always greedy loop reaches clean iters 2+.
      * WINNER POOL — only the HARD-veto-passing children (chirality <= threshold AND no
        composition violation). The final select / anneal draws from THIS clean pool, so a
        vetoed intermediate can never be SELECTED as a winner.

    The binding term inside the panel is RELATIVE (min-max within this batch) — the allowed
    fixed-L-target ipTM gradient — never an absolute cutoff. ``c.composite`` is set from the
    veto-zeroed composite (so clean pool members rank by the true panel composite); ``c.soft``
    carries the un-vetoed soft score used only for advancement. Returns
    ``(advance_top_b, clean_pool, panel_result)``.
    """
    if not children:
        return [], [], None
    scores = [referee_fn(c) for c in children]
    result = panel.combine(scores)
    # Fall back to the veto-zeroed composite if an older PanelResult lacks the soft field.
    soft = result.soft_composite_scores or result.composite_scores
    for c, comp, sc in zip(children, result.composite_scores, soft):
        c.composite = float(comp)
        c.soft = float(sc)
    # WINNER POOL: hard-veto-passing children only (these alone are selectable).
    clean_pool = [c for c, v in zip(children, result.vetoed) if not v]
    # BEAM ADVANCEMENT: top-B by SOFT score over ALL children (vetoed included), so a fully
    # dirty cycle still advances the least-dirty B children.
    advance = sorted(children, key=lambda c: c.soft, reverse=True)[:keep_b]
    return advance, clean_pool, result


def beam_search(seed: BeamState, design_fn, predict_fn, extract_fn, referee_fn, panel,
                beam_width: int = _DEFAULT_BEAM_WIDTH,
                children_per_branch: int = _DEFAULT_CHILDREN_PER_BRANCH,
                cycles: int = _DEFAULT_CYCLES, cost: Optional[CostAccount] = None,
                anchor_fn: Optional[Callable] = None, dedup_on: bool = True,
                ref_time_steps: int = 50, out_dir="/tmp/beam",
                next_id: Optional[Callable[[], int]] = None,
                known_seq_fn: Optional[Callable] = None,
                encode_fn: Optional[Callable] = None):
    """Run the beam search and return ``(pool, cost)``.

    The seed is predicted once (1 predict), then ``cycles`` cycles each expand the live beam,
    dedup, predict the survivors, and prune. The next beam is the top-``beam_width`` by the SOFT
    composite (so the search advances through dirty intermediates, even an all-vetoed cycle);
    ``pool`` is every HARD-veto-PASSING child across all cycles (the clean candidate set
    ``anneal_best`` polishes — a vetoed intermediate can never be selected as a winner). The ONLY
    stop is literally zero children to expand (e.g. all deduped).

    Predict budget (no dedup collisions): ``1 + m + (C-1)*B*m`` — the seed, then the first
    cycle's m children (single beam at the seed), then ``B*m`` per subsequent cycle. The soft
    advancement does NOT change this accounting (every non-deduped child is still predicted once).
    """
    cost = cost or CostAccount()
    next_id = next_id or _next_id
    seen: set = set()

    # 1) Score the seed once so its backbone is real for the first expansion. Route it through
    #    predict_children (DRY) so it gets coords/iptm/chirality/cif_path/prediction set the SAME
    #    way cycle children do — crucially cif_path, which expand_state(seed)->extract_fn needs
    #    (the real _extract raises if cif_path is None). Charges exactly the 1 seed predict.
    predict_children([seed], predict_fn, cost, ref_time_steps=ref_time_steps, out_dir=out_dir)
    if seed.l_seq:
        seen.add(seed.l_seq)

    beam: list[BeamState] = [seed]
    pool: list[BeamState] = []

    for c in range(cycles):
        # Expand every live state; collect all children, then dedup globally before predicting.
        children: list[BeamState] = []
        for parent in beam:
            kids = expand_state(parent, design_fn, extract_fn, anchor_fn=anchor_fn,
                                next_id=next_id, known_seq_fn=known_seq_fn, encode_fn=encode_fn)
            if dedup_on:
                kids = dedup(kids, seen, parent_l_seq=parent.l_seq, cost=cost)
            children.extend(kids)
        if not children:
            logger.warning("beam_search cycle %d: no surviving children (all deduped).", c)
            break

        predict_children(children, predict_fn, cost, ref_time_steps=ref_time_steps,
                         out_dir=out_dir)
        advance, clean_pool, result = prune(children, referee_fn, panel, keep_b=beam_width)
        # WINNER POOL accumulates ONLY hard-veto-passing children across cycles. A vetoed
        # intermediate can NEVER enter the pool, so it can never be SELECTED as a winner
        # (the fallback_used parity-bias guard stays — at selection, not as a search-killer).
        pool.extend(clean_pool)

        if result is not None and result.fallback_used:
            # Whole cycle all-vetoed. This is NO LONGER a stop: the beam GENERALIZES the greedy
            # loop and ADVANCES through dirty intermediates (the accept-always greedy run reaches
            # clean basins at iter2+ the same way). No vetoed child entered the pool, so none can
            # be selected — we just keep exploring on the SOFT score.
            logger.warning(
                "beam_search cycle %d: ALL children vetoed (PanelResult.fallback_used). "
                "Advancing on the soft score (least-dirty children) — no vetoed child can "
                "enter the winner pool.", c)

        # The ONLY stop condition is literally zero children to expand into the next beam
        # (e.g. all deduped). ``advance`` is non-empty whenever ``children`` is non-empty.
        if not advance:
            break
        beam = advance

    return pool, cost


def anneal_best(pool: list[BeamState], make_loop_fn, make_init_fn, score_fn,
                panel, referee_fn, top_n: int = 3, anneal_steps: int = 5,
                anneal_ref_start: int = 200, base_ref_time_steps: int = 50,
                anneal_temp_start: float = 0.3, base_temperature: float = 0.1,
                cost: Optional[CostAccount] = None, out_dir="/tmp/beam_anneal"):
    """Polish the global top-``top_n`` non-vetoed pool states with a short ``HalluLoop`` run.

    For each seed state ``make_loop_fn(state)`` builds a ``HalluLoop`` and ``make_init_fn(state)``
    its initial ``LoopState``; the loop runs ``anneal_steps`` iterations with the three cooling
    levers (decreasing ref_time_steps + cooling MPNN temperature via ``AnnealSchedule`` +
    greedy zero-T accept). The final pick is ``JudgePanel.combine`` over the union of anneal
    states. Returns ``(best_step, anneal_states, cost)``.

    ``cost`` is charged ``anneal_steps`` predicts per seed (the loop's per-iteration structure
    step is a Chai predict in the real wiring).
    """
    from pathlib import Path as _Path

    from xenodesign.loop import greedy_iptm_accept
    from xenodesign.schedule import AnnealSchedule

    cost = cost or CostAccount()
    seeds = sorted([c for c in pool if c.composite is not None],
                   key=lambda c: c.composite, reverse=True)[:top_n]
    schedule = AnnealSchedule(base_ref_time_steps=base_ref_time_steps,
                              anneal_start=anneal_ref_start, temp_start=anneal_temp_start)
    accept = greedy_iptm_accept(min_delta=0.0)

    anneal_states = []
    base = _Path(out_dir)
    for i, state in enumerate(seeds):
        loop = make_loop_fn(state)
        init = make_init_fn(state)
        history = loop.run(init, iterations=anneal_steps, ref_time_steps=base_ref_time_steps,
                           out_dir=base / f"anneal_{i:02d}", accept_fn=accept,
                           schedule=schedule)
        cost.add_predicts(anneal_steps)
        anneal_states.extend(history)

    if not anneal_states:
        return None, [], cost

    scores = [referee_fn(step) for step in anneal_states]
    result = panel.combine(scores)
    best_step = anneal_states[result.selected_idx]
    return best_step, anneal_states, cost
