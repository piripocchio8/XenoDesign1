"""Diff-steps <-> fitness-fidelity calibration (ABC §5.1 — the GATING step).

The ABC fitness is a FAST low-diffusion-step Chai cycle whose objective reads
geometry (BSA/contacts/Zn-N/amide-planarity/pLDDT/pTM). If too few diffusion
steps wreck the structure, the objective stops *ranking* candidates correctly and
the search optimizes noise. This module answers the make-or-break question:

    What is the LOWEST diffusion-step count K* whose objective RANKING still
    matches the full-200-step reference (and preserves good-vs-bad separation)?

It is intentionally pure / numpy-free / CPU-testable. ``spearman``,
``summarize_calibration`` and ``select_k_star`` operate on plain dicts of
objective values; the real low-step Chai predictions are produced by
``calibrate_fast_oracle`` (GPU body, wired in T3) and fed through these helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence


# ── Spearman rank correlation (pure, no numpy/scipy) ──────────────────────────

def _rankdata(values: Sequence[float]) -> list[float]:
    """Average-rank of each value (ties share the mean of their rank span)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j (inclusive) are tied -> share the average rank (1-based)
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation of two equal-length sequences.

    Pure Python (Pearson on the average-ranks). A constant input (zero rank
    variance) yields ``0.0`` by convention — there is no order to correlate.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} != {len(ys)}")
    n = len(xs)
    if n == 0:
        raise ValueError("empty inputs")
    rx = _rankdata(list(xs))
    ry = _rankdata(list(ys))
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return cov / (vx * vy) ** 0.5


# ── per-step summary: rank-corr vs reference + good/bad margin ────────────────

def summarize_calibration(
    objective_by_step: Mapping[int, Mapping[str, float]],
    *,
    labels: Mapping[str, bool],
    reference_step: int = 200,
) -> dict[int, dict[str, float]]:
    """For every step, report rank-corr vs the reference step + good/bad margin.

    Args:
        objective_by_step: ``{step: {seq_id: objective_value}}``. Every step must
            score the SAME set of seq_ids (so the rankings are comparable).
        labels: ``{seq_id: is_good}`` — ``True`` for a known-good sequence,
            ``False`` for a known-bad one. Used for the separation margin.
        reference_step: the full-fidelity step count to rank-correlate against
            (default 200). Must be present in ``objective_by_step``.

    Returns:
        ``{step: {"rank_corr": float, "margin": float}}`` where ``rank_corr`` is
        the Spearman correlation of that step's objective vector with the
        reference step's (over a fixed seq_id order), and ``margin`` is
        ``mean(good) - mean(bad)`` at that step. The reference step itself is
        included (rank_corr == 1.0 by construction).
    """
    if reference_step not in objective_by_step:
        raise ValueError(f"reference_step {reference_step} not in objective_by_step")
    ref = objective_by_step[reference_step]
    seq_ids = list(ref.keys())  # fixed order for all comparisons
    if not seq_ids:
        raise ValueError("reference step has no sequences")

    out: dict[int, dict[str, float]] = {}
    for step, values in objective_by_step.items():
        missing = [s for s in seq_ids if s not in values]
        if missing:
            raise ValueError(f"step {step} missing sequences {missing}")
        xs = [values[s] for s in seq_ids]
        ys = [ref[s] for s in seq_ids]
        rank_corr = spearman(xs, ys)

        goods = [values[s] for s in seq_ids if labels.get(s)]
        bads = [values[s] for s in seq_ids if not labels.get(s)]
        if goods and bads:
            margin = sum(goods) / len(goods) - sum(bads) / len(bads)
        else:
            margin = float("nan")
        out[step] = {"rank_corr": float(rank_corr), "margin": float(margin)}
    return out


# ── K* selection — two compatible call forms ──────────────────────────────────

def _is_summary_table(table: Mapping) -> bool:
    """True if values are {"rank_corr", "margin"} dicts (vs bare corr floats)."""
    for v in table.values():
        return isinstance(v, Mapping) and "rank_corr" in v
    return False


def select_k_star(
    table: Mapping[int, object],
    *,
    threshold: float | None = None,
    min_rank_corr: float = 0.9,
    min_margin: float | None = None,
) -> int | None:
    """Smallest step count whose objective ranks like full-200 (and separates).

    Two compatible forms:

    1. **Per-step correlation** — ``table`` maps ``{step: rank_corr_float}``.
       Pass ``threshold`` (or rely on ``min_rank_corr``); margin is not checked.
       Returns the smallest step whose corr ``>= threshold``, else ``None``.

    2. **Summary table** — ``table`` maps ``{step: {"rank_corr", "margin"}}``
       (the :func:`summarize_calibration` output). Returns the smallest step that
       clears BOTH ``min_rank_corr`` and ``min_margin`` (default ``min_margin``
       0.0, i.e. good strictly >= bad). A NaN margin fails the gate.

    Returns ``None`` when no step passes — the spec §7 STOP/rethink signal.
    """
    corr_gate = threshold if threshold is not None else min_rank_corr

    if _is_summary_table(table):
        margin_gate = 0.0 if min_margin is None else min_margin
        for step in sorted(table):
            row = table[step]
            corr = row["rank_corr"]
            margin = row["margin"]
            if margin != margin:  # NaN
                continue
            if corr >= corr_gate and margin >= margin_gate:
                return step
        return None

    # Per-step bare-correlation form.
    for step in sorted(table):
        if table[step] >= corr_gate:
            return step
    return None


# ── orchestrator (pure body for T2; GPU predict wired in T3) ──────────────────

def calibrate_fast_oracle(
    cases: Sequence[Mapping],
    steps: Sequence[int],
    *,
    predict_fn: Callable[..., object],
    objective_fn: Callable[[object], float],
    reference_step: int = 200,
    min_rank_corr: float = 0.9,
    min_margin: float | None = None,
) -> dict:
    """Run each case at each step, score, and report per-step corr/margin + K*.

    Pure orchestration over INJECTED ``predict_fn`` / ``objective_fn`` so it is
    CPU-testable with fakes (T2) and GPU-real with the chai backends (T3).

    Args:
        cases: each a mapping with at least ``{"id": str, "is_good": bool, ...}``
            plus whatever ``predict_fn`` needs (e.g. ``d_fasta``, ``restrained``,
            ``constraint_path``). The extra keys are forwarded to ``predict_fn``.
        steps: the diffusion-step counts to sweep (must include ``reference_step``).
        predict_fn: ``predict_fn(case, step) -> prediction`` (the structure at
            that step count). Exceptions are caught and scored ``-inf``.
        objective_fn: ``objective_fn(prediction) -> float`` (the geometry objective).
        reference_step: full-fidelity step to rank against (default 200).
        min_rank_corr / min_margin: the K* gates.

    Returns:
        ``{"objective_by_step": {...}, "labels": {...}, "summary": {...},
           "k_star": int | None}``.
    """
    if reference_step not in steps:
        raise ValueError(f"steps {list(steps)} must include reference_step {reference_step}")

    labels = {c["id"]: bool(c.get("is_good", False)) for c in cases}
    objective_by_step: dict[int, dict[str, float]] = {s: {} for s in steps}
    for case in cases:
        cid = case["id"]
        for step in steps:
            try:
                pred = predict_fn(case, step)
                score = float(objective_fn(pred))
            except Exception:
                score = float("-inf")
            objective_by_step[step][cid] = score

    summary = summarize_calibration(
        objective_by_step, labels=labels, reference_step=reference_step
    )
    k_star = select_k_star(
        summary, min_rank_corr=min_rank_corr, min_margin=min_margin
    )
    return {
        "objective_by_step": objective_by_step,
        "labels": labels,
        "summary": summary,
        "k_star": k_star,
    }


# ── GPU-real predict + objective (the T3 body; deferred heavy imports) ─────────

def chai_predict_fn(device: "str | None" = None, seed: int = 0) -> Callable[[Mapping, int], object]:
    """Build a ``predict_fn(case, step) -> Prediction`` over the real Chai backend.

    Each ``case`` is a mapping with:
        - ``id`` (str), ``is_good`` (bool)
        - ``d_fasta`` (str): the pre-emitted mixed-chirality binder sequence
          (paren ``(DXX)`` blocks for D positions, bare letters for L)
        - ``restrained`` (bool): if True, add the Zn ligand entity + pass
          ``constraint_path`` (full-predict route); else unrestrained single chain.
        - ``constraint_path`` (str|Path|None), ``zn_smiles`` (str, default ``[Zn+2]``)
        - ``out_root`` (str|Path): where per-(case,step) chai dirs are written.

    The diffusion-step count is passed as ``num_diffn_timesteps=step`` to the full
    ``ChaiBackend.predict`` for BOTH routes — the truncated sampler does NOT accept a
    ``constraint_path`` (chai_truncated TODO #27), so to keep the restrained vs
    unrestrained comparison on the SAME predict path we vary ``num_diffn_timesteps``
    rather than ``ref_time_steps``. (Unrestrained could use the truncated refine, but
    a full predict at K steps is the apples-to-apples control and is what the campaign
    fitness can also run when a constraint is present.)
    """
    from xenodesign.backends.chai_backend import ChaiBackend  # deferred (torch/chai)

    backend = ChaiBackend(device=device, seed=seed)

    def _predict(case: Mapping, step: int):  # pragma: no cover (gpu)
        out_root = Path(case["out_root"])
        out_dir = out_root / f"{case['id']}_s{step}"
        entities = [
            {"type": "protein", "name": "binder", "sequence": case["d_fasta"],
             "chirality": "L"},  # already paren-emitted; do NOT re-convert
        ]
        constraint_path = None
        if case.get("restrained"):
            entities.append({
                "type": "ligand", "name": "ZN",
                "smiles": case.get("zn_smiles", "[Zn+2]"),
            })
            constraint_path = case.get("constraint_path")
        pred = backend.predict(
            entities, out_dir,
            num_diffn_timesteps=int(step),
            constraint_path=constraint_path,
        )
        # Attach the scored CIF so the geometry objective can re-parse per-atom data
        # (the Prediction dataclass only carries pooled coords/pLDDT).
        from xenodesign.cif_io import _best_cif_path
        try:
            pred._cif_path = _best_cif_path(out_dir / "chai_out")
        except Exception:
            pred._cif_path = None
        return pred

    return _predict


def intramolecular_objective_fn(chain_name: str = "A") -> Callable[[object], float]:
    """Build an ``objective_fn(prediction) -> float`` over the no-target objective.

    Re-parses the binder chain CIF that ChaiBackend just wrote (the Prediction's
    ``last_out_dir`` is not exposed here, so we re-read from disk is avoided — instead
    we rely on the prediction's parsed coords being insufficient for the per-atom
    geometry terms, so this wrapper expects the caller to have set ``out_dir`` on the
    prediction). The simplest robust contract: the predict_fn writes to a known dir and
    the objective reads the best CIF there. We attach the dir on the prediction object.
    """
    from xenodesign.classes.cyclic import (  # deferred (gemmi at call time)
        combine_intramolecular_terms,
        cyclic_records_from_cif,
        intramolecular_terms_from_records,
        INTRAMOLECULAR_WEIGHTS,
    )

    def _objective(prediction) -> float:  # pragma: no cover (gpu)
        ptm = float(getattr(prediction, "ptm", 0.0) or 0.0)
        cif = getattr(prediction, "_cif_path", None)
        if cif is None:
            return float(INTRAMOLECULAR_WEIGHTS["ptm"] * max(0.0, min(1.0, ptm)))
        records = cyclic_records_from_cif(cif, chain_name=chain_name)
        terms = intramolecular_terms_from_records(records, ptm=ptm)
        return combine_intramolecular_terms(terms)

    return _objective


def intramolecular_per_term_fn(chain_name: str = "A") -> Callable[[object], dict]:
    """Like :func:`intramolecular_objective_fn`, but emit the PER-TERM vector (#cyclization-calib).

    Returns ``per_term_fn(prediction) -> dict`` with, in addition to the aggregate objective,
    EACH named objective term alone AND the GROUND-TRUTH structural closure of the head-to-tail
    seam, so the calibration can report per-term (per-term breakdown, not just aggregate):

        objective        : float — the weighted-sum intramolecular objective (the same scalar
                                    ``intramolecular_objective_fn`` returns).
        mainchain_plddt  : float — term (1), seam (termini N/CA/C/O) pLDDT / 100.
        chirality        : float — term (2), 1 - chirality-violation frac.
        geometry         : float — term (3), closure-amide planarity + backbone valence sanity.
        ptm              : float — term (4), prediction.ptm clamped to [0,1].
        cn_distance      : float|None — GROUND TRUTH: |C(res L) - N(res 1)| (A). ~1.33 = closed.
        closure_omega    : float|None — closure-amide dihedral (deg).
        omega_planarity  : float|None — amide_omega_score of that omega in [0,1].
        closed           : bool — cn_distance <= 1.6 A (a closed ring).

    Pure-pTM fallback (no CIF on the prediction) yields the objective + ptm term only; the
    geometry/closure fields are None so a single missing CIF never crashes the sweep.
    """
    from xenodesign.classes.cyclic import (  # deferred (gemmi at call time)
        combine_intramolecular_terms,
        cyclic_records_from_cif,
        head_to_tail_closure_geometry_from_cif,
        intramolecular_terms_from_records,
        INTRAMOLECULAR_WEIGHTS,
    )

    def _per_term(prediction) -> dict:  # pragma: no cover (gpu)
        ptm = float(getattr(prediction, "ptm", 0.0) or 0.0)
        cif = getattr(prediction, "_cif_path", None)
        if cif is None:
            ptm_term = float(max(0.0, min(1.0, ptm)))
            return {
                "objective": float(INTRAMOLECULAR_WEIGHTS["ptm"] * ptm_term),
                "mainchain_plddt": None, "chirality": None, "geometry": None,
                "ptm": ptm_term, "cn_distance": None, "closure_omega": None,
                "omega_planarity": None, "closed": False,
            }
        records = cyclic_records_from_cif(cif, chain_name=chain_name)
        terms = intramolecular_terms_from_records(records, ptm=ptm)
        closure = head_to_tail_closure_geometry_from_cif(cif, chain_name=chain_name) or {}
        out = {"objective": combine_intramolecular_terms(terms)}
        out.update(terms)
        out.update({
            "cn_distance": closure.get("cn_distance"),
            "closure_omega": closure.get("closure_omega"),
            "omega_planarity": closure.get("omega_planarity"),
            "closed": bool(closure.get("closed", False)),
        })
        return out

    return _per_term
