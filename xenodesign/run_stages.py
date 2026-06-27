"""Strategy-uniform run stages for the unified pipeline spine (S3a).

These are FREE functions (not BinderClass hooks) precisely because the concerns they own —
sequence_constraints production (spec §3.3), Restraints emission (§3.4), and the composed Gates —
must be identical across greedy / beam / ABC, none of which share a class loop. ``run_design`` and
``dispatch._run_abc`` both call them. Pure CPU; heavy imports (chai/gemmi/MetalHawk) are deferred.

All S3a routing is gated on ``XENO_SEQ_STAGE`` (default OFF) at the CALL SITES (dispatch); these
helpers themselves are pure and flag-agnostic so they are independently unit-testable.
"""
from __future__ import annotations

import logging
from typing import Optional

from pathlib import Path

from xenodesign.seq_stage import FrozenPosition

logger = logging.getLogger(__name__)


def frozen_from_coord_residues(coord_residues) -> set:
    """The spec §3.3 metal-coordination producer: declared coordinator tuples -> the FrozenPosition
    set carrying IDENTITY + CHIRALITY (not the position-only set S2 used).

    Each tuple is ``(pos1based, one_letter[, three_letter, chirality, atom])`` as stored by the
    ``--coord_residues`` flag (see ``classes/cyclic.py:_coord_residues``). The 4th element
    (chirality 'L'/'D') is carried so ``encode_d_fasta`` / ``ensure_canonical_anchor`` honour the
    donor handedness; the 2nd (one_letter, e.g. 'H') is carried as ``identity`` so
    ``build_known_seq`` pins the donor identity. Back-compat: a 2-tuple yields ``chirality=None``.
    """
    out = set()
    for t in (coord_residues or ()):
        pos0 = int(t[0]) - 1
        identity = t[1] if len(t) > 1 else None
        chirality = t[3] if len(t) > 3 else None
        out.add(FrozenPosition(position0=pos0, identity=identity, chirality=chirality))
    return out


def build_run_restraints(cfg, *, out_dir, case=None, target_ctx=None, roles=None):
    """The spec §3.4 uniform restraint emitter: coordination + closure rows for ONE run, reusing the
    cyclic class's metal/closure logic and the benchmark @atom covalent grammar.

    Returns the written ``.restraints`` Path, or ``None`` when restraints are off. The metal case
    (target_type != 'none' with declared coordinators) gets native-covalent ``@atom`` coordination
    rows + a head-to-tail closure row; a non-metal cyclic gets closure-only. Delegates to the SAME
    ``Cyclic.restraints`` hook the greedy path uses, so the rows are byte-identical across strategies
    — the unification the spec asks for (ABC currently emits NONE of these).
    """
    if not getattr(cfg, "restraints_on", False):
        return None

    from xenodesign.benchmark.cases import get_case
    from xenodesign.classes.cyclic import Cyclic

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = case or get_case("cyclic")
    # Cyclic.restraints owns the metal-coordination + closure emission (and the declared-coordinator
    # REQUIRE guard); it reads coord_residues + the closure default from cfg. target_ctx carries the
    # assembled entity list when available (drives the Zn/binder chain split); None -> legacy order.
    return Cyclic().restraints(cfg, case, out_dir, target_ctx)


def build_run_gates(cfg, *, roles=None, panel=None, last_out_dir_fn=None):
    """The spec §3 uniform GATE composer: pick the accept gates for this run from (binder_class,
    target_type, gate knobs) and AND-compose them. Returns a single ``accept_fn`` or ``None``.

    Flat, fully-overridable lookup (ponytail: no rules engine — same shape as resolve_defaults will
    take in S3b):
      * class == 'alpha'  AND cfg.gates.periodicity   -> periodicity_gated_accept
      * class == 'non_alpha'                            -> alpha_demote_gated_accept (anti-alpha)
      * target == 'metal' AND cfg.gates.metal_geometry -> metalhawk_gated_accept
      * cfg.gates.pll_veto                              -> (pLM veto; the existing referee channel —
                                                            applied at panel time, so NOT a loop gate
                                                            here; documented, no double-application)

    ``compose_accept_fns`` is None-safe: no surviving gate -> None (the loop's accept-always default,
    byte-identical to legacy when no gate applies).

    Args:
        cfg: DesignConfig (from resolve_config).
        roles: ChainRoles or None (threaded from dispatch; not used directly here yet).
        panel: Optional JudgePanel with score_fn for alpha_demote_gated_accept. When None a stub
               panel is constructed (returns helix_fraction=None -> always-accept, best-effort).
        last_out_dir_fn: Optional ``() -> Path`` that returns the most recent per-step output dir;
               required for MetalHawk to locate the predicted CIF. None -> pass-through accept.
    """
    from xenodesign.loop import (
        alpha_demote_gated_accept, compose_accept_fns, metalhawk_gated_accept,
        periodicity_gated_accept,
    )

    gates = []
    if cfg.binder_class == "alpha" and cfg.gates.periodicity:
        gates.append(periodicity_gated_accept(heptad_thresh=cfg.gates.heptad_thresh))
    if cfg.binder_class == "non_alpha":
        # alpha_demote_gated_accept requires a JudgePanel with score_fn. When no real panel is
        # injected (e.g. CPU-only unit tests or pre-S3a.4 callers) supply a best-effort stub:
        # helix_fraction=None -> gate always accepts (mirrors the "never crash a trajectory" rule).
        _panel = panel
        if _panel is None:
            from xenodesign.judges.panel import JudgePanel, RefereeScore
            _panel = JudgePanel(score_fn=lambda step: RefereeScore(helix_fraction=None))
        gates.append(alpha_demote_gated_accept(_panel))
    if cfg.target.target_type == "metal" and cfg.gates.metal_geometry:
        gates.append(metalhawk_gated_accept(_make_metalhawk_score_fn(cfg, last_out_dir_fn)))
    return compose_accept_fns(*gates)


def make_helix_panel_for_gates(last_out_dir_fn, roles=None):
    """Build a JudgePanel whose ``score_fn`` reads helix_fraction from the per-step CIF.

    This is the REAL panel for ``alpha_demote_gated_accept`` in the greedy dispatch path
    (S3a.3d).  The helix source mirrors ``_make_referee_fn`` / ``_binder_helix_fraction``:
    read the binder chain's CA trace from the CIF that the wrapper just wrote and compute
    ``secondary_structure.helix_fraction`` over it.

    Args:
        last_out_dir_fn: ``() -> Path`` — returns the most-recent per-step output dir.
            The same lambda supplied to ``build_run_gates`` as ``last_out_dir_fn``.  The
            wrapper records ``last_out_dir`` BEFORE calling the refine fn (i.e. before
            ``accept_fn`` is called), so the CIF is present at call time.
        roles: ChainRoles or None.  None → binder chain 'B' (alpha default).
    """
    from xenodesign.judges.panel import JudgePanel, RefereeScore

    binder_chain = roles.binder if roles is not None else "B"

    def _score(step):
        """Return RefereeScore.helix_fraction from the last-written CIF (None on any failure)."""
        try:
            from xenodesign.cif_io import _best_cif_path
            from xenodesign.classes._alpha_internals import _binder_helix_fraction
            cif = _best_cif_path(last_out_dir_fn())
            helix = _binder_helix_fraction(cif, chain=binder_chain)
        except Exception as _exc:
            logger.debug("make_helix_panel_for_gates: helix read failed — degrading to None "
                         "(gate will accept); reason: %s: %s", type(_exc).__name__, _exc)
            helix = None  # never crash a trajectory — gate will accept on None
        return RefereeScore(chirality_violation=0.0, iptm=0.0, helix_fraction=helix)

    return JudgePanel(score_fn=_score)


def _make_metalhawk_score_fn(cfg, last_out_dir_fn):
    """A ``score_fn(step) -> GateResult`` reading the step's predicted CIF and running the (best-
    effort, subprocess-isolated) MetalHawk geometry gate. ``last_out_dir_fn() -> dir`` locates the
    per-step CIF (the wrapper records ``last_out_dir``); a missing CIF -> a pass-through GateResult
    (ok=False), which the gate accepts. Monkeypatched in CPU tests (MetalHawk env never invoked)."""
    from xenodesign.eval.metal_geometry_gate import GateResult, metal_geometry_gate

    thresh = float(cfg.gates.metal_perplexity_thresh)

    def _score(step):
        try:
            if last_out_dir_fn is None:
                return GateResult(passed=True, ok=False, error="no out_dir fn")
            from xenodesign.cif_io import _best_cif_path
            cif = _best_cif_path(last_out_dir_fn())
            return metal_geometry_gate(cif, threshold=thresh)
        except Exception as exc:  # never crash a design run
            return GateResult(passed=True, ok=False, error=f"{type(exc).__name__}: {exc}")

    return _score
