"""α-case (trimer D/L-ABLE) BEAM+ANNEAL design driver — frozen experiment (ADR-018, WF-1 spec).

A self-contained SUPERSET of ``scripts/design_alpha.py``: it reuses that driver's reviewed
helpers (seed/restraint construction, the C-terminal-Gly anchor, the panel referee, the run
scaffolding) by IMPORT ONLY — ``design_alpha.py`` is never edited — and replaces the single
greedy ``HalluLoop.run`` with the beam+anneal search in ``xenodesign/beam.py``:

  * BEAM (``beam.beam_search``) widens each greedy step into ``beam_width`` live states, each
    expanded with a ``MultiCandidate(top_k=children_per_branch)`` over the context-aware
    LigandMPNN (cheap, MPNN-only), deduped, predicted once each with the real
    ``_PredictBackendWrapper`` (one Chai predict/child), then pruned to the top-``beam_width`` by
    the ``JudgePanel`` COMPOSITE — hard chirality + composition veto FIRST; binding is a RELATIVE
    min-max gradient within the cycle's batch, never an absolute ipTM cutoff.
  * ANNEAL (``beam.anneal_best``) polishes the global top-``anneal_top_n`` non-vetoed states with
    a short ``HalluLoop`` run under three cooling levers (decreasing ref_time_steps + cooling MPNN
    temperature via ``AnnealSchedule`` + greedy zero-T accept).

Every expensive op is INJECTED into ``beam`` as a callable; this file is the wiring that builds
the real callables from the ``design_alpha`` seams. ``beam.py`` itself never imports torch/chai.

Cost (``CostAccount`` counts Chai predicts only): defaults B=3,m=3,C=3,anneal_steps=5,top_n=3 ->
``1 + m + (C-1)*B*m`` (= 22) beam predicts + ``top_n*anneal_steps`` (= 15) = 37 predicts (vs 8 for
a 7-iter greedy run). ``--smoke`` forces B=2,m=2,C=2,anneal_steps=2 = 7 + 6 = 13 predicts.

Usage (inside the gradio_design / chai Docker container, PYTHONPATH=/work, --network host for
PepMLM; GPU required for the real run — the CPU suite only exercises CLI parsing):
    python scripts/design_alpha_beam.py --device cuda:0
    python scripts/design_alpha_beam.py --smoke                 # 13-predict wiring smoke
    python scripts/design_alpha_beam.py --beam_width 4 --cycles 4 --no_dedup
    python scripts/design_alpha_beam.py --help

D-peptide reporting: the selected binder's one-letter sequence is the L (mirror) projection
emitted by the backend; any D-peptide surfaced is reported LOWERCASE with Gly as ``G`` (project
convention) via :func:`_report_d_peptide`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse design_alpha's reviewed seams by IMPORT ONLY (ADR-018 — design_alpha.py is NOT edited).
from scripts.design_alpha import (
    _DEFAULT_DEVICE,
    _DEFAULT_REF_TIME_STEPS,
    _RUN_BINDER_CHAIN,
    _RUN_TARGET_CHAIN,
    _TARGET_RECORD,
    _backbone_array_from_residues,
    _best_cif_path,
    _binder_helix_fraction,
    _chirality_violation_frac_from_cif,
    _cterm_gly_anchor,
    _ensure_cterm_glycine,
    _loop_score_fn,
    binder_seq_from_cif,
    build_alpha_restraint,
    build_alpha_seed,
    composition_violation,
    make_mixed_loop_score_fn,
    mixed_objective_from_cif,
)
from scripts.design_demo import (
    _all_atoms_from_chain,
    _PredictBackendWrapper,
)

# Beam defaults (the CLI overrides these; --smoke shrinks them). The panel weights mirror
# design_alpha's localized _ALPHA_WEIGHTS exactly (chir/bind 40/40, pll/mir 0.05, ss_bias 0.10).
_DEFAULT_BEAM_WIDTH = 3
_DEFAULT_CHILDREN_PER_BRANCH = 3
_DEFAULT_CYCLES = 3
_DEFAULT_NUM_SEQS = 8                  # MultiCandidate oversample (top_k <= num_seqs)
_DEFAULT_ANNEAL_STEPS = 5
_DEFAULT_ANNEAL_TOP_N = 3
_DEFAULT_ANNEAL_REF_START = 200        # ref_time_steps anneal start (-> base ref_time_steps)
_DEFAULT_ANNEAL_TEMP_START = 0.3       # MPNN temperature anneal start (-> base 0.1)
_DEFAULT_BASE_TEMPERATURE = 0.1

_ALPHA_WEIGHTS = {"chirality": 0.40, "binding": 0.40, "pll": 0.05, "mirror": 0.05,
                  "ss_bias": 0.10}


def _effective_children(children_per_branch: int, num_seqs: int) -> int:
    """Cap children_per_branch (the MultiCandidate top_k) at num_seqs so 1 <= top_k <= num_seqs
    always holds (MultiCandidate asserts it). Lets the CLI over-ask without crashing."""
    return max(1, min(int(children_per_branch), int(num_seqs)))


def _smoke_knobs():
    """The --smoke beam knobs: (B, m, C, anneal_steps) = (2, 2, 2, 2) -> 7 + top_n*2 predicts.
    Returns (B, m, C, anneal_steps, anneal_top_n)."""
    return 2, 2, 2, 2, _DEFAULT_ANNEAL_TOP_N


def _report_d_peptide(l_seq: str) -> str:
    """Report a designed binder's D-peptide sequence LOWERCASE with Gly as 'G' (project
    convention). The backend emits the L (mirror) projection; the physical chain is all-D, so we
    lowercase every residue except achiral glycine."""
    return "".join(c if c.upper() == "G" else c.lower() for c in (l_seq or ""))


# ── Beam-state extraction (reads the SCORED CIF of the parent, not wrapper.last_out_dir) ──

def _make_extract_fn(roles=None):
    """Build the beam ``extract_fn(parent) -> inputs dict`` replicating design_alpha._extract but
    reading from ``parent.cif_path`` (the parent's own scored out_dir) — beam states are scored
    out of loop order, so the wrapper's mutable ``last_out_dir`` is NOT a safe source.

    ``roles`` (a ``ChainRoles``) is the dispatch chain contract: when supplied (the dispatch beam
    path) the binder/context chains come from it, so a non-'B' binder (non_alpha 'C', no-target 'A')
    reads the right chain. ``roles=None`` keeps the standalone α beam driver byte-identical
    (binder ``_RUN_BINDER_CHAIN``, context ``_RUN_TARGET_CHAIN``)."""
    from xenodesign.eval.gate_tier0a import backbone_by_residue_from_cif

    binder_chain = roles.binder if roles is not None else _RUN_BINDER_CHAIN
    context_chain = roles.context if roles is not None else _RUN_TARGET_CHAIN

    def _extract(parent) -> dict:
        if parent.cif_path is None:
            raise RuntimeError("extract_fn called on an unpredicted parent (cif_path is None)")
        cif = _best_cif_path(Path(parent.cif_path))
        binder_res = (backbone_by_residue_from_cif(cif, binder_chain)
                      or backbone_by_residue_from_cif(cif, binder_chain.lower()))
        if not binder_res:
            raise RuntimeError(f"cannot extract binder chain {binder_chain!r} from {cif}")
        design_backbone = _backbone_array_from_residues(binder_res)
        ctx_coords, ctx_elements = _all_atoms_from_chain(cif, context_chain)
        if ctx_coords.shape[0] == 0:
            ctx_coords, ctx_elements = _all_atoms_from_chain(cif, context_chain.lower())
        return {
            "design_backbone": design_backbone,
            "design_codes": ["DAL"] * design_backbone.shape[0],
            "context_coords": ctx_coords,
            "context_elements": ctx_elements,
        }

    return _extract


# ── Beam referee (adapts design_alpha._make_referee_fn to a BeamState) ─────────────────

def _make_beam_referee_fn(esm_judge=None):
    """Build the beam ``referee_fn(child) -> RefereeScore``, mirroring design_alpha's referee but
    reading from a ``BeamState``: chirality / scored-seq / helix from ``child.cif_path``, and
    iptm / plddt / token_index from ``child.prediction`` (set by ``beam.predict_children``).

    Chirality always carries the TRUE measured value (#37 de-conflation); a low-complexity scored
    sequence is vetoed on the panel's INDEPENDENT composition channel, never by overwriting
    chirality."""
    from xenodesign.judges.panel import RefereeScore

    def _score(child) -> RefereeScore:
        cif = None
        try:
            cif = _best_cif_path(Path(child.cif_path))
            chir = _chirality_violation_frac_from_cif(cif)
        except Exception:
            chir = 1.0   # unverifiable -> treat as a chirality violation, never false-clean

        pred = child.prediction
        ti = np.asarray(pred.token_index)
        mask = ti == 1
        iface_plddt = (float(pred.plddt[mask].mean()) if mask.any()
                       else float(pred.plddt.mean()))

        helix = None
        if cif is not None:
            try:
                helix = _binder_helix_fraction(cif)
            except Exception:
                helix = None

        scored_l_seq = None
        if cif is not None:
            try:
                scored_l_seq = binder_seq_from_cif(cif, _RUN_BINDER_CHAIN)
            except Exception:
                scored_l_seq = None

        comp_viol = (scored_l_seq is not None and composition_violation(scored_l_seq))

        pll = None
        if esm_judge is not None and scored_l_seq is not None:
            try:
                pll = esm_judge(scored_l_seq)
            except Exception:
                pll = None

        return RefereeScore(
            chirality_violation=chir, iptm=pred.iptm,
            interface_plddt=iface_plddt, pll=pll, mirror_discrepancy=None,
            composition_violation=comp_viol, helix_fraction=helix,
        )

    return _score


# ── Main driver ────────────────────────────────────────────────────────────────

def run_alpha_beam_design(
    beam_width: int = _DEFAULT_BEAM_WIDTH,
    children_per_branch: int = _DEFAULT_CHILDREN_PER_BRANCH,
    cycles: int = _DEFAULT_CYCLES,
    num_seqs: int = _DEFAULT_NUM_SEQS,
    anneal_steps: int = _DEFAULT_ANNEAL_STEPS,
    anneal_top_n: int = _DEFAULT_ANNEAL_TOP_N,
    anneal_ref_start: int = _DEFAULT_ANNEAL_REF_START,
    anneal_temp_start: float = _DEFAULT_ANNEAL_TEMP_START,
    ref_time_steps: int = _DEFAULT_REF_TIME_STEPS,
    prune_metric: str = "composite",
    dedup_on: bool = True,
    device: str = _DEFAULT_DEVICE,
    seed: int = 42,
    out_dir: "Path | str | None" = None,
    use_pepmlm: bool = True,
    seed_seq: "str | None" = None,
    restraints: bool = True,
    use_pll: bool = True,
    esm_device: "str | None" = None,
    restraint_file: "str | Path | None" = None,
    target_fasta: "str | None" = None,
    objective: str = "iptm",
    periodicity_gate: bool = False,
    heptad_thresh: float = 0.35,
) -> dict:
    """Run the α BEAM+ANNEAL design end-to-end (GPU-only) and return a result dict.

    Wires the real ``design_alpha`` callables into ``xenodesign.beam``: the seed is predicted
    once, ``cycles`` beam cycles expand/dedup/predict/prune to the top-``beam_width`` by panel
    composite, then ``anneal_best`` polishes the global top-``anneal_top_n`` non-vetoed states.
    The final pick is a ``JudgePanel.combine`` over the union of anneal states.

    GPU/network only — the CPU test-suite exercises only the CLI (``_parse_args``) and the
    budget math; this body imports ChaiBackend / PepMLM / ESM lazily so importing the module is
    CPU-clean.
    """
    import tempfile

    from xenodesign.backends.chai_backend import ChaiBackend
    from xenodesign.benchmark.cases import get_case, ss_bias_config_for_case
    from xenodesign.io_spec import d_fasta_to_one_letter, to_d_fasta
    from xenodesign.judges.panel import JudgePanel
    from xenodesign.loop import HalluLoop, LoopState
    from xenodesign.scorer import sequence_quality_key
    from xenodesign.seed import read_target_sequence, reflect_binder_in_complex_from_cif
    from xenodesign.sequence_update import _ligandmpnn_design_fn
    from xenodesign.inverse_folding import MultiCandidate

    from xenodesign.beam import BeamState, CostAccount, anneal_best, beam_search

    m = _effective_children(children_per_branch, num_seqs)

    case = get_case("alpha")
    if target_fasta is not None:
        import dataclasses
        case = dataclasses.replace(case, fasta_path=target_fasta)
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="xd_alpha_beam_"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Restraint (same RUN-chain emission as design_alpha) ────────────────────
    constraint_path = None
    if restraints:
        if restraint_file is not None:
            import shutil
            constraint_path = out_dir / "alpha.restraints"
            shutil.copyfile(restraint_file, constraint_path)
        else:
            constraint_path = build_alpha_restraint(case, out_dir)

    # ── ESM-PLL judge (lazy; only when running) ────────────────────────────────
    esm_judge = None
    if use_pll:
        from xenodesign.judges.plm_judge import ESMPseudoLogLikelihood
        esm_judge = ESMPseudoLogLikelihood(device=esm_device or "cpu")

    # ── Target + seed ──────────────────────────────────────────────────────────
    target_seq = read_target_sequence(case.fasta_path, name=_TARGET_RECORD)
    seed_l_seq = build_alpha_seed(case, target_seq, use_pepmlm=use_pepmlm, seed_seq=seed_seq,
                                  pepmlm_seed=seed)
    target_entity = {"type": "protein", "name": "target",
                     "sequence": target_seq, "chirality": "L"}

    t0 = time.time()
    print(f"\n{'='*78}\nXenoDesign1 — α (trimer D/L-ABLE) BEAM+ANNEAL design\n{'='*78}")
    print(f"  target (chain A, L) : {len(target_seq)} aa")
    print(f"  seed   (chain B, D) : {_report_d_peptide(seed_l_seq)}  ({len(seed_l_seq)} aa)")
    print(f"  beam B={beam_width} m={m} (asked {children_per_branch}) C={cycles} "
          f"num_seqs={num_seqs} dedup={dedup_on} prune={prune_metric}")
    print(f"  anneal steps={anneal_steps} top_n={anneal_top_n} "
          f"ref_start={anneal_ref_start} temp_start={anneal_temp_start}")
    print(f"  device {device} | ref_time_steps {ref_time_steps} | restraints {restraints} "
          f"({constraint_path}) | use_pll {use_pll}")
    print(f"  out_dir {out_dir}\n{'='*78}\n")

    # ── Backend + L-seed predict + double-flip → D-correct seed coords ─────────
    print("[1/4] Loading ChaiBackend (weights load once) ...")
    backend = ChaiBackend(device=device, seed=seed)

    print("[2/4] L-seed predict + double-flip → D-correct seed coords ...")
    l_seed_entities = [
        target_entity,
        {"type": "protein", "name": "binder", "sequence": seed_l_seq, "chirality": "L"},
    ]
    l_seed_pred = backend.predict(l_seed_entities, out_dir / "p0_l_seed",
                                  num_diffn_timesteps=200, constraint_path=constraint_path)
    l_seed_cif = _best_cif_path(out_dir / "p0_l_seed")
    d_seed_coords = reflect_binder_in_complex_from_cif(l_seed_cif, binder_chain="B", axis=0)
    print(f"    L-seed ipTM={l_seed_pred.iptm:.4f} ; D-seed {d_seed_coords.shape[0]} atoms")

    # ── Wire the injected callables for beam.py ────────────────────────────────
    # predict_fn: the SAME _PredictBackendWrapper design_alpha uses as its restrained refine_fn.
    predict_fn = _PredictBackendWrapper(backend, target_entity, constraint_path=constraint_path)

    # design_fn: MultiCandidate(top_k=m) over the C-term-Gly-anchored context-aware LigandMPNN,
    # re-ranked by the pure-sequence de-gaming key (cheap; multiplies only MPNN passes).
    design_fn = MultiCandidate(_cterm_gly_anchor(_ligandmpnn_design_fn), num_seqs=num_seqs,
                               key_fn=sequence_quality_key, top_k=m)
    extract_fn = _make_extract_fn()
    referee_fn = _make_beam_referee_fn(esm_judge=esm_judge)
    panel = JudgePanel(weights=_ALPHA_WEIGHTS, ss_bias=ss_bias_config_for_case(case))
    cost = CostAccount()

    # ── [3/4] BEAM search ──────────────────────────────────────────────────────
    print(f"[3/4] beam_search (B={beam_width}, m={m}, C={cycles}) ...")
    seed_state = BeamState(d_fasta=to_d_fasta(seed_l_seq), coords=d_seed_coords,
                           l_seq=seed_l_seq, cycle=0)
    pool, cost = beam_search(
        seed_state, design_fn=design_fn, predict_fn=predict_fn, extract_fn=extract_fn,
        referee_fn=referee_fn, panel=panel, beam_width=beam_width, children_per_branch=m,
        cycles=cycles, cost=cost, anchor_fn=_ensure_cterm_glycine, dedup_on=dedup_on,
        ref_time_steps=ref_time_steps, out_dir=out_dir / "beam",
    )
    print(f"    beam pool: {len(pool)} non-vetoed states | {cost.summary()}")

    # ── [4/4] ANNEAL polish over the global top-anneal_top_n ───────────────────
    print(f"[4/4] anneal_best (top_n={anneal_top_n}, steps={anneal_steps}) ...")
    seq_update_fn = _make_anneal_seq_update_fn(predict_fn, num_seqs=num_seqs)

    # --objective: the anneal loop's per-iteration score_fn is the mixed parity-aware composite of
    # the candidate CIF when objective=='mixed' (graceful ipTM fallback), else the reproducible
    # ipTM+pLDDT design_score. This is the beam analogue of the greedy in-loop score_fn routing.
    anneal_score_fn = make_mixed_loop_score_fn(predict_fn) if objective == "mixed" else _loop_score_fn

    def _make_loop_fn(_state):
        return HalluLoop(backend=predict_fn, sequence_update_fn=seq_update_fn,
                         score_fn=anneal_score_fn, refine_fn=predict_fn)

    def _make_init_fn(state):
        return LoopState(d_fasta=state.d_fasta, coords=state.coords)

    best_step, anneal_states, cost = anneal_best(
        pool, make_loop_fn=_make_loop_fn, make_init_fn=_make_init_fn,
        score_fn=anneal_score_fn, panel=panel, referee_fn=_make_anneal_referee_fn(esm_judge),
        top_n=anneal_top_n, anneal_steps=anneal_steps, anneal_ref_start=anneal_ref_start,
        base_ref_time_steps=ref_time_steps, anneal_temp_start=anneal_temp_start,
        base_temperature=_DEFAULT_BASE_TEMPERATURE, cost=cost, out_dir=out_dir / "anneal",
    )

    # ── beam-dep final selection: periodicity gate (reject register-UNachievable picks) +
    # mixed re-selection over the anneal states. The gate is a DESIGN-TIME sequence-only filter on
    # the SELECTED design (heptad-periodic -> register not scorable). Mixed re-ranks the surviving
    # anneal states by the parity-aware composite of each state's scored CIF. Both default-OFF:
    # objective=='iptm' + periodicity_gate==False reproduces the byte-for-byte original pick.
    if (objective == "mixed" or periodicity_gate) and anneal_states:
        best_step = _select_beam_final(
            anneal_states, _make_anneal_referee_fn(esm_judge), panel,
            objective=objective, periodicity_gate=periodicity_gate, heptad_thresh=heptad_thresh,
            default_step=best_step)

    # ── Report the final pick ──────────────────────────────────────────────────
    wall = time.time() - t0
    sel_l_seq = None
    if best_step is not None:
        try:
            sel_l_seq = binder_seq_from_cif(
                _best_cif_path(Path(predict_fn.last_out_dir)), _RUN_BINDER_CHAIN)
        except Exception:
            sel_l_seq = d_fasta_to_one_letter(best_step.state.d_fasta)

    result = {
        "case_id": "alpha",
        "selected_l_seq": sel_l_seq,
        "selected_d_peptide": _report_d_peptide(sel_l_seq) if sel_l_seq else None,
        "selected_iptm": float(best_step.prediction.iptm) if best_step is not None else None,
        "beam_pool_size": len(pool),
        "predicts": cost.predicts,
        "dedup_hits": cost.dedup_hits,
        "cost_summary": cost.summary(),
        "l_seed_iptm": float(l_seed_pred.iptm),
        "beam_width": beam_width, "children_per_branch": m, "cycles": cycles,
        "num_seqs": num_seqs, "anneal_steps": anneal_steps, "anneal_top_n": anneal_top_n,
        "prune_metric": prune_metric, "dedup": bool(dedup_on),
        "objective": objective,
        "periodicity_gate": bool(periodicity_gate),
        "heptad_thresh": float(heptad_thresh),
        "restraints": bool(restraints),
        "constraint_path": str(constraint_path) if constraint_path is not None else None,
        "use_pll": bool(use_pll), "wall_time_s": wall, "out_dir": str(out_dir),
    }
    (out_dir / "alpha_beam_result.json").write_text(
        json.dumps(result, indent=2, default=lambda o: getattr(o, "tolist", lambda: str(o))()))

    print(f"\n{'='*78}")
    if sel_l_seq is not None:
        print(f"SELECTED D-peptide: {_report_d_peptide(sel_l_seq)}  "
              f"ipTM {result['selected_iptm']:.4f}")
    print(f"  {cost.summary()}")
    print(f"  wall {wall/60:.1f} min | result -> {out_dir/'alpha_beam_result.json'}\n{'='*78}")
    return result


def _select_beam_final(anneal_states, referee_fn, panel, objective: str,
                       periodicity_gate: bool, heptad_thresh: float, default_step):
    """beam-dep final pick: filter anneal states by the DESIGN-TIME register gate, then rank the
    survivors by the requested objective.

    * periodicity_gate: drop any anneal state whose binder sequence (decoded from
      ``step.state.d_fasta``, sequence-only) is register-UNachievable (heptad-periodic). If the gate
      removes EVERY survivor, it is treated as inert (we keep all states) — the gate must never
      leave the run with no design to report.
    * objective=='mixed': rank survivors by the panel COMPOSITE of their referee scores (the beam's
      composite selection, which already blends binding/chirality/etc.) — the anneal states have no
      addressable per-step CIF, so we reuse the panel composite rather than rebuild a score_complex
      panel per step. objective=='iptm': rank survivors by ipTM.

    Returns the chosen ``LoopStep`` (or ``default_step`` if nothing survives / scores)."""
    from xenodesign.io_spec import d_fasta_to_one_letter
    from scripts.seq_periodicity import compute as _periodicity_compute

    survivors = list(anneal_states)
    if periodicity_gate:
        kept = []
        for step in survivors:
            d_fasta = getattr(step.state, "d_fasta", "") or ""
            try:
                l_seq = d_fasta_to_one_letter(d_fasta) if d_fasta else ""
            except Exception:
                l_seq = ""
            if not l_seq or _periodicity_compute(l_seq, heptad_thresh=heptad_thresh)["register_achievable"]:
                kept.append(step)
        if kept:                       # never leave zero designs to report (gate is inert if it would)
            survivors = kept

    if not survivors:
        return default_step

    if objective == "mixed":
        scores = [referee_fn(step) for step in survivors]
        result = panel.combine(scores)
        return survivors[result.selected_idx]
    # iptm: pick the highest-ipTM survivor.
    return max(survivors, key=lambda s: float(getattr(s.prediction, "iptm", 0.0)))


def _make_anneal_seq_update_fn(wrapper: _PredictBackendWrapper, num_seqs: int = _DEFAULT_NUM_SEQS):
    """The anneal loop's sequence_update_fn(prediction) -> one-letter L seq — the SAME drift-fixed
    MultiCandidate-over-LigandMPNN closure design_alpha uses, reading the binder chain from the
    wrapper's last scored CIF. Mirrors design_alpha.make_alpha_seq_update_fn (top_k=1: anneal is a
    single-path greedy polish, not a beam expansion)."""
    from xenodesign.eval.gate_tier0a import backbone_by_residue_from_cif
    from xenodesign.inverse_folding import MultiCandidate
    from xenodesign.scorer import sequence_quality_key
    from xenodesign.sequence_update import (
        SequenceUpdater, _ligandmpnn_design_fn, make_sequence_update_fn,
    )

    design_fn = MultiCandidate(_cterm_gly_anchor(_ligandmpnn_design_fn), num_seqs=num_seqs,
                               key_fn=sequence_quality_key)
    updater = SequenceUpdater(design_fn=design_fn)

    def _extract(prediction) -> dict:
        out_dir = wrapper.last_out_dir
        if out_dir is None:
            raise RuntimeError("seq_update called before any structure step")
        cif = _best_cif_path(out_dir)
        binder_res = (backbone_by_residue_from_cif(cif, _RUN_BINDER_CHAIN)
                      or backbone_by_residue_from_cif(cif, _RUN_BINDER_CHAIN.lower()))
        if not binder_res:
            raise RuntimeError(f"cannot extract binder chain from {cif}")
        design_backbone = _backbone_array_from_residues(binder_res)
        ctx_coords, ctx_elements = _all_atoms_from_chain(cif, _RUN_TARGET_CHAIN)
        if ctx_coords.shape[0] == 0:
            ctx_coords, ctx_elements = _all_atoms_from_chain(cif, _RUN_TARGET_CHAIN.lower())
        return {
            "design_backbone": design_backbone,
            "design_codes": ["DAL"] * design_backbone.shape[0],
            "context_coords": ctx_coords,
            "context_elements": ctx_elements,
        }

    base_fn = make_sequence_update_fn(updater, _extract, emit="one_letter")

    def _guarded(prediction) -> str:
        return _ensure_cterm_glycine(base_fn(prediction))

    return _guarded


def _make_anneal_referee_fn(esm_judge=None):
    """Referee for an anneal ``LoopStep`` — ``anneal_best`` passes a bare ``LoopStep`` (not the
    ``(step, idx)`` pair design_alpha._make_referee_fn expects, and no per-step CIF path is in
    scope here), so this scores directly from ``step.prediction``: ipTM / interface-pLDDT (over
    the token_index==1 design mask) and the chirality read from the predict wrapper's
    ``prediction.chirality_violation_frac``. pll/helix are imputed (None) for the anneal polish —
    the heavy CIF-derived terms already drove beam selection; anneal only hill-climbs ipTM."""
    from xenodesign.judges.panel import RefereeScore

    def _score(step) -> RefereeScore:
        pred = step.prediction
        ti = np.asarray(pred.token_index)
        mask = ti == 1
        iface_plddt = (float(pred.plddt[mask].mean()) if mask.any()
                       else float(pred.plddt.mean()))
        chir = float(getattr(pred, "chirality_violation_frac", 0.0))
        return RefereeScore(
            chirality_violation=chir, iptm=pred.iptm,
            interface_plddt=iface_plddt, pll=None, mirror_discrepancy=None,
            composition_violation=False, helix_fraction=None,
        )

    return _score


# ── CLI (superset of design_alpha's parser) ─────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="α (trimer D/L-ABLE) BEAM+ANNEAL design driver (ADR-018, WF-1 spec)")

    # ── new beam/anneal knobs ──
    p.add_argument("--beam_width", type=int, default=_DEFAULT_BEAM_WIDTH,
                   help="live states kept per cycle (B)")
    p.add_argument("--children_per_branch", type=int, default=_DEFAULT_CHILDREN_PER_BRANCH,
                   help="MultiCandidate top_k per parent (m); capped at --num_seqs")
    p.add_argument("--cycles", type=int, default=_DEFAULT_CYCLES,
                   help="beam expand/predict/prune cycles (C)")
    p.add_argument("--num_seqs", type=int, default=_DEFAULT_NUM_SEQS,
                   help="MultiCandidate oversample count (top_k <= num_seqs)")
    p.add_argument("--anneal_steps", type=int, default=_DEFAULT_ANNEAL_STEPS,
                   help="HalluLoop iterations per anneal seed")
    p.add_argument("--anneal_top_n", type=int, default=_DEFAULT_ANNEAL_TOP_N,
                   help="how many top-composite pool states to anneal")
    p.add_argument("--anneal_ref_start", type=int, default=_DEFAULT_ANNEAL_REF_START,
                   help="ref_time_steps anneal start (-> base ref_time_steps)")
    p.add_argument("--anneal_temp_start", type=float, default=_DEFAULT_ANNEAL_TEMP_START,
                   help="MPNN temperature anneal start (-> base 0.1)")
    p.add_argument("--prune_metric", default="composite", choices=["composite"],
                   help="pruning key (composite only — relative-binding gradient; never abs ipTM)")
    p.add_argument("--no_dedup", dest="dedup", action="store_false",
                   help="disable the global pre-predict sequence dedup (charges full predicts)")
    p.set_defaults(dedup=True)
    p.add_argument("--ref_time_steps", type=int, default=_DEFAULT_REF_TIME_STEPS,
                   help="base diffusion truncation depth (anneal floor)")

    # ── inherited design_alpha knobs (superset CLI) ──
    p.add_argument("--device", default=_DEFAULT_DEVICE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--no_pepmlm", action="store_true", help="skip the network PepMLM seed")
    p.add_argument("--seed_seq", default=None, help="explicit 21-res L seed (offline/repeat)")
    p.add_argument("--no_restraints", action="store_true",
                   help="disable the α pin-polarity restraint (revert to truncated_refine)")
    p.add_argument("--no_pll", action="store_true",
                   help="disable the ESM-2 pseudo-log-likelihood judge (impute the pll term)")
    p.add_argument("--esm_device", default=None,
                   help="device for the ESM-PLL judge (default cpu, to keep VRAM free)")
    p.add_argument("--restraint_file", default=None,
                   help="override the case-default pin with this .restraints CSV")
    p.add_argument("--target_fasta", default=None,
                   help="override the L-HLH target FASTA (record trimer_DL_ABLE_B)")
    p.add_argument("--objective", choices=("iptm", "mixed"), default="iptm",
                   help="selection objective: iptm (DEFAULT, reproducible) or mixed (parity-aware "
                        "mixed_objective panel — anneal score_fn + final re-selection)")
    p.add_argument("--periodicity_gate", action="store_true",
                   help="DESIGN-TIME register gate on the final pick: reject heptad-periodic "
                        "(register-UNachievable) designs via seq_periodicity (the '-dep' variant)")
    p.add_argument("--heptad_thresh", type=float, default=0.35,
                   help="lag-7 hydropathy-autocorr threshold for the periodicity gate (default 0.35)")
    p.add_argument("--smoke", action="store_true",
                   help="quick wiring smoke: B=2,m=2,C=2,anneal_steps=2 (13 predicts)")
    return p.parse_args(argv)


if __name__ == "__main__":
    import os
    args = _parse_args()
    if args.smoke:
        B, m_smoke, C, a_smoke, top_n_smoke = _smoke_knobs()
        beam_width, children_per_branch, cycles = B, m_smoke, C
        anneal_steps, anneal_top_n = a_smoke, top_n_smoke
        num_seqs = max(args.num_seqs, m_smoke)
    else:
        beam_width, children_per_branch, cycles = args.beam_width, args.children_per_branch, args.cycles
        anneal_steps, anneal_top_n = args.anneal_steps, args.anneal_top_n
        num_seqs = args.num_seqs

    out = Path(args.out_dir) if args.out_dir else Path(f"/home/tmp/xd_alpha_beam_{os.getpid()}")
    run_alpha_beam_design(
        beam_width=beam_width, children_per_branch=children_per_branch, cycles=cycles,
        num_seqs=num_seqs, anneal_steps=anneal_steps, anneal_top_n=anneal_top_n,
        anneal_ref_start=args.anneal_ref_start, anneal_temp_start=args.anneal_temp_start,
        ref_time_steps=args.ref_time_steps, prune_metric=args.prune_metric, dedup_on=args.dedup,
        device=args.device, seed=args.seed, out_dir=out, use_pepmlm=not args.no_pepmlm,
        seed_seq=args.seed_seq, restraints=not args.no_restraints, use_pll=not args.no_pll,
        esm_device=args.esm_device, restraint_file=args.restraint_file,
        target_fasta=args.target_fasta, objective=args.objective,
        periodicity_gate=args.periodicity_gate, heptad_thresh=args.heptad_thresh,
    )
    sys.exit(0)
