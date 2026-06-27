"""α binder class — the validated trimer D/L-ABLE design loop, as a BinderClass.

This module is the public CONTRACT for the α case: the :class:`Alpha` BinderClass adapter
(the 9 hooks the dispatcher calls) + the standalone ``run_alpha_design`` driver. The
restraint / seed / backend-selector / sequence-update / objective / referee / result-assembly
INTERNALS live in :mod:`xenodesign.classes._alpha_internals` (MOD-3 split) and are re-exported
here so every previously-importable name (``from xenodesign.classes.alpha import …`` and
``scripts.design_alpha``'s re-exports) keeps working byte-for-byte.

``scripts/design_alpha.py`` is a thin shim that re-exports every public name from here and keeps
its ``run_alpha_design`` / ``_parse_args`` / CLI.

Behaviour-preservation note (monkeypatch contract): several helpers in ``_alpha_internals`` read
their collaborators (``_ligandmpnn_design_fn`` / ``carbonara_design_fn`` / ``_make_base_backend`` /
``_cterm_gly_anchor`` / ``_best_cif_path`` / ``_all_atoms_from_chain`` / ``binder_seq_from_cif`` /
``make_alpha_seq_update_fn`` / ``build_alpha_restraint``) at CALL TIME through ``_self()``, which
returns THIS public module. So a test patching one of those names on this module is honoured even
though the body lives in the internals module. (MOD-1 moved the CIF/backend plumbing into the
package; MOD-2 removed the former ``scripts.design_alpha`` ``_shim()`` indirection; MOD-3 split
the internals out — the package never imports from ``scripts``.)
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

# Re-export every α internal so existing imports (``from xenodesign.classes.alpha import X`` and
# the ``scripts.design_alpha`` shim's re-exports) keep working, AND so the call-time ``_self().X``
# lookups in ``_alpha_internals`` resolve against THIS module (the monkeypatch surface).
from xenodesign.classes._alpha_internals import (  # noqa: F401
    _ALPHA_WEIGHTS,
    _COMP_MAX_ALA_GLY_FRAC,
    _COMP_MAX_HOMOPOLYMER_RUN,
    _COMP_MAX_SINGLE_AA_FRAC,
    _COMP_MIN_NORM_ENTROPY,
    _DEFAULT_DEVICE,
    _DEFAULT_N_ITERS,
    _DEFAULT_NUM_SEQS,
    _DEFAULT_REF_TIME_STEPS,
    _LoopBackendWrapper,
    _MixedBackend,
    _RUN_BINDER_CHAIN,
    _RUN_TARGET_CHAIN,
    _TARGET_RECORD,
    _all_atoms_from_chain,
    _assemble_alpha_result,
    _backbone_array_from_residues,
    _best_cif_path,
    _binder_helix_fraction,
    _chirality_violation_frac_from_cif,
    _cterm_gly_anchor,
    _ensure_cterm_glycine,
    _ligandmpnn_design_fn,
    _loop_score_fn,
    _make_base_backend,
    _make_referee_fn,
    _self,
    binder_seq_from_cif,
    build_alpha_restraint,
    build_alpha_seed,
    carbonara_design_fn,
    composition_violation,
    ipsae_objective_from_cif,
    make_alpha_seq_update_fn,
    make_ipsae_loop_score_fn,
    make_mixed_loop_score_fn,
    mixed_objective_from_cif,
)


# ── Main driver (the standalone α CLI delegates here) ───────────────────────────

def run_alpha_design(
    n_iters: int = _DEFAULT_N_ITERS,
    ref_time_steps: int = _DEFAULT_REF_TIME_STEPS,
    num_seqs: int = _DEFAULT_NUM_SEQS,
    device: str = _DEFAULT_DEVICE,
    seed: int = 42,
    out_dir=None,
    use_pepmlm: bool = True,
    seed_seq: str | None = None,
    chirality_gate: bool = False,
    esm_device: str | None = None,
    restraints: bool = True,
    use_pll: bool = True,
    restraint_file=None,
    target_fasta: str | None = None,
    backend: str = "ligandmpnn",
    objective: str = "iptm",
    periodicity_gate: bool = False,
    heptad_thresh: float = 0.35,
) -> dict:
    """Run the α design loop end-to-end and score the selected design vs the baseline.

    Returns a result dict (also written to out_dir/alpha_result.json) with the selected
    design, its interface metrics, the vs-baseline deltas, beats_baseline, and the full
    per-iter trajectory (chirality / ipTM / composite).

    Behaviour-preserving migration of the legacy ``scripts.design_alpha.run_alpha_design``:
    monkeypatch-sensitive collaborators (``make_alpha_seq_update_fn`` / ``_best_cif_path``) are
    resolved through THIS module at call time so the legacy regression test
    ``test_run_alpha_design_backend_string_not_shadowed_by_chai_object`` (which patches both here)
    is honoured.
    """
    from xenodesign.backends.chai_backend import ChaiBackend
    from xenodesign.benchmark.cases import get_case, ss_bias_config_for_case
    from xenodesign.io_spec import to_d_fasta
    from xenodesign.judges.panel import JudgePanel
    from xenodesign.loop import (
        HalluLoop, LoopState, chirality_gated_accept, compose_accept_fns,
        periodicity_gated_accept,
    )
    from xenodesign.seed import read_target_sequence, reflect_binder_in_complex_from_cif

    from xenodesign.backends.wrappers import _PredictBackendWrapper

    shim = _self()

    from xenodesign.config import resolve_device
    device = device or resolve_device()  # None -> XENO_DEVICE / cuda:0 if avail / mps / cpu

    case = get_case("alpha")
    if target_fasta is not None:
        import dataclasses
        case = dataclasses.replace(case, fasta_path=target_fasta)
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="xd_alpha_"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Restraint (TASK 3, #27): emit with the RUN's chains (binder=B, target=A) ──
    constraint_path = None
    if restraints:
        if restraint_file is not None:
            import shutil
            constraint_path = out_dir / "alpha.restraints"
            shutil.copyfile(restraint_file, constraint_path)
        else:
            constraint_path = build_alpha_restraint(case, out_dir)

    # ── ESM-PLL judge (TASK 4): lazy — only imported/instantiated when running ────
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
    print(f"\n{'='*78}\nXenoDesign1 — α (trimer D/L-ABLE) design loop\n{'='*78}")
    print(f"  target (chain A, L) : {len(target_seq)} aa")
    print(f"  seed   (chain B, D) : {seed_l_seq}  ({len(seed_l_seq)} aa)")
    print(f"  baseline-to-beat    : interface ipTM {case.baseline.interface_iptm} "
          f"(beat >0.50), ipAE {case.baseline.ipae} (beat <10), chirality "
          f"{case.baseline.chirality}")
    print(f"  device {device} | iters {n_iters} | ref_time_steps {ref_time_steps} | "
          f"num_seqs {num_seqs} | chirality_gate {chirality_gate}")
    print(f"  restraints {restraints} ({constraint_path}) | use_pll {use_pll}")
    print(f"  backend {backend} | objective {objective} | periodicity_gate {periodicity_gate} "
          f"(heptad_thresh {heptad_thresh})")
    print(f"  out_dir {out_dir}\n{'='*78}\n")

    # ── Backend + L-seed predict + double-flip ────────────────────────────────
    print("[1/4] Loading ChaiBackend (weights load once) ...")
    chai_backend = ChaiBackend(device=device, seed=seed)

    print("[2/4] L-seed predict + double-flip → D-correct seed coords ...")
    l_seed_entities = [
        target_entity,
        {"type": "protein", "name": "binder", "sequence": seed_l_seq, "chirality": "L"},
    ]
    l_seed_pred = chai_backend.predict(l_seed_entities, out_dir / "p0_l_seed",
                                  num_diffn_timesteps=200, constraint_path=constraint_path)
    l_seed_cif = shim._best_cif_path(out_dir / "p0_l_seed")
    d_seed_coords = reflect_binder_in_complex_from_cif(l_seed_cif, binder_chain="B", axis=0)
    print(f"    L-seed ipTM={l_seed_pred.iptm:.4f} ; D-seed {d_seed_coords.shape[0]} atoms")

    # ── Loop ──────────────────────────────────────────────────────────────────
    if restraints:
        print(f"[3/4] HalluLoop × {n_iters} (PREDICT mode — restrained, constraints honoured) ...")
        wrapper = _PredictBackendWrapper(chai_backend, target_entity,
                                         constraint_path=constraint_path)
        refine_fn = wrapper
    else:
        print(f"[3/4] HalluLoop × {n_iters} (truncated_refine {ref_time_steps} steps/iter) ...")
        wrapper = _LoopBackendWrapper(chai_backend, target_entity)
        refine_fn = None
    seq_update_fn = shim.make_alpha_seq_update_fn(wrapper, num_seqs=num_seqs, backend=backend)
    if objective == "mixed":
        loop_score_fn = make_mixed_loop_score_fn(wrapper)
    elif objective == "ipsae":
        loop_score_fn = make_ipsae_loop_score_fn(wrapper)
    else:
        loop_score_fn = _loop_score_fn
    loop = HalluLoop(backend=wrapper, sequence_update_fn=seq_update_fn,
                     score_fn=loop_score_fn, refine_fn=refine_fn)
    loop_dir = out_dir / "loop"

    referee_fn = _make_referee_fn(loop_dir, esm_judge=esm_judge)   # PLL + composition floor
    chir_gate = None
    if chirality_gate:
        def _gate_score(step):
            chir = _chirality_violation_frac_from_cif(shim._best_cif_path(wrapper.last_out_dir))
            from xenodesign.judges.panel import RefereeScore
            return RefereeScore(chirality_violation=chir, iptm=step.prediction.iptm,
                                interface_plddt=0.0, pll=None, mirror_discrepancy=None)
        chir_gate = chirality_gated_accept(JudgePanel(score_fn=_gate_score), max_violation=0.1)
    period_gate = periodicity_gated_accept(heptad_thresh=heptad_thresh) if periodicity_gate else None
    accept_fn = compose_accept_fns(chir_gate, period_gate)

    init_state = LoopState(d_fasta=to_d_fasta(seed_l_seq), coords=d_seed_coords)
    history = loop.run(init=init_state, iterations=n_iters, ref_time_steps=ref_time_steps,
                       out_dir=loop_dir, accept_fn=accept_fn)

    # ── Per-iter trajectory + panel selection ─────────────────────────────────
    referee_scores = [referee_fn(step, i) for i, step in enumerate(history)]
    panel = JudgePanel(weights=_ALPHA_WEIGHTS, score_fn=lambda step: None,
                       ss_bias=ss_bias_config_for_case(case))
    panel_result = panel.combine(referee_scores)

    wall = time.time() - t0
    return _assemble_alpha_result(
        history, referee_scores, panel_result, case,
        loop_dir=loop_dir, out_dir=out_dir, l_seed_iptm=l_seed_pred.iptm,
        n_iters=n_iters, num_seqs=num_seqs, ref_time_steps=ref_time_steps,
        chirality_gate=chirality_gate, objective=objective, periodicity_gate=periodicity_gate,
        heptad_thresh=heptad_thresh, restraints=restraints, constraint_path=constraint_path,
        use_pll=use_pll, backend=backend, wall_time_s=wall,
    )


# ── BinderClass adapter ────────────────────────────────────────────────────────

class Alpha:
    """α binder class implementing :class:`xenodesign.classes.base.BinderClass`.

    Each hook delegates to the migrated module-level helpers above (the SAME callables the
    validated ``run_alpha_design`` driver uses), so the dispatcher reproduces the α loop's
    behaviour. ``case_id == 'alpha'``.
    """

    case_id = "alpha"

    def seed(self, cfg, target_seq) -> "SeedSpec":
        """FROM-SCRATCH unified PepMLM seed conditioned on the L-HLH target.

        Routes through the ONE ``seed.unified_seed`` path (same generator every class uses) at
        ``resolve_binder_length(cfg)`` (default 21 — the validated α length, a re-baseline-safe
        DEFAULT, overridable via --binder_length). The seed NEVER inherits the reference binder's
        sequence or length. The C-terminal Gly is a chai tokenization anchor (≥1 canonical residue
        per all-D chain), NOT binder-derived scaffold, so it is preserved."""
        from xenodesign.classes.base import SeedSpec
        from xenodesign.config import resolve_binder_length
        from xenodesign.seed import make_configured_generator, unified_seed

        length = resolve_binder_length(cfg)
        gen = make_configured_generator(cfg)
        result = unified_seed(gen, target_seq=target_seq or "", length=length, reverse=True)
        return SeedSpec(one_letter=_ensure_cterm_glycine(result.one_letter.upper()))

    def ss_bias(self, cfg, case):
        from xenodesign.benchmark.cases import ss_bias_config_for_case
        return ss_bias_config_for_case(case)

    def restraints(self, cfg, case, out_dir, target_ctx):
        return build_alpha_restraint(case, out_dir) if cfg.restraints_on else None

    def closure(self, cfg, seed_spec) -> list:
        return []

    def seq_update(self, cfg, wrapper, seed_spec, roles=None):
        return make_alpha_seq_update_fn(wrapper, num_seqs=cfg.loop.num_seqs,
                                        backend=cfg.loop.backend, roles=roles)

    def accept_fns(self, cfg):
        from xenodesign.loop import compose_accept_fns, periodicity_gated_accept
        period = (periodicity_gated_accept(heptad_thresh=cfg.gates.heptad_thresh)
                  if cfg.gates.periodicity else None)
        return compose_accept_fns(period)   # chirality gate wired in dispatch when cfg.gates.chirality

    def objective(self, cfg, wrapper):
        if cfg.objective == "mixed":
            return make_mixed_loop_score_fn(wrapper)
        if cfg.objective == "ipsae":
            return make_ipsae_loop_score_fn(wrapper)
        return _loop_score_fn

    def referee(self, cfg, loop_dir, esm_judge, roles=None):
        return _make_referee_fn(loop_dir, esm_judge=esm_judge, roles=roles)

    def report(self, cfg, history, panel_result, case, out_dir,
               *, l_seed_iptm: float = 0.0, wall_time_s: float = 0.0) -> dict:
        referee_scores = list(getattr(panel_result, "raw_scores", []))
        return _assemble_alpha_result(
            history, referee_scores, panel_result, case,
            loop_dir=Path(out_dir) / "loop", out_dir=out_dir,
            l_seed_iptm=l_seed_iptm, n_iters=cfg.loop.iters, num_seqs=cfg.loop.num_seqs,
            ref_time_steps=cfg.loop.ref_time_steps, chirality_gate=cfg.gates.chirality,
            objective=cfg.objective, periodicity_gate=cfg.gates.periodicity,
            heptad_thresh=cfg.gates.heptad_thresh, restraints=cfg.restraints_on,
            constraint_path=None, use_pll=cfg.use_pll, backend=cfg.loop.backend,
            wall_time_s=wall_time_s,
        )
