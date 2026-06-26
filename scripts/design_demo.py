"""End-to-end HalluDesign-on-Chai-1 D-peptide design demo.

Demonstrates the full pipeline for a single D-peptide:L-target design case:

  1. Double-flip D-correct seeding (reflect_binder_in_complex_from_cif)
  2. HalluLoop.run() — truncated_refine (50 steps) + context-aware LigandMPNN
  3. HalluLoop.select_by_panel() — adversarial JudgePanel (chirality veto + composite)
  4. Trajectory report: per-iter ipTM / chirality / ESM-PLL / composite + vetoed flag
  5. Selected design: D-CCD sequence, scores, narrative

Reuses wiring from tests/gpu/test_loop_end_to_end_gpu.py — no reinvented plumbing.
Weights load ONCE (ChaiBackend) and ESM-2 loads once; both are reused across iterations.

Usage (inside the gradio_design Docker container with PYTHONPATH=/work):
    CUDA_VISIBLE_DEVICES=0 python scripts/design_demo.py
    CUDA_VISIBLE_DEVICES=0 python scripts/design_demo.py --device cuda:0 --iters 7
    CUDA_VISIBLE_DEVICES=0 python scripts/design_demo.py --help
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Canonical demo case (same as test_loop_end_to_end_gpu.py) ─────────────────
_DEFAULT_TARGET_SEQ = "GSHMKVLITGGAGFIGSHLVDRL"   # 23-residue L-target
_DEFAULT_BINDER_SEQ = "ACDEFGHIK"                  # 9-residue starting D-binder (has G)
_DEFAULT_N_ITERS = 7
_DEFAULT_REF_TIME_STEPS = 50
_DEFAULT_DEVICE = None  # unset -> resolve_device() (XENO_DEVICE / cuda:0 if avail / mps / cpu)

_TARGET_ENTITY = {
    "type": "protein", "name": "target",
    "sequence": _DEFAULT_TARGET_SEQ, "chirality": "L",
}


# ── Helpers reused from test_loop_end_to_end_gpu.py ───────────────────────────

def _best_cif_path(out_dir: Path) -> Path:
    import re
    score_files = sorted(out_dir.rglob("scores.model_idx_*.npz"))
    if score_files:
        best_idx, best_agg = 0, -np.inf
        for f in score_files:
            d = np.load(f)
            agg = float(np.asarray(d["aggregate_score"]).reshape(-1)[0])
            idx = int(re.search(r"idx_(\d+)", f.name).group(1))
            if agg > best_agg:
                best_agg = agg
                best_idx = idx
        cif = next(out_dir.rglob(f"pred.model_idx_{best_idx}.cif"), None)
        if cif is not None:
            return cif
    return sorted(out_dir.rglob("*.cif"))[0]


def _all_atoms_from_chain(cif_path: Path, chain_name: str):
    import gemmi
    structure = gemmi.read_structure(str(cif_path))
    coords_list, elements_list = [], []
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                for atom in res:
                    coords_list.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    elements_list.append(atom.element.name)
        break
    if not coords_list:
        return np.zeros((0, 3), dtype=np.float32), []
    return np.array(coords_list, dtype=np.float32), elements_list


def _backbone_array_from_residues(residues: list[dict]) -> np.ndarray:
    arr = np.zeros((len(residues), 4, 3), dtype=np.float32)
    for i, res in enumerate(residues):
        arr[i, 0] = res["N"]
        arr[i, 1] = res["CA"]
        arr[i, 2] = res["C"]
        arr[i, 3] = res.get("CB", res["CA"])
    return arr


def _chirality_violation_frac_from_cif(cif_path: Path) -> float:
    from xenodesign.eval.gate_tier0a import backbone_by_residue_from_cif
    from xenodesign.chirality import is_chirality_violation

    residues = backbone_by_residue_from_cif(cif_path, "B")
    if not residues:
        residues = backbone_by_residue_from_cif(cif_path, "b")
    if not residues:
        return 0.0

    total = violations = 0
    for res in residues:
        if "CB" not in res:
            continue
        total += 1
        if is_chirality_violation(res["N"], res["CA"], res["C"], res["CB"], "D"):
            violations += 1
    return violations / total if total > 0 else 0.0


# ── Backend wrappers (same as test_loop_end_to_end_gpu.py) ────────────────────

def _as_target_list(target_entity):
    """Normalise the target arg to a chain-ordered entity list (the binder is appended LAST).

    Accepts a single dict (legacy single-target callers — α), a list of dicts (multi-chain /
    ligand+metal targets), or ``None``/empty (no-target free-peptide mode). Keeping a single
    dict == ``[dict]`` keeps the α path byte-identical (binder still becomes chain B)."""
    if target_entity is None:
        return []
    if isinstance(target_entity, dict):
        return [target_entity]
    return list(target_entity)


def _build_entities(target_entities, binder_l_seq):
    """[*target chains, binder] — binder is ALWAYS the LAST chain (Chai labels A,B,C… in order),
    so multi-chain protein / DNA-RNA / ligand+metal targets all order correctly and the binder's
    chain letter is ``chr(ord('A') + len(target_entities))``."""
    return [
        *target_entities,
        {"type": "protein", "name": "binder",
         "sequence": binder_l_seq, "chirality": "D"},
    ]


class _LoopBackendWrapper:
    """Translates LoopState API → ChaiBackend.truncated_refine."""

    def __init__(self, chai_backend, target_entity=None, *, msa_directory=None):
        self._backend = chai_backend
        self._target_entities = _as_target_list(target_entity)
        self._msa_directory = msa_directory
        self.last_out_dir: Path | None = None

    def truncated_refine(self, state, ref_time_steps, out_dir):
        from xenodesign.io_spec import d_fasta_to_one_letter

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        binder_l_seq = d_fasta_to_one_letter(state.d_fasta)
        entities = _build_entities(self._target_entities, binder_l_seq)

        self.last_out_dir = out_dir
        return self._backend.truncated_refine(
            structure={"entities": entities, "coords": state.coords},
            ref_time_steps=ref_time_steps,
            out_dir=out_dir,
        )


class _PredictBackendWrapper:
    """Per-iteration structure step that runs a FULL ChaiBackend.predict (not truncated).

    Drop-in ``refine_fn`` for ``HalluLoop`` (signature ``(state, ref_time_steps, out_dir) ->
    Prediction``). Unlike ``_LoopBackendWrapper.truncated_refine``, this re-folds the binder
    sequence from scratch (200-step predict) each iteration: slower, but (a) it preserves
    D-chirality better and (b) it supports ``constraint_path`` — the vendored truncated sampler
    does NOT (see chai_truncated.py TODO #27). Used by the RESTRAINED α run so the pin-polarity
    restraint is honoured on every iteration. ``ref_time_steps`` is ignored (predict is full).

    Like ``_LoopBackendWrapper`` it builds entities=[target, binder] so Chai labels
    TARGET=chain A, BINDER=chain B, and records ``last_out_dir`` for the sequence-update step.
    The binder is emitted with chirality 'D' (same as the truncated path).
    """

    def __init__(self, chai_backend, target_entity=None,
                 constraint_path: "Path | str | None" = None,
                 num_diffn_timesteps: int = 200, *, msa_directory=None):
        self._backend = chai_backend
        self._target_entities = _as_target_list(target_entity)
        self._constraint_path = Path(constraint_path) if constraint_path is not None else None
        self._num_diffn_timesteps = num_diffn_timesteps
        self._msa_directory = msa_directory
        self.last_out_dir: Path | None = None

    def __call__(self, state, ref_time_steps, out_dir):
        from xenodesign.io_spec import d_fasta_to_one_letter

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        binder_l_seq = d_fasta_to_one_letter(state.d_fasta)
        entities = _build_entities(self._target_entities, binder_l_seq)

        self.last_out_dir = out_dir
        return self._backend.predict(
            entities,
            out_dir,
            num_diffn_timesteps=self._num_diffn_timesteps,
            constraint_path=self._constraint_path,
            msa_directory=self._msa_directory,
        )


def _make_sequence_update_fn(wrapper: _LoopBackendWrapper):
    from xenodesign.sequence_update import SequenceUpdater, _ligandmpnn_design_fn
    from xenodesign.eval.gate_tier0a import backbone_by_residue_from_cif

    updater = SequenceUpdater(design_fn=_ligandmpnn_design_fn)

    def _seq_update_fn(prediction) -> str:
        out_dir = wrapper.last_out_dir
        if out_dir is None:
            raise RuntimeError("_seq_update_fn called before any structure step")

        cif_path = _best_cif_path(out_dir)
        print(f"    [seq_update] CIF: {cif_path.name}")

        binder_residues = backbone_by_residue_from_cif(cif_path, "B")
        if not binder_residues:
            binder_residues = backbone_by_residue_from_cif(cif_path, "b")
        if not binder_residues:
            raise RuntimeError(f"Cannot extract binder chain from {cif_path}")

        design_backbone = _backbone_array_from_residues(binder_residues)

        ctx_coords, ctx_elements = _all_atoms_from_chain(cif_path, "A")
        if ctx_coords.shape[0] == 0:
            ctx_coords, ctx_elements = _all_atoms_from_chain(cif_path, "a")

        n_binder = design_backbone.shape[0]
        d_codes = ["DAL"] * n_binder

        result = updater.update(
            design_backbone=design_backbone,
            design_codes=d_codes,
            context_coords=ctx_coords,
            context_elements=ctx_elements,
        )
        one_letter = result.one_letter

        # Chai constraint: ≥1 canonical residue per chain
        if "G" not in one_letter:
            gly_pos = len(one_letter) // 2
            one_letter = one_letter[:gly_pos] + "G" + one_letter[gly_pos + 1:]
            print(f"    [seq_update] forced G at pos {gly_pos}: {one_letter}")

        print(f"    [seq_update] designed: {one_letter}")
        return one_letter

    return _seq_update_fn


def _score_fn(prediction) -> float:
    from xenodesign.scorer import design_score

    ti = np.asarray(prediction.token_index)
    binder_mask = ti == 1
    if binder_mask.any():
        interface_plddt = float(prediction.plddt[binder_mask].mean())
    else:
        interface_plddt = float(prediction.plddt.mean())

    return design_score(
        iptm=prediction.iptm,
        interface_plddt=interface_plddt,
        chirality_violation_frac=0.0,
    )


# ── Main demo function (importable by the GPU test) ───────────────────────────

def run_design_demo(
    target_seq: str = _DEFAULT_TARGET_SEQ,
    binder_seq: str = _DEFAULT_BINDER_SEQ,
    n_iters: int = _DEFAULT_N_ITERS,
    ref_time_steps: int = _DEFAULT_REF_TIME_STEPS,
    device: str = _DEFAULT_DEVICE,
    out_dir: Path | str | None = None,
    seed: int = 42,
    esm_device: str | None = None,
) -> dict:
    """Run the full pipeline; return a result dict with selected design + trajectory.

    Returns
    -------
    dict with keys:
        selected_d_fasta : str   — D-CCD sequence of the panel-selected design
        selected_iptm    : float
        selected_chirality : float
        selected_pll     : float
        selected_composite : float
        trajectory       : list[dict] — per-iter {iter, d_fasta, iptm, chirality, pll, composite, vetoed}
        naive_best_idx   : int   — index (0-based) of HalluLoop.best() selection
        panel_best_idx   : int   — index (0-based) of panel selection
        wall_time_s      : float
        out_dir          : Path
    """
    import os
    import tempfile

    from xenodesign.backends.chai_backend import ChaiBackend
    from xenodesign.config import resolve_device
    from xenodesign.io_spec import to_d_fasta
    from xenodesign.judges.panel import JudgePanel, RefereeScore
    from xenodesign.judges.plm_judge import ESMPseudoLogLikelihood
    from xenodesign.loop import HalluLoop, LoopState
    from xenodesign.seed import reflect_binder_in_complex_from_cif

    device = device or resolve_device()  # None -> XENO_DEVICE / cuda:0 if avail / mps / cpu

    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="xd_demo_"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_entity = {
        "type": "protein", "name": "target",
        "sequence": target_seq, "chirality": "L",
    }

    t0 = time.time()

    # ── Step 1: Load ChaiBackend (weights load once here) ─────────────────────
    print(f"\n{'='*70}")
    print(f"XenoDesign1 — End-to-End Design Demo")
    print(f"  Target : {target_seq} ({len(target_seq)} aa, L)")
    print(f"  Binder : {binder_seq} ({len(binder_seq)} aa, D)")
    print(f"  Device : {device}  |  Iterations: {n_iters}  |  Seed: {seed}")
    print(f"{'='*70}\n")

    print("[1/5] Loading ChaiBackend (weights load once) ...")
    backend = ChaiBackend(device=device, seed=seed)

    # ── Step 2: L-seed predict (chai in-manifold for L → clean chirality) ─────
    print("\n[2/5] L-seed prediction (double-flip seeding Step 1) ...")
    l_seed_entities = [
        target_entity,
        {"type": "protein", "name": "binder",
         "sequence": binder_seq, "chirality": "L"},
    ]
    l_seed_dir = out_dir / "p0_l_seed"
    l_seed_pred = backend.predict(
        l_seed_entities,
        l_seed_dir,
        num_diffn_timesteps=200,
    )
    initial_iptm = l_seed_pred.iptm
    print(f"    L-seed ipTM = {initial_iptm:.4f}")

    # ── Step 3: Double-flip → D-correct seed coords ───────────────────────────
    print("\n[3/5] Double-flip: reflect binder chain → D-correct seed ...")
    l_seed_cif = _best_cif_path(l_seed_dir)
    d_seed_coords = reflect_binder_in_complex_from_cif(
        l_seed_cif, binder_chain="B", axis=0
    )
    print(f"    D-seed: {d_seed_coords.shape[0]} atoms, binder chain B reflected (axis=x)")

    # ── Step 4: Wire HalluLoop + run ──────────────────────────────────────────
    print(f"\n[4/5] Running HalluLoop ({n_iters} iters, {ref_time_steps} refine steps/iter) ...")
    wrapper = _LoopBackendWrapper(backend, target_entity)
    seq_update_fn = _make_sequence_update_fn(wrapper)

    loop = HalluLoop(
        backend=wrapper,
        sequence_update_fn=seq_update_fn,
        score_fn=_score_fn,
    )

    init_state = LoopState(
        d_fasta=to_d_fasta(binder_seq),
        coords=d_seed_coords,
    )
    loop_dir = out_dir / "loop"
    history = loop.run(
        init=init_state,
        iterations=n_iters,
        ref_time_steps=ref_time_steps,
        out_dir=loop_dir,
    )

    # ── Step 5: Collect chirality + PLL per iter ──────────────────────────────
    print(f"\n[5/5] Adversarial panel: ESM-2 PLL + chirality + composite scoring ...")
    esm_judge = ESMPseudoLogLikelihood(
        device=esm_device or device,
    )

    per_iter: list[dict] = []
    for i, step in enumerate(history):
        iter_dir = loop_dir / f"iter_{i:03d}"
        try:
            cif_path = _best_cif_path(iter_dir)
            chir = _chirality_violation_frac_from_cif(cif_path)
        except Exception:
            chir = 0.0

        # PLL on the L-equivalent (ESM-2 is chirality-agnostic; we score sequence naturalness)
        from xenodesign.io_spec import d_fasta_to_one_letter
        l_seq = d_fasta_to_one_letter(step.state.d_fasta)
        try:
            pll = esm_judge(l_seq)
        except Exception as exc:
            print(f"    [PLL] iter {i}: ESM error: {exc} — using None")
            pll = None

        per_iter.append({
            "iter": i,
            "d_fasta": step.state.d_fasta,
            "l_seq": l_seq,
            "iptm": step.prediction.iptm,
            "chirality": chir,
            "pll": pll,
            "score": step.score,
        })
        pll_str_inline = f"{pll:.3f}" if pll is not None else "N/A"
        print(f"    iter {i+1:d}: ipTM={step.prediction.iptm:.4f}  chir={chir:.3f}  "
              f"PLL={pll_str_inline}  seq={l_seq}")

    # ── Panel: build RefereeScore list + select ───────────────────────────────
    ti = np.asarray(history[0].prediction.token_index)
    binder_mask = ti == 1

    def _referee_score_fn(step):
        # Find index in history
        idx = history.index(step)
        pi = per_iter[idx]
        # interface_plddt from binder chain
        ti_ = np.asarray(step.prediction.token_index)
        mask_ = ti_ == 1
        iface_plddt = float(step.prediction.plddt[mask_].mean()) if mask_.any() else float(step.prediction.plddt.mean())
        return RefereeScore(
            chirality_violation=pi["chirality"],
            iptm=step.prediction.iptm,
            interface_plddt=iface_plddt,
            pll=pi["pll"],
            mirror_discrepancy=None,
        )

    panel = JudgePanel(score_fn=_referee_score_fn)
    referee_scores = [_referee_score_fn(step) for step in history]
    panel_result = panel.combine(referee_scores)

    # Enrich per_iter with composite + vetoed
    for i, pi in enumerate(per_iter):
        pi["composite"] = panel_result.composite_scores[i]
        pi["vetoed"] = panel_result.vetoed[i]

    # ── Naive best vs panel best ───────────────────────────────────────────────
    naive_best_step = HalluLoop.best(history)
    naive_best_idx = history.index(naive_best_step)
    panel_best_idx = panel_result.selected_idx
    panel_best_step = history[panel_best_idx]

    wall_time = time.time() - t0

    # ── Print trajectory table ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"{'iter':>4}  {'sequence (L)':>16}  {'ipTM':>6}  {'chir':>6}  {'PLL':>7}  {'composite':>9}  {'vetoed':>6}  {'select':>8}")
    print(f"{'─'*90}")
    for pi in per_iter:
        naive_marker = "<-naive" if pi["iter"] == naive_best_idx else ""
        panel_marker = "<-PANEL" if pi["iter"] == panel_best_idx else ""
        select_marker = panel_marker or naive_marker
        pll_str = f"{pi['pll']:7.3f}" if pi['pll'] is not None else f"{'N/A':>7}"
        print(f"  {pi['iter']+1:2d}  {pi['l_seq']:>16}  {pi['iptm']:6.4f}  "
              f"{pi['chirality']:6.3f}  {pll_str}  {pi['composite']:9.4f}  "
              f"{'YES' if pi['vetoed'] else 'no':>6}  {select_marker}")
    print(f"{'='*90}")

    # ── Selected design summary ───────────────────────────────────────────────
    sel_pi = per_iter[panel_best_idx]
    naive_pll = per_iter[naive_best_idx]["pll"]
    naive_pll_str = f"{naive_pll:.3f}" if naive_pll is not None else "N/A"
    sel_pll_str = f"{sel_pi['pll']:.3f}" if sel_pi["pll"] is not None else "N/A"
    print(f"\nNaive best()   → iter {naive_best_idx+1}  ipTM={per_iter[naive_best_idx]['iptm']:.4f}  "
          f"chir={per_iter[naive_best_idx]['chirality']:.3f}  PLL={naive_pll_str}")
    print(f"Panel selected → iter {panel_best_idx+1}  ipTM={sel_pi['iptm']:.4f}  "
          f"chir={sel_pi['chirality']:.3f}  PLL={sel_pll_str}  "
          f"composite={sel_pi['composite']:.4f}")
    print(f"\nSELECTED DESIGN (D-CCD): {sel_pi['d_fasta']}")
    print(f"SELECTED DESIGN (L-seq): {sel_pi['l_seq']}")
    if panel_result.fallback_used:
        print("\nWARNING: all steps were chirality-vetoed — panel fell back to naive best()")
    print(f"\nWall-clock: {wall_time/60:.1f} min   GPU: {device}   Out: {out_dir}")

    return {
        "selected_d_fasta": sel_pi["d_fasta"],
        "selected_l_seq": sel_pi["l_seq"],
        "selected_iptm": sel_pi["iptm"],
        "selected_chirality": sel_pi["chirality"],
        "selected_pll": sel_pi["pll"],
        "selected_composite": sel_pi["composite"],
        "initial_iptm": initial_iptm,
        "trajectory": per_iter,
        "naive_best_idx": naive_best_idx,
        "panel_best_idx": panel_best_idx,
        "panel_fallback_used": panel_result.fallback_used,
        "panel_result": panel_result,
        "wall_time_s": wall_time,
        "out_dir": out_dir,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="XenoDesign1 end-to-end D-peptide design demo "
                    "(double-flip seed + adversarial panel)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", default=_DEFAULT_TARGET_SEQ,
                   help="L-target sequence (one-letter)")
    p.add_argument("--binder", default=_DEFAULT_BINDER_SEQ,
                   help="Initial D-binder sequence (one-letter L codes)")
    p.add_argument("--iters", type=int, default=_DEFAULT_N_ITERS,
                   help="Number of HalluLoop iterations")
    p.add_argument("--ref_time_steps", type=int, default=_DEFAULT_REF_TIME_STEPS,
                   help="Truncated-refine diffusion steps per iteration")
    p.add_argument("--device", default=_DEFAULT_DEVICE,
                   help="CUDA device (e.g. cuda:0 or cuda:1)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--out_dir", default=None,
                   help="Output root dir (default: /home/tmp/xd_demo_<pid>)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"/home/tmp/xd_demo_{os.getpid()}")
    import os
    result = run_design_demo(
        target_seq=args.target,
        binder_seq=args.binder,
        n_iters=args.iters,
        ref_time_steps=args.ref_time_steps,
        device=args.device,
        out_dir=out_dir,
        seed=args.seed,
    )
    sys.exit(0)
