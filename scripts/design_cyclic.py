"""CYCLIC 6UFA case — single-chain Zn-macrocycle design + geometry-RECALL driver (task #9).

The 6UFA case (benchmark.cases._CYCLIC) is the metal/holo geometry-recovery test:
a SINGLE-chain mixed-chirality peptide that chelates one Zn(II) through a set of His
imidazole nitrogens, with NO target / NO interface. Success = RECALL of the deposited
0.77-A macrocycle (backbone heavy-atom RMSD <= 1 A) + correct Zn-N coordination geometry.
Because 6UFA is a 2019 deposit, Chai has very likely seen it in training, so this is a
RECALL (memorization) probe, not a novelty test — the deposit is the ground truth to recover.

This driver composes the EXISTING primitives rather than reinventing them:
  - benchmark.seeding.build_seed_for_case("cyclic")  -> seed.insert_fixed_chirality places the
    coordinating His (mixed L/D) at the case's his_resnums (the unconditioned cyclic path; PepMLM
    cannot condition on a metal). The per-position D/L map is _CYCLIC_HIS_CHIRALITY.
  - benchmark.restraints.build_for_case(case)        -> metal_coordination_rows: one inter-chain
    His<->Zn CONTACT per coordinating His (His identity 'H', Zn token 'X'/UNK). Works post the #1
    restraint-column fix (*_angstrom columns). The Zn enters as a SEPARATE ligand chain so these
    His<->Zn restraints are inter-chain and valid for Chai (intra-chain bonds are unsupported).
  - geometry.kabsch_rmsd                              -> backbone heavy-atom RMSD to the deposit.

NEW here (kept local because the shared modules do not cover them):
  - mixed_chirality_fasta : per-position L-vs-D FASTA (io_spec.to_d_fasta is ALL-D; this case is
    MIXED, so only the D-marked positions get the (DXX) D-CCD block, L/unmarked stay bare).
  - build_cyclic_input_fasta : appends the Zn as a chai `>ligand|name=zn` SMILES entity (the
    HETATM/metal context). io_spec.build_fasta only emits protein chains — see the BLOCKER note.
  - backbone_rmsd_to_deposit / zn_coordination_geometry : the RECALL + Zn-N geometry scorers.


6UFA DEPOSIT REALITY (verified on the RCSB mmCIF; load-bearing for interpreting results)
----------------------------------------------------------------------------------------
The 6UFA deposit is a SINGLE chain A of 24 residues = the 12-mer repeat unit TWICE:
    KL(DGN)(DGL)(AIB)H(DLY)(DLE)QE(AIB)(DHI)   (x2)
Per 12-mer repeat the residues are:
    1 K(L) 2 L(L) 3 DGN(D-Gln) 4 DGL(D-Glu) 5 AIB(achiral) 6 HIS(L-His)
    7 DLY(D-Lys) 8 DLE(D-Leu) 9 GLN(L) 10 GLU(L) 11 AIB(achiral) 12 DHI(D-His)
One Zn(II) is chelated tetrahedrally by FOUR His ND1 atoms — residues 6, 12, 18, 24 — i.e.
the L-His (pos 6) and D-His (pos 12) of EACH repeat: [Zn(L-His)2(D-His)2], Zn-N ~2.02 A.

REGISTRY RE-SYNCED TO THE DEPOSIT (P2a, 2026-06-15): the case restraint his_resnums and seeding
_CYCLIC_HIS_CHIRALITY are now (6, 12) / {6:'L', 12:'D'} — the deposit's true coordinating His
within the modeled 12-mer (L-His@6, D-His@12). The earlier registry encoded a STYLIZED layout
(3,6,8,11); that discrepancy is RESOLVED. The full tetrahedral [Zn(His)4] site uses 6/12/18/24
across both 12-mer repeats; this single-12-mer model therefore carries 2 of the 4 coordinating
His (one [Zn(L-His)(D-His)] pair) — a sequence-faithful 24-mer run is a follow-up. The
``test_design_cyclic.py`` guard now asserts the corrected positions + seeding/restraint
self-consistency. Also note AIB (achiral 2-aminoisobutyrate, pos 5/11) and the
D-Gln/D-Glu/D-Lys/D-Leu backbone are non-canonical; the phase-1 seed approximates them with
canonical residues (a known approximation — see the "SEQUENCE APPROXIMATION" note below).


CLOSURE: head-to-tail COVALENT bond (#23) is now AVAILABLE (opt-in via --closure / closure=True)
-----------------------------------------------------------------------------------------------
chai 0.6.1 DOES support intra-chain covalent bonds: a COVALENT row in the constraint table is
wired as a real N-to-C backbone bond via bond_utils.get_atom_covalent_bond_pairs_from_constraints
(NOT through the contact/pocket restraint path, which skips covalent rows). So head-to-tail
macrocyclization is a real closure bond — correcting the earlier "Chai not trained on intra-chain
bonds" assumption. With closure OFF (default) the driver keeps the phase-1 LINEAR predict and
relies on the His<->Zn restraints + the deposit's intrinsic ring geometry for EMERGENT closure
(the N/C-terminus distance is REPORTED as a proxy). CAVEAT: chai matches a COVALENT row's
one-letter residue code against the token name. CONFIRMED ON GPU (2026-06-15): the C-terminal
D-His (token 'DHI') does NOT match the L 'HIS' code (rc.restype_1to3 is L-only), so the closure
COVALENT bond is REJECTED for this D-terminus (bond_utils assert: left_residue_idx empty). The
default run therefore stays LINEAR + emergent-closure; --closure is retained for the L-terminus
case / future chai versions. The CyclicBoltz1 relative-position-encoding offset is a separate,
heavier alternative that would not hit the name-match wall.


SEQUENCE APPROXIMATION (phase-1): the design seed is a canonical-residue stand-in for the
deposit's AIB / D-Gln / D-Glu / D-Lys / D-Leu macrocycle. We pin only the coordinating His
handedness (the chemistry that defines the site); the rest is designed/approximated. RMSD-to-
deposit is therefore a BACKBONE recall metric over the matched-length 12-mer, not a per-atom
sequence-identical overlay. A sequence-faithful run (exact AIB/D-CCD seed) is a follow-up.


GPU RUN COMMAND (on ifrit, inside the chai 0.6.1 container; see RUNBOOK.local.md for the tag):
    PYTHONPATH=/work python scripts/design_cyclic.py --device cuda:0           # full recall run
    PYTHONPATH=/work python scripts/design_cyclic.py --smoke                   # 1-predict wiring
    PYTHONPATH=/work python scripts/design_cyclic.py --deposit_cif /tmp/6UFA.cif  # score vs deposit
The deposit CIF (RCSB 6UFA) is needed for the RMSD-recall metric; without it the run still
predicts + scores Zn-N geometry and the closure proxy, and reports RMSD as unavailable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# T7: the cyclic-class logic was migrated to xenodesign/classes/cyclic.py. This script is now
# a thin re-export shim — the seed / mixed-chirality FASTA / Zn-ligand FASTA / metal-coordination
# restraint / closure / RMSD-recall / Zn-N geometry helpers (and the CIF parsers) all live there,
# so tests/test_design_cyclic.py keeps importing the same names. The GPU driver
# run_cyclic_design + the CLI stay HERE (they compose the migrated helpers).
from xenodesign.classes.cyclic import (  # noqa: F401
    CYCLIC_HIS_CHIRALITY,
    ZN_SMILES,
    _DEFAULT_DEVICE,
    _ZN_N_CUTOFF,
    backbone_heavy_atoms_from_cif,
    backbone_rmsd_to_deposit,
    build_closure_row,
    build_cyclic_input_fasta,
    build_cyclic_restraint_rows,
    build_cyclic_seed,
    mixed_chirality_fasta,
    termini_distance_from_cif,
    write_cyclic_restraints,
    zn_and_his_nitrogens_from_cif,
    zn_coordination_geometry,
)
from xenodesign.benchmark.cases import get_case



# ── GPU driver ──────────────────────────────────────────────────────────────────

def run_cyclic_design(
    device: str = _DEFAULT_DEVICE,
    seed: int = 42,
    out_dir: Path | str | None = None,
    seed_seq: str | None = None,
    rng_seed: int = 0,
    deposit_cif: str | Path | None = None,
    num_diffn_timesteps: int = 200,
    restraints: bool = True,
    closure: bool = False,
) -> dict:  # pragma: no cover (gpu)
    """Predict the single-chain 6UFA Zn-macrocycle and score it for geometry RECALL.

    Phase-1 LINEAR predict (cyclic offset #23 deferred): build the mixed-chirality seed
    (His L/D pinned), emit the peptide + Zn-ligand FASTA, run a constrained Chai predict with
    the His<->Zn metal_coordination restraints, then score:
      * backbone heavy-atom RMSD to the 6UFA deposit (RECALL) — needs deposit_cif;
      * Zn-N coordination geometry (n_coordinating, Zn-N distances, N-Zn-N angles);
      * the N/C-terminus distance (emergent-closure proxy).

    NOTE on checkpoint_noise (case knob 0.10): the case knob 'checkpoint_noise' is a
    truncated-diffusion warm-start σ for a refine path. Phase-1 uses a FULL predict (the only
    path that honours constraints — chai_truncated.py does not support constraint_path, same
    limitation as the α restrained run), so checkpoint_noise is recorded in the result but not
    applied here; it becomes live when a constrained truncated-refine path exists.
    """
    import tempfile

    from xenodesign.backends.chai_backend import ChaiBackend, load_prediction
    from xenodesign.config import resolve_device

    device = device or resolve_device()  # None -> XENO_DEVICE / cuda:0 if avail / mps / cpu

    case = get_case("cyclic")
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="xd_cyclic_"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Seed (mixed L/D His pinned) ──────────────────────────────────────────────
    seed_result = build_cyclic_seed(case, seed_seq=seed_seq, rng_seed=rng_seed)
    mixed_seq = mixed_chirality_fasta(seed_result.one_letter, seed_result.fixed_chirality)
    input_fasta = build_cyclic_input_fasta(mixed_seq)

    # ── Restraint (His<->Zn metal coordination) ──────────────────────────────────
    constraint_path = (write_cyclic_restraints(case, out_dir, seed_result=seed_result,
                                               closure=closure)
                       if restraints else None)

    t0 = time.time()
    print(f"\n{'='*78}\nXenoDesign1 — CYCLIC (6UFA Zn macrocycle) single-chain RECALL\n{'='*78}")
    print(f"  seed (L backbone) : {seed_result.one_letter}")
    print(f"  mixed-chirality   : {mixed_seq}")
    print(f"  His chirality     : {seed_result.fixed_chirality}")
    print(f"  Zn ligand         : {ZN_SMILES}  (chai chain B)")
    print(f"  restraints        : {restraints} ({constraint_path})")
    print(f"  baseline-to-beat  : backbone RMSD <= {case.baseline.backbone_rmsd} A to the "
          f"0.77-A deposit; Zn-N geometry secondary")
    print(f"  knobs             : {case.knobs}  (checkpoint_noise recorded, not applied — "
          f"full predict honours constraints)")
    print(f"  device {device} | timesteps {num_diffn_timesteps} | out {out_dir}\n{'='*78}\n")

    # ── Predict (FULL, constrained — phase-1 LINEAR) ─────────────────────────────
    # io_spec.build_fasta has no ligand path, so we write the Zn-ligand FASTA ourselves and run
    # chai's run_inference directly on it (ChaiBackend.predict would re-emit a protein-only FASTA).
    from chai_lab.chai1 import run_inference
    from xenodesign.backends.chai_backend import _save_confidence_npz

    fasta_path = out_dir / "input.fasta"
    fasta_path.write_text(input_fasta)
    chai_out = out_dir / "chai_out"
    chai_out.mkdir(parents=True, exist_ok=True)
    candidates = run_inference(
        fasta_file=fasta_path, output_dir=chai_out, device=device, seed=seed,
        num_diffn_timesteps=num_diffn_timesteps, use_esm_embeddings=True,
        use_msa_server=False,
        constraint_path=Path(constraint_path) if constraint_path is not None else None,
    )
    _save_confidence_npz(candidates, chai_out)
    pred = load_prediction(chai_out)
    best_cif = _best_pred_cif(chai_out)

    # ── Score: RMSD-recall + Zn-N geometry + closure proxy ───────────────────────
    rmsd = None
    if deposit_cif is not None:
        dep_bb = backbone_heavy_atoms_from_cif(deposit_cif, chain_name="A")
        des_bb = backbone_heavy_atoms_from_cif(best_cif, chain_name="A")
        # The deposit AND the design are now both the full 24-mer (S2-symmetric 6UFA); overlay
        # them at matched length (min guards against any tokenization length mismatch).
        n = min(dep_bb.shape[0], des_bb.shape[0])
        if n > 0 and des_bb.shape[0] > 0:
            try:
                rmsd = backbone_rmsd_to_deposit(des_bb[:n], dep_bb[:n])
            except Exception as exc:
                print(f"    [rmsd] WARNING: {exc}")

    zn_pos, his_ns = zn_and_his_nitrogens_from_cif(best_cif, chain_name="A")
    zn_geom = (zn_coordination_geometry(zn_pos, his_ns)
               if zn_pos is not None else {"n_coordinating": 0})
    termini_distance = termini_distance_from_cif(best_cif, chain_name="A")

    wall = time.time() - t0
    result = {
        "case_id": "cyclic",
        "seed_l_backbone": seed_result.one_letter,
        "mixed_chirality_seq": mixed_seq,
        "his_chirality": seed_result.fixed_chirality,
        "iptm": float(pred.iptm),
        "ptm": float(pred.ptm),
        "backbone_rmsd_to_deposit": rmsd,
        "baseline_backbone_rmsd": case.baseline.backbone_rmsd,
        "recall_meets_baseline": (rmsd is not None and rmsd <= case.baseline.backbone_rmsd),
        "zn_coordination_geometry": zn_geom,
        "termini_distance_closure_proxy": termini_distance,
        "knobs": dict(case.knobs),
        "restraints": bool(restraints),
        "closure": bool(closure),   # the actual closure-bond flag (was shadowed by the float)
        "constraint_path": str(constraint_path) if constraint_path is not None else None,
        "phase": "linear+emergent-closure (cyclic offset #23 DEFERRED)",
        "deposit_cif": str(deposit_cif) if deposit_cif is not None else None,
        "wall_time_s": wall,
        "out_dir": str(out_dir),
    }
    (out_dir / "cyclic_result.json").write_text(
        json.dumps(result, indent=2, default=lambda o: getattr(o, "tolist", lambda: str(o))()))

    print(f"\n{'='*78}")
    print(f"  ptm {pred.ptm:.4f}  iptm {pred.iptm:.4f}")
    print(f"  backbone RMSD to deposit: {rmsd}  (baseline <= {case.baseline.backbone_rmsd})  "
          f"-> recall {'MET' if result['recall_meets_baseline'] else 'not met / N/A'}")
    print(f"  Zn coordination: {zn_geom}")
    print(f"  termini distance (closure proxy): {termini_distance}")
    print(f"  wall {wall/60:.1f} min | result -> {out_dir/'cyclic_result.json'}\n{'='*78}")
    return result


def _best_pred_cif(chai_out: Path) -> Path:  # pragma: no cover (gpu)
    """Best-model CIF in a chai output dir (highest aggregate_score; lexical fallback)."""
    import re

    chai_out = Path(chai_out)
    score_files = sorted(chai_out.glob("scores.model_idx_*.npz"))
    if score_files:
        best_idx, best_agg = 0, -np.inf
        for f in score_files:
            agg = float(np.asarray(np.load(f)["aggregate_score"]).reshape(-1)[0])
            if agg > best_agg:
                best_agg = agg
                best_idx = int(re.search(r"idx_(\d+)", f.name).group(1))
        cif = chai_out / f"pred.model_idx_{best_idx}.cif"
        if cif.exists():
            return cif
    return sorted(chai_out.glob("*.cif"))[0]


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CYCLIC 6UFA Zn-macrocycle single-chain RECALL")
    p.add_argument("--device", default=_DEFAULT_DEVICE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rng_seed", type=int, default=0, help="RNG seed for the random backbone")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed_seq", default=None, help="explicit 12-res L backbone seed")
    p.add_argument("--deposit_cif", default=None,
                   help="path to the 6UFA deposit CIF for the RMSD-recall metric")
    p.add_argument("--num_diffn_timesteps", type=int, default=200)
    p.add_argument("--no_restraints", action="store_true",
                   help="disable the His<->Zn metal-coordination restraints")
    p.add_argument("--closure", action="store_true",
                   help="add the head-to-tail COVALENT macrocycle closure bond (#23)")
    p.add_argument("--smoke", action="store_true",
                   help="quick wiring smoke: 60 diffusion timesteps")
    return p.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover (gpu)
    import os

    args = _parse_args()
    timesteps = 60 if args.smoke else args.num_diffn_timesteps
    out = Path(args.out_dir) if args.out_dir else Path(f"/home/tmp/xd_cyclic_{os.getpid()}")
    run_cyclic_design(
        device=args.device, seed=args.seed, out_dir=out, seed_seq=args.seed_seq,
        rng_seed=args.rng_seed, deposit_cif=args.deposit_cif,
        num_diffn_timesteps=timesteps, restraints=not args.no_restraints,
        closure=args.closure,
    )
    sys.exit(0)
