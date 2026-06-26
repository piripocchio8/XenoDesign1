"""NON-ALPHA 9DXX (D-knottin : influenza HA) — MSA'd 2-chain target + all-D cystine-knot
binder driver (task #29 / spec §2.3, P3b).

This is a PURE-SOFTWARE wiring driver for a structure-prediction / design problem:
  * Target  : the influenza HA protomer = TWO L chains, HA1 (328 aa) + HA2 (176 aa). The
              MSA-free fold is wrong (gate #29: pTM 0.43, HA1-HA2 ipTM 0.15); with a PRECOMPUTED
              MSA Chai reproduces it near-natively (pTM 0.92). We therefore feed the cached MSA
              (chai .aligned.pqt, keyed by sequence hash) through the new
              ``ChaiBackend.predict(msa_directory=...)`` seam (#29, P3a). The target is FIXED
              context; only the binder is designed.
  * Binder  : a 31-residue ALL-D cystine-knot peptide. An inhibitor-cystine-knot (ICK) scaffold
              has 6 Cys forming 3 disulfides with connectivity I-IV, II-V, III-VI. We place the
              6 Cys on the binder and emit the 3 disulfide bonds as chai COVALENT restraints
              (``benchmark.restraints.disulfide_rows``); chai wires COVALENT rows as real
              ``atom_covalent_bond_indices`` (bond_utils), NOT soft distance restraints.
  * SS-bias : anti-alpha (knottins are NON-helical) — the per-case SS-bias config (#21, P1b).

The exotic deposited DP93 binder (chain E of 9DXX) uses non-canonical residues (7YO, F9D) and is
NOT reconstructed here; we design a CANONICAL all-D cystine-knot binder (the deposit is the
prior-art reference to beat, not a recall target — interface baseline is a future measurement).

PRIMARY path = a constrained MSA'd predict of [HA1, HA2, binder] that VALIDATES the full
wiring (MSA target + 2-chain + D-binder + disulfide bonds) end-to-end and reports the binder
interface ipTM + binder chirality + disulfide geometry. A multi-trajectory DESIGN loop
(redesign the non-Cys binder positions, reusing the α loop wiring) is the documented next step.

GPU RUN (ifrit, chai 0.6.1 container; tag in RUNBOOK.local.md):
    PYTHONPATH=/work python scripts/design_nonalpha.py --device cuda:0          # full predict
    PYTHONPATH=/work python scripts/design_nonalpha.py --smoke                  # 40-step wiring

CAVEAT (D-residue covalent name-match) — CONFIRMED ON GPU 2026-06-15: chai matches a COVALENT
row's one-letter residue code against the token name via rc.restype_1to3 (L-only). An all-D Cys
token is stored as the D-CCD name (DCY), which does NOT match the L 'CYS' code, so the disulfide
COVALENT bond is REJECTED (bond_utils assert left_residue_idx.numel() > 0 fails — verified with
the analogous D-His closure). Therefore the cystine-knot DISULFIDES cannot be imposed as chai
covalent bonds on an all-D binder; the default run uses ``--no_disulfides`` and the binder is
designed under the anti-α SS-bias + the MSA'd target only. The disulfide_rows builder is correct
(it works for L-Cys / future chai) and is kept for that path. Imposing D-Cys disulfides needs a
chai-side fix (D-aware covalent name-match) or a structure-level bond input — a documented
follow-up, NOT faked here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from xenodesign.benchmark.cases import get_case, ss_bias_config_for_case
from xenodesign.benchmark.restraints import write_restraints

# T6: the ICK scaffold + HA-target helpers were promoted into the real BinderClass module
# ``xenodesign.classes.non_alpha`` (single source of truth). This script is now a thin shim that
# re-exports them and keeps its predict-only ``run_nonalpha_design`` CLI (the wiring validator).
# ``tests/test_design_nonalpha.py`` and ``xenodesign.targets`` import these names from here.
from xenodesign.classes.non_alpha import (  # noqa: F401
    _BINDER_CHAIN,
    _DEFAULT_BINDER_LEN,
    _DEFAULT_HA_FASTA,
    _DEFAULT_MSA_DIR,
    build_binder_seed,
    build_nonalpha_disulfide_rows,
    disulfide_geometry_from_cif,
    ick_disulfide_pairs,
    knottin_cys_positions,
    load_ha_entities,
    place_cys,
)

_DEFAULT_DEVICE = None  # unset -> resolve_device() (XENO_DEVICE / cuda:0 if avail / mps / cpu)


# ── GPU driver ──────────────────────────────────────────────────────────────────────

def run_nonalpha_design(
    device: str = _DEFAULT_DEVICE,
    seed: int = 42,
    out_dir: Path | str | None = None,
    ha_fasta: str | Path = _DEFAULT_HA_FASTA,
    msa_dir: str | Path | None = _DEFAULT_MSA_DIR,
    binder_length: int = _DEFAULT_BINDER_LEN,
    cys_positions=None,
    seed_seq: str | None = None,
    rng_seed: int = 0,
    disulfides: bool = True,
    num_diffn_timesteps: int = 200,
    deposit_cif: str | Path | None = None,
) -> dict:  # pragma: no cover (gpu)
    """Predict the MSA'd HA target + all-D cystine-knot binder complex (P3b wiring validation).

    Assembles entities [HA1, HA2, binder(all-D)], emits the ICK disulfide COVALENT bonds on the
    binder chain, feeds the cached HA MSA via ``ChaiBackend.predict(msa_directory=...)``, runs a
    constrained predict, and reports the binder interface ipTM, binder chirality, and the SG-SG
    disulfide geometry. ``disulfides=False`` runs the bondless control (and is the documented
    fallback if the D-Cys covalent name-match is rejected by chai)."""
    import tempfile

    from xenodesign.backends.chai_backend import ChaiBackend, per_chain_plddt
    from xenodesign.config import resolve_device

    device = device or resolve_device()  # None -> XENO_DEVICE / cuda:0 if avail / mps / cpu

    case = get_case("nonalpha")
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="xd_nonalpha_"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cys_positions is None:
        cys_positions = knottin_cys_positions(binder_length)
    binder_seed = build_binder_seed(binder_length, cys_positions, seed_seq=seed_seq,
                                    rng_seed=rng_seed)

    ha_entities = load_ha_entities(ha_fasta)
    binder_entity = {"type": "protein", "name": "binder",
                     "sequence": binder_seed, "chirality": "D"}
    entities = [*ha_entities, binder_entity]

    constraint_path = None
    if disulfides:
        rows = build_nonalpha_disulfide_rows(cys_positions, binder_chain=_BINDER_CHAIN)
        constraint_path = write_restraints(out_dir / "nonalpha.restraints", rows)

    ss_bias = ss_bias_config_for_case(case)   # anti-alpha (target_helix_frac=0.0)

    t0 = time.time()
    print(f"\n{'='*78}\nXenoDesign1 — NON-ALPHA 9DXX (D-knottin : HA) MSA'd predict\n{'='*78}")
    print(f"  HA target chains : {[e['name'] for e in ha_entities]} "
          f"({sum(len(e['sequence']) for e in ha_entities)} aa, L, MSA={msa_dir})")
    print(f"  binder (chain {_BINDER_CHAIN}, D) : {binder_seed}  ({binder_length} aa)")
    print(f"  Cys positions    : {cys_positions}  ICK pairs {ick_disulfide_pairs(cys_positions)}")
    print(f"  disulfides       : {disulfides} ({constraint_path})")
    print(f"  ss_bias          : anti_alpha (target_helix_frac={ss_bias.target_helix_frac})")
    print(f"  device {device} | timesteps {num_diffn_timesteps} | out {out_dir}\n{'='*78}\n")

    backend = ChaiBackend(device=device, seed=seed)
    pred = backend.predict(entities, out_dir / "pred",
                           num_diffn_timesteps=num_diffn_timesteps,
                           constraint_path=constraint_path,
                           msa_directory=msa_dir)

    # Best-model CIF for the geometry read.
    from scripts.design_demo import _best_cif_path, _chirality_violation_frac_from_cif
    cif = _best_cif_path(out_dir / "pred")
    try:
        binder_chir = _chirality_violation_frac_from_cif(cif)
    except Exception:
        binder_chir = None
    try:
        ss_geom = disulfide_geometry_from_cif(cif, _BINDER_CHAIN, cys_positions)
    except Exception as exc:
        ss_geom = {"error": str(exc)}

    wall = time.time() - t0
    result = {
        "case_id": "nonalpha",
        "binder_seed_l": binder_seed,
        "binder_length": binder_length,
        "cys_positions": list(cys_positions),
        "ick_disulfide_pairs": ick_disulfide_pairs(cys_positions),
        "iptm": float(pred.iptm),
        "ptm": float(pred.ptm),
        "binder_chirality": binder_chir,
        "disulfide_geometry": ss_geom,
        "disulfides": bool(disulfides),
        "msa_dir": str(msa_dir) if msa_dir is not None else None,
        "constraint_path": str(constraint_path) if constraint_path is not None else None,
        "ss_bias_target_helix_frac": ss_bias.target_helix_frac,
        "phase": "MSA'd predict wiring-validation (design loop = documented next step)",
        "wall_time_s": wall,
        "out_dir": str(out_dir),
    }
    (out_dir / "nonalpha_result.json").write_text(
        json.dumps(result, indent=2, default=lambda o: getattr(o, "tolist", lambda: str(o))()))

    print(f"\n{'='*78}")
    print(f"  ptm {pred.ptm:.4f}  iptm {pred.iptm:.4f}  binder chirality {binder_chir}")
    print(f"  disulfide geometry: {ss_geom}")
    print(f"  wall {wall/60:.1f} min | result -> {out_dir/'nonalpha_result.json'}\n{'='*78}")
    return result


# ── CLI ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NON-ALPHA 9DXX D-knottin:HA MSA'd predict")
    p.add_argument("--device", default=_DEFAULT_DEVICE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--ha_fasta", default=_DEFAULT_HA_FASTA)
    p.add_argument("--msa_dir", default=_DEFAULT_MSA_DIR)
    p.add_argument("--binder_length", type=int, default=_DEFAULT_BINDER_LEN)
    p.add_argument("--seed_seq", default=None, help="explicit binder L backbone seed")
    p.add_argument("--rng_seed", type=int, default=0)
    p.add_argument("--no_disulfides", action="store_true",
                   help="bondless control / fallback if D-Cys covalent name-match is rejected")
    p.add_argument("--no_msa", action="store_true", help="disable the HA target MSA (gate #29 off)")
    p.add_argument("--num_diffn_timesteps", type=int, default=200)
    p.add_argument("--smoke", action="store_true", help="quick wiring smoke: 40 timesteps")
    return p.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover (gpu)
    import os

    args = _parse_args()
    timesteps = 40 if args.smoke else args.num_diffn_timesteps
    out = Path(args.out_dir) if args.out_dir else Path(f"/home/tmp/xd_nonalpha_{os.getpid()}")
    run_nonalpha_design(
        device=args.device, seed=args.seed, out_dir=out,
        ha_fasta=args.ha_fasta, msa_dir=None if args.no_msa else args.msa_dir,
        binder_length=args.binder_length, seed_seq=args.seed_seq, rng_seed=args.rng_seed,
        disulfides=not args.no_disulfides, num_diffn_timesteps=timesteps,
    )
    sys.exit(0)
