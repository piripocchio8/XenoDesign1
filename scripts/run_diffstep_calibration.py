#!/usr/bin/env python
"""ABC T3 — diff-steps <-> fitness-fidelity calibration (GPU GATE).

Runs a small panel of KNOWN good-vs-bad mixed-chirality sequences through Chai-1
at several diffusion-step counts, scores each with the no-target intramolecular
objective, and reports per-step Spearman rank-correlation vs full-200 plus the
good-vs-bad margin -> the lowest viable step count K*.

Panel (real 6UFA His6/12/18/24 L/D/L/D zinc macrocycle):
  GOOD = the actual 6UFA deposit sequence (KL(DGN)(DGL)(AIB)H... — His on the
         coordinating register 6/12/18/24 with the correct L/D/L/D handedness).
  BAD  = composition-preserving scrambles (same residue multiset, His shuffled
         OFF the coordinating register so the Zn site cannot form).

Two routes are compared:
  - UNRESTRAINED: single binder chain, full predict at K steps.
  - RESTRAINED:   binder chain + Zn ligand + His(L 6,18)->Zn contact restraints
                  (the two D-His 12/24 are NOT anchored — chai's pocket/contact
                  name-check rejects D residues, RUNBOOK §8.3).

Run INSIDE the chai 0.6.1 container (see RUNBOOK.local.md §2). Writes a JSON
results blob to --out; the results doc is assembled from it.

Usage (inside container):
    python scripts/run_diffstep_calibration.py \
        --steps 10 25 50 100 200 \
        --device cuda:0 \
        --out_root /work/XenoDesign1_local_ref/diffstep_calib \
        --out /work/XenoDesign1_local_ref/diffstep_calib/results.json \
        [--restrained] [--quick]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Establish import order (cyclic <-> base circular import guard).
import xenodesign.classes.base  # noqa: F401

from xenodesign.abc.calibration import (
    chai_predict_fn,
    intramolecular_objective_fn,
    select_k_star,
    summarize_calibration,
)


# ── Panel construction from the real 6UFA deposit ─────────────────────────────

def _deposit_resnames(cif_path: Path) -> list[str]:
    import gemmi
    s = gemmi.read_structure(str(cif_path))
    return [r.name for r in s[0]["A"] if r.name not in ("HOH", "ZN")]


def _emit_fasta(resnames: list[str]) -> str:
    """Per-residue chai FASTA: canonical L -> bare letter; D/ncAA -> (CCD) block."""
    from xenodesign.io_spec import AA3_TO_AA1
    out: list[str] = []
    for nm in resnames:
        one = AA3_TO_AA1.get(nm)
        out.append(one if one is not None else f"({nm})")
    return "".join(out)


def build_panel(cif_path: Path, n_bad: int = 3) -> list[dict]:
    """GOOD = the deposit; BAD = composition-preserving scrambles (His off register)."""
    import random

    resnames = _deposit_resnames(cif_path)
    panel = [{
        "id": "GOOD_6UFA", "is_good": True, "d_fasta": _emit_fasta(resnames),
        "note": "real 6UFA deposit; His L/D/L/D on coordinating register 6/12/18/24",
    }]
    for seed in range(1, n_bad + 1):
        perm = resnames[:]
        random.Random(seed).shuffle(perm)
        his = [i + 1 for i, nm in enumerate(perm) if nm in ("HIS", "DHI")]
        panel.append({
            "id": f"BAD_scramble{seed}", "is_good": False,
            "d_fasta": _emit_fasta(perm),
            "note": f"composition-preserving scramble; His at {his} (off register)",
        })
    return panel


def write_zn_restraints(path: Path) -> Path:
    """His(L 6,18) -> Zn contact restraints. D-His (12,24) excluded (chai name-check).

    Binder = chain A; Zn ligand = chain B (FASTA record order). The Zn ligand token
    has no standard one-letter code, so its token is ``X1`` (X = UNK wildcard, the
    ``res_idxB`` form ``<one-letter><pos>`` with pos=1 = first/only ligand token), per
    ``benchmark.restraints.metal_coordination_rows`` (RUNBOOK §8.1/§8.3). We anchor only
    the two L-His (6, 18) — chai's contact name-check rejects the D-His (12, 24).
    """
    from xenodesign.benchmark.restraints import (
        RESTRAINT_HEADER, contact_row, UNKNOWN_RES,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        contact_row(chain_a="A", resnum_a=hr, chain_b="B", resnum_b=1,
                    confidence=0.8, max_distance=2.6,
                    res_one_letter_a="H", res_one_letter_b=UNKNOWN_RES,
                    comment=f"His{hr}(L)-Zn", restraint_id=f"zn_coord_{hr}")
        for hr in (6, 18)
    ]
    path.write_text("\n".join([RESTRAINT_HEADER, *rows]) + "\n")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    ap.add_argument("--reference_step", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cif", default="XenoDesign1_local_ref/6UFA.cif")
    ap.add_argument("--n_bad", type=int, default=3)
    ap.add_argument("--out_root", required=True, help="dir for per-(case,step) chai outputs")
    ap.add_argument("--out", required=True, help="results JSON path")
    ap.add_argument("--restrained", action="store_true",
                    help="add Zn ligand + His(L)->Zn contacts (full-predict route)")
    ap.add_argument("--min_rank_corr", type=float, default=0.9)
    ap.add_argument("--min_margin", type=float, default=0.0)
    ap.add_argument("--quick", action="store_true",
                    help="GOOD + 1 BAD, steps [10,200] only (smoke)")
    args = ap.parse_args()

    steps = [10, 200] if args.quick else args.steps
    n_bad = 1 if args.quick else args.n_bad

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    panel = build_panel(Path(args.cif), n_bad=n_bad)
    constraint_path = None
    if args.restrained:
        constraint_path = str(write_zn_restraints(out_root / "zn.restraints"))

    # Attach run-time fields each case needs for predict_fn.
    for case in panel:
        case["out_root"] = str(out_root / ("restr" if args.restrained else "unrestr"))
        case["restrained"] = bool(args.restrained)
        case["constraint_path"] = constraint_path

    predict_fn = chai_predict_fn(device=args.device, seed=args.seed)
    objective_fn = intramolecular_objective_fn(chain_name="A")

    objective_by_step: dict[int, dict[str, float]] = {s: {} for s in steps}
    for case in panel:
        for step in steps:
            print(f"[run] {case['id']} steps={step} restrained={args.restrained}", flush=True)
            try:
                pred = predict_fn(case, step)
                score = float(objective_fn(pred))
                ptm = float(getattr(pred, "ptm", 0.0) or 0.0)
                iptm = float(getattr(pred, "iptm", 0.0) or 0.0)
                print(f"[score] {case['id']} s={step} obj={score:.4f} "
                      f"ptm={ptm:.3f} iptm={iptm:.3f}", flush=True)
            except Exception as e:  # graceful: never crash the sweep
                score = float("-inf")
                print(f"[ERR] {case['id']} s={step}: {type(e).__name__}: {e}", flush=True)
            objective_by_step[step][case["id"]] = score

    labels = {c["id"]: bool(c["is_good"]) for c in panel}
    summary = summarize_calibration(
        objective_by_step, labels=labels, reference_step=args.reference_step)
    k_star = select_k_star(
        summary, min_rank_corr=args.min_rank_corr, min_margin=args.min_margin)

    result = {
        "route": "restrained" if args.restrained else "unrestrained",
        "steps": steps,
        "reference_step": args.reference_step,
        "min_rank_corr": args.min_rank_corr,
        "min_margin": args.min_margin,
        "panel": [{k: v for k, v in c.items() if k != "out_root"} for c in panel],
        "objective_by_step": objective_by_step,
        "labels": labels,
        "summary": summary,
        "k_star": k_star,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("\n=== SUMMARY (route=%s) ===" % result["route"], flush=True)
    for step in sorted(summary):
        r = summary[step]
        print(f"  step={step:>3}  rank_corr={r['rank_corr']:+.3f}  margin={r['margin']:+.4f}",
              flush=True)
    print(f"  K* = {k_star}", flush=True)
    print(f"[written] {args.out}", flush=True)


if __name__ == "__main__":
    main()
