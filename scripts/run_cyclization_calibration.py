#!/usr/bin/env python
"""ABC cyclization-calibration GATE — diffusion (steps × start) vs RING CLOSURE.

The signal the fast-cycle fitness must detect is CYCLIZATION (proper head-to-tail
ring closure), NOT real-vs-scramble register specificity. This sweep answers:

  Is there a (diffusion-step-count, starting-noise) MUCH cheaper than the full
  200-step predict at which the KNOWN mixed-chirality cycles (POS-6, POS-24)
  actually CLOSE, AND an objective term ranks proper cyclization ABOVE a strained
  full-L homochiral control (NEG-6, NEG-24)?

Restraint policy (GENERAL — must generalize to ANY mixed-chirality design):
  ONLY the head-to-tail COVALENT closure bond (C(res L) -> N(res 1)), via
  ``benchmark.restraints.head_to_tail_closure_row`` (== cyclic.build_closure_row).
  NO Zn, NO coordination/contact restraints. ``chai_patches.ensure_patches()`` is
  installed so the D-residue covalent closure name-match works.

Two starting points per (case, K):
  - start='full'   : a FULL predict at K diffusion steps from pure noise, WITH the
                     closure bond (constraint_path) applied. The closure is ENFORCED.
  - start='refine' : structure-conditioned truncated refinement — start from a
                     less-noised seed (a cheap 25-step full predict) and run only the
                     trailing K steps (``truncated_refine(ref_time_steps=K)``). The
                     vendored truncated sampler does NOT accept constraint_path
                     (TODO #27), so closure here is EMERGENT (measured, not enforced).
                     This isolates whether WHERE you start matters, not just how many.

Per (case × K × start) we MEASURE and REPORT EACH SEPARATELY:
  1. GROUND-TRUTH closure: C(res L)<->N(res 1) distance + closure-amide omega planarity.
  2. Each objective TERM alone (+ aggregate): mainchain-pLDDT of the cyclizing seam,
     per-position chirality, amide-planarity+backbone-valence geometry, pTM.

Run INSIDE the chai 0.6.1 container (RUNBOOK §2). GPU split: GPU 0 = the two
REAL mixed-chirality cycles (POS-6, POS-24); GPU 1 = the two FAKE full-L cycles
(NEG-6, NEG-24); identical sweep on both, one heavy job per GPU.

Usage (inside container, one lane per GPU):
    python scripts/run_cyclization_calibration.py --lane pos --device cuda:0 \
        --steps 10 25 50 100 200 --out_root XenoDesign1_local_ref/cyc_calib/pos \
        --out XenoDesign1_local_ref/cyc_calib/pos.json
    python scripts/run_cyclization_calibration.py --lane neg --device cuda:0 \
        --steps 10 25 50 100 200 --out_root XenoDesign1_local_ref/cyc_calib/neg \
        --out XenoDesign1_local_ref/cyc_calib/neg.json
(each lane pinned to its GPU via -e CUDA_VISIBLE_DEVICES on the docker run, so the
in-container device is always cuda:0.)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

# Establish import order (cyclic <-> base circular import guard).
import xenodesign.classes.base  # noqa: F401


# ── Panel construction ────────────────────────────────────────────────────────
#
# POS = genuinely mixed-chirality macrocycles (close well even strained-free):
#   POS-24 : 6UFA 24-mer KL(DGN)(DGL)(AIB)H(DLY)(DLE)QE(AIB)(DHI) x2, AIB->ALA proxy.
#   POS-6  : EPpKPp = EP(DPR)KP(DPR).
# NEG = full-L homochiral controls of the SAME length (a homochiral ring is strained,
#   so it should close worse / score lower):
#   NEG-24 : a deterministic full-L random 24-mer (closure bond applied).
#   NEG-6  : a deterministic full-L random 6-mer  (closure bond applied).

# AIB proxied as ALA ('A'); the explicit per-position D blocks are the 6UFA handedness.
_POS24_UNIT = "KL(DGN)(DGL)AH(DLY)(DLE)QEA(DHI)"   # 12-mer repeat (AIB->A)
POS24_FASTA = _POS24_UNIT * 2                       # 24-mer
POS6_FASTA = "EP(DPR)KP(DPR)"                        # 6-mer EPpKPp

# Full-L random controls — canonical L residues only (no D blocks, no Gly so every
# position is a stereocenter), deterministic per seed.
_L_ALPHABET = "ADEFHIKLMNQRSTVWY"  # excl. C/G/P: no disulfide, no achiral, no kink bias


def _random_l_fasta(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(_L_ALPHABET) for _ in range(length))


def build_lane(lane: str) -> list[dict]:
    """Return the two cases for a lane: 'pos' (REAL mixed-chirality) or 'neg' (full-L)."""
    if lane == "pos":
        return [
            {"id": "POS-6", "is_good": True, "length": 6, "d_fasta": POS6_FASTA,
             "note": "EPpKPp = EP(DPR)KP(DPR) (real mixed-chirality hexamer)"},
            {"id": "POS-24", "is_good": True, "length": 24, "d_fasta": POS24_FASTA,
             "note": "6UFA 24-mer KL(DGN)(DGL)AH(DLY)(DLE)QEA(DHI)x2 (AIB->ALA proxy)"},
        ]
    if lane == "neg":
        return [
            {"id": "NEG-6", "is_good": False, "length": 6,
             "d_fasta": _random_l_fasta(6, seed=601),
             "note": "full-L random hexamer (homochiral ring -> strained closure)"},
            {"id": "NEG-24", "is_good": False, "length": 24,
             "d_fasta": _random_l_fasta(24, seed=2401),
             "note": "full-L random 24-mer (homochiral ring -> strained closure)"},
        ]
    raise ValueError(f"unknown lane {lane!r} (expected 'pos' or 'neg')")


# ── Closure restraint (general head-to-tail; NO Zn/coordination) ──────────────

def _termini_one_letters(d_fasta: str) -> tuple[str, int, str]:
    """(n_term_one_letter, length, c_term_one_letter) from a chai mixed-chirality seq.

    Parses the (CCD) blocks vs bare letters. For the COVALENT closure row we need the
    canonical L one-letter at the termini (chai_patches accepts the D-CCD synonym), so a
    D block at a terminus is mapped back to its L parent one-letter."""
    from xenodesign.io_spec import AA3_TO_AA1
    from xenodesign.mirror import L_TO_D

    D_TO_L3 = {d: l for l, d in L_TO_D.items()}
    tokens: list[str] = []
    i = 0
    while i < len(d_fasta):
        if d_fasta[i] == "(":
            j = d_fasta.index(")", i)
            ccd = d_fasta[i + 1:j]
            l3 = D_TO_L3.get(ccd, ccd)        # D-CCD -> L parent 3-letter (e.g. DPR->PRO)
            tokens.append(AA3_TO_AA1.get(l3, "X"))
            i = j + 1
        else:
            tokens.append(d_fasta[i])
            i += 1
    return tokens[0], len(tokens), tokens[-1]


def write_closure_restraint(path: Path, d_fasta: str, chain: str = "A") -> Path:
    """Write the .restraints CSV carrying ONLY the head-to-tail COVALENT closure row."""
    from xenodesign.benchmark.restraints import head_to_tail_closure_row, write_restraints
    n_one, length, c_one = _termini_one_letters(d_fasta)
    row = head_to_tail_closure_row(chain, length=length,
                                   n_term_one_letter=n_one, c_term_one_letter=c_one)
    return write_restraints(path, [row])


# ── predict / refine drivers ──────────────────────────────────────────────────

def _entities(d_fasta: str) -> list[dict]:
    # already paren-emitted mixed-chirality; single binder chain, no target, no Zn.
    return [{"type": "protein", "name": "binder", "sequence": d_fasta, "chirality": "L"}]


def _attach_cif(pred, chai_out_dir: Path):
    from scripts.design_demo import _best_cif_path
    try:
        pred._cif_path = _best_cif_path(chai_out_dir)
    except Exception:
        pred._cif_path = None
    return pred


def main() -> None:  # pragma: no cover (gpu)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", choices=["pos", "neg"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    ap.add_argument("--reference_step", type=int, default=200)
    ap.add_argument("--starts", nargs="+", default=["full", "refine"],
                    choices=["full", "refine"])
    ap.add_argument("--refine_seed_steps", type=int, default=25,
                    help="full-predict step count used to make the truncated-refine SEED")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="one case (the 6-mer), steps [10,200], start full only (smoke)")
    args = ap.parse_args()

    from xenodesign.backends.chai_backend import ChaiBackend
    from xenodesign import chai_patches
    from xenodesign.abc.calibration import intramolecular_per_term_fn

    chai_patches.ensure_patches()  # D-residue COVALENT closure name-match
    backend = ChaiBackend(device=args.device, seed=args.seed)
    per_term = intramolecular_per_term_fn(chain_name="A")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = build_lane(args.lane)
    steps = [10, 200] if args.quick else list(args.steps)
    starts = ["full"] if args.quick else list(args.starts)
    if args.quick:
        cases = cases[:1]

    records: list[dict] = []
    t0 = time.time()
    for case in cases:
        cid = case["id"]
        d_fasta = case["d_fasta"]
        constraint_path = str(write_closure_restraint(
            out_root / f"{cid}_closure.restraints", d_fasta))

        # Build the truncated-refine SEED once per case (a cheap full predict whose coords
        # seed every 'refine' run). The seed itself carries the closure bond.
        refine_seed = None
        if "refine" in starts:
            seed_dir = out_root / f"{cid}_seed{args.refine_seed_steps}"
            try:
                pred = backend.predict(_entities(d_fasta), seed_dir,
                                       num_diffn_timesteps=int(args.refine_seed_steps),
                                       constraint_path=constraint_path)
                refine_seed = {"entities": _entities(d_fasta),
                               "coords": _attach_cif(pred, seed_dir / "chai_out").coords}
                print(f"[seed] {cid} refine-seed @ {args.refine_seed_steps} steps ready",
                      flush=True)
            except Exception as e:
                print(f"[ERR seed] {cid}: {type(e).__name__}: {e}", flush=True)
                refine_seed = None

        for start in starts:
            for K in steps:
                tag = f"{cid}_{start}_s{K}"
                out_dir = out_root / tag
                try:
                    if start == "full":
                        pred = backend.predict(
                            _entities(d_fasta), out_dir,
                            num_diffn_timesteps=int(K),
                            constraint_path=constraint_path)
                        pred = _attach_cif(pred, out_dir / "chai_out")
                    else:  # refine: trailing K steps from the less-noised seed (no constraint)
                        if refine_seed is None:
                            raise RuntimeError("no refine seed available")
                        pred = backend.truncated_refine(refine_seed, int(K), out_dir)
                        pred = _attach_cif(pred, out_dir / "chai_out")
                    vec = per_term(pred)
                    vec.update({
                        "ptm_raw": float(getattr(pred, "ptm", 0.0) or 0.0),
                        "iptm_raw": float(getattr(pred, "iptm", 0.0) or 0.0),
                    })
                    err = None
                except Exception as e:
                    vec = {"objective": float("-inf"), "mainchain_plddt": None,
                           "chirality": None, "geometry": None, "ptm": None,
                           "cn_distance": None, "closure_omega": None,
                           "omega_planarity": None, "closed": False}
                    err = f"{type(e).__name__}: {e}"
                rec = {"case_id": cid, "is_good": bool(case["is_good"]),
                       "length": case["length"], "start": start, "steps": int(K),
                       "error": err, **vec}
                records.append(rec)
                cn = vec.get("cn_distance")
                print(f"[run] {tag}  obj={vec.get('objective')}  "
                      f"cn={cn if cn is None else round(cn, 2)}A  "
                      f"closed={vec.get('closed')}  err={err}", flush=True)

    wall = time.time() - t0
    result = {
        "lane": args.lane, "device": args.device, "seed": args.seed,
        "steps": steps, "starts": starts, "reference_step": args.reference_step,
        "refine_seed_steps": args.refine_seed_steps,
        "cases": [{k: v for k, v in c.items()} for c in cases],
        "records": records, "wall_time_s": wall,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[written] {args.out}  ({len(records)} records, {wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
