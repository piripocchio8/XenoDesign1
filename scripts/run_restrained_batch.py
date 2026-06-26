"""GPU batch driver for the POCKET-RESTRAINED register experiment (8GQP + 7YH8 ONLY).

Re-predict each True binder AND its register/scramble decoys POCKET-RESTRAINED to the
SAME deposited target pocket, so a composition-preserving register-shift / scramble decoy
cannot dock elsewhere and confound the fit (free re-pred had 7YH8 shift3 re-dock to
ipTM 0.732 > real 0.637).

Reads a manifest JSON (list of jobs), filters to one --lane, and runs each job sequentially
with chai_lab 0.6.1 run_inference (constraint_path=<system pocket restraints>, seed, cuda:0).
Idempotent: a job whose out_dir/chai_out already has 5 pred CIFs is skipped.

Run (inside the chai 0.6.1 container, one process per GPU, each pinned via CUDA_VISIBLE_DEVICES):
  python scripts/run_restrained_batch.py --manifest <path> --lane 0   # 8GQP (CUDA_VISIBLE_DEVICES=0)
  python scripts/run_restrained_batch.py --manifest <path> --lane 1   # 7YH8 (CUDA_VISIBLE_DEVICES=1)

ponytail: device is HARD-CODED "cuda:0" — the container is launched with CUDA_VISIBLE_DEVICES=<lane>
so the single visible GPU is always "cuda:0" inside (the CUDA_VISIBLE_DEVICES=1 -> "cuda:0" gotcha).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEVICE = "cuda:0"  # ponytail: see module docstring — always cuda:0 inside the pinned container.
N_MODELS = 5       # chai num_diffn_samples default; 5 pred CIFs per job.


def _patch_pocket_name_check() -> None:  # pragma: no cover (gpu)
    """Back-compat shim → the shared ``chai_patches._patch_pocket_name_check``.

    The pocket residue-name relaxation was promoted into ``xenodesign.chai_patches`` so it can be
    installed off the shared ``ensure_patches`` dispatch path. Behaviour is byte-identical; this
    thin re-export keeps any caller importing it from this module working."""
    from xenodesign.chai_patches import _patch_pocket_name_check as _shared
    _shared()


def _done(chai_out: Path) -> bool:
    """Idempotency: a job is done when its chai_out already holds 5 pred CIFs."""
    return len(list(chai_out.glob("pred.model_idx_*.cif"))) >= N_MODELS


def run_job(job: dict) -> dict:  # pragma: no cover (gpu)
    from chai_lab.chai1 import run_inference

    # _save_confidence_npz persists per-token pae/pde/plddt (chai does NOT auto-save these);
    # scripts/score_complex.confidence() reads confidence.model_idx_*.npz for ipAE/ipSAE.
    from xenodesign.backends.chai_backend import _save_confidence_npz, load_prediction

    out_dir = Path(job["out_dir"])
    chai_out = out_dir / "chai_out"
    fasta = Path(job["fasta"])
    # FREE re-predict support: falsy/missing "restraint" -> no constraint (backward compatible).
    restraint = Path(job["restraint"]) if job.get("restraint") else None
    seed = int(job["seed"])

    if _done(chai_out):
        print(f"[skip] {job['system']}/{job['item']}/seed{seed}: {N_MODELS} CIFs present.",
              flush=True)
        return {"status": "skipped", **{k: job[k] for k in ("system", "item", "seed")}}

    # chai 0.6.1 asserts output_dir is empty/nonexistent -> route into out_dir/chai_out.
    out_dir.mkdir(parents=True, exist_ok=True)
    chai_out.mkdir(parents=True, exist_ok=True)
    print(f"[run ] {job['system']}/{job['item']}/seed{seed}  fasta={fasta.name} "
          f"restraint={restraint.name if restraint else 'NONE'} na={job['na']}", flush=True)

    cands = run_inference(
        fasta_file=fasta,
        output_dir=chai_out,
        constraint_path=restraint,  # None -> FREE (unrestrained) re-predict
        device=DEVICE,
        seed=seed,
        use_esm_embeddings=True,
        use_msa_server=False,
    )
    _save_confidence_npz(cands, chai_out)

    pred = load_prediction(chai_out)
    res = {"status": "ok", "system": job["system"], "item": job["item"], "seed": seed,
           "na": job["na"], "nb": job["nb"], "iptm": round(float(pred.iptm), 3),
           "fasta": str(fasta), "restraint": str(restraint) if restraint else None,
           "chai_out": str(chai_out)}
    (out_dir / "complex_result.json").write_text(json.dumps(res, indent=2))
    print(f"[done] {job['system']}/{job['item']}/seed{seed}  iptm={res['iptm']}", flush=True)
    return res


def main(argv=None):  # pragma: no cover (gpu)
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lane", type=int, required=True, choices=[0, 1])
    args = ap.parse_args(argv)

    jobs = json.loads(Path(args.manifest).read_text())
    lane_jobs = [j for j in jobs if j["lane"] == args.lane]
    sys_label = sorted({j["system"] for j in lane_jobs})
    print(f"=== lane {args.lane}: {len(lane_jobs)} jobs, systems={sys_label} ===", flush=True)

    from xenodesign.chai_patches import ensure_patches
    ensure_patches()  # pocket-name relaxation + token-dist residue-match repair (shared installer)

    n_ok = n_skip = n_err = 0
    for i, job in enumerate(lane_jobs, 1):
        print(f"--- [{i}/{len(lane_jobs)}] ---", flush=True)
        try:
            r = run_job(job)
            n_skip += (r["status"] == "skipped")
            n_ok += (r["status"] == "ok")
        except Exception as e:
            n_err += 1
            print(f"[ERR ] {job['system']}/{job['item']}/seed{job['seed']}: "
                  f"{type(e).__name__}: {e}", flush=True)
    print(f"=== lane {args.lane} finished: ok={n_ok} skip={n_skip} err={n_err} ===", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":  # pragma: no cover (gpu)
    sys.exit(main())
