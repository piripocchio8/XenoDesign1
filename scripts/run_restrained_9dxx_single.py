"""Standalone CPU-prep / GPU-run driver for the SINGLE-ANCHOR restrained re-prediction of 9DXX.

Extra benchmark point for the complex-scoring pipeline: 9DXX is a 3-chain complex — a 31-residue
all-D cystine-knot mini-peptide ("DP93", pred chain C) docked on a two-chain influenza-HA receptor
(HA1=chain A, HA2=chain B). An UNRESTRAINED Chai prediction found the right surface patch but placed
the peptide ~6 A from the deposited pose (wrong orientation). This re-predicts WITH one light POCKET
anchor (peptide whole-chain <-> the single most-contacted receptor residue, ASN21 of HA1 = FASTA
token A:N13, max_distance 6.0 A) to test whether that anchor recovers the deposited pose so 9DXX can
join 8GQP/7YH8 as a fit case.

Recipe reused from chai_real_v2 (the cached-MSA real run): run_inference(9dxx_complex.fasta,
use_esm_embeddings=True, use_msa_server=False, device="cuda:0", seed=...) PLUS the cached MSA dir
from chai_real_v2_seed42/msas (offline, sequence-hash matched -> HA1/HA2 hit cached .aligned.pqt;
the all-D peptide is non-protein and gets an empty MSA) PLUS constraint_path=<single anchor>.

This is a SEPARATE script — scripts/run_restrained_batch.py is in use and must NOT be edited.
NO name-check patch: the POCKET token-level partner is the L-standard receptor (ASN -> matches FASTA
'N'); the all-D peptide is the chain-level partner with no named token, so it is never name-checked.

Run inside the chai 0.6.1 container, GPU pinned via CUDA_VISIBLE_DEVICES so the single visible GPU is
"cuda:0" inside:
  python scripts/run_restrained_9dxx_single.py --seeds 42 43 44
  python scripts/run_restrained_9dxx_single.py --dry-run        # print planned run_inference calls, no GPU

Idempotent: a seed whose chai_out already holds 5 pred CIFs is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo-relative anchors (script lives in scripts/, repo root is its parent).
REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "XenoDesign1_local_ref" / "9dxx_target_gate"

FASTA = GATE / "9dxx_complex.fasta"
RESTRAINT = REPO / "XenoDesign1_local_ref" / "benchmarks" / "pocket_restraints_single" / "9DXX.restraints"
# Cached real-run MSA (offline). NOTE: chai_real_v2's output_dir IS the seed dir; MSAs live in
# <seed>/msas (there is no chai_out subdir under the v2 cache). HA1/HA2 .aligned.pqt are matched by
# sequence hash; the all-D peptide has no MSA (non-protein -> empty), which is expected.
MSA_DIR = GATE / "chai_real_v2_seed42" / "msas"

OUT_ROOT = REPO / "XenoDesign1_local_ref" / "benchmarks" / "restrained_single" / "9DXX" / "real"

DEVICE = "cuda:0"  # container pinned via CUDA_VISIBLE_DEVICES -> single visible GPU is cuda:0 inside.
N_MODELS = 5       # chai num_diffn_samples default -> 5 pred CIFs per seed.


def _done(chai_out: Path) -> bool:
    """Idempotency: done when chai_out already holds 5 pred CIFs."""
    return len(list(chai_out.glob("pred.model_idx_*.cif"))) >= N_MODELS


def plan(seeds: list[int]) -> list[dict]:
    """Build the per-seed job plan (no GPU touched)."""
    jobs = []
    for seed in seeds:
        out_dir = OUT_ROOT / f"seed{seed}"
        jobs.append(
            {
                "system": "9DXX",
                "item": "real",
                "seed": seed,
                "fasta": str(FASTA),
                "restraint": str(RESTRAINT),
                "msa_directory": str(MSA_DIR),
                "out_dir": str(out_dir),
                "chai_out": str(out_dir / "chai_out"),
            }
        )
    return jobs


def run_job(job: dict) -> dict:  # pragma: no cover (gpu)
    from chai_lab.chai1 import run_inference

    # _save_confidence_npz persists per-token pae/pde/plddt (chai does NOT auto-save these);
    # the gate/score code reads confidence.model_idx_*.npz alongside scores.model_idx_*.npz.
    from xenodesign.backends.chai_backend import _save_confidence_npz, load_prediction

    out_dir = Path(job["out_dir"])
    chai_out = Path(job["chai_out"])
    seed = int(job["seed"])

    if _done(chai_out):
        print(f"[skip] 9DXX/real/seed{seed}: {N_MODELS} CIFs present.", flush=True)
        return {"status": "skipped", "system": "9DXX", "item": "real", "seed": seed}

    # chai 0.6.1 asserts output_dir is empty/nonexistent -> route into out_dir/chai_out.
    out_dir.mkdir(parents=True, exist_ok=True)
    chai_out.mkdir(parents=True, exist_ok=True)
    print(
        f"[run ] 9DXX/real/seed{seed}  fasta={Path(job['fasta']).name} "
        f"restraint={Path(job['restraint']).name} msa={Path(job['msa_directory']).parent.name}/msas",
        flush=True,
    )

    cands = run_inference(
        fasta_file=Path(job["fasta"]),
        output_dir=chai_out,
        constraint_path=Path(job["restraint"]),
        msa_directory=Path(job["msa_directory"]),
        use_esm_embeddings=True,
        use_msa_server=False,
        device=DEVICE,
        seed=seed,
    )
    _save_confidence_npz(cands, chai_out)

    pred = load_prediction(chai_out)
    res = {
        "status": "ok",
        "system": "9DXX",
        "item": "real",
        "seed": seed,
        "iptm": round(float(pred.iptm), 3),
        "fasta": job["fasta"],
        "restraint": job["restraint"],
        "msa_directory": job["msa_directory"],
        "chai_out": str(chai_out),
    }
    (out_dir / "complex_result.json").write_text(json.dumps(res, indent=2))
    print(f"[done] 9DXX/real/seed{seed}  iptm={res['iptm']}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned run_inference calls and exit (no GPU, no chai import).",
    )
    args = ap.parse_args(argv)

    jobs = plan(args.seeds)

    if args.dry_run:
        print(f"=== DRY RUN: {len(jobs)} planned run_inference call(s) for 9DXX real ===")
        for j in jobs:
            done = _done(Path(j["chai_out"]))
            print(f"--- seed {j['seed']}  ({'SKIP: 5 CIFs present' if done else 'WOULD RUN'}) ---")
            print(
                "  run_inference(\n"
                f"      fasta_file={j['fasta']!r},\n"
                f"      output_dir={j['chai_out']!r},\n"
                f"      constraint_path={j['restraint']!r},\n"
                f"      msa_directory={j['msa_directory']!r},\n"
                "      use_esm_embeddings=True, use_msa_server=False,\n"
                f"      device={DEVICE!r}, seed={j['seed']},\n"
                "  )"
            )
        # Light validation so a dry-run also flags missing inputs early.
        miss = [p for p in (FASTA, RESTRAINT, MSA_DIR) if not Path(p).exists()]
        if miss:
            print("\n[WARN] missing input(s):", *[str(m) for m in miss])
            return 1
        print("\n[ok] FASTA, restraint, and cached MSA dir all present.")
        return 0

    n_ok = n_skip = n_err = 0
    for i, job in enumerate(jobs, 1):
        print(f"--- [{i}/{len(jobs)}] seed {job['seed']} ---", flush=True)
        try:
            r = run_job(job)
            n_skip += r["status"] == "skipped"
            n_ok += r["status"] == "ok"
        except Exception as e:  # pragma: no cover (gpu)
            n_err += 1
            print(f"[ERR ] 9DXX/real/seed{job['seed']}: {type(e).__name__}: {e}", flush=True)
    print(f"=== 9DXX single-anchor finished: ok={n_ok} skip={n_skip} err={n_err} ===", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
