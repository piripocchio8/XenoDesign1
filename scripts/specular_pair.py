"""T16: specular-pair internal-consistency control (the mirror test).

The designed complex is L-target . D-binder. Its EXACT mirror image is D-target . L-binder — by
parity these are physically identical (same structure mirrored, same energy), so a self-consistent
oracle must return the SAME interface ipTM and a mirror-image fold. We predict BOTH with two default
Chai-1 calls (no coordinate mirroring; each predicted from scratch) and report the gap:

    gap = |ipTM(L-target.D-binder) - ipTM(D-target.L-binder)|

A gap ~0 = Chai is mirror-self-consistent on this interface; a large gap = Chai's L-handed training
bias (the real worry, since Chai is both designer and judge). This is DISTINCT from the binder-only
`mirror_discrepancy` referee — here the WHOLE complex is mirrored and re-predicted.

Run (GPU; chai 0.6.1 container):
    python scripts/specular_pair.py --design seireriekaGkeleeilkrf --device cuda:0 \
        --out_dir XenoDesign1_local_ref/specular/p1
The --design sequence is the one-letter binder (case-insensitive; chirality is applied by this
script, NOT taken from the case of the letters).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_TARGET_RECORD = "trimer_DL_ABLE_B"


def _iptm_ptm(backend, target_seq, target_chir, binder_seq, binder_chir, out_dir, n_steps):  # pragma: no cover (gpu)
    entities = [
        {"type": "protein", "name": "target", "sequence": target_seq, "chirality": target_chir},
        {"type": "protein", "name": "binder", "sequence": binder_seq, "chirality": binder_chir},
    ]
    pred = backend.predict(entities, out_dir, num_diffn_timesteps=n_steps)
    return float(pred.iptm), float(pred.ptm)


def run_specular(design: str, device: str = "cuda:0", out_dir=None, seed: int = 42,
                 n_steps: int = 200) -> dict:  # pragma: no cover (gpu)
    import tempfile

    from xenodesign.backends.chai_backend import ChaiBackend
    from xenodesign.benchmark.cases import get_case
    from xenodesign.seed import read_target_sequence

    design = design.upper()
    case = get_case("alpha")
    target_seq = read_target_sequence(case.fasta_path, name=_TARGET_RECORD)
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="xd_specular_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = ChaiBackend(device=device, seed=seed)
    t0 = time.time()
    print(f"[specular] design (D) {design}  vs L-target {len(target_seq)}aa  | device {device}")
    # original: L-target . D-binder (the actual design)
    iptm_o, ptm_o = _iptm_ptm(backend, target_seq, "L", design, "D", out_dir / "orig_Ltgt_Dbinder", n_steps)
    # mirror: D-target . L-binder (the exact specular image)
    iptm_m, ptm_m = _iptm_ptm(backend, target_seq, "D", design, "L", out_dir / "mirror_Dtgt_Lbinder", n_steps)

    gap = abs(iptm_o - iptm_m)
    result = {
        "design": design,
        "iptm_orig_Ltarget_Dbinder": round(iptm_o, 4),
        "iptm_mirror_Dtarget_Lbinder": round(iptm_m, 4),
        "iptm_gap": round(gap, 4),
        "ptm_orig": round(ptm_o, 4),
        "ptm_mirror": round(ptm_m, 4),
        "mirror_self_consistent": gap <= 0.05,  # ~Chai's ipTM non-determinism band
        "interpretation": "gap~0 => mirror-consistent; large gap => Chai L-handed bias",
        "wall_time_s": round(time.time() - t0, 1),
        "out_dir": str(out_dir),
    }
    (out_dir / "specular_result.json").write_text(json.dumps(result, indent=2))
    print(f"[specular] orig {iptm_o:.4f} | mirror {iptm_m:.4f} | gap {gap:.4f} "
          f"| {'CONSISTENT' if result['mirror_self_consistent'] else 'L-BIAS GAP'}")
    return result


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="specular-pair internal-consistency control (T16)")
    p.add_argument("--design", required=True, help="binder one-letter sequence (chirality applied here)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--smoke", action="store_true", help="40 timesteps")
    return p.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover (gpu)
    import os
    args = _parse_args()
    steps = 40 if args.smoke else args.n_steps
    out = Path(args.out_dir) if args.out_dir else Path(f"/home/tmp/xd_specular_{os.getpid()}")
    run_specular(args.design, device=args.device, out_dir=out, seed=args.seed, n_steps=steps)
    sys.exit(0)
