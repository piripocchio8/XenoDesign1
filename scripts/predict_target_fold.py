"""Predict the α target (HLH) ALONE, L or D, and report fold COMPACTNESS — to test the
observation that Chai folds the mirrored D-HLH to one long helix, and whether it is a no-MSA /
trunk-recycle artifact rather than a real D failure.

A 41-res single long helix -> end-to-end ~60 Å; a folded HLH (helix-loop-helix) bends back ->
end-to-end short + smaller Rg. We compare L vs D at default and higher trunk recycles, MSA-free
(exactly the specular setting). Calls run_inference directly so num_trunk_recycles is exposed.

Run (GPU; chai 0.6.1 container):
  python scripts/predict_target_fold.py --chirality D --recycles 3  --out_dir .../dhlh_r3
  python scripts/predict_target_fold.py --chirality D --recycles 10 --out_dir .../dhlh_r10
  python scripts/predict_target_fold.py --chirality L --recycles 3  --out_dir .../lhlh_r3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_TARGET_RECORD = "trimer_DL_ABLE_B"


def run(chirality: str, recycles: int, steps: int, device: str, out_dir,
        seq_override: str | None = None, seed: int = 42):  # pragma: no cover (gpu)
    from chai_lab.chai1 import run_inference

    from xenodesign.backends.chai_backend import _save_confidence_npz, write_inputs
    from xenodesign.benchmark.cases import get_case
    from xenodesign.seed import read_target_sequence

    seq = seq_override or read_target_sequence(get_case("alpha").fasta_path, name=_TARGET_RECORD)
    ent = [{"type": "protein", "name": "target", "sequence": seq, "chirality": chirality}]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta = write_inputs(ent, out_dir)
    chai_out = out_dir / "chai_out"
    chai_out.mkdir(parents=True, exist_ok=True)
    cands = run_inference(fasta_file=fasta, output_dir=chai_out, device=device, seed=seed,
                          num_diffn_timesteps=steps, num_trunk_recycles=recycles,
                          use_esm_embeddings=True, use_msa_server=False)
    _save_confidence_npz(cands, chai_out)

    import gemmi

    from xenodesign.secondary_structure import helix_fraction
    sf = sorted(chai_out.glob("scores.model_idx_*.npz"))
    bi, bv = 0, -np.inf
    for f in sf:
        a = float(np.load(f)["aggregate_score"].reshape(-1)[0])
        if a > bv:
            bv, bi = a, int(re.search(r"idx_(\d+)", f.name).group(1))
    cif = chai_out / f"pred.model_idx_{bi}.cif"
    st = gemmi.read_structure(str(cif))
    ca = []
    for m in st:
        for ch in m:
            for r in ch:
                x = r.find_atom("CA", "*")
                if x is not None:
                    ca.append([x.pos.x, x.pos.y, x.pos.z])
        break
    ca = np.asarray(ca, float)
    e2e = float(np.linalg.norm(ca[0] - ca[-1]))
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
    ptm = None
    sc = chai_out / f"scores.model_idx_{bi}.npz"
    if sc.exists():
        d = np.load(sc)
        if "ptm" in d:
            ptm = float(np.asarray(d["ptm"]).reshape(-1)[0])
    result = {
        "chirality": chirality, "recycles": recycles, "steps": steps, "seed": seed, "msa": False,
        "n_res": int(len(ca)), "ptm": None if ptm is None else round(ptm, 3),
        "end_to_end_A": round(e2e, 1), "radius_of_gyration_A": round(rg, 1),
        "helix_fraction": round(float(helix_fraction(ca)), 3),
        "fold": "ONE LONG HELIX" if e2e > 45 else ("compact / HLH-like" if e2e < 30 else "intermediate"),
        "cif": str(cif),
    }
    (out_dir / "fold_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def _parse(argv=None):
    p = argparse.ArgumentParser(description="predict the HLH target alone (L/D) + compactness")
    p.add_argument("--chirality", choices=["L", "D"], required=True)
    p.add_argument("--recycles", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sequence", default=None, help="override: predict THIS sequence (else the α target)")
    p.add_argument("--seed", type=int, default=42, help="diffusion seed (for seed-robustness checks)")
    return p.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover (gpu)
    a = _parse()
    run(a.chirality, a.recycles, a.steps, a.device, a.out_dir, seq_override=a.sequence, seed=a.seed)
    sys.exit(0)
