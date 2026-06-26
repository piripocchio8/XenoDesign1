"""Reconstruct the ABC pilot convergence curve from per-eval chai outputs.

For each abc_eval_NNNNN dir (in evaluation order), recompute the bee nectar with the SAME
objective the live fitness uses: 0.7*pTM + 0.3*termini_proximity, K*=15, closure-restrained.
Emits the running-best (best_nectar vs eval index) — the per-eval convergence curve — plus the
best design (sequence + chirality + termini geometry). Pure post-hoc read of on-disk artifacts.
"""
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_PEPTIDE_BOND_A = 1.33
_PROXIMITY_SCALE = 6.0


def termini_proximity(cn):
    if cn is None:
        return 0.0
    return math.exp(-max(0.0, float(cn) - _PEPTIDE_BOND_A) / _PROXIMITY_SCALE)


def best_score_npz(chai_out):
    fs = sorted(glob.glob(str(Path(chai_out) / "scores.model_idx_*.npz")))
    if not fs:
        return None, None
    best_f, best_agg = None, -1e30
    for f in fs:
        d = np.load(f)
        agg = float(np.asarray(d["aggregate_score"]).reshape(-1)[0])
        if agg > best_agg:
            best_agg, best_f = agg, f
    d = np.load(best_f)
    idx = int(re.search(r"idx_(\d+)", Path(best_f).name).group(1))
    ptm = float(np.asarray(d["ptm"]).reshape(-1)[0])
    return max(0.0, min(1.0, ptm)), idx


def cn_distance_from_cif(cif_path, chain_name="A"):
    try:
        import gemmi
    except Exception:
        return None
    st = gemmi.read_structure(str(cif_path))
    res = []
    for model in st:
        for chain in model:
            if chain.name != chain_name:
                continue
            for r in chain:
                if r.name in ("HOH", "ZN"):
                    continue
                if r.find_atom("CA", "*") is None:
                    continue
                res.append(r)
        break
    if len(res) < 2:
        return None
    def xyz(r, n):
        a = r.find_atom(n, "*")
        return None if a is None else np.array([a.pos.x, a.pos.y, a.pos.z])
    c = xyz(res[-1], "C")
    nn = xyz(res[0], "N")
    if c is None or nn is None:
        return None
    return float(np.linalg.norm(c - nn))


def fasta_seq(eval_dir):
    p = Path(eval_dir) / "input.fasta"
    if not p.exists():
        return None
    return "".join(l.strip() for l in p.read_text().splitlines() if not l.startswith(">"))


def main(out_dir, w_ptm=0.7, w_termini=0.3):
    out_dir = Path(out_dir)
    eval_dirs = sorted(glob.glob(str(out_dir / "abc_evals" / "abc_eval_*")),
                       key=lambda d: int(re.search(r"_(\d+)$", d).group(1)))
    curve = []  # (eval_idx, nectar, running_best)
    best = {"nectar": -1e30}
    for i, ed in enumerate(eval_dirs, 1):
        chai_out = Path(ed) / "chai_out"
        ptm, idx = best_score_npz(chai_out)
        if ptm is None:
            continue
        cif = chai_out / f"pred.model_idx_{idx}.cif"
        cn = cn_distance_from_cif(cif) if cif.exists() else None
        nectar = w_ptm * ptm + w_termini * termini_proximity(cn)
        if nectar > best["nectar"]:
            best = {"nectar": nectar, "eval": i, "ptm": ptm, "cn": cn,
                    "prox": termini_proximity(cn), "fasta": fasta_seq(ed), "dir": ed}
        curve.append({"eval": i, "ptm": ptm, "cn": cn, "nectar": nectar,
                      "running_best": best["nectar"]})
    res_json = out_dir / "abc_result.json"
    result = json.loads(res_json.read_text()) if res_json.exists() else {}
    summary = {
        "out_dir": str(out_dir),
        "variant": result.get("abc_variant"),
        "selected_nectar_engine": result.get("selected_nectar"),
        "selected_d_fasta": result.get("selected_d_fasta"),
        "selected_chirality_pattern": result.get("selected_chirality_pattern"),
        "n_evals_on_disk": len(curve),
        "best_reconstructed": best,
        "curve": curve,
    }
    analysis_dir = Path("abc_runs/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / f"convergence_{out_dir.name}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}, indent=2, default=str))
    print("--- running-best by eval ---")
    for c in curve:
        cn = f"{c['cn']:.2f}" if c['cn'] is not None else "None"
        print(f"eval {c['eval']:3d}  pTM {c['ptm']:.3f}  cn {cn:>6}  nectar {c['nectar']:.4f}  best {c['running_best']:.4f}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1])
