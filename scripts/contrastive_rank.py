"""Score contrastive (register-shift) decoy predictions and rank designs by register-specificity (T01/T18).

For each design: margin = mixed_objective(real) - max_over_shifts mixed_objective(shift)  (worst-case decoy,
per the ADR-008 honesty rule). Also reports the ipTM contrastive margin. A positive margin => the binder binds
its register better than any shifted register = register-specific (within Chai). Reads the dirs produced by
contrastive_decoys.py + predict_batch.py.

Usage (chai container, freesasa+gemmi):
  python scripts/contrastive_rank.py --root <out_root> --target_na 41 --out <rank.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_complex import structural, confidence   # noqa: E402
from mixed_objective import score as obj_score      # noqa: E402

import numpy as np  # noqa: E402


def _best_cif(chai):
    import re
    sf = sorted(Path(chai).glob("scores.model_idx_*.npz"))
    bi = max(sf, key=lambda f: float(np.load(f)["aggregate_score"].reshape(-1)[0]))
    idx = re.search(r"idx_(\d+)", bi.name).group(1)
    return str(Path(chai) / f"pred.model_idx_{idx}.cif")


def _score_dir(d, na):
    chai = Path(d) / "chai_out"
    if not chai.exists():
        return None
    cif = _best_cif(chai)
    panel = structural(cif, "A", "B")          # A = binder, B = target (predict_complex order)
    panel.update(confidence(str(chai), na))
    o, _ = obj_score(panel)
    return {"obj": round(o, 3), "iptm": panel.get("iptm"), "bsa": panel.get("bsa_A2"),
            "sc": panel.get("sc_normal_opp")}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--na", type=int, default=21, help="chain-A (binder) token count for ipAE/ipSAE split")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rows = []
    for design_dir in sorted(Path(a.root).iterdir()):
        if not design_dir.is_dir():
            continue
        real = _score_dir(design_dir / "real", a.na)
        if real is None:
            continue
        shifts = {}
        for sd in sorted(design_dir.glob("shift*")):
            sc = _score_dir(sd, a.na)
            if sc:
                shifts[sd.name] = sc
        if not shifts:
            continue
        worst_obj = max(s["obj"] for s in shifts.values())
        worst_iptm = max((s["iptm"] or 0) for s in shifts.values())
        rows.append({
            "design": design_dir.name,
            "real_obj": real["obj"], "worst_shift_obj": worst_obj,
            "obj_margin": round(real["obj"] - worst_obj, 3),
            "real_iptm": real["iptm"], "worst_shift_iptm": round(worst_iptm, 3),
            "iptm_margin": round((real["iptm"] or 0) - worst_iptm, 3),
            "register_specific": real["obj"] - worst_obj > 0,
            "shifts": shifts,
        })
    rows.sort(key=lambda r: -r["obj_margin"])
    print(f"{'design':18} {'real_obj':>8} {'worstShift':>10} {'OBJmargin':>9} {'iptm_m':>7} {'spec':>5}")
    for r in rows:
        print(f"{r['design']:18} {r['real_obj']:>8} {r['worst_shift_obj']:>10} {r['obj_margin']:>9} "
              f"{r['iptm_margin']:>7} {str(r['register_specific']):>5}")
    n_spec = sum(r["register_specific"] for r in rows)
    print(f"\nregister-specific (obj_margin>0): {n_spec}/{len(rows)}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main()
