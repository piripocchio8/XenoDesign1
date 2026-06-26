"""Score the FOCUSED (<=3 pocket-restraint) 3L35 predictions and compute the NEW
real-vs-scram per-term separation vector (8 normalized terms).

Mirrors _score_gp41_panels.py exactly: best chai model per seed/item by aggregate_score,
structural() + confidence(), then mixed_objective.normalize() for the 8 terms.

Separation per term = mean_norm(real over 3 seeds) - mean_norm(scram over 6 seedxscram panels).

  PYTHONPATH=$PWD python3 scripts/_score_focused_3L35.py
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import score_complex as SC
from mixed_objective import normalize

NA = 16
FEATURES = ["bsa", "contacts", "pack", "sc", "ipsae", "iptm", "ipae", "hbond"]
ROOT = Path("XenoDesign1_local_ref/benchmarks/focused/3L35")
OUT = Path("XenoDesign1_local_ref/benchmarks/panels_focused")


def best_idx(chai_dir):
    bv, bi = -np.inf, None
    for f in sorted(glob.glob(str(chai_dir / "scores.model_idx_*.npz"))):
        a = float(np.load(f)["aggregate_score"].reshape(-1)[0])
        if a > bv:
            bv, bi = a, int(re.search(r"idx_(\d+)", f).group(1))
    return bi, bv


def panel(chai_dir):
    bi, bv = best_idx(chai_dir)
    cif = chai_dir / f"pred.model_idx_{bi}.cif"
    res = SC.structural(str(cif), "A", "B")
    res.update(SC.confidence(str(chai_dir), NA))
    res.update({"na": NA, "best_model_idx": bi, "best_cif": cif.name,
                "aggregate_score": round(bv, 4)})
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = {"real": [42, 43, 44],
             "scram1": [42, 43, 44],
             "scram2": [42, 43, 44]}
    norms = {}  # (item,seed) -> normalized 8-term dict
    for item, seeds in items.items():
        for seed in seeds:
            cd = ROOT / item / f"seed{seed}" / "chai_out"
            res = panel(cd)
            n = normalize(res)
            norms[(item, seed)] = n
            (OUT / f"focused_3L35__{item}__seed{seed}.json").write_text(json.dumps(res, indent=2))
            print(f"{item:7} seed{seed}  "
                  + "  ".join(f"{k}={n[k]:.3f}" for k in FEATURES)
                  + f"   (raw sc={res.get('sc_normal_opp')} iptm={res.get('iptm')} "
                  + f"ipae={res.get('ipae')} ipsae={res.get('ipsae')} "
                  + f"hbden={res.get('hbond_density')} bsa={res.get('bsa_A2')})")

    real_keys = [("real", s) for s in items["real"]]
    scram_keys = [("scram1", s) for s in items["scram1"]] + [("scram2", s) for s in items["scram2"]]
    mean_real = {f: sum(norms[k][f] for k in real_keys) / len(real_keys) for f in FEATURES}
    mean_scram = {f: sum(norms[k][f] for k in scram_keys) / len(scram_keys) for f in FEATURES}
    sep = {f: round(mean_real[f] - mean_scram[f], 4) for f in FEATURES}

    print("\nmean_real :", {f: round(mean_real[f], 4) for f in FEATURES})
    print("mean_scram:", {f: round(mean_scram[f], 4) for f in FEATURES})
    print("\nNEW 3L35 separation (real - scram):")
    print(json.dumps(sep, indent=2))

    sep_path = OUT / "focused_3L35_separation.json"
    sep_path.write_text(json.dumps({
        "system": "3L35", "na": NA, "n_real": len(real_keys), "n_scram": len(scram_keys),
        "mean_real": {f: round(mean_real[f], 4) for f in FEATURES},
        "mean_scram": {f: round(mean_scram[f], 4) for f in FEATURES},
        "separation": sep,
    }, indent=2))
    print(f"\nseparation -> {sep_path}")
    return sep


if __name__ == "__main__":
    main()
