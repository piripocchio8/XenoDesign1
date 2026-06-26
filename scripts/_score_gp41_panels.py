"""Score the gp41 panels: best model per item/seed by aggregate_score, all 8 terms + confidence.
Writes panels in the same dir/naming as the MDM2 panels in panels_expanded/.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for xenodesign
import score_complex as SC

NA = {"3L35": 16, "2R5B": 15, "2R3C": 15, "3MGN": 15, "1CZQ": 16, "2Q3I": 16}
ROOT = Path("XenoDesign1_local_ref/benchmarks/gp41_mdm2_r2")
OUT = Path("XenoDesign1_local_ref/benchmarks/panels_expanded")


def best_idx(chai_dir):
    bv, bi = -np.inf, None
    for f in sorted(glob.glob(str(chai_dir / "scores.model_idx_*.npz"))):
        a = float(np.load(f)["aggregate_score"].reshape(-1)[0])
        if a > bv:
            bv, bi = a, int(re.search(r"idx_(\d+)", f).group(1))
    return bi, bv


def main():
    written = []
    for sysid in ["3L35", "2R5B", "2R3C", "3MGN", "1CZQ", "2Q3I"]:
        na = NA[sysid]
        for item in ["real", "scram1", "scram2"]:
            for seed in [42, 43, 44]:
                chai_dir = ROOT / sysid / item / f"seed{seed}" / "chai_out"
                if not chai_dir.is_dir():
                    print("MISSING", chai_dir)
                    continue
                bi, bv = best_idx(chai_dir)
                cif = chai_dir / f"pred.model_idx_{bi}.cif"
                res = SC.structural(str(cif), "A", "B")
                res.update(SC.confidence(str(chai_dir), na))
                res.update({
                    "system": sysid, "item": item, "seed": seed, "na": na,
                    "best_model_idx": bi, "best_cif": cif.name,
                    "aggregate_score": round(bv, 4),
                })
                outp = OUT / f"gp41_{sysid}__{item}__seed{seed}.json"
                outp.write_text(json.dumps(res, indent=2))
                written.append((outp.name, res.get("sc_normal_opp"), res.get("iptm"),
                                res.get("ipsae"), res.get("ipae"), res.get("bsa_A2")))
    print(f"\nWrote {len(written)} panels.")
    for n, sc, iptm, ipsae, ipae, bsa in written:
        print(f"  {n:45s} sc={sc} iptm={iptm} ipsae={ipsae} ipae={ipae} bsa={bsa}")


if __name__ == "__main__":
    main()
