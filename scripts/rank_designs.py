"""Rank design_alpha sweep outputs by what matters for an all-D HLH/H binder (P3): clean-D chirality,
helix cross-angle / topology (coil vs bundle), correct Tyr-29 outer face, and the mixed objective. Per
ADR-018 this is a standalone driver. Reads each run's alpha_result.json, finds the selected iteration's
complex CIF (loop/iter_<sel>/chai_out, best aggregate), and aggregates analyze_interface_face + score_complex
+ mixed_objective.

Usage (in the chai container, freesasa+gemmi): python scripts/rank_designs.py <dir1> <dir2> ... --out rank.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_interface_face import analyze          # noqa: E402
from score_complex import structural, confidence    # noqa: E402
from mixed_objective import score as obj_score       # noqa: E402


def best_cif(chai):
    sf = sorted(Path(chai).glob("scores.model_idx_*.npz"))
    bi = max(sf, key=lambda f: float(np.load(f)["aggregate_score"].reshape(-1)[0]))
    idx = re.search(r"idx_(\d+)", bi.name).group(1)
    return Path(chai) / f"pred.model_idx_{idx}.cif"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rows = []
    for d in a.dirs:
        res = json.load(open(Path(d) / "alpha_result.json"))
        it = res["selected_iter"]
        chai = Path(d) / f"loop/iter_{it:03d}/chai_out"
        if not chai.exists():
            print(f"[skip] {Path(d).name}: no chai_out for iter {it}")
            continue
        cif = best_cif(chai)
        face = analyze(cif, "B", "A", Path(d).name)           # binder=B, target=A (run convention)
        panel = structural(str(cif), "A", "B")
        panel.update(confidence(str(chai), 41))               # target=chain A = 41 tokens
        obj, _ = obj_score(panel)
        ang = [x for x in (face["angle_binder_vs_HLH_helix1_deg"], face["angle_binder_vs_HLH_helix2_deg"]) if x is not None]
        rows.append({
            "design": Path(d).name, "iptm": round(res["selected_iptm"], 3),
            "chir": res["selected_chirality"], "orientation": face["orientation"],
            "min_cross_angle_deg": round(min(ang), 1) if ang else None,
            "face": face["nearer_face"], "binder_helix": face["binder_helix_fraction"],
            "bsa_A2": panel.get("bsa_A2"), "sc": panel.get("sc_normal_opp"),
            "n_res_contacts": panel.get("n_residue_contacts"), "iptm_panel": panel.get("iptm"),
            "objective": round(obj, 3), "seq": res["selected_l_seq"],
        })
    # rank: clean-D first, then bundle (low cross-angle), correct Tyr face, then objective
    rows.sort(key=lambda r: (r["chir"] > 0.1, r["orientation"] != "parallel/bundle",
                             r["face"] != "Tyr29(outside)", -r["objective"]))
    print(f"{'design':18} {'iptm':>5} {'chir':>4} {'orient':>16} {'Xang':>5} {'face':>14} {'bsa':>6} {'sc':>5} {'obj':>5}")
    for r in rows:
        print(f"{r['design']:18} {r['iptm']:>5} {r['chir']:>4} {str(r['orientation']):>16} "
              f"{str(r['min_cross_angle_deg']):>5} {str(r['face']):>14} {str(r['bsa_A2']):>6} {str(r['sc']):>5} {r['objective']:>5}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main()
