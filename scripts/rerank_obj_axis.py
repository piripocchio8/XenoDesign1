"""Re-rank the existing contrastive pool by the OBJ axis (T20) AND the ipTM axis, side by side.

The census/pool ranked.json was produced with --margin_axis iptm, so its `obj_margin`/`worst_shift`
fields actually carry the IPTM-axis values (real_obj_mean == real_iptm_mean there). This re-scores the
already-predicted real+shift CIFs with contrastive_rank._score_dir (T20: obj + iptm in one panel) and
re-runs select_contrastive.select_by_margin on BOTH axes, so we can compare obj-axis vs ipTM-axis
register margins for the same designs WITHOUT any GPU prediction.

CPU only (reuses existing CIFs). Needs gemmi + freesasa -> run in the gradio_design Docker.

  python scripts/rerank_obj_axis.py \
      --root XenoDesign1_local_ref/contrastive_val/pool1 \
      --designs seed_45 seed_46 seed_47 GT seed_42 \
      --reps 3 --shifts 3,4,7 --na 21 --out runs/logs/contrastive_obj_axis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contrastive_rank as cr            # noqa: E402  (_score_dir)
import select_contrastive as sc          # noqa: E402  (select_by_margin)


def score_pool(root, designs, reps, na, shifts):
    """Score every (design, kind, rep) dir under <root>/rep{r}/<design>/<kind> with _score_dir.

    Returns {(design, kind, rep): {"obj", "iptm"}} for select_by_margin, plus a raw per-dir
    dump so we can show exactly what each CIF scored.
    """
    kinds = ["real"] + [f"shift{s}" for s in shifts]
    per_dir = {}
    raw = []
    for d in designs:
        for kind in kinds:
            for r in range(reps):
                rep_dir = Path(root) / f"rep{r}" / d / kind
                res = cr._score_dir(str(rep_dir), na)
                if res is None:
                    raw.append({"design": d, "kind": kind, "rep": r, "error": "no chai_out / unscorable"})
                    continue
                per_dir[(d, kind, r)] = {"obj": res["obj"], "iptm": res.get("iptm")}
                raw.append({"design": d, "kind": kind, "rep": r,
                            "obj": res["obj"], "iptm": res.get("iptm"),
                            "bsa": res.get("bsa"), "sc": res.get("sc")})
    return per_dir, raw


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--designs", nargs="+", required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--shifts", default="3,4,7")
    p.add_argument("--na", type=int, default=21)
    p.add_argument("--k", type=float, default=1.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    shifts = tuple(int(s) for s in a.shifts.split(","))

    per_dir, raw = score_pool(a.root, a.designs, a.reps, a.na, shifts)

    rows_obj = sc.select_by_margin(per_dir, reps=a.reps, k=a.k, shifts=shifts, axis="obj")
    rows_iptm = sc.select_by_margin(per_dir, reps=a.reps, k=a.k, shifts=shifts, axis="iptm")

    # compact per-design comparison: obj-axis margin/worst-shift vs iptm-axis margin/worst-shift
    by_design = {}
    for r in rows_obj:
        by_design.setdefault(r["design"], {})["obj_axis"] = {
            "real_obj_mean": round(r["real_obj_mean"], 4),
            "real_obj_std": round(r["real_obj_std"], 4),
            "worst_shift": r["worst_shift"],
            "worst_shift_obj_mean": round(r["worst_shift_obj_mean"], 4),
            "obj_margin": round(r["obj_margin"], 4),
            "obj_margin_std": round(r["obj_margin_std"], 4),
            "selected_obj_axis": r["selected"],
        }
    for r in rows_iptm:
        # on the iptm axis, select_by_margin's "obj_*" fields carry the IPTM-axis values
        by_design.setdefault(r["design"], {})["iptm_axis"] = {
            "real_iptm_mean": round(r["real_iptm_mean"], 4),
            "real_iptm_std": round(r["real_iptm_std"], 4),
            "worst_shift_iptm": r["worst_shift"],
            "worst_shift_iptm_mean": round(r["worst_shift_obj_mean"], 4),
            "iptm_margin": round(r["iptm_margin"], 4),
            "iptm_margin_std": round(r["iptm_margin_std"], 4),
            "selected_iptm_axis": r["selected"],
        }

    out = {
        "root": a.root, "na": a.na, "reps": a.reps, "shifts": list(shifts), "k": a.k,
        "comparison": by_design,
        "rows_obj_axis": rows_obj,
        "rows_iptm_axis": rows_iptm,
        "raw_per_dir": raw,
    }

    # readable table
    order = sorted(by_design, key=lambda d: -by_design[d].get("obj_axis", {}).get("obj_margin", -9))
    print(f"\n{'design':10} {'OBJaxis margin':>16} {'worstSh':>8} {'sel':>4}  | "
          f"{'IPTMaxis margin':>16} {'worstSh':>8} {'sel':>4}")
    print("-" * 86)
    for d in order:
        o = by_design[d].get("obj_axis", {})
        i = by_design[d].get("iptm_axis", {})
        print(f"{d:10} "
              f"{o.get('obj_margin', 0):+.3f}±{o.get('obj_margin_std', 0):.3f}  "
              f"{o.get('worst_shift', '?'):>8} {('Y' if o.get('selected_obj_axis') else '.'):>4}  | "
              f"{i.get('iptm_margin', 0):+.3f}±{i.get('iptm_margin_std', 0):.3f}  "
              f"{i.get('worst_shift_iptm', '?'):>8} {('Y' if i.get('selected_iptm_axis') else '.'):>4}")
    n_obj = sum(1 for d in by_design if by_design[d].get("obj_axis", {}).get("selected_obj_axis"))
    n_ipt = sum(1 for d in by_design if by_design[d].get("iptm_axis", {}).get("selected_iptm_axis"))
    print(f"\nselected obj-axis: {n_obj}/{len(by_design)}   selected iptm-axis: {n_ipt}/{len(by_design)}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.out}")
    return out


if __name__ == "__main__":
    main()
