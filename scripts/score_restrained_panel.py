#!/usr/bin/env python3
"""Score a pocket-restrained register panel (one system) and report intra-system register separation.

For each manifest job of --system, pick the best chai model (max aggregate_score), run the current
score_complex (confidence + H-bonds + Sc), write a per-(item,seed) panel JSON, then aggregate
mean+/-std over seeds per item and print real-vs-worst-shift deltas per axis.

  PYTHONPATH=$PWD python3 scripts/score_restrained_panel.py 7YH8
"""
import glob, json, statistics, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "XenoDesign1_local_ref/benchmarks/restrained_panel/manifest.json"
OUTDIR = REPO / "XenoDesign1_local_ref/benchmarks/panels_t20_restrained"
ITEMS = ["real", "scram1", "shift3", "shift4", "shift7"]
KEYS = ["iptm", "ipae", "ipsae", "sc_normal_opp", "n_interchain_hbonds", "hbond_density"]
# axis -> True if HIGHER is better (real should exceed decoys); ipae is lower-is-better
HIGHER_BETTER = {"iptm": True, "ipae": False, "ipsae": True, "sc_normal_opp": True,
                 "n_interchain_hbonds": True, "hbond_density": True}


def best_model(chai_dir: Path):
    best, bs = None, -1e9
    for npz in glob.glob(str(chai_dir / "scores.model_idx_*.npz")):
        d = np.load(npz)
        s = float(np.asarray(d["aggregate_score"]).mean()) if "aggregate_score" in d else 0.0
        idx = npz.split("model_idx_")[1].split(".npz")[0]
        if s > bs:
            bs, best = s, idx
    return best


def agg(items, key):
    vals = [r.get(key) for r in items if r.get(key) is not None]
    if not vals:
        return (None, None)
    return (round(statistics.mean(vals), 3), round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0)


def main(system, manifest=MANIFEST):
    jobs = json.load(open(manifest))
    jobs = jobs if isinstance(jobs, list) else jobs.get("jobs", [])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = {}
    for j in jobs:
        if j["system"] != system:
            continue
        cd = REPO / j["out_dir"] / "chai_out"
        bm = best_model(cd)
        if bm is None:
            print(f"  WARN no model in {cd}")
            continue
        cif = cd / f"pred.model_idx_{bm}.cif"
        pj = OUTDIR / f"{system}__{j['item']}__seed{j['seed']}.json"
        r = subprocess.run([sys.executable, str(REPO / "scripts/score_complex.py"),
                            "--cif", str(cif), "--chain_a", "A", "--chain_b", "B",
                            "--chai_dir", str(cd), "--na", str(j.get("na", 62)), "--out", str(pj)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ERR {system}/{j['item']}/seed{j['seed']}: {r.stderr.strip()[-200:]}")
            continue
        p = json.load(open(pj))
        p["_best_model"] = bm
        rows.setdefault(j["item"], []).append(p)

    print(f"\nsystem {system}: per-item mean(+/-std) over seeds")
    for item in ITEMS:
        if item not in rows:
            continue
        its = rows[item]
        s = "  ".join(f"{k}={agg(its,k)[0]}±{agg(its,k)[1]}" for k in KEYS)
        print(f"  {item:7s} n={len(its)}  {s}")

    print(f"\nsystem {system}: real-vs-worst-shift intra delta (>0 => real separates; worst-shift = most-favorable decoy)")
    real = rows.get("real", [])
    for axis in KEYS:
        rm = agg(real, axis)[0]
        sm = [agg(rows[s], axis)[0] for s in ("shift3", "shift4", "shift7")
              if s in rows and agg(rows[s], axis)[0] is not None]
        if rm is None or not sm:
            continue
        if HIGHER_BETTER[axis]:
            worst = max(sm); delta = round(rm - worst, 3)
        else:
            worst = min(sm); delta = round(worst - rm, 3)
        print(f"  {axis:20s} real={rm} worst_shift={round(worst,3)} delta={delta} {'SEP' if delta > 0 else 'no'}")


if __name__ == "__main__":
    _sys = sys.argv[1] if len(sys.argv) > 1 else "7YH8"
    _man = sys.argv[2] if len(sys.argv) > 2 else MANIFEST
    main(_sys, _man)
