"""Harvest the generate-and-select pool across N α trajectories (Gate-A anti-survivorship).

Reads every run dir's alpha_result.json, pools ALL per-iter candidates, reports the FULL
cross-trajectory distribution (not the survivor), and ranks the chirality-clean +
composition-passing candidates by ipTM. The top-K are the designs to re-score multi-seed +
run controls on. Pure CPU (reads JSON + recomputes composition); no GPU.

    python scripts/harvest_alpha.py XenoDesign1_local_ref/campaign/seed_*  --top 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", help="campaign run dirs (each with alpha_result.json)")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--chir_max", type=float, default=0.10)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    from xenodesign.eval.controls import composition_matched_scramble  # noqa: F401 (ensure module ok)
    from scripts.design_alpha import composition_violation

    pool = []          # every non-vetoed-by-chirality candidate across all trajectories
    all_chir = []      # FULL chirality list across every iter of every trajectory (anti-survivorship)
    n_traj = 0
    for rd in args.run_dirs:
        rj = Path(rd) / "alpha_result.json"
        if not rj.exists():
            print(f"  [skip] no result.json in {rd}")
            continue
        n_traj += 1
        d = json.loads(rj.read_text())
        seed = d.get("seed", "?")
        for t in d.get("trajectory", []):
            chir = t.get("chirality")
            if chir is not None:
                all_chir.append(chir)
            seq = t.get("l_seq", "")
            comp_bad = composition_violation(seq) if seq else True
            cand = {
                "run": str(rd), "iter": t.get("iter"), "l_seq": seq,
                "iptm": t.get("iptm"), "chirality": chir,
                "composite": t.get("composite"), "composition_violation": comp_bad,
                "chai_out": str(Path(rd) / "loop" / f"iter_{t.get('iter'):03d}" / "chai_out"),
            }
            # Keep only chirality-clean AND composition-passing for the ranked pool.
            if chir is not None and chir <= args.chir_max and not comp_bad and t.get("iptm") is not None:
                pool.append(cand)

    pool.sort(key=lambda c: c["iptm"], reverse=True)
    n_pass = sum(1 for c in all_chir if c <= args.chir_max)
    summary = {
        "n_trajectories": n_traj,
        "n_total_iters": len(all_chir),
        "chirality_pass_fraction": (n_pass / len(all_chir)) if all_chir else None,
        "chirality_mean": (sum(all_chir) / len(all_chir)) if all_chir else None,
        "chirality_max": max(all_chir) if all_chir else None,
        "n_harvested_clean_diverse": len(pool),
        "harvest_yield": (len(pool) / len(all_chir)) if all_chir else None,
        "top": pool[: args.top],
        "best_iptm": pool[0]["iptm"] if pool else None,
        "best_seq": pool[0]["l_seq"] if pool else None,
    }
    out = Path(args.out) if args.out else (Path(args.run_dirs[0]).parent / "harvest_summary.json")
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*78}\nCAMPAIGN HARVEST ({n_traj} trajectories, {len(all_chir)} total iters)")
    print(f"  chirality: pass_fraction {summary['chirality_pass_fraction']}  mean {summary['chirality_mean']}  max {summary['chirality_max']}")
    print(f"  harvested chirality-clean + composition-passing: {len(pool)}  (yield {summary['harvest_yield']})")
    print(f"\n  TOP {args.top} by ipTM:")
    for c in pool[: args.top]:
        print(f"   ipTM {round(c['iptm'],4)}  chir {c['chirality']}  {c['l_seq']}  ({Path(c['run']).name} iter {c['iter']})")
    print(f"\n  -> {out}\n{'='*78}")
    return summary


if __name__ == "__main__":
    main()
