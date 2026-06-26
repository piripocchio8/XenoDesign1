"""Re-fit the mixed objective on the STRUCTURALLY-COMPLIANT subset, side by side with all-11.

Goal 1 (structure_compliance.py) showed only some of the 11 fit systems have a Chai-predicted
"real" complex that actually recapitulates the deposited interface. The ADR-026 objective was fit
on all 11 -- including ~4 whose "real" panel is a mis-docked pose (DockQ Incorrect). Compute
the compliant-subset re-fit AND keep the all-11 side by side (decide later, also pending the focused
re-predictions of the failing systems).

This reuses the EXACT fitter algorithm (mean of intra-system deltas -> clip negatives -> renormalize)
applied to the per-system separations already stored in fit_expanded.json, so the only thing that
changes between columns is WHICH systems are averaged. No re-scoring, no GPU.

  PYTHONPATH=$PWD python3 scripts/refit_compliant_subset.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIT = REPO / "XenoDesign1_local_ref" / "benchmarks" / "fit_expanded.json"
OUT = REPO / "docs" / "results" / "2026-06-23-refit-compliant-subset.json"
# NEW 3L35 separation from the FOCUSED (<=3 pocket-restraint) re-predictions that rescued
# 3L35 structurally (DockQ 0.50). Computed by scripts/_score_focused_3L35.py ->
# XenoDesign1_local_ref/benchmarks/panels_focused/focused_3L35_separation.json
FOCUSED_3L35 = REPO / "XenoDesign1_local_ref" / "benchmarks" / "panels_focused" / "focused_3L35_separation.json"

FEATURES = ["bsa", "contacts", "pack", "sc", "ipsae", "iptm", "ipae", "hbond"]

# Goal-1 structural-compliance tiers (best DockQ; see 2026-06-23-structure-compliance.md)
ALL11 = ["1CZQ", "2Q3I", "2R3C", "2R5B", "3IWY", "3L35", "3LNJ", "3MGN", "7KJM", "7YH8v2", "8GQP"]
COMPLIANT7 = ["3LNJ", "7KJM", "3IWY", "7YH8v2", "1CZQ", "2Q3I", "2R5B"]   # DockQ Medium+
# compliant8 = compliant7 + 3L35-rescued (focused restraints -> DockQ 0.50). 3L35 uses the
# NEW focused separation; the other 7 keep their fit_expanded per_system_separation.
COMPLIANT8 = COMPLIANT7 + ["3L35"]
RECAP4 = ["3LNJ", "7KJM", "3IWY", "7YH8v2"]                               # DockQ Med-High, fnat>=0.5, LRMSD 2-3A
FAILING4 = ["3L35", "3MGN", "2R3C", "8GQP"]                               # excluded (mis-docked)


def fit_subset(per_sys, systems):
    """Fitter algorithm: per-term mean over `systems` -> clip<0 -> renormalize to sum 1."""
    means = {f: sum(per_sys[s][f] for s in systems) / len(systems) for f in FEATURES}
    raw = {f: max(0.0, means[f]) for f in FEATURES}
    tot = sum(raw.values()) or 1.0
    weights = {f: round(raw[f] / tot, 4) for f in FEATURES}
    return means, weights


def main():
    d = json.loads(FIT.read_text())
    per_sys = dict(d["per_system_separation"])

    # Overlay 3L35 with the NEW focused separation for the compliant8 column ONLY.
    # We compute compliant8 against a per_sys copy whose 3L35 entry is the focused vector.
    focused = json.loads(FOCUSED_3L35.read_text())["separation"]
    per_sys_focused = dict(per_sys)
    per_sys_focused["3L35"] = {f: focused[f] for f in FEATURES}

    cols = {"all11": ALL11, "compliant7": COMPLIANT7, "compliant8": COMPLIANT8, "recap4": RECAP4}
    out = {"subsets": {k: v for k, v in cols.items()}, "failing_excluded": FAILING4,
           "compliant8_3L35_separation_source": "focused (DockQ 0.50)", "fits": {}}
    for name, systems in cols.items():
        ps = per_sys_focused if name == "compliant8" else per_sys
        means, weights = fit_subset(ps, systems)
        out["fits"][name] = {"n": len(systems), "per_feature_intra_mean": {f: round(means[f], 4) for f in FEATURES},
                             "weights": weights}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    # --- pretty side-by-side ---
    print("Per-feature intra-system mean separation (mean of within-system real-minus-scram deltas):\n")
    print(f"{'feature':9} {'all-11':>9} {'compliant-7':>12} {'compliant-8':>12} {'recap-4':>9}")
    for f in FEATURES:
        print(f"{f:9} {out['fits']['all11']['per_feature_intra_mean'][f]:>9.4f} "
              f"{out['fits']['compliant7']['per_feature_intra_mean'][f]:>12.4f} "
              f"{out['fits']['compliant8']['per_feature_intra_mean'][f]:>12.4f} "
              f"{out['fits']['recap4']['per_feature_intra_mean'][f]:>9.4f}")

    print("\nFitted weights (clip-neg, renormalize):\n")
    print(f"{'feature':9} {'all-11':>9} {'compliant-7':>12} {'compliant-8':>12} {'recap-4':>9}")
    for f in FEATURES:
        w = out["fits"]
        a = w['all11']['weights'][f]; c7 = w['compliant7']['weights'][f]
        c8 = w['compliant8']['weights'][f]; r = w['recap4']['weights'][f]
        star = "  <-- 8 vs 7 moves" if abs(c8 - c7) >= 0.03 else ""
        print(f"{f:9} {a:>9.4f} {c7:>12.4f} {c8:>12.4f} {r:>9.4f}{star}")

    print(f"\nall-11 (ADR-026): " + "  ".join(f"{f} {out['fits']['all11']['weights'][f]:.2f}"
          for f in FEATURES if out['fits']['all11']['weights'][f] > 0))
    print(f"compliant-7    : " + "  ".join(f"{f} {out['fits']['compliant7']['weights'][f]:.2f}"
          for f in FEATURES if out['fits']['compliant7']['weights'][f] > 0))
    print(f"compliant-8    : " + "  ".join(f"{f} {out['fits']['compliant8']['weights'][f]:.2f}"
          for f in FEATURES if out['fits']['compliant8']['weights'][f] > 0))
    print(f"recap-4        : " + "  ".join(f"{f} {out['fits']['recap4']['weights'][f]:.2f}"
          for f in FEATURES if out['fits']['recap4']['weights'][f] > 0))
    print(f"\nJSON -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
