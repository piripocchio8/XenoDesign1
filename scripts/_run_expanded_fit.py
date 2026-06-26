"""Assemble the 11-system expanded re-fit (scrambles-only) and invoke fit_objective.fit().

Systems (11):
  8GQP, 7YH8v2          -> panels_t20_restrained  (real + scram1/scram2 ONLY, no shift*)
  gp41 6 (3L35 2R5B 2R3C 3MGN 1CZQ 2Q3I), MDM2 3 (3IWY 3LNJ 7KJM) -> panels_expanded

pos = real panels per system ; neg = scram1 + scram2 per system ; group by SYSTEM.
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_objective as FO
from mixed_objective import normalize

T20 = Path("XenoDesign1_local_ref/benchmarks/panels_t20_restrained")
EXP = Path("XenoDesign1_local_ref/benchmarks/panels_expanded")

# (system_id, glob_for_real, glob_for_scrams)
SPECS = [
    ("8GQP",   sorted(T20.glob("8GQP__real__*.json")),
               sorted(T20.glob("8GQP__scram1__*.json")) + sorted(T20.glob("8GQP__scram2__*.json"))),
    ("7YH8v2", sorted(T20.glob("7YH8v2__real__*.json")),
               sorted(T20.glob("7YH8v2__scram1__*.json")) + sorted(T20.glob("7YH8v2__scram2__*.json"))),
]
for s in ["3L35", "2R5B", "2R3C", "3MGN", "1CZQ", "2Q3I"]:
    SPECS.append((s, sorted(EXP.glob(f"gp41_{s}__real__*.json")),
                  sorted(EXP.glob(f"gp41_{s}__scram1__*.json")) + sorted(EXP.glob(f"gp41_{s}__scram2__*.json"))))
for s in ["3IWY", "3LNJ", "7KJM"]:
    SPECS.append((s, sorted(EXP.glob(f"MDM2_{s}__real__*.json")),
                  sorted(EXP.glob(f"MDM2_{s}__scram1__*.json")) + sorted(EXP.glob(f"MDM2_{s}__scram2__*.json"))))


def main():
    pos_entries, neg_entries = [], []
    print("System assembly (pos/neg counts):")
    for sysid, reals, scrams in SPECS:
        assert reals, f"{sysid}: no real panels"
        assert scrams, f"{sysid}: no scram panels"
        print(f"  {sysid:8s} pos={len(reals)} neg={len(scrams)}")
        for p in reals:
            pos_entries.append((sysid, p.name, normalize(json.load(open(p))), "real"))
        for p in scrams:
            kind = "scramble"
            neg_entries.append((sysid, p.name, normalize(json.load(open(p))), kind))

    feats = FO.active_features(pos_entries + neg_entries)
    out = FO.fit(pos_entries, neg_entries, feats)
    out["neg_kind_filter"] = "scramble"
    out["n_systems"] = len(SPECS)
    Path("XenoDesign1_local_ref/benchmarks/fit_expanded.json").write_text(json.dumps(out, indent=2))
    print("\nFeatures:", feats)
    print("Wrote XenoDesign1_local_ref/benchmarks/fit_expanded.json")
    print(json.dumps({
        "fitted_weights": out["fitted_weights"],
        "per_feature_intra_mean": out["per_feature_intra_mean"],
        "single_metric_ranking": out["single_metric_ranking"],
        "per_system_margins": out["per_system_margins"],
        "all_systems_separate": out["all_systems_separate"],
    }, indent=2))


if __name__ == "__main__":
    main()
