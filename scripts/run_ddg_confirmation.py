"""Orthogonal-physics (OpenMM ddG-proxy) confirmation of Chai register-specificity.

For designs {seed_45,46,47,GT} (Chai register-specific) + control {seed_42}
(Chai-promiscuous), score the BEST chai model CIF (max aggregate_score) per
(design, kind, rep) with score_ddg -> E_int. Then:

  per-design ddG register margin = mean_reps[E_int(real)]
                                   - min_over_shifts( mean_reps[E_int(shift)] )

Sign convention: more-NEGATIVE E_int = better binding. A register-SPECIFIC
design has real MORE NEGATIVE than every shifted decoy -> NEGATIVE margin.
(margin < 0 == specific; margin > 0 == NOT specific / a shift binds better.)

Reports mean +- std over reps for real and the best (lowest-E) decoy.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contrastive_rank import _best_cif          # noqa: E402
from score_ddg import score                      # noqa: E402

ROOT = Path("/home/user/claude_projects/XenoDesign1/XenoDesign1_local_ref/contrastive_val")
PLAN = {  # design -> pool subdir
    "seed_45": "pool1", "seed_46": "pool1", "seed_47": "pool1", "GT": "pool1",
    "seed_42": "census_a",
}
KINDS = ["real", "shift3", "shift4", "shift7"]
SHIFTS = ["shift3", "shift4", "shift7"]
REPS = ["rep0", "rep1", "rep2"]
# Chain layout in these CIFs: A = D-binder, B = L-target.
BINDER, TARGET = "A", "B"
# Chai ipTM contrastive margins (given), positive = Chai says register-specific.
CHAI_IPTM_MARGIN = {
    "seed_47": 0.165, "seed_46": 0.156, "seed_45": 0.103, "GT": 0.058, "seed_42": -0.063,
}


def score_one(design, pool, rep, kind):
    chai = ROOT / pool / rep / design / kind / "chai_out"
    cif = _best_cif(chai)
    res = score(cif, target_chain=TARGET, binder_chain=BINDER)
    return cif, res


def main():
    per_cell = {}      # (design,kind) -> list of e_int over reps
    methods = set()
    detail = []
    for design, pool in PLAN.items():
        for kind in KINDS:
            vals = []
            for rep in REPS:
                cif, res = score_one(design, pool, rep, kind)
                e = res["e_int_kcal"]
                methods.add(res["method"])
                vals.append(e)
                detail.append({"design": design, "kind": kind, "rep": rep,
                               "e_int_kcal": e, "method": res["method"],
                               "cif": str(Path(cif).relative_to(ROOT))})
                print(f"{design:8} {kind:7} {rep:5} E_int={e:>10.3f}  [{res['method']}]")
            per_cell[(design, kind)] = vals

    fallback_used = any("fallback" in m for m in methods)

    results = {"sign_convention": ("more-negative E_int = better binding; "
                                   "register-SPECIFIC design has real MORE NEGATIVE than shifts "
                                   "=> NEGATIVE ddG margin = specific (agrees with Chai)."),
               "methods_used": sorted(methods),
               "fallback_used": fallback_used,
               "chain_layout": {"binder": BINDER, "target": TARGET},
               "designs": {}}

    print("\n=== PER-DESIGN SUMMARY ===")
    for design in PLAN:
        real = np.array(per_cell[(design, "real")], float)
        real_mean, real_std = float(real.mean()), float(real.std(ddof=0))
        shift_means = {s: float(np.mean(per_cell[(design, s)])) for s in SHIFTS}
        shift_stds = {s: float(np.std(per_cell[(design, s)], ddof=0)) for s in SHIFTS}
        # best decoy = LOWEST (most negative / most favorable) mean E_int among shifts
        best_shift = min(shift_means, key=lambda s: shift_means[s])
        best_decoy_mean = shift_means[best_shift]
        best_decoy_std = shift_stds[best_shift]
        # margin = real - best(most favorable) decoy; negative => real binds better => specific
        margin_mean = real_mean - best_decoy_mean
        # propagate std over reps for the margin (real and best-decoy are independent rep means;
        # use per-rep paired difference real - best_shift_per_rep for an honest rep std)
        best_shift_vals = np.array(per_cell[(design, best_shift)], float)
        paired = real - best_shift_vals
        margin_std = float(paired.std(ddof=0))

        chai = CHAI_IPTM_MARGIN[design]
        ddg_says_specific = margin_mean < 0
        chai_says_specific = chai > 0
        agree = (ddg_says_specific == chai_says_specific)

        results["designs"][design] = {
            "pool": PLAN[design],
            "E_int_real_mean": round(real_mean, 3), "E_int_real_std": round(real_std, 3),
            "shift_means": {k: round(v, 3) for k, v in shift_means.items()},
            "best_decoy_shift": best_shift,
            "best_decoy_E_int_mean": round(best_decoy_mean, 3),
            "best_decoy_E_int_std": round(best_decoy_std, 3),
            "ddg_register_margin_mean": round(margin_mean, 3),
            "ddg_register_margin_std": round(margin_std, 3),
            "ddg_says_register_specific": ddg_says_specific,
            "chai_iptm_margin": chai,
            "chai_says_register_specific": chai_says_specific,
            "agree": agree,
        }
        print(f"{design:8} real={real_mean:8.2f}+-{real_std:5.2f}  "
              f"bestDecoy({best_shift})={best_decoy_mean:8.2f}  "
              f"ddGmargin={margin_mean:8.2f}+-{margin_std:5.2f}  "
              f"ddG_spec={ddg_says_specific!s:5}  chai={chai:+.3f}  agree={agree}")

    results["detail"] = detail
    n_agree = sum(d["agree"] for d in results["designs"].values())
    results["n_agree"] = n_agree
    results["n_total"] = len(results["designs"])
    out = ROOT.parent / "ddg_confirmation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nAGREE {n_agree}/{len(results['designs'])}  (fallback_used={fallback_used})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
