"""Goal 2 — per-term reliability scatter: does the DEPOSIT separate from the chai-scramble per term,
and does the CHAI-predicted real reproduce that separation? (8 terms x 11 systems, for the Thomas deck.)

Intent: the scramble is ALWAYS the CHAI-1-predicted scramble. For each of the
8 normalized terms and each system, score:
  (a) the DEPOSIT real cif, scored directly  (ground-truth structure)
  (b) the CHAI-predicted real                (existing real panels)
  (c) the CHAI-predicted scramble            (existing scram1+scram2 panels)
then plot  x = norm(a) - mean norm(c)   vs   y = mean norm(b) - mean norm(c).
  near y = x  => the prediction reproduces the deposit's term separation.
  x large, y small  => the term separates on the TRUTH but the prediction loses it.

CRUCIAL CONSTRAINT: 3 of the 8 terms (iptm, ipae, ipsae) are chai CONFIDENCE outputs (npz) -- they do
NOT exist for a static deposited structure. So for the deposit only the 5 STRUCTURAL terms
(sc, bsa, contacts, pack, hbond) are scoreable; iptm/ipae/ipsae have NO deposit value (x undefined).
That is itself a finding: confidence terms are model self-estimates, not validatable against ground truth.
y (chai real-vs-scram separation) is the per-system separation already in fit_expanded.json.

CPU-only; gemmi + freesasa (via score_complex) + the existing panels.
  PYTHONPATH=$PWD python3 scripts/per_term_reliability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemmi  # noqa: E402
from gate_mdm2_restrained import _aa_residues, _chain, _one, pick_deposit_pair  # noqa: E402
from mixed_objective import normalize  # noqa: E402
import score_complex  # noqa: E402
from structure_compliance import system_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "XenoDesign1_local_ref" / "benchmarks"
PANELS_EXP = BENCH / "panels_expanded"
PANELS_T20 = BENCH / "panels_t20_restrained"
OUT_JSON = REPO / "docs" / "results" / "2026-06-23-per-term-reliability.json"
OUT_PNG = REPO / "docs" / "reports" / "2026-06-23-per-term-reliability.png"

STRUCTURAL = ["sc", "bsa", "contacts", "pack", "hbond"]   # scoreable on the deposit
CONFIDENCE = ["iptm", "ipae", "ipsae"]                    # chai-only (no deposit value)
ALL_TERMS = STRUCTURAL + CONFIDENCE

# panel file prefix per system (panels_expanded uses gp41_/MDM2_; t20 uses bare names)
GP41 = ("3L35", "2R5B", "2R3C", "3MGN", "1CZQ", "2Q3I")
MDM2 = ("3LNJ", "7KJM", "3IWY")
FAMILY = {**{s: "gp41" for s in GP41}, **{s: "MDM2" for s in MDM2},
          "8GQP": "8GQP/7YH8", "7YH8v2": "8GQP/7YH8"}
ALL_SYSTEMS = list(GP41) + list(MDM2) + ["8GQP", "7YH8v2"]


def panel_glob(system, item):
    if system in GP41:
        return sorted(PANELS_EXP.glob(f"gp41_{system}__{item}__*.json"))
    if system in MDM2:
        return sorted(PANELS_EXP.glob(f"MDM2_{system}__{item}__*.json"))
    if system in ("8GQP", "7YH8v2"):
        return sorted(PANELS_T20.glob(f"{system}__{item}__*.json"))
    return []


def mean_norm(panels):
    """Mean of normalize() over a list of panel JSON paths -> {term: mean_norm}."""
    norms = [normalize(json.loads(p.read_text())) for p in panels]
    if not norms:
        return None
    return {t: sum(n[t] for n in norms) / len(norms) for t in ALL_TERMS}


def score_deposit(system):
    """Score the cognate deposit interface -> normalized structural terms (confidence = None)."""
    dep_cif, dirs, pb, pt = system_config(system)
    dep_model = gemmi.read_structure(str(dep_cif))[0]
    # pred target seq picks the cognate deposit copy
    first_cif = next((c for _, d in dirs for c in sorted(d.glob("pred.model_idx_*.cif"))), None)
    pm0 = gemmi.read_structure(str(first_cif))[0]
    pred_target_seq = "".join(_one(r) for r in _aa_residues(_chain(pm0, pt)))
    dep_t, dep_b = pick_deposit_pair(dep_model, pred_target_seq)

    # score_complex.structural on the deposit, cognate chains (binder=ca, target=cb)
    raw = score_complex.structural(str(dep_cif), dep_b.name, dep_t.name)
    nrm = normalize(raw)
    return {t: nrm[t] for t in STRUCTURAL}, (dep_b.name, dep_t.name)


def main():
    rows = {}
    for system in ALL_SYSTEMS:
        real = mean_norm(panel_glob(system, "real"))
        scram_panels = panel_glob(system, "scram1") + panel_glob(system, "scram2")
        scram = mean_norm(scram_panels)
        if real is None or scram is None:
            print(f"[skip] {system}: missing panels (real={real is not None}, scram={scram is not None})")
            continue
        dep, chains = score_deposit(system)
        entry = {"family": FAMILY[system], "deposit_chains": chains, "terms": {}}
        for t in ALL_TERMS:
            y = round(real[t] - scram[t], 4)               # chai_real - chai_scram (always defined)
            if t in STRUCTURAL:
                x = round(dep[t] - scram[t], 4)            # deposit - chai_scram
            else:
                x = None                                    # confidence: no deposit value
            entry["terms"][t] = {"x_deposit_minus_scram": x, "y_chaireal_minus_scram": y,
                                 "norm_deposit": (round(dep[t], 4) if t in STRUCTURAL else None),
                                 "norm_chai_real": round(real[t], 4),
                                 "norm_chai_scram": round(scram[t], 4)}
        rows[system] = entry

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    _plot(rows)

    # summary table for structural terms: how many systems have the prediction reproduce the deposit sep?
    print(f"\n{'term':9} {'#dep>0.02':>9} {'#pred>0.02':>10} {'#both(near y=x)':>15}")
    for t in STRUCTURAL:
        dep_pos = pred_pos = both = 0
        for s in rows:
            x = rows[s]["terms"][t]["x_deposit_minus_scram"]
            y = rows[s]["terms"][t]["y_chaireal_minus_scram"]
            if x is not None and x > 0.02:
                dep_pos += 1
                if y > 0.02:
                    both += 1
            if y > 0.02:
                pred_pos += 1
        print(f"{t:9} {dep_pos:>9} {pred_pos:>10} {both:>15}")
    print(f"\nJSON -> {OUT_JSON}\nPNG  -> {OUT_PNG}")
    return 0


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"MDM2": "#2c7fb8", "gp41": "#d95f0e", "8GQP/7YH8": "#31a354"}
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    for ax, t in zip(axes, STRUCTURAL):
        lim = 0.3
        ax.plot([-lim, lim], [-lim, lim], ls="--", color="grey", lw=0.8, zorder=0)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        for s, e in rows.items():
            x = e["terms"][t]["x_deposit_minus_scram"]
            y = e["terms"][t]["y_chaireal_minus_scram"]
            ax.scatter(x, y, s=45, color=colors[e["family"]], edgecolor="k", linewidth=0.4, zorder=3)
            ax.annotate(s, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.set_title(f"{t}  (x=deposit−scram, y=chai_real−scram)", fontsize=10)
        ax.set_xlabel("Δ deposit − chai_scram"); ax.set_ylabel("Δ chai_real − chai_scram")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    # confidence terms: only y is defined -> strip plot of y per system
    ax = axes[5]
    ax.set_title("confidence terms (deposit unscoreable):\ny = chai_real − chai_scram only", fontsize=10)
    syslist = list(rows.keys())
    for j, t in enumerate(CONFIDENCE):
        ys = [rows[s]["terms"][t]["y_chaireal_minus_scram"] for s in syslist]
        cs = [colors[rows[s]["family"]] for s in syslist]
        ax.scatter([j] * len(ys), ys, c=cs, s=40, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(CONFIDENCE))); ax.set_xticklabels(CONFIDENCE)
    ax.set_ylabel("Δ chai_real − chai_scram")

    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=f) for f, c in colors.items()]
    axes[6].legend(handles=handles, loc="center", fontsize=11); axes[6].axis("off")
    axes[7].axis("off")
    axes[7].text(0.0, 0.5, "near y=x: chai-real reproduces\nthe deposit's term separation.\n\n"
                 "x>0, y≈0: term separates on the\nTRUTH but the prediction loses it\n"
                 "(structural non-compliance).", fontsize=10, va="center")
    fig.suptitle("Goal 2 — per-term reliability: deposit vs chai-real vs chai-scramble (8 terms × 11 systems)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
