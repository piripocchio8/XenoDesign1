"""Plot Goal-1 structural-compliance distributions (DockQ, fnat, placement RMSD) per system.

Reads the JSON from structure_compliance.py and draws per-system distributions over all
seed x model predictions, grouped/colored by family (MDM2, gp41, 8GQP/7YH8), with DockQ
quality bins and a 5 A placement-RMSD reference.

  python3 scripts/plot_structure_compliance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
JSON = REPO / "docs" / "results" / "2026-06-23-structure-compliance.json"
OUT = REPO / "docs" / "reports" / "2026-06-23-structure-compliance.png"

FAMILY = {
    "3LNJ": "MDM2", "7KJM": "MDM2", "3IWY": "MDM2",
    "3L35": "gp41", "2R5B": "gp41", "2R3C": "gp41",
    "3MGN": "gp41", "1CZQ": "gp41", "2Q3I": "gp41",
    "8GQP": "8GQP/7YH8", "7YH8v2": "8GQP/7YH8",
}
COLORS = {"MDM2": "#2c7fb8", "gp41": "#d95f0e", "8GQP/7YH8": "#31a354"}
# order: best-recapitulating first
ORDER = ["3LNJ", "7KJM", "3IWY", "7YH8v2", "2R5B", "2Q3I", "1CZQ",
         "3L35", "3MGN", "8GQP", "2R3C"]


def main():
    data = json.loads(JSON.read_text())
    systems = [s for s in ORDER if s in data and data[s].get("per_pred")]
    cols = [COLORS[FAMILY[s]] for s in systems]

    def series(key):
        return [[r[key] for r in data[s]["per_pred"] if key in r] for s in systems]

    dockq = series("dockq")
    fnat = series("fnat")
    lrmsd = series("lrmsd")

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    x = np.arange(len(systems))

    # --- DockQ ---
    ax = axes[0]
    for b, lab in [(0.23, "Acceptable"), (0.49, "Medium"), (0.80, "High")]:
        ax.axhline(b, ls="--", lw=0.8, color="grey")
        ax.text(len(systems) - 0.4, b + 0.01, lab, fontsize=8, color="grey", ha="right")
    for i, (vals, c) in enumerate(zip(dockq, cols)):
        jit = (np.random.RandomState(i).rand(len(vals)) - 0.5) * 0.3
        ax.scatter(x[i] + jit, vals, s=28, color=c, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.scatter(x[i], np.median(vals), s=90, color=c, marker="_", linewidth=2.5)
    ax.set_ylabel("DockQ (capri_peptide)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Goal 1 — Structural compliance of the 11 'real' Trues vs deposit "
                 "(per seed x chai-model; bar = median)")

    # --- fnat ---
    ax = axes[1]
    ax.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax.text(len(systems) - 0.4, 0.51, "interface recovered", fontsize=8, color="grey", ha="right")
    for i, (vals, c) in enumerate(zip(fnat, cols)):
        jit = (np.random.RandomState(i + 100).rand(len(vals)) - 0.5) * 0.3
        ax.scatter(x[i] + jit, vals, s=28, color=c, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.scatter(x[i], np.median(vals), s=90, color=c, marker="_", linewidth=2.5)
    ax.set_ylabel("fnat (native interface contacts recovered)")
    ax.set_ylim(0, 1.0)

    # --- placement RMSD (LRMSD) ---
    ax = axes[2]
    ax.axhline(5.0, ls="--", lw=0.8, color="grey")
    ax.text(len(systems) - 0.4, 5.3, "5 A", fontsize=8, color="grey", ha="right")
    for i, (vals, c) in enumerate(zip(lrmsd, cols)):
        jit = (np.random.RandomState(i + 200).rand(len(vals)) - 0.5) * 0.3
        ax.scatter(x[i] + jit, vals, s=28, color=c, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.scatter(x[i], np.median(vals), s=90, color=c, marker="_", linewidth=2.5)
    ax.set_ylabel("binder placement RMSD = DockQ LRMSD (A)")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=30, ha="right")

    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=f) for f, c in COLORS.items()]
    axes[0].legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
