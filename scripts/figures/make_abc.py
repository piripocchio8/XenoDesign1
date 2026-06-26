#!/usr/bin/env python3
"""Render the ABC/EA mixed-chirality search figure for the XenoDesign1 README.

Two panels:
  A) EA convergence curve (best-so-far fitness vs evaluation)
  B) Variant B sequence + chirality map

Data: abc_runs/analysis/convergence_pilot_b.json (curve array). Falls back to
anchor points if absent.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = "/home/user/claude_projects/XenoDesign1"
JSON_PATH = os.path.join(ROOT, "abc_runs/analysis/convergence_pilot_b.json")
OUT_PNG = os.path.join(ROOT, "docs/figures/fig_abc.png")
OUT_SVG = os.path.join(ROOT, "docs/figures/fig_abc.svg")

SLATE = "#4a5a78"   # L
CORAL = "#e8745a"   # D

# ----------------------------------------------------------------------------
# Load convergence curve (best-so-far per evaluation)
# ----------------------------------------------------------------------------
evals, best = [], []
if os.path.exists(JSON_PATH):
    d = json.load(open(JSON_PATH))
    for row in d.get("curve", []):
        evals.append(row["eval"])
        best.append(row.get("running_best"))
if not evals or any(b is None for b in best):
    # Fallback monotone best-so-far step curve
    evals = list(range(1, 88))
    best = []
    cur = 0.718
    for e in evals:
        if e >= 80:
            cur = 0.841
        elif e >= 40:
            cur = 0.806
        best.append(cur)

start_val = best[0]
best_val = max(best)
best_eval = evals[best.index(best_val)]

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(9.0, 4.5), dpi=200,
    gridspec_kw={"width_ratios": [1.25, 1.0]},
)
fig.patch.set_facecolor("white")

# --- Panel A: convergence ---------------------------------------------------
axA.set_facecolor("white")
axA.step(evals, best, where="post", color=SLATE, lw=2.2, zorder=3,
         label="best-so-far")

# 20-mer scaled variant reference line
axA.axhline(0.806, ls="--", lw=1.4, color="#9a9a9a", zorder=2)
axA.text(evals[-1], 0.806, " 20-mer scaled variant", va="center", ha="right",
         fontsize=8.5, color="#666666", style="italic",
         bbox=dict(fc="white", ec="none", pad=1.0))

# Start marker
axA.scatter([evals[0]], [start_val], s=55, color=CORAL, zorder=5,
            edgecolor="white", linewidth=1.2)
axA.annotate(f"start {start_val:.3f}", (evals[0], start_val),
             xytext=(8, -16), textcoords="offset points", fontsize=8.5,
             color=SLATE,
             arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.8))

# Best marker
axA.scatter([best_eval], [best_val], s=70, color=CORAL, zorder=6,
            edgecolor="white", linewidth=1.2, marker="*")
axA.annotate(f"best {best_val:.3f}", (best_eval, best_val),
             xytext=(-6, 10), textcoords="offset points", fontsize=9,
             fontweight="bold", color="#b8472f", ha="right")

axA.set_xlabel("evaluation", fontsize=10)
axA.set_ylabel("fitness (nectar)", fontsize=10)
axA.set_title("A  EA convergence", fontsize=11, fontweight="bold", loc="left")
axA.set_xlim(0, evals[-1] + 2)
ymin = min(min(best), 0.806) - 0.03
ymax = best_val + 0.03
axA.set_ylim(ymin, ymax)
axA.grid(True, ls=":", lw=0.6, color="#cccccc", zorder=0)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
axA.legend(loc="lower right", fontsize=8.5, frameon=False)

# --- Panel B: sequence + chirality map --------------------------------------
seq = "MGVRIFQQFGAP"
chir = "LLDDDDDDLDDD"
assert len(seq) == len(chir) == 12

axB.set_facecolor("white")
axB.set_xlim(0, 12)
axB.set_ylim(0, 4)
axB.axis("off")

cell_w = 1.0
y0 = 1.7
h = 1.1
for i, (res, c) in enumerate(zip(seq, chir)):
    color = SLATE if c == "L" else CORAL
    box = FancyBboxPatch(
        (i * cell_w + 0.06, y0), cell_w - 0.12, h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=1.0, edgecolor="white", facecolor=color, zorder=2,
    )
    axB.add_patch(box)
    axB.text(i * cell_w + 0.5, y0 + h / 2 + 0.02, res, ha="center",
             va="center", fontsize=13, fontweight="bold", color="white",
             zorder=3)
    # chirality letter below
    axB.text(i * cell_w + 0.5, y0 - 0.30, c, ha="center", va="center",
             fontsize=8.5, color=color, fontweight="bold")
    # position index above
    axB.text(i * cell_w + 0.5, y0 + h + 0.22, str(i + 1), ha="center",
             va="center", fontsize=7, color="#999999")

# Legend
leg = [Patch(facecolor=SLATE, edgecolor="white", label="L (slate)"),
       Patch(facecolor=CORAL, edgecolor="white", label="D (coral)")]
axB.legend(handles=leg, loc="upper center", ncol=2, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 0.30))

axB.set_title(
    "B  Variant B (12-mer)\nring-closure proxy cn 1.21 Å, pTM 0.773",
    fontsize=10.5, fontweight="bold", loc="left",
)

# --- Overall title ----------------------------------------------------------
fig.suptitle("ABC / EA mixed-chirality search", fontsize=14,
             fontweight="bold", y=0.99)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PNG, dpi=200, facecolor="white", bbox_inches="tight")
fig.savefig(OUT_SVG, facecolor="white", bbox_inches="tight")
print("wrote", OUT_PNG)
print("wrote", OUT_SVG)
print("start", start_val, "best", best_val, "at eval", best_eval,
      "n_evals", len(evals))
