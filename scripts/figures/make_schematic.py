#!/usr/bin/env python3
"""Render the XenoDesign1 design-loop schematic (matplotlib only).

Outputs:
  docs/figures/design_loop.svg
  docs/figures/design_loop.png  (dpi 200, ~1800x1100 px)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

# ---- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "figures"))
os.makedirs(OUT, exist_ok=True)
PNG = os.path.join(OUT, "design_loop.png")
SVG = os.path.join(OUT, "design_loop.svg")

# ---- palette (restrained, modern) ------------------------------------------
SLATE = "#334e68"   # stages
TEAL = "#2c7a7b"    # feedback / seq-update
CORAL = "#e07a5f"   # gates
AMBER = "#d9a531"   # classes
INK = "#1f2933"     # text / outlines
BG = "white"
EDGE = "#0f1b24"

FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "text.color": INK,
})

# ---- canvas: 1800x1100 px @ dpi200 -> 9.0 x 5.5 in -------------------------
FIG_W, FIG_H = 9.0, 5.5
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
# logical coordinate grid (margins so nothing clips)
ax.set_xlim(-2, 118)
ax.set_ylim(0, 100)
ax.axis("off")

# ---- helpers ---------------------------------------------------------------

def stage_box(cx, cy, w, h, title, sub=None, face=SLATE, tcolor="white",
              tfs=11.5):
    """Rounded stage box centered at (cx, cy). Returns (cx, cy, w, h)."""
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.0,rounding_size=1.6",
        linewidth=1.6, edgecolor=EDGE, facecolor=face, zorder=3,
    )
    ax.add_patch(box)
    if sub is None:
        ax.text(cx, cy, title, ha="center", va="center", color=tcolor,
                fontsize=tfs, fontweight="bold", zorder=4, wrap=True)
    else:
        ax.text(cx, cy + h * 0.20, title, ha="center", va="center",
                color=tcolor, fontsize=tfs, fontweight="bold", zorder=4)
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center",
                color=tcolor, fontsize=7.6, zorder=4, wrap=True)
    return cx, cy, w, h


def diamond(cx, cy, w, h, label, face=CORAL):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    poly = plt.Polygon(pts, closed=True, linewidth=1.4, edgecolor=EDGE,
                       facecolor=face, zorder=3)
    ax.add_patch(poly)
    ax.text(cx, cy, label, ha="center", va="center", color="white",
            fontsize=7.4, fontweight="bold", zorder=4, wrap=True)


def pill(cx, cy, w, h, label, face=AMBER):
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.0,rounding_size=2.6",
        linewidth=1.3, edgecolor=EDGE, facecolor=face, zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center", color=INK,
            fontsize=8.0, fontweight="bold", zorder=4)


def arrow(p0, p1, color=INK, lw=2.0, style="-|>", rad=0.0, ls="-", z=2):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, zorder=z,
        connectionstyle=f"arc3,rad={rad}", linestyle=ls,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


# ---- title -----------------------------------------------------------------
ax.text(58, 97.0, "XenoDesign1 — mixed-chirality binder design loop on Chai-1",
        ha="center", va="center", fontsize=15.0, fontweight="bold", color=INK)

# ===========================================================================
# CORE LOOP (left -> right), main row at y = 66
# ===========================================================================
ROW = 66
SW, SH = 16.5, 13.0   # stage box default w/h

seed = stage_box(11, ROW, SW, SH, "SEED",
                 "unconditional PepMLM /\ndeclared scaffold\n(free length 6-50)")
restr = stage_box(31.5, ROW, SW, SH, "RESTRAINTS (opt)",
                  "coordinators,\ncyclization, contacts", face="#5b7a96",
                  tfs=8.6)
chai = stage_box(52, ROW, SW, SH, "CHAI-1 PREDICT", face=SLATE, tfs=9.6)
score = stage_box(72.5, ROW, SW, SH, "SCORE",
                  "ipTM / pTM / pLDDT")
select = stage_box(93, ROW, 13.5, SH, "SELECT", face="#3a6b6b")

# forward arrows along the core row
arrow((seed[0] + SW / 2, ROW), (restr[0] - SW / 2, ROW))
arrow((restr[0] + SW / 2, ROW), (chai[0] - SW / 2, ROW))
arrow((chai[0] + SW / 2, ROW), (score[0] - SW / 2, ROW))
arrow((score[0] + SW / 2, ROW), (select[0] - 13.5 / 2, ROW))

# ===========================================================================
# FEEDBACK LOOP: SELECT -> seq-update (LigandMPNN/CARBonAra) -> CHAI-1
# ===========================================================================
TOP = 86
UW = 32
upd = stage_box(72, TOP, UW, 8.5, "sequence update\n(LigandMPNN / CARBonAra)",
                face=TEAL, tfs=8.6)

# SELECT up to update node (curved), then update node down/left into CHAI-1
arrow((select[0], ROW + SH / 2), (upd[0] + UW / 2, TOP), color=TEAL, lw=2.2,
      rad=-0.30)
arrow((upd[0] - UW / 2, TOP), (chai[0], ROW + SH / 2), color=TEAL, lw=2.2,
      rad=-0.22)
ax.text(72, TOP - 6.2, "iterate", ha="center", va="center",
        fontsize=8.5, style="italic", color=TEAL)

# ===========================================================================
# GATES (diamonds) attached beneath SCORE / SELECT, row at y = 40
# ===========================================================================
GROW = 38
DW, DH = 18.5, 15
gate_labels = [
    "Chirality\nveto",
    "Coiled-coil\nperiodicity\n(alpha)",
    "MetalHawk\nmetal-geometry\ngate",
    "ABC / EA\nmixed-chirality\nsearch",
]
# four diamonds with clear gaps across the right half of the frame
gx = [48.5, 68.5, 88.5, 108.5]   # 20-unit pitch, DW=18.5 -> 1.5-unit gaps
for x, lab in zip(gx, gate_labels):
    diamond(x, GROW, DW, DH, lab, face=CORAL)

# connector from SCORE/SELECT region down to the gate band
gmid = (gx[0] + gx[-1]) / 2
arrow(((score[0] + select[0]) / 2, ROW - SH / 2),
      (gmid, GROW + DH / 2 + 1.5),
      color=CORAL, lw=2.0, rad=0.0)
ax.text(gmid, GROW + DH / 2 + 4.5,
        "class-specific gates", ha="center", va="center", fontsize=8.6,
        style="italic", color=CORAL)

# ===========================================================================
# CLASSES (pills) branching off SELECT, stacked on the left, row band y ~ 10-40
# ===========================================================================
class_labels = [
    "alpha (helical, vs target)",
    "non_alpha (cystine-knot / ICK)",
    "cyclic (free or metal)",
    "no-target (intramolecular)",
]
PW, PH = 34, 8.0
cy_list = [31, 22, 13, 4]
cx_pill = 21
for cyp, lab in zip(cy_list, class_labels):
    pill(cx_pill, cyp, PW, PH, lab, face=AMBER)

# branch hub: from SELECT down-left to the class stack
hub = (cx_pill + PW / 2 + 9, 17.5)
arrow((select[0], ROW - SH / 2), hub, color=AMBER, lw=2.2, rad=0.34)
for cyp in cy_list:
    arrow(hub, (cx_pill + PW / 2, cyp), color=AMBER, lw=1.6, rad=0.0)
ax.text(hub[0] + 3, 34, "branch per\nbinder class",
        ha="center", va="center", fontsize=8.4, style="italic", color=AMBER)

# ===========================================================================
# LEGEND
# ===========================================================================
legend_handles = [
    Patch(facecolor=SLATE, edgecolor=EDGE, label="stage"),
    Patch(facecolor=CORAL, edgecolor=EDGE, label="gate"),
    Patch(facecolor=AMBER, edgecolor=EDGE, label="class"),
    Line2D([0], [0], color=TEAL, lw=2.4, label="iterative feedback"),
]
leg = ax.legend(handles=legend_handles, loc="lower left",
                bbox_to_anchor=(0.47, 0.02), frameon=True, fontsize=9,
                title="key", title_fontsize=9, borderpad=0.8,
                labelspacing=0.7)
leg.get_frame().set_edgecolor(EDGE)
leg.get_frame().set_facecolor("#f7f9fa")

# ---- save ------------------------------------------------------------------
# No bbox_inches="tight": the axes already fill the figure with deliberate
# margins baked into the xlim/ylim, so tight cropping would clip text that
# overhangs patch edges. Keep the fixed canvas instead.
fig.savefig(PNG, dpi=200, facecolor=BG)
fig.savefig(SVG, facecolor=BG)
print("WROTE", PNG, SVG)
