#!/usr/bin/env python3
"""The rolling-residency swap schematic: what the router does between two tokens.

Redrawn from the talk deck's slide04 SVG (LEGACY_paper/talk_figures/) so the paper
carries a vector figure from a committed producer rather than a raster lifted from
the slides. Two changes against the slide version, both driven by the paper's page
budget rather than taste:

  * a key along the bottom, since the slide relied on the speaker to explain that a
    light green fill means resident, a dashed border means leaving, a solid fill
    means arriving, and grey means not resident (a reviewer read the colours as
    "two experts active" without it);
  * a fixed pool of five slots, small enough to read at half a column.

Sized so its natural width is about half the text column, which means LaTeX places
it without downscaling and the font sizes set below are the sizes that print.

Writes results/ablations/figures/swap_diagram.pdf (vector, for the paper) and .png
(for preview). No data inputs; this is a schematic.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

FIG = os.path.join(ABLATIONS, "figures")

BLUE = "#0B5394"
GREEN = "#1f9d55"
GREEN_FILL = "#e6f6ec"
GREY_FILL = "#eef2f6"
GREY_EDGE = "#d3dbe4"
POOL_FILL = "#f6f9fc"
POOL_EDGE = "#e1e9f1"
INK = "#4a4f57"

N_SLOTS = 5
STAYS, EVICTED, ADMITTED = 0, 3, 4      # slot indices used by the walkthrough

SLOT_W, SLOT_H, GAP = 52.0, 11.5, 2.6
TOP = 18.0                              # first slot's top edge
PITCH = SLOT_H + GAP
BOTTOM = TOP + (N_SLOTS - 1) * PITCH + SLOT_H
LEFT_X, RIGHT_X = 10.0, 196.0


def slot_y(i):
    return TOP + i * PITCH


def slot_mid(i):
    return slot_y(i) + SLOT_H / 2


def box(ax, x, y, w, h, fill, edge, lw=0.8, dashed=False, r=2.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fill, edgecolor=edge, linewidth=lw,
                       linestyle=(0, (2.4, 1.7)) if dashed else "solid", zorder=2)
    ax.add_patch(p)
    return p


def arrow(ax, a, b, dashed=False, lw=1.1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                 color=BLUE, linewidth=lw, zorder=3,
                                 linestyle=(0, (2.6, 2.0)) if dashed else "solid",
                                 shrinkA=0, shrinkB=0))


def draw():
    W, H = 258.0, 110.0
    fig, ax = plt.subplots(figsize=(W / 95, H / 95))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                    # SVG-style: y grows downward
    ax.axis("off")

    for x, label in ((LEFT_X, "TOKEN $t$"), (RIGHT_X, "TOKEN $t{+}1$")):
        box(ax, x, 17, SLOT_W + 4, BOTTOM - 17 + 5, POOL_FILL, POOL_EDGE, 0.7, r=3)
        ax.text(x + (SLOT_W + 4) / 2, 10, label, ha="center", va="center",
                fontsize=7.0, fontweight="bold", color=BLUE)

    for i in range(N_SLOTS):
        for x, side in ((LEFT_X + 2, "L"), (RIGHT_X + 2, "R")):
            if i == STAYS:
                box(ax, x, slot_y(i), SLOT_W, SLOT_H, GREEN_FILL, GREEN, 1.1)
            elif side == "L" and i == EVICTED:
                box(ax, x, slot_y(i), SLOT_W, SLOT_H, GREEN_FILL, GREEN, 1.1,
                    dashed=True)
            elif side == "R" and i == ADMITTED:
                box(ax, x, slot_y(i), SLOT_W, SLOT_H, GREEN, GREEN, 1.1)
            else:
                box(ax, x, slot_y(i), SLOT_W, SLOT_H, GREY_FILL, GREY_EDGE, 0.7)

    # the expert that is active for both tokens never moves
    # drawn in two segments so the router's input label sits in the gap rather
    # than on top of the line
    y = slot_mid(STAYS)
    ax.plot([LEFT_X + SLOT_W + 4, RIGHT_X], [y, y], color="#8fbf9f", lw=0.7,
            dashes=(2.0, 2.4), zorder=1)

    rx, ry, rw, rh = 104.0, 36.0, 80.0, 24.0
    box(ax, rx, ry, rw, rh, "#ffffff", BLUE, 1.3, r=3)
    ax.text(rx + rw / 2, ry + rh / 2, "ROUTER", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=BLUE)
    arrow(ax, (rx + rw / 2, ry - 7.0), (rx + rw / 2, ry - 0.8), lw=0.9)

    arrow(ax, (LEFT_X + SLOT_W + 5, slot_mid(EVICTED)), (rx - 1.5, ry + rh - 6),
          dashed=True)
    ax.text(82, slot_mid(EVICTED) + 8, "evict", ha="center", va="center",
            fontsize=6.6, color=BLUE)

    arrow(ax, (rx + rw + 1.5, ry + rh - 4), (RIGHT_X + 1, slot_mid(ADMITTED)))
    ax.text(170, slot_mid(ADMITTED) + 6, "admit", ha="center", va="center",
            fontsize=6.6, color=BLUE)


    # key: one swatch per colour, spread along the bottom edge
    ky, kw, kh = 101.0, 9.0, 5.6
    for x, label, fill, edge, dashed in ((16, "resident", GREEN_FILL, GREEN, False),
                                         (74, "evicted", GREEN_FILL, GREEN, True),
                                         (134, "admitted", GREEN, GREEN, False),
                                         (192, "not resident", GREY_FILL, GREY_EDGE, False)):
        box(ax, x, ky, kw, kh, fill, edge, 0.9 if fill != GREY_FILL else 0.7, dashed=dashed,
            r=1.2)
        ax.text(x + kw + 3, ky + kh / 2, label, ha="left", va="center", fontsize=6.4,
                color=INK)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext in ("pdf", "png"):
        p = f"{FIG}/swap_diagram.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.01, transparent=False)
        print("wrote", p)


if __name__ == "__main__":
    draw()
