#!/usr/bin/env python3
"""Bandwidth-masking timeline (schematic, no measured data): one decode step of one MoE layer
under rolling residency. The k-1 resident experts are read from fast memory back to back while
the one incoming expert streams from slower storage underneath; the stream is hidden when
(k-1) * b/BW_fast >= b/BW_slow, i.e. once k-1 reaches the fast-to-slow bandwidth ratio.
Successor to the talk's slide05_bandwidth_timeline.svg (LEGACY_paper/talk_figures/), redrawn in
matplotlib with the paper's notation so it regenerates from a committed producer.
Output: results/phase0/figures/bandwidth_timeline.png (+ _nocaption paper variant).
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "phase0", "figures",
                   "bandwidth_timeline_nocaption.png" if PAPER else "bandwidth_timeline.png")

FS = 15 if PAPER else 11.5
LIGHT, EDGE, DARK, GREY = "#cfe3f5", "#4a7fb5", "#0d3b66", "#666666"

fig, ax = plt.subplots(figsize=(9.0, 2.7) if PAPER else (10.0, 3.4))
ax.set_xlim(0, 10.0)
ax.set_ylim(0, 3.2)
ax.axis("off")

# k-1 = 5 resident-expert reads from fast memory, back to back.
n, x0, w, gap, y, h = 5, 0.15, 1.30, 0.06, 1.95, 0.62
for i in range(n):
    x = x0 + i * (w + gap)
    ax.add_patch(Rectangle((x, y), w, h, facecolor=LIGHT, edgecolor=EDGE, lw=1.4,
                           joinstyle="round"))
    ax.text(x + w / 2, y + h / 2, r"$b/\mathrm{BW}_{\mathrm{fast}}$",
            ha="center", va="center", fontsize=FS - 2.5, color=DARK)
top_end = x0 + n * (w + gap) - gap
ax.text(x0, y + h + 0.14, r"$k-1$ resident experts read from fast memory",
        fontsize=FS, color=DARK, weight="bold", va="bottom")

# Wall-clock arrow between the rows.
ax.add_patch(FancyArrow(x0, 1.55, top_end - x0 + 0.55, 0, width=0.012,
                        head_width=0.16, head_length=0.18, color=DARK))
ax.text(top_end + 0.72, 1.55, "time", fontsize=FS - 2, color=GREY, va="center")

# The one incoming expert streaming from slow storage, hidden under the reads.
sw = top_end - x0 - 0.55                     # ends before the reads do: the masked case
ax.add_patch(Rectangle((x0, 0.55), sw, h, facecolor=DARK, edgecolor=DARK))
ax.text(x0 + sw / 2, 0.55 + h / 2, r"incoming expert streams $b/\mathrm{BW}_{\mathrm{slow}}$",
        ha="center", va="center", fontsize=FS - 1.5, color="white")
ax.text(x0 + sw + 0.12, 0.55 + h / 2, "hidden", fontsize=FS - 2.5, color=GREY,
        va="center", style="italic")

# The condition, right-aligned under the rows.
ax.text(x0, 0.08,
        r"the swap is free when $(k-1)\,b/\mathrm{BW}_{\mathrm{fast}} \geq b/\mathrm{BW}_{\mathrm{slow}}$"
        r",  i.e. $k-1 \geq \mathrm{BW}_{\mathrm{fast}}/\mathrm{BW}_{\mathrm{slow}}$",
        fontsize=FS - 1, color=DARK, ha="left", va="bottom")

if not PAPER:
    fig.suptitle("One decode step of one MoE layer: the single swap hides behind resident compute",
                 fontsize=FS + 1, y=0.99)
    fig.text(0.01, 0.005,
             "Schematic. Expert size b cancels from the condition, so only the routing width k "
             "matters; fine-graining raises k at unchanged active parameters.",
             fontsize=FS - 3, color=GREY)

fig.tight_layout(rect=(0, 0.05, 1, 0.93) if not PAPER else (0, 0, 1, 1))
fig.savefig(OUT, dpi=220)
print(OUT)
