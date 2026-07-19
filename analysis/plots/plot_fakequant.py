#!/usr/bin/env python3
"""Fake-quant degradation: delta test CE vs the 16-bit baseline at 8/4/3-bit RTN (group 128,
routed experts only). Data: results/ablations/stability_fakequant.csv (h100@8f5064e6 38M,
h100@ec9007c8 1e19). Panels are independent scales (1e19 models are ~2-3x more robust)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTD = f"{REPO}/results/phase0/figures"

MOE_C, MOE_F = "#5aa0dd", "#0d3b66"
TMP_C, TMP_F = "#5cc85c", "#145a14"
BITS = [8, 4, 3]

P19 = [
    ("MoE · coarse", MOE_C, [0.0000, 0.0076, 0.0452]),
    ("temporal · coarse", TMP_C, [0.0000, 0.0066, 0.0383]),
    ("temporal · fine", TMP_F, [0.0000, 0.0059, 0.0340]),
]
P18 = [
    ("MoE · coarse", MOE_C, [0.0001, 0.0180, 0.1077]),
    ("temporal · coarse", TMP_C, [0.0000, 0.0144, 0.0866]),
    ("MoE · fine", MOE_F, [0.0000, 0.0126, 0.0722]),
    ("temporal · fine", TMP_F, [0.0001, 0.0106, 0.0618]),
]

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, data, title in [(axes[0], P19, "$10^{19}$ FLOPs"), (axes[1], P18, "$10^{18}$ FLOPs · 38M")]:
    for label, color, ys in data:
        ax.plot(range(len(BITS)), ys, "-o", color=color, ms=5, lw=1.8)
    ax.set_xticks(range(len(BITS))); ax.set_xticklabels([f"{b}-bit" for b in BITS])
    ax.set_title(title)
    ax.set_ylabel("$\\Delta$ test CE vs 16-bit")
    ax.grid(True, ls=":", alpha=0.4)
handles = [Line2D([0], [0], color=c, lw=2.2) for _, c, _ in P18]
fig.legend(handles, [l for l, _, _ in P18], ncol=4, loc="upper center", fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = f"{OUTD}/fakequant_degradation_nocaption.png"
fig.savefig(out, dpi=200)
print("wrote", out)
