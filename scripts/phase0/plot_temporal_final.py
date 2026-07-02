#!/usr/bin/env python3
"""Final temporal-MoE figure: full IsoFLOP curves for dense floor, full MoE (1 shared), and temporal
(rolling residency, min_logit eviction, 1 shared = 6/64 experts resident) at both FLOP budgets.

Combined single-axes only: color = method, linestyle = budget (dashed 1e16, solid 1e17). All BPB
measured (results/phase0/log.md); lower is better. Each budget trimmed to the shapes all three curves
share so an unmatched extreme doesn't stretch the y-scale.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --no-caption = paper mode: drop the baked-in caption (detail goes in the LaTeX caption), use a
# compact figsize + short title/labels + large fonts so the figure stays legible after it is
# downscaled into a paper column. Output gets a _nocaption suffix. Default mode is unchanged.
PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PAPER:
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})

N = {"sm1": 0.770, "s0": 1.361, "s1": 3.812, "s2": 8.115, "s3": 14.774}

DENSE = {
    "1e16": {"sm1": 1.534, "s0": 1.519, "s1": 1.591, "s2": 1.848},
    "1e17": {"s1": 1.361, "s2": 1.341, "s3": 1.408},
}
MOE = {  # full MoE, 1 shared expert
    "1e16": {"sm1": 1.478, "s0": 1.447, "s1": 1.540, "s2": 1.819},
    "1e17": {"s1": 1.284, "s2": 1.269, "s3": 1.289},
}
TEMPORAL = {  # rolling residency, min_logit eviction, 1 shared (K=6 resident of 64)
    "1e16": {"sm1": 1.4891, "s0": 1.4599, "s1": 1.5488, "s2": 1.8260},
    "1e17": {"s1": 1.3039, "s2": 1.2821, "s3": 1.3073},
}
SERIES = [("dense baseline", DENSE, "C3"), ("full MoE", MOE, "C0"),
          ("temporal", TEMPORAL, "C2")]

def pts(d):
    xs = np.array([N[k] for k in d]); ys = np.array([d[k] for k in d])
    o = np.argsort(xs); return xs[o], ys[o]

# ---------- combined single axes ----------
fig, ax = plt.subplots(figsize=(5.6, 3.9) if PAPER else (9.5, 6.2))
STYLE = {"1e16": "--", "1e17": "-"}
for name, data, color in SERIES:
    for b in ["1e16", "1e17"]:
        x, y = pts(data[b])
        ax.plot(x, y, STYLE[b] + "o", color=color, lw=2.0, ms=6,
                label=(None if PAPER else f"{name}  ({b} FLOPs)"))
if not PAPER:
    for b in ["1e16", "1e17"]:
        items = TEMPORAL[b]; smin = min(items, key=items.get)
        ax.annotate(f"{items[smin]:.4f}", (N[smin], items[smin]), textcoords="offset points",
                    xytext=(8, -12), fontsize=9, color="C2", fontweight="bold")
ax.set_xscale("log")
ax.grid(True, which="both", ls=":", alpha=0.4)

if PAPER:
    ax.set_xlabel("active params (M)")
    ax.set_ylabel("validation BPB")
    ax.set_title("Quality at fixed compute")
    # method-by-color legend; dashed = 1e16 / solid = 1e17 is explained in the LaTeX caption
    handles = [Line2D([0], [0], color=c, marker="o", lw=2) for _, _, c in SERIES]
    ax.legend(handles, [n for n, _, _ in SERIES], loc="upper left")
    fig.tight_layout()
    out = f"{REPO}/results/phase0/figures/temporal_vs_dense_and_full_moe_isoflop_nocaption.png"
else:
    ax.set_xlabel("active non-embedding params N (millions)")
    ax.set_ylabel("validation BPB (bits/byte, lower better)")
    ax.set_title("Temporal routing (rolling residency: keep top-k experts resident, swap 1 per token)\n"
                 "vs the dense baseline and full MoE, across compute budgets — temporal tracks just above full MoE")
    ax.legend(fontsize=8.5, ncol=3, loc="upper left")
    fig.text(0.5, 0.01,
             "IsoFLOP curves: validation bits-per-byte (BPB, lower is better) vs active non-embedding "
             "parameters N (millions, log x-axis) at two compute budgets (dashed = 10^16 FLOPs, solid = 10^17 "
             "FLOPs). 'temporal' = rolling residency (6 of 64 experts resident); 'full MoE' = standard top-k "
             "routing; 'dense baseline' = a plain feed-forward model. Green labels mark the temporal minimum "
             "at each budget.", ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = f"{REPO}/results/phase0/figures/temporal_vs_dense_and_full_moe_isoflop.png"
fig.savefig(out, dpi=200)
print("wrote", out)
