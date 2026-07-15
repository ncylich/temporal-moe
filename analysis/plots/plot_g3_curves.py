#!/usr/bin/env python3
"""THE standard IsoFLOP figure: coarse + fine-grained, all methods, both budgets, one axes.

Encoding standard (keep this for every isoFLOP-style graph going forward):
  color  = method       (dense gray, MoE blue, temporal green)
  shade  = granularity  (coarse = normal color, fine-grained = dark color)
  marker = compute budget (10^16 = circle, 10^17 = triangle)
Equal line weight + opacity so no series overpowers another. All BPB measured
(results/phase0/G3_RESULTS.md + baseline docs); lower is better.

--no-caption = paper mode: compact figsize + short title/labels + large fonts + no baked caption
(detail goes in the LaTeX caption). Output gets a _nocaption suffix.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PAPER:
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 9.5})

N_G1 = {"sm1": 0.770, "s0": 1.361, "s1": 3.812, "s2": 8.115, "s3": 14.774}
N_G3 = {"sm1": 0.81, "s0": 1.42, "s1": 3.91, "s2": 8.23, "s3": 15.09}

DENSE_G1 = {"1e16": {"sm1": 1.534, "s0": 1.519, "s1": 1.591},
            "1e17": {"s1": 1.361, "s2": 1.341, "s3": 1.408}}
# Values = end-of-training TEST-set BPB (canonical series; see results/ablations/FINDINGS.md).
MOE_G1   = {"1e16": {"sm1": 1.4766, "s0": 1.447, "s1": 1.540},
            "1e17": {"s1": 1.2803, "s2": 1.269, "s3": 1.289}}
MOE_G3   = {"1e16": {"sm1": 1.4786, "s0": 1.4585, "s1": 1.5352},
            "1e17": {"s1": 1.2846, "s2": 1.2708, "s3": 1.2815}}
TMP_G1   = {"1e16": {"sm1": 1.4872, "s0": 1.4599, "s1": 1.5473},
            "1e17": {"s1": 1.3027, "s2": 1.2821, "s3": 1.3061}}
TMP_G3   = {"1e16": {"sm1": 1.4976, "s0": 1.4753, "s1": 1.5861},
            "1e17": {"s1": 1.3065, "s2": 1.2873, "s3": 1.3129}}

# color = method, shade = granularity (coarse normal / fine dark)
DENSE_C = "#7f7f7f"
MOE_COARSE, MOE_FINE = "#5aa0dd", "#0d3b66"
TMP_COARSE, TMP_FINE = "#5cc85c", "#145a14"
MK = {"1e16": "o", "1e17": "^"}   # marker = compute budget

# data, N-table, color, lw, label, short-label (paper)
SERIES = [
    (DENSE_G1, N_G1, DENSE_C,    1.4, "dense baseline",                     "dense"),
    (MOE_G1,   N_G1, MOE_COARSE, 1.9, "full MoE, coarse (6 of 64)",         "MoE · coarse"),
    (MOE_G3,   N_G3, MOE_FINE,   1.9, "full MoE, fine-grained (18 of 192)", "MoE · fine"),
    (TMP_G1,   N_G1, TMP_COARSE, 1.9, "temporal, coarse (6 of 64)",         "temporal · coarse"),
    (TMP_G3,   N_G3, TMP_FINE,   1.9, "temporal, fine-grained (18 of 192)", "temporal · fine"),
]

fig, ax = plt.subplots(figsize=(6.1, 4.2) if PAPER else (10.0, 6.3))
for data, Ntab, color, lw, label, short in SERIES:
    for b in ["1e16", "1e17"]:
        d = data[b]
        xs = np.array([Ntab[k] for k in d]); ys = np.array([d[k] for k in d]); o = np.argsort(xs)
        ax.plot(xs[o], ys[o], "-", color=color, marker=MK[b], mfc=color, mec=color,
                ms=7, lw=lw, alpha=0.9)
ax.set_xscale("log")
ax.grid(True, which="both", ls=":", alpha=0.4)

# legend: color+shade = method x granularity; then marker shape = budget
handles = [Line2D([0], [0], color=s[2], lw=2.2) for s in SERIES]
labels  = [(s[5] if PAPER else s[4]) for s in SERIES]
handles += [Line2D([0], [0], color="0.4", marker="o", ls="", label="$10^{16}$"),
            Line2D([0], [0], color="0.4", marker="^", ls="", label="$10^{17}$")]
labels  += ["$10^{16}$ FLOPs", "$10^{17}$ FLOPs"]

if PAPER:
    ax.set_xlabel("active params (M)")
    ax.set_ylabel("test BPB")
    ax.set_title("Quality at fixed compute")
    ax.legend(handles, labels, ncol=1, loc="lower left")
    fig.tight_layout()
    out = f"{REPO}/results/phase0/figures/fine_grained_vs_coarse_experts_isoflop_nocaption.png"
else:
    ax.set_xlabel("active non-embedding params  N  (millions)")
    ax.set_ylabel("test BPB  (bits/byte, lower better)")
    ax.set_title("IsoFLOP: coarse (6 of 64) vs fine-grained (18 of 192) experts, all methods")
    ax.legend(handles, labels, ncol=2, loc="lower left")
    fig.text(0.5, 0.01,
             "IsoFLOP curves (test-set BPB, lower is better, vs active non-embedding params). "
             "Color = method (dense gray, MoE blue, temporal green); shade = granularity (coarse = "
             "normal, fine-grained = dark); marker = compute budget (circle 10^16, triangle 10^17). "
             "'temporal' = rolling residency (keep top-k experts resident, swap 1 per token).",
             ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = f"{REPO}/results/phase0/figures/fine_grained_vs_coarse_experts_isoflop.png"
fig.savefig(out, dpi=200 if PAPER else 140)
print("wrote", out)
