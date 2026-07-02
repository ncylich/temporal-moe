#!/usr/bin/env python3
"""Combined single-axes IsoFLOP figure (matches plot_temporal_final.py style):
  color   = method   (dense gray, MoE blue, temporal green)
  linestyle = budget (dashed 1e16, solid 1e17)
  weight/marker = granularity (G1 6/64 thin+open circle+faded, G3 18/192 bold+filled square)
All BPB measured (results/phase0/G3_RESULTS.md + baseline docs); lower is better.
All 12 G3 runs complete (G3 temporal s1@1e17 = 1.3065, the last A6000 point).

--no-caption = paper mode: compact figsize + short title/labels + large fonts + no baked caption
(detail goes in the LaTeX caption). Output gets a _nocaption suffix. Default mode is unchanged.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PAPER:
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11})

N_G1 = {"sm1": 0.770, "s0": 1.361, "s1": 3.812, "s2": 8.115, "s3": 14.774}
N_G3 = {"sm1": 0.81, "s0": 1.42, "s1": 3.91, "s2": 8.23, "s3": 15.09}

DENSE_G1 = {"1e16": {"sm1": 1.534, "s0": 1.519, "s1": 1.591},
            "1e17": {"s1": 1.361, "s2": 1.341, "s3": 1.408}}
MOE_G1   = {"1e16": {"sm1": 1.478, "s0": 1.447, "s1": 1.540},
            "1e17": {"s1": 1.284, "s2": 1.269, "s3": 1.289}}
MOE_G3   = {"1e16": {"sm1": 1.4786, "s0": 1.4585, "s1": 1.5352},
            "1e17": {"s1": 1.2846, "s2": 1.2708, "s3": 1.2815}}
TMP_G1   = {"1e16": {"sm1": 1.4891, "s0": 1.4599, "s1": 1.5488},
            "1e17": {"s1": 1.3039, "s2": 1.2821, "s3": 1.3073}}
TMP_G3   = {"1e16": {"sm1": 1.4976, "s0": 1.4753, "s1": 1.5861},
            "1e17": {"s1": 1.3065, "s2": 1.2873, "s3": 1.3129}}

# name, data, N-table, color, marker, facecolor, lw, alpha, label, short-label (paper)
SERIES = [
    ("dense_g1", DENSE_G1, N_G1, "0.55", "x", "0.55", 1.3, 0.9,  "dense baseline",                    "dense"),
    ("moe_g1",   MOE_G1,   N_G1, "C0",   "o", "none", 1.4, 0.55, "full MoE, coarse (6 of 64)",        "MoE · coarse"),
    ("tmp_g1",   TMP_G1,   N_G1, "C2",   "o", "none", 1.4, 0.55, "temporal, coarse (6 of 64)",        "temporal · coarse"),
    ("moe_g3",   MOE_G3,   N_G3, "C0",   "s", "C0",   2.6, 1.0,  "full MoE, fine-grained (18 of 192)","MoE · fine"),
    ("tmp_g3",   TMP_G3,   N_G3, "C2",   "s", "C2",   2.6, 1.0,  "temporal, fine-grained (18 of 192)","temporal · fine"),
]
LSTYLE = {"1e16": "--", "1e17": "-"}

fig, ax = plt.subplots(figsize=(6.1, 4.2) if PAPER else (10.5, 6.6))
for key, data, Ntab, color, mk, fc, lw, alpha, label, short in SERIES:
    for b in ["1e16", "1e17"]:
        d = data[b]
        xs = np.array([Ntab[k] for k in d]); ys = np.array([d[k] for k in d])
        o = np.argsort(xs)
        ax.plot(xs[o], ys[o], LSTYLE[b], color=color, marker=mk, mfc=fc,
                ms=(6 if PAPER else 6.5), lw=(lw*0.8 if PAPER else lw), alpha=alpha,
                label=((short if PAPER else label) if b == "1e17" else None))
ax.set_xscale("log")
ax.grid(True, which="both", ls=":", alpha=0.4)

if PAPER:
    ax.set_xlabel("active params (M)")
    ax.set_ylabel("validation BPB")
    ax.set_title("Quality at fixed compute")
    ax.legend(ncol=1, loc="lower left")   # lower-left is empty (1e17 curves start at ~4M params)
    fig.tight_layout()
    out = f"{REPO}/results/phase0/figures/fine_grained_vs_coarse_experts_isoflop_nocaption.png"
else:
    ax.set_xlabel("active non-embedding params  N  (millions)")
    ax.set_ylabel("validation BPB  (bits/byte, lower better)")
    ax.set_title("Splitting each expert 3x (coarse: 6 of 64  ->  fine-grained: 18 of 192 experts)\n"
                 "full MoE and temporal (rolling residency) vs the dense baseline, across compute budgets")
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    fig.text(0.5, 0.01,
             "IsoFLOP curves: validation bits-per-byte (BPB, lower is better) vs active non-embedding "
             "parameters N (millions, log x-axis). Dashed = 10^16 FLOPs, solid = 10^17 FLOPs. Color = method. "
             "Bold filled squares = fine-grained experts (each of 64 experts split 3-ways into 192, top-18 "
             "active); thin open circles = coarse experts (64 experts, top-6 active). 'temporal' = rolling "
             "residency (keep top-k experts resident, swap 1 per token).", ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = f"{REPO}/results/phase0/figures/fine_grained_vs_coarse_experts_isoflop.png"
fig.savefig(out, dpi=200 if PAPER else 140)
print("wrote", out)
