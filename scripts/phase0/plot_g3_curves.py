#!/usr/bin/env python3
"""Combined single-axes IsoFLOP figure (matches plot_temporal_final.py style):
  color   = method   (dense gray, MoE blue, temporal green)
  linestyle = budget (dashed 1e16, solid 1e17)
  weight/marker = granularity (G1 6/64 thin+open circle+faded, G3 18/192 bold+filled square)
All BPB measured (results/phase0/G3_RESULTS.md + baseline docs); lower is better.
G3 temporal s1@1e17 still running on the A6000 -> omitted.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
            "1e17": {"s2": 1.2873, "s3": 1.3129}}   # s1@1e17 running -> omitted

# name, data, N-table, color, marker, facecolor, lw, alpha, label
SERIES = [
    ("dense_g1", DENSE_G1, N_G1, "0.55",     "x", "0.55",     1.3, 0.9, "dense floor (G1)"),
    ("moe_g1",   MOE_G1,   N_G1, "C0",       "o", "none",     1.4, 0.55, "MoE  G1 (6/64)"),
    ("tmp_g1",   TMP_G1,   N_G1, "C2",       "o", "none",     1.4, 0.55, "temporal  G1 (6/64)"),
    ("moe_g3",   MOE_G3,   N_G3, "C0",       "s", "C0",       2.6, 1.0, "MoE  G3 (18/192)"),
    ("tmp_g3",   TMP_G3,   N_G3, "C2",       "s", "C2",       2.6, 1.0, "temporal  G3 (18/192)"),
]
LSTYLE = {"1e16": "--", "1e17": "-"}

fig, ax = plt.subplots(figsize=(10.5, 6.6))
for key, data, Ntab, color, mk, fc, lw, alpha, label in SERIES:
    for b in ["1e16", "1e17"]:
        d = data[b]
        xs = np.array([Ntab[k] for k in d]); ys = np.array([d[k] for k in d])
        o = np.argsort(xs)
        ax.plot(xs[o], ys[o], LSTYLE[b], color=color, marker=mk, mfc=fc, ms=6.5,
                lw=lw, alpha=alpha, label=(label if b == "1e17" else None))
ax.set_xscale("log")
ax.set_xlabel("active non-embedding params  N  (millions)")
ax.set_ylabel("validation BPB  (bits/byte, lower better)")
ax.set_title("Fine-graining the experts (G1 6/64  →  G3 18/192): MoE & temporal vs dense floor\n"
             "color = method,  dashed = 1e16 / solid = 1e17,  bold-square = G3 / thin-open = G1")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=9, ncol=2, loc="upper left")
fig.tight_layout()
out = "/workspace/FLAME-MoE/results/phase0/g3_vs_g1_combined.png"
fig.savefig(out, dpi=140)
print("wrote", out)
