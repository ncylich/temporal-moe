#!/usr/bin/env python3
"""Final temporal-MoE figure: full IsoFLOP curves for dense floor, full MoE (1 shared), and temporal
(rolling residency, min_logit eviction, 1 shared = 6/64 experts resident) at both FLOP budgets.

Combined single-axes only: color = method, linestyle = budget (dashed 1e16, solid 1e17). All BPB
measured (results/phase0/log.md); lower is better. Each budget trimmed to the shapes all three curves
share so an unmatched extreme doesn't stretch the y-scale.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
SERIES = [("dense", DENSE, "C3"), ("MoE (1 shared)", MOE, "C0"),
          ("temporal (min_logit, 1 shared)", TEMPORAL, "C2")]

def pts(d):
    xs = np.array([N[k] for k in d]); ys = np.array([d[k] for k in d])
    o = np.argsort(xs); return xs[o], ys[o]

# ---------- combined single axes ----------
fig, ax = plt.subplots(figsize=(9.5, 6.2))
STYLE = {"1e16": "--", "1e17": "-"}
for name, data, color in SERIES:
    for b in ["1e16", "1e17"]:
        x, y = pts(data[b])
        ax.plot(x, y, STYLE[b] + "o", color=color, lw=1.8, ms=6,
                label=f"{name}  {b}")
for b in ["1e16", "1e17"]:
    # mark temporal minimum
    items = TEMPORAL[b]; smin = min(items, key=items.get)
    ax.annotate(f"{items[smin]:.4f}", (N[smin], items[smin]), textcoords="offset points",
                xytext=(8, -12), fontsize=9, color="C2", fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("active non-embedding params N (millions)")
ax.set_ylabel("validation BPB (bits/byte, lower better)")
ax.set_title("Temporal MoE (6/64 experts resident) vs dense floor & full MoE — both budgets\n"
             "dashed = 1e16, solid = 1e17;  temporal tracks just above MoE, well inside the band")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=8.5, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig("results/phase0/temporal_minlogit_final_combined.png", dpi=130)
print("wrote results/phase0/temporal_minlogit_final_combined.png")
