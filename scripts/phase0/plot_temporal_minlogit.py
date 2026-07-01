#!/usr/bin/env python3
"""Dense floor vs full-MoE (1 shared) IsoFLOP curves, with the temporal (rolling-residency,
min_logit eviction, 1 shared) points overlaid — one per budget at its compute-optimal shape.

Shows where temporal lands in the dense..MoE quality band. Output: results/phase0/temporal_minlogit_vs_baselines.png
All BPB are measured (results/phase0/log.md); lower is better.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = {"sm1": 0.770, "s0": 1.361, "s1": 3.812, "s2": 8.115, "s3": 14.774, "s4": 24.292}

MOE = {  # full MoE, 1 shared expert (the s=1 baseline)
    "1e16": {"sm1": 1.478, "s0": 1.447, "s1": 1.540, "s2": 1.819, "s3": 2.187},
    "1e17": {"s1": 1.284, "s2": 1.269, "s3": 1.289},
}
DENSE = {
    "1e16": {"sm1": 1.534, "s0": 1.519, "s1": 1.591, "s2": 1.848},
    "1e17": {"s1": 1.361, "s2": 1.341, "s3": 1.408, "s4": 1.485},
}
# temporal: rolling residency, min_logit eviction, 1 shared (K=6). one point per budget @ optimal shape.
TEMPORAL = {"1e16": ("s0", 1.4599), "1e17": ("s2", 1.2821)}

def pts(d):
    xs = np.array([N[k] for k in d]); ys = np.array([d[k] for k in d])
    o = np.argsort(xs); return xs[o], ys[o]

LSTYLE = {"1e16": "--", "1e17": "-"}
fig, ax = plt.subplots(figsize=(9.5, 6.2))
for b in ["1e16", "1e17"]:
    for label, data, color in [("dense floor", DENSE[b], "C3"), ("MoE (1 shared)", MOE[b], "C0")]:
        x, y = pts(data)
        ax.plot(x, y, LSTYLE[b], color=color, marker="o", lw=1.8, ms=6, label=f"{label}  {b}")
        lx = np.log10(x); c = np.polyfit(lx, y, 2); xx = np.linspace(lx.min(), lx.max(), 100)
        ax.plot(10**xx, np.polyval(c, xx), ":", color=color, lw=1, alpha=0.35)
    shape, bpb = TEMPORAL[b]
    ax.plot(N[shape], bpb, "*", color="C2", ms=20, zorder=5, markeredgecolor="k",
            markeredgewidth=0.5, label=f"temporal (min_logit, 1 shared)  {b}")
    ax.annotate(f"{bpb:.4f}", (N[shape], bpb), textcoords="offset points",
                xytext=(10, -4), fontsize=9, color="C2", fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("active non-embedding params N (millions)")
ax.set_ylabel("validation BPB (bits/byte, lower better)")
ax.set_title("Temporal MoE (6/64 experts resident) lands inside the dense↔MoE band — both budgets\n"
             "color = method, dashed = 1e16 / solid = 1e17; lower is better")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig("results/phase0/temporal_minlogit_vs_baselines.png", dpi=130)
print("wrote results/phase0/temporal_minlogit_vs_baselines.png")
