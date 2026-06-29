#!/usr/bin/env python3
"""Two views of temporal (rolling-residency, min_logit, 1 shared) vs the dense floor and full MoE
(1 shared) IsoFLOP curves. All BPB measured (results/phase0/log.md); lower is better.

  1) temporal_minlogit_2panel.png   — 1e16 and 1e17 in separate panels, each trimmed to the shapes
     BOTH curves share (drop the unmatched extreme: MoE s3@1e16, dense s4@1e17) so the y-scale isn't
     stretched by a point only one curve has.
  2) temporal_minlogit_combined.png — both budgets on a SINGLE axes (color = method, linestyle = budget).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = {"sm1": 0.770, "s0": 1.361, "s1": 3.812, "s2": 8.115, "s3": 14.774, "s4": 24.292}

# trimmed to shared shapes per budget (MoE s3@1e16 and dense s4@1e17 dropped — unmatched extremes)
MOE = {
    "1e16": {"sm1": 1.478, "s0": 1.447, "s1": 1.540, "s2": 1.819},
    "1e17": {"s1": 1.284, "s2": 1.269, "s3": 1.289},
}
DENSE = {
    "1e16": {"sm1": 1.534, "s0": 1.519, "s1": 1.591, "s2": 1.848},
    "1e17": {"s1": 1.361, "s2": 1.341, "s3": 1.408},
}
TEMPORAL = {"1e16": ("s0", 1.4599), "1e17": ("s2", 1.2821)}  # min_logit, 1 shared, optimal shape

def pts(d):
    xs = np.array([N[k] for k in d]); ys = np.array([d[k] for k in d])
    o = np.argsort(xs); return xs[o], ys[o]

# ---------- 1) trimmed 2-panel ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, b in zip(axes, ["1e16", "1e17"]):
    for label, data, color in [("dense floor", DENSE[b], "C3"), ("MoE (1 shared)", MOE[b], "C0")]:
        x, y = pts(data); ax.plot(x, y, "o-", color=color, label=label, lw=1.7, ms=7)
    shape, bpb = TEMPORAL[b]
    ax.plot(N[shape], bpb, "*", color="C2", ms=20, zorder=5, label="temporal (min_logit, 1 shared)")
    ax.annotate(f"{bpb:.4f}", (N[shape], bpb), textcoords="offset points", xytext=(10, -4),
                fontsize=10, color="C2", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("active non-embedding params N (millions)")
    ax.set_ylabel("validation BPB (bits/byte, lower better)")
    ax.set_title(f"IsoFLOP @ {b} FLOPs")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(fontsize=9)
fig.suptitle("Temporal MoE (6/64 experts resident) sits inside the dense↔MoE band", fontsize=12)
fig.tight_layout(); fig.savefig("results/phase0/temporal_minlogit_2panel.png", dpi=130)
print("wrote results/phase0/temporal_minlogit_2panel.png")

# ---------- 2) combined single axes ----------
fig, ax = plt.subplots(figsize=(9, 6))
STYLE = {"1e16": "--", "1e17": "-"}
for b in ["1e16", "1e17"]:
    for label, data, color in [("dense", DENSE[b], "C3"), ("MoE (1 shared)", MOE[b], "C0")]:
        x, y = pts(data)
        ax.plot(x, y, STYLE[b] + "o", color=color, lw=1.7, ms=6, label=f"{label}  {b}")
    shape, bpb = TEMPORAL[b]
    ax.plot(N[shape], bpb, "*", color="C2", ms=20, zorder=5,
            markeredgecolor="k", markeredgewidth=0.5, label=f"temporal (min_logit, 1sh)  {b}")
    ax.annotate(f"{bpb:.4f}", (N[shape], bpb), textcoords="offset points", xytext=(9, 4),
                fontsize=9, color="C2", fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("active non-embedding params N (millions)")
ax.set_ylabel("validation BPB (bits/byte, lower better)")
ax.set_title("Dense vs MoE (1 shared) vs temporal — both FLOP budgets\n"
             "(dashed = 1e16, solid = 1e17; lower is better)")
ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(fontsize=9, ncol=2)
fig.tight_layout(); fig.savefig("results/phase0/temporal_minlogit_combined.png", dpi=130)
print("wrote results/phase0/temporal_minlogit_combined.png")
