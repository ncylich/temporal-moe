#!/usr/bin/env python3
"""Plot the dense IsoFLOP floor against the FLAME-MoE frontier at 1e16 and 1e17 FLOPs.

Two panels (one per budget); each shows the dense (vanilla SwiGLU) parabola and the MoE
parabola over active non-embedding params N. Metric is bits-per-byte (lower better). All points
are measured (results/phase0/log.md). Output: results/phase0/dense_vs_moe.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# N_active (non-embedding) per shape, millions
N = {"sm1": 0.770, "s0": 1.361, "s1": 3.812, "s2": 8.115, "s3": 14.774, "s4": 24.292}

# measured BPB (bits-per-byte, lower is better). MoE: PASS.md; dense: log.md.
MOE = {
    "1e16": {"sm1": 1.478, "s0": 1.447, "s1": 1.540, "s2": 1.819, "s3": 2.187},  # sm1==s_-1 geom
    "1e17": {"s1": 1.284, "s2": 1.269, "s3": 1.289},
}
DENSE = {
    "1e16": {"sm1": 1.534, "s0": 1.519, "s1": 1.591, "s2": 1.848},
    "1e17": {"s1": 1.361, "s2": 1.341, "s3": 1.408, "s4": 1.485},
}

def pts(d):
    xs = [N[k] for k in d]
    ys = [d[k] for k in d]
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order], [list(d.keys())[i] for i in order]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, budget in zip(axes, ["1e16", "1e17"]):
    for label, data, color in [("MoE (FLAME)", MOE[budget], "C0"),
                               ("dense floor", DENSE[budget], "C3")]:
        x, y, names = pts(data)
        ax.plot(x, y, "o-", color=color, label=label, lw=1.6, ms=7)
        # parabola fit in log10(N) for a smooth guide + located minimum
        lx = np.log10(x)
        c = np.polyfit(lx, y, 2)
        xx = np.linspace(lx.min(), lx.max(), 100)
        ax.plot(10**xx, np.polyval(c, xx), "--", color=color, lw=1, alpha=0.5)
        imin = int(np.argmin(y))
        ax.annotate(names[imin], (x[imin], y[imin]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("active non-embedding params N (millions)")
    ax.set_ylabel("validation BPB (bits/byte, lower better)")
    ax.set_title(f"IsoFLOP @ {budget} FLOPs")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
fig.suptitle("FLAME-MoE beats the dense IsoFLOP floor at every shape (Phase 0, single A6000)",
             fontsize=12)
fig.tight_layout()
fig.savefig("results/phase0/dense_vs_moe.png", dpi=130)
print("wrote results/phase0/dense_vs_moe.png")
