#!/usr/bin/env python3
"""Plot the dense IsoFLOP floor against the FLAME-MoE frontier at 1e16 and 1e17 FLOPs.

Combined single axes: dense (vanilla SwiGLU) and MoE parabolas over active non-embedding params N,
both budgets overlaid (color = method, linestyle = budget: dashed 1e16 / solid 1e17). Metric is
bits-per-byte (lower better). All points measured (results/phase0/log.md). Output: dense_vs_moe.png
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

LSTYLE = {"1e16": "--", "1e17": "-"}
fig, ax = plt.subplots(figsize=(9.5, 6.2))
for budget in ["1e16", "1e17"]:
    for label, data, color in [("full MoE", MOE[budget], "C0"),
                               ("dense baseline", DENSE[budget], "C3")]:
        x, y, names = pts(data)
        ax.plot(x, y, LSTYLE[budget], color=color, marker="o", lw=1.8, ms=6,
                label=f"{label}  ({budget} FLOPs)")
        # parabola fit in log10(N) for a smooth guide + located minimum
        lx = np.log10(x)
        c = np.polyfit(lx, y, 2)
        xx = np.linspace(lx.min(), lx.max(), 100)
        ax.plot(10**xx, np.polyval(c, xx), ":", color=color, lw=1, alpha=0.4)
        imin = int(np.argmin(y))
        ax.annotate(f"{N[names[imin]]:.2f}M active", (x[imin], y[imin]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=color)
ax.set_xscale("log")
ax.set_xlabel("active non-embedding params N (millions)")
ax.set_ylabel("validation BPB (bits/byte, lower better)")
ax.set_title("Full MoE beats the dense-baseline compute floor at every model size\n"
             "IsoFLOP curves at two compute budgets; lower bits-per-byte is better")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=9, ncol=2)
fig.text(0.5, 0.01,
         "IsoFLOP curves: validation bits-per-byte (BPB, lower is better) vs active non-embedding "
         "parameters N (millions, log x-axis). Dashed = 10^16 FLOPs, solid = 10^17 FLOPs. 'full MoE' = "
         "standard top-k routing; 'dense baseline' = a plain feed-forward model. Dotted lines are quadratic "
         "fits; labels mark the best (minimum-BPB) model size on each curve.", ha="center", fontsize=8, wrap=True)
fig.tight_layout(rect=[0, 0.07, 1, 1])
out = "/workspace/FLAME-MoE/results/phase0/figures/dense_floor_vs_full_moe_isoflop.png"
fig.savefig(out, dpi=130)
print("wrote", out)
