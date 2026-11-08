#!/usr/bin/env python3
"""Share of the MoE-over-dense gain the fine-grained Temporal MoE keeps, against the strongest
standard MoE at each compute budget. Test BPB at each budget's compute-optimal size, the same
values plotted in Fig. 3 (analysis/plots/plot_isoflop_panels.py). Writes
paper/figures/recovery_vs_compute.pdf."""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
B = {1e16: dict(dense=1.519, moe_c=1.447, moe_f=1.4585, tmp_c=1.4599, tmp_f=1.4753),
     1e17: dict(dense=1.341, moe_c=1.269, moe_f=1.2708, tmp_c=1.2821, tmp_f=1.2873),
     1e18: dict(dense=1.3911, moe_c=1.3175, moe_f=1.3478, tmp_c=1.3122, tmp_f=1.3339),
     1e19: dict(dense=1.1260, moe_c=1.0514, moe_f=1.0604, tmp_c=1.0680, tmp_f=1.0655)}
C = sorted(B)
def kept(d, t):
    best = min(d["moe_c"], d["moe_f"])
    return 100 * (d["dense"] - d[t]) / (d["dense"] - best)
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(4.4, 2.6))
for t, col, lab, lw in [("tmp_f", "#0b5c1c", "Temporal MoE, fine (18 of 192)", 2.6)]:
    ys = [kept(B[c], t) for c in C]
    ax.plot(C, ys, "-o", color=col, lw=lw, label=lab)
    for c, y in zip(C, ys):
        ax.annotate(f"{y:.0f}%", (c, y), textcoords="offset points", xytext=(0, -14) if (t == "tmp_c" and c == 1e19) else (0, 7), ha="center", fontsize=9.5, color=col)
ax.axhline(100, ls="--", color="#888", lw=1)
ax.text(1.2e16, 101.5, "matches the strongest standard MoE", fontsize=9, color="#666")
ax.set_xscale("log"); ax.set_xticks(C); ax.set_xticklabels(["$10^{16}$", "$10^{17}$", "$10^{18}$", "$10^{19}$"])
ax.minorticks_off(); ax.set_ylim(50, 108)
ax.set_xlabel("training compute (FLOPs)"); ax.set_ylabel("MoE gain kept (%)")
ax.grid(ls=":", alpha=0.4); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
for d in (os.path.join(REPO, "results/phase0/figures"), os.path.join(REPO, "paper/figures")):
    fig.savefig(os.path.join(d, "recovery_vs_compute.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(d, "recovery_vs_compute.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote recovery_vs_compute")
