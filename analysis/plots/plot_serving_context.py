#!/usr/bin/env python3
"""Serving throughput + memory vs context length: temporal (deploy) vs all-resident MoE.
Two panels (prefill | decode), X = context, left Y = throughput (tok/s, higher better),
right Y = peak VRAM (GB). Data: results/ablations/serving_benchmarks.csv context_sweep rows
(fine model, -ub 2048, RTX A6000). 8 GB reference line = entry-level consumer GPU.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = f"{REPO}/results/phase0/figures"

ctx = [1024, 2048, 4096, 8192, 16384]
# ceiling (A), deploy (C): prefill tok/s, decode tok/s, VRAM MiB
A_pp = [6756, 6625, 6568, 6284, 5608]; C_pp = [2906, 3570, 3715, 3618, 3568]  # deploy = temporal residency-masked prefill
A_tg = [203.4, 203.0, 201.9, 201.1, 202.4]; C_tg = [161.7, 159.6, 159.2, 159.5, 159.3]
A_vram = [8.174, 8.840, 8.936, 9.128, 9.512]; C_vram = [2.172, 2.918, 3.014, 3.206, 3.590]

CEIL, DEPL, VRC = "#5aa0dd", "#2ca02c", "0.45"
PAPER = True
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.5})

def panel(ax, tps_A, tps_C, title, show_8gb=True):
    axr = ax.twinx()
    # VRAM (right, dashed) first so throughput draws on top
    axr.plot(ctx, A_vram, "--o", color=CEIL, ms=4, lw=1.4, alpha=0.85)
    axr.plot(ctx, C_vram, "--s", color=DEPL, ms=4, lw=1.4, alpha=0.85)
    if show_8gb:
        axr.axhline(8.0, color="crimson", ls=":", lw=1.2)
        axr.text(1024, 8.15, "8 GB GPU", color="crimson", fontsize=7, va="bottom")
    axr.set_ylim(0, 11); axr.set_ylabel("peak VRAM (GB)", color=VRC)
    axr.tick_params(axis="y", labelcolor=VRC)
    # throughput (left, solid)
    ax.plot(ctx, tps_A, "-o", color=CEIL, ms=5, lw=2.0, label="all-resident MoE")
    ax.plot(ctx, tps_C, "-s", color=DEPL, ms=5, lw=2.0, label="temporal (ours)")
    ax.set_xscale("log", base=2); ax.set_xticks(ctx); ax.set_xticklabels(["1k","2k","4k","8k","16k"])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("throughput (tok/s, higher better)")
    ax.set_ylim(0, max(tps_A)*1.12); ax.set_title(title); ax.grid(True, ls=":", alpha=0.35)
    return axr

fig, (axp, axd) = plt.subplots(1, 2, figsize=(7.4, 3.1))
panel(axp, A_pp, C_pp, "Prefill")
panel(axd, A_tg, C_tg, "Decode (100-token gen)")
# legend: method = color, metric = linestyle (kept separate so the dotted 8 GB line reads as VRAM)
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color=CEIL, marker="o", ls="", ms=8),
           Line2D([0],[0], color=DEPL, marker="s", ls="", ms=8),
           Line2D([0],[0], color="0.25", lw=2.0, ls="-"),
           Line2D([0],[0], color="0.45", lw=1.6, ls="--")]
labels  = ["all-resident MoE", "temporal (ours)", "throughput (left axis)", "peak VRAM (right axis)"]
fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=[0,0,1,0.93])
out = f"{OUT}/serving_context_sweep_nocaption.png"
fig.savefig(out, dpi=200); print("wrote", out)
