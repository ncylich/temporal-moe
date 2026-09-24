#!/usr/bin/env python3
"""Serving comparison across devices: RTX A6000, Pixel 10a, laptop (i7-11370H, Windows native).

Columns are devices, rows are prefill throughput, decode throughput and peak memory, each against
context length; a bottom panel overlays decode deploy/ceiling for every device. Fine 18-of-192
model, deploy = R = k (only the active experts resident).

Sources (values copied, not re-measured):
  A6000   analysis/plots/plot_serving_context.py (results/ablations/serving_benchmarks.csv,
          context_sweep rows, ubatch 2048). Memory is peak VRAM only; the CPU expert pool is not counted.
  Pixel   androidbench/results/ctx_designpoint_2026-08-24.csv, fine shape. Ceiling is E=80, the largest
          model that fits resident. No prefill and no memory were recorded.
  Laptop  comms/laptop/w1/runs.jsonl (decode tok/s and VmHWM per depth, tag w1 sitting, ledger L1-7)
          and ledger L1-10/L1-11 (prefill, prompt 512, ubatch 512, deploy uncapped). Memory is process
          peak working set. The packed CSV omits the depth-2048/4096 ceiling rows; use runs.jsonl.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "serving_figures_by_device.png")
CEIL, DEPL = "#5aa0dd", "#2ca02c"
POS = {0: 0, 512: 0.5, 1024: 1, 2048: 2, 4096: 3, 8192: 4, 16384: 5}
TICKS = ([0, 1, 2, 3, 4, 5], ["0", "1k", "2k", "4k", "8k", "16k"])
C6, C4 = [1024, 2048, 4096, 8192, 16384], [0, 1024, 2048, 4096]

A6000 = dict(pp=([6756, 6625, 6568, 6284, 5608], [2906, 3570, 3715, 3618, 3568]),
             tg=([203.4, 203.0, 201.9, 201.1, 202.4], [161.7, 159.6, 159.2, 159.5, 159.3]),
             mem=([8.174, 8.840, 8.936, 9.128, 9.512], [2.172, 2.918, 3.014, 3.206, 3.590]))
PIXEL_TG = ([32.416, 25.440, 20.223, 14.578], [20.701, 17.804, 15.656, 12.158])
LAPTOP = dict(pp=(74.39, 66.7),
              tg=([32.84, 25.89, 20.42, 13.17], [22.82, 16.74, 13.83, 9.93]),
              mem=([5.703, 5.970, 6.151, 6.511], [0.799, 2.563, 2.821, 3.257]))

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
xs = lambda c: [POS[x] for x in c]


def axes(ax, xl):
    ax.set_xticks(*TICKS); ax.set_xlim(-0.4, 5.4); ax.set_xlabel(xl); ax.grid(True, ls=":", alpha=0.4)


def na(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="0.45")
    ax.set_xticks([]); ax.set_yticks([])


def pair(ax, c, ceil, depl):
    ax.plot(xs(c), ceil, "-o", color=CEIL, ms=6, lw=2)
    ax.plot(xs(c), depl, "-s", color=DEPL, ms=6, lw=2)


fig = plt.figure(figsize=(15, 15.3))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.95], hspace=0.35, top=0.935)
A = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
for j, t in enumerate(["RTX A6000 (CUDA, PCIe pool)", "Pixel 10a (CPU, UFS)", "Laptop i7-11370H (CPU, NVMe, Windows)"]):
    A[0][j].set_title(t, fontsize=13, fontweight="bold", pad=12)

pair(A[0][0], C6, *A6000["pp"])
na(A[0][1], "not measured")
A[0][2].plot([POS[512]], [LAPTOP["pp"][0]], "o", color=CEIL, ms=7)
A[0][2].plot([POS[512]], [LAPTOP["pp"][1]], "s", color=DEPL, ms=7)
A[0][2].text(0.97, 0.06, "prompt 512 only; deploy ran uncapped", ha="right",
             transform=A[0][2].transAxes, fontsize=8.5, color="0.35")
pair(A[1][0], C6, *A6000["tg"]); pair(A[1][1], C4, *PIXEL_TG); pair(A[1][2], C4, *LAPTOP["tg"])
pair(A[2][0], C6, *A6000["mem"]); na(A[2][1], "not measured\n(the phone runner records no memory)")
pair(A[2][2], C4, *LAPTOP["mem"])

for j in (0, 2):
    axes(A[0][j], "prompt length (tokens)"); axes(A[2][j], "context length (tokens)"); A[2][j].set_ylim(0, 10.5)
for j in range(3):
    axes(A[1][j], "context length (tokens)")
A[0][0].set_ylim(0, 7600); A[0][2].set_ylim(0, 85)
A[1][0].set_ylim(0, 230); A[1][1].set_ylim(0, 38); A[1][2].set_ylim(0, 38)
for i, t in enumerate(["Prefill\nthroughput (tok/s)", "Decode\nthroughput (tok/s)", "Peak memory (GB)"]):
    A[i][0].set_ylabel(t, fontsize=11)

ov = fig.add_subplot(gs[3, :])
ov.plot([1, 2, 3, 4, 5], [d / c for d, c in zip(A6000["tg"][1], A6000["tg"][0])], "-o", color="#5aa0dd", lw=2.2, ms=7, label="RTX A6000")
ov.plot([0, 1, 2, 3], [d / c for d, c in zip(PIXEL_TG[1], PIXEL_TG[0])], "-s", color="#e0801f", lw=2.2, ms=7, label="Pixel 10a")
ov.plot([0, 1, 2, 3], [d / c for d, c in zip(LAPTOP["tg"][1], LAPTOP["tg"][0])], "-^", color="#2ca02c", lw=2.2, ms=7, label="Laptop i7-11370H")
axes(ov, "context depth (tokens)"); ov.set_ylim(0.5, 1.0); ov.set_ylabel("decode: deploy / ceiling")
ov.set_title("Overlay: decode deploy/ceiling vs context depth, R = k on every device", fontsize=12)
ov.legend(loc="lower right", ncol=3, fontsize=10)

H = [Line2D([0], [0], color=CEIL, marker="o", lw=2, ms=7), Line2D([0], [0], color=DEPL, marker="s", lw=2, ms=7)]
fig.legend(H, ["standard MoE, all experts resident (ceiling)", "Temporal MoE, R = k (deploy)"],
           ncol=2, loc="upper center", frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.99))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight"); print("wrote", OUT)
