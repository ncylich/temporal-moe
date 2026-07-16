#!/usr/bin/env python3
"""Per-budget isoFLOP panels for Figure 1: one INDEPENDENT graph per compute decade.

Each budget is plotted alone (own axes, own y-range) because the sweeps belong to two model
families that are NOT cross-comparable at these scales: 1e16/1e17 use the 16k-BPE tokenizer,
1e18/1e19 use pythia-50k (FLAME paper protocol). BPB is tokenizer-invariant as a loss unit, but a
50k-vocab model at micro scale spends most of its params/FLOPs on the vocab (h192: 19.3M embed on a
6.9M transformer), so cross-family absolute comparisons on one axis are misleading — within-panel
comparisons are the honest ones.

Values = end-of-training TEST evals (canonical; see results/ablations/FINDINGS.md):
- 1e16/1e17: BPB = CE/2.7568 (16k), per-point data in phase0_isoflop_points.csv.
- 1e18: BPB = CE/2.9780 (50k), single seed (1234; for the two-seed 38M coarse cells this is
  seed 1) — flame192_leftflank_1e18.csv, flame38m_1e18_cells.csv, flame512_1e18_rightflank.csv.
  x = non-embed active params (6.88/12.19/48.50M from the run configs).
- 1e19: BPB = CE/2.9780, t19_1e19_curves.csv (one shape; bars). No fine full-MoE cell was trained
  at 1e19 (1e18 already showed fine-graining hurts the full MoE; temporal is the fine contender).

Outputs results/phase0/figures/isoflop_panel_1e{16,17,18,19}_nocaption.png (paper tiles) and a
captioned 2x2 overview isoflop_panels_all.png for the repo.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTD = f"{REPO}/results/phase0/figures"

DENSE_C = "#7f7f7f"
MOE_COARSE, MOE_FINE = "#5aa0dd", "#0d3b66"
TMP_COARSE, TMP_FINE = "#5cc85c", "#145a14"

# per-budget: series -> {N_active_M: test BPB}
P16 = {
    "dense":      {0.770: 1.534, 1.361: 1.519, 3.812: 1.591},
    "moe_c":      {0.770: 1.4766, 1.361: 1.447, 3.812: 1.540},
    "moe_f":      {0.81: 1.4786, 1.42: 1.4585, 3.91: 1.5352},
    "tmp_c":      {0.770: 1.4872, 1.361: 1.4599, 3.812: 1.5473},
    "tmp_f":      {0.81: 1.4976, 1.42: 1.4753, 3.91: 1.5861},
}
P17 = {
    "dense":      {3.812: 1.361, 8.115: 1.341, 14.774: 1.408},
    "moe_c":      {3.812: 1.2803, 8.115: 1.269, 14.774: 1.289},
    "moe_f":      {3.91: 1.2846, 8.23: 1.2708, 15.09: 1.2815},
    "tmp_c":      {3.812: 1.3027, 8.115: 1.2821, 14.774: 1.3061},
    "tmp_f":      {3.91: 1.3065, 8.23: 1.2873, 15.09: 1.3129},
}
P18 = {
    "dense":      {6.88: 1.4072, 12.19: 1.3893, 48.50: 1.4256},
    "moe_c":      {6.88: 1.3272, 12.19: 1.3158, 48.50: 1.3766},
    "moe_f":      {6.88: 1.3570, 12.19: 1.3461, 48.50: 1.4174},
    "tmp_c":      {6.88: 1.3198, 12.19: 1.3128, 48.50: 1.3762},
    "tmp_f":      {6.88: 1.3379, 12.19: 1.3354, 48.50: 1.4047},
}
P19 = [("dense", 1.1260, DENSE_C), ("temporal\ncoarse", 1.0680, TMP_COARSE),
       ("temporal\nfine", 1.0655, TMP_FINE), ("full MoE\ncoarse", 1.0514, MOE_COARSE)]

STYLE = [("dense", DENSE_C, 1.4), ("moe_c", MOE_COARSE, 1.9), ("moe_f", MOE_FINE, 1.9),
         ("tmp_c", TMP_COARSE, 1.9), ("tmp_f", TMP_FINE, 1.9)]
LEG = ["dense", "MoE · coarse", "MoE · fine", "temporal · coarse", "temporal · fine"]


def curve_panel(ax, data, title, legend=False, ticks=None, xlabel=True, ylabel=True):
    for key, color, lw in STYLE:
        d = data[key]
        xs = sorted(d); ys = [d[x] for x in xs]
        ax.plot(xs, ys, "-o", color=color, mfc=color, mec=color, ms=6, lw=lw, alpha=0.9)
    ax.set_xscale("log")
    # clean ticks at the swept sizes only — no garbled log-minor labels
    if ticks:
        ax.set_xticks(ticks[0]); ax.set_xticklabels(ticks[1])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(True, which="major", ls=":", alpha=0.4)
    ax.set_title(title)
    if xlabel: ax.set_xlabel("active params (M)")
    if ylabel: ax.set_ylabel("test BPB")
    if legend:
        ax.legend([Line2D([0], [0], color=c, lw=2.2) for _, c, _ in STYLE], LEG,
                  fontsize=7.5, framealpha=0.9)


def bar_panel(ax, title, ylabel=True):
    labels = [p[0] for p in P19]; vals = [p[1] for p in P19]; cols = [p[2] for p in P19]
    bars = ax.bar(labels, vals, color=cols, width=0.66, edgecolor="k", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + 0.002, f"{v:.3f}", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_ylim(1.0, 1.16)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.set_title(title)
    if ylabel: ax.set_ylabel("test BPB")
    ax.tick_params(axis="x", labelsize=8.5)


# paper tiles (one file per budget, no baked caption)
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10})
TICKS = {"1e16": ([0.77, 1.4, 3.85], ["0.8", "1.4", "3.9"]),
         "1e17": ([3.85, 8.17, 14.9], ["3.9", "8.1", "15"]),
         "1e18": ([6.88, 12.19, 48.50], ["6.9", "12", "49"])}
for name, data, vocab, legend in [("1e16", P16, "16k", True), ("1e17", P17, "16k", False), ("1e18", P18, "50k", False)]:
    fig, ax = plt.subplots(figsize=(3.7, 2.35))
    curve_panel(ax, data, f"$10^{{{name[2:]}}}$ FLOPs · {vocab} vocab", legend=legend, ticks=TICKS[name])
    fig.tight_layout()
    out = f"{OUTD}/isoflop_panel_{name}_nocaption.png"
    fig.savefig(out, dpi=200); print("wrote", out); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.7, 2.35))
bar_panel(ax, "$10^{19}$ FLOPs · 50k vocab")
fig.tight_layout()
out = f"{OUTD}/isoflop_panel_1e19_nocaption.png"
fig.savefig(out, dpi=200); print("wrote", out); plt.close(fig)

# PAPER GRID: one tight 2x2 image with a single shared legend (used by paper/main.tex)
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5))
curve_panel(axes[0][0], P16, "$10^{16}$ FLOPs · 16k vocab", ticks=TICKS["1e16"], xlabel=False)
curve_panel(axes[0][1], P17, "$10^{17}$ FLOPs · 16k vocab", ticks=TICKS["1e17"], xlabel=False, ylabel=False)
curve_panel(axes[1][0], P18, "$10^{18}$ FLOPs · 50k vocab", ticks=TICKS["1e18"])
bar_panel(axes[1][1], "$10^{19}$ FLOPs · 50k vocab", ylabel=False)
fig.legend([Line2D([0], [0], color=c, lw=2.4) for _, c, _ in STYLE], LEG,
           ncol=5, loc="upper center", fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.5, 1.005), columnspacing=1.4, handlelength=1.6)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.subplots_adjust(wspace=0.16, hspace=0.42)
out = f"{OUTD}/isoflop_panels_2x2_nocaption.png"
fig.savefig(out, dpi=200); print("wrote", out); plt.close(fig)

# captioned 2x2 overview for the repo
fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.6))
curve_panel(axes[0][0], P16, "$10^{16}$ FLOPs · 16k vocab", legend=True, ticks=TICKS["1e16"])
curve_panel(axes[0][1], P17, "$10^{17}$ FLOPs · 16k vocab", ticks=TICKS["1e17"])
curve_panel(axes[1][0], P18, "$10^{18}$ FLOPs · 50k vocab", ticks=TICKS["1e18"])
bar_panel(axes[1][1], "$10^{19}$ FLOPs · 50k vocab")
fig.suptitle("Quality at fixed compute — one independent panel per budget", fontsize=13)
fig.text(0.5, 0.005,
         "Test-set bits-per-byte (lower better) at four compute budgets. Hue = method (dense gray, full MoE blue, "
         "temporal green); shade = granularity (coarse 6-of-64 normal, fine 18-of-192 dark). Panels are independent: "
         "1e16/1e17 use the 16k-BPE tokenizer, 1e18/1e19 use pythia-50k, so compare within a panel, not across. "
         "'temporal' = rolling residency (top-k resident, swap 1/token).",
         ha="center", fontsize=8, color="dimgray", wrap=True)
fig.tight_layout(rect=[0, 0.045, 1, 0.97])
out = f"{OUTD}/isoflop_panels_all.png"
fig.savefig(out, dpi=200); print("wrote", out)
