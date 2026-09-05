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
- 1e18: BPB = CE/2.9780 (50k); flanks single seed (1234), the 38M midpoint plots 3-seed means
  on the h100 split with its local dense floor —
  flame192_leftflank_1e18.csv, flame38m_1e18_cells.csv, flame512_1e18_rightflank.csv.
  x = non-embed active params (6.88/12.19/48.50M from the run configs).
- 1e19: BPB = CE/2.9780, t19_1e19_curves.csv (one shape; bars). The fine full-MoE cell was added
  on 2026-09-03 (moe_fine_g3_1e19, 1.0604): it beats the fine temporal model by 0.005 BPB, a third
  of the coarse pair's gap, so the 1e19 panel now shows both grains for both paradigms.

Outputs results/phase0/figures/isoflop_panel_1e{16,17,18,19}_nocaption.png (paper tiles) and a
captioned 2x2 overview isoflop_panels_all.png for the repo.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTD = f"{REPO}/results/phase0/figures"

# NON-DEFAULT deck variant: --highlight-deck fades two series (temporal coarse, fine MoE)
# and emphasizes the other three (dense, coarse MoE, temporal fine) for a talk slide. Without
# the flag every output below is byte-identical to the paper figures. --no-caption is accepted
# only for CLI parity (these tiles are already caption-less).
_ap = argparse.ArgumentParser()
_ap.add_argument("--highlight-deck", action="store_true")
_ap.add_argument("--no-caption", action="store_true")
_ap.add_argument("--out", default=f"{REPO}/paper/talk_figures/slide06_isoflop_highlight.png")
ARGS, _ = _ap.parse_known_args()
FADE_KEYS = {"moe_f", "tmp_c"}          # curve series faded on the deck variant
FADE_BARS = {"temporal\ncoarse"}        # 1e19 bar faded on the deck variant

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
    "dense":      {6.88: 1.4072, 12.19: 1.3911, 48.50: 1.4256},
    "moe_c":      {6.88: 1.3272, 12.19: 1.3175, 48.50: 1.3766},
    "moe_f":      {6.88: 1.3570, 12.19: 1.3478, 48.50: 1.4174},
    "tmp_c":      {6.88: 1.3198, 12.19: 1.3122, 48.50: 1.3762},
    "tmp_f":      {6.88: 1.3379, 12.19: 1.3339, 48.50: 1.4047},
}
# 38M midpoint seeds (h100 split, 3/arm; plotted values = means, per-seed table in the paper):
# tmp_c {1.3128,1.3111,1.3128}  moe_c {1.3158,1.3197,1.3169}
# tmp_f {1.3354,1.3339,1.3323}  moe_f {1.3461,1.3489,1.3483}
P19 = [("dense", 1.1260, DENSE_C), ("temporal\ncoarse", 1.0680, TMP_COARSE),
       ("temporal\nfine", 1.0655, TMP_FINE), ("full MoE\ncoarse", 1.0514, MOE_COARSE),
       ("full MoE\nfine", 1.0604, MOE_FINE)]   # moe_fine_g3_1e19, trained 2026-09-03 (t19_1e19_curves.csv)

STYLE = [("dense", DENSE_C, 1.4), ("moe_c", MOE_COARSE, 1.9), ("moe_f", MOE_FINE, 1.9),
         ("tmp_c", TMP_COARSE, 1.9), ("tmp_f", TMP_FINE, 1.9)]
LEG = ["dense", "MoE · coarse", "MoE · fine", "temporal · coarse", "temporal · fine"]


def curve_panel(ax, data, title, legend=False, ticks=None, xlabel=True, ylabel=True):
    for key, color, lw in STYLE:
        d = data[key]
        xs = sorted(d); ys = [d[x] for x in xs]
        a, lwx, ms = 0.9, lw, 6
        if ARGS.highlight_deck:
            if key in FADE_KEYS:
                a, lwx, ms = 0.15, 1.2, 4
            else:
                a, lwx, ms = 1.0, lw + 0.8, 7
        ax.plot(xs, ys, "-o", color=color, mfc=color, mec=color, ms=ms, lw=lwx, alpha=a)
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
                  fontsize=9.5, framealpha=0.9)


def bar_panel(ax, title, ylabel=True):
    labels = [p[0] for p in P19]; vals = [p[1] for p in P19]; cols = [p[2] for p in P19]
    bars = ax.bar(labels, vals, color=cols, width=0.66, edgecolor="k", linewidth=0.6)
    for b, v, lab in zip(bars, vals, labels):
        faded = ARGS.highlight_deck and lab in FADE_BARS
        if faded:
            b.set_alpha(0.2)
        ax.text(b.get_x()+b.get_width()/2, v + 0.002, f"{v:.3f}", ha="center",
                fontsize=11, fontweight="bold", alpha=0.25 if faded else 1.0)
    ax.set_ylim(1.0, 1.16)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.set_title(title)
    if ylabel: ax.set_ylabel("test BPB")
    ax.tick_params(axis="x", labelsize=9.5)


# paper tiles (one file per budget, no baked caption)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12.5})
TICKS = {"1e16": ([0.77, 1.4, 3.85], ["0.8", "1.4", "3.9"]),
         "1e17": ([3.85, 8.17, 14.9], ["3.9", "8.1", "15"]),
         "1e18": ([6.88, 12.19, 48.50], ["6.9", "12", "49"])}

# --highlight-deck: emit ONLY the faded talk-slide 2x2 and stop (paper outputs untouched).
if ARGS.highlight_deck:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.15))
    curve_panel(axes[0][0], P16, "$10^{16}$ FLOPs · 16k vocab", ticks=TICKS["1e16"], xlabel=False)
    curve_panel(axes[0][1], P17, "$10^{17}$ FLOPs · 16k vocab", ticks=TICKS["1e17"], xlabel=False, ylabel=False)
    curve_panel(axes[1][0], P18, "$10^{18}$ FLOPs · 50k vocab", ticks=TICKS["1e18"])
    bar_panel(axes[1][1], "$10^{19}$ FLOPs · 50k vocab", ylabel=False)
    fig.legend([Line2D([0], [0], color=c, lw=2.4) for _, c, _ in STYLE], LEG,
               ncol=5, loc="upper center", fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 1.005), columnspacing=1.4, handlelength=1.6)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(wspace=0.16, hspace=0.42)
    os.makedirs(os.path.dirname(ARGS.out), exist_ok=True)
    fig.savefig(ARGS.out, dpi=200); print("wrote", ARGS.out); plt.close(fig)
    sys.exit(0)
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
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.15))
curve_panel(axes[0][0], P16, "$10^{16}$ FLOPs · 16k vocab", ticks=TICKS["1e16"], xlabel=False)
curve_panel(axes[0][1], P17, "$10^{17}$ FLOPs · 16k vocab", ticks=TICKS["1e17"], xlabel=False, ylabel=False)
curve_panel(axes[1][0], P18, "$10^{18}$ FLOPs · 50k vocab", ticks=TICKS["1e18"])
bar_panel(axes[1][1], "$10^{19}$ FLOPs · 50k vocab", ylabel=False)
fig.legend([Line2D([0], [0], color=c, lw=2.6) for _, c, _ in STYLE], LEG,
           ncol=5, loc="upper center", fontsize=12.5, frameon=False,
           bbox_to_anchor=(0.5, 1.008), columnspacing=1.5, handlelength=1.7)
fig.tight_layout(rect=[0, 0, 1, 0.925])
fig.subplots_adjust(wspace=0.16, hspace=0.42)
out = f"{OUTD}/isoflop_panels_2x2_nocaption.png"
# tight bbox: the shared legend sits above the axes and was clipped at the right edge otherwise
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02); print("wrote", out); plt.close(fig)

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
