#!/usr/bin/env python3
"""Locus-by-depth figure: does routing get more contextual as you go deeper?

y = median over experts of (context AUC - token AUC) for one MoE layer. Above 0 = that layer's
routing is better predicted by the excluded-context mean than by the current token; below 0 =
lexical. x = MoE layer index (layer 1 is dense in every config -- moe-layer-freq [0]*1+[1]*(L-1) --
so there is no layer-1 point to plot).

Colors follow the isoFLOP standard used by plot_mechinterp.py: hue = method (MoE blue, temporal
green), shade = granularity (fine 18/192 dark, coarse 6/64 normal). Marker = compute budget.

Sources, and the window each row was measured at (w = context half-width):
  results/ablations/mechinterp_locus.csv       variant kfull = w=k, kwin = w=k/2, base = w=32
  results/ablations/mechinterp_locus_1e19.csv  variant base  = w=k (delex_locus.py default)
We plot w=k everywhere it exists. s0_SOFTMAX_BASELINE was only ever run at w=32, so it is drawn
dashed; its sigmoid-router sibling (s0_FULL, same budget/granularity) does have w=k and is drawn
solid, which brackets what the missing measurement would show.

Broken y-axis: the two regimes are ~0.3 AUC apart, so a single scale hides the per-layer slopes
that are the point of the figure.

Output: results/phase0/figures/locus_by_layer[_nocaption].png
"""
import csv, math, os, statistics as st, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "results", "ablations")
OUT = os.path.join(REPO, "results", "phase0", "figures")

MOE_FINE, MOE_COARSE = "#0d3b66", "#5aa0dd"
TMP_FINE, TMP_COARSE = "#145a14", "#5cc85c"
BUDGET_MARKER = {"1e16": "o", "1e17": "s", "1e19": "^"}

# (file, label, variant, color, budget, legend, linestyle)
SERIES = [
    ("mechinterp_locus.csv",      "s0_TEMPORAL",         "kfull", TMP_FINE,   "1e16", "temporal 18/192", "-"),
    ("mechinterp_locus_1e19.csv", "temporal_fine_1e19",  "base",  TMP_FINE,   "1e19", "temporal 18/192", "-"),
    ("mechinterp_locus.csv",      "s2_TEMPORAL",         "kfull", TMP_COARSE, "1e17", "temporal 6/64",   "-"),
    ("mechinterp_locus_1e19.csv", "temporal_coarse_1e19", "base", TMP_COARSE, "1e19", "temporal 6/64",   "-"),
    ("mechinterp_locus.csv",      "s0_FULL",             "kfull", MOE_FINE,   "1e16", "full MoE 18/192 (sigmoid)", "-"),
    ("mechinterp_locus.csv",      "s0_SOFTMAX_BASELINE", "base",  MOE_FINE,   "1e16", "full MoE 18/192 (w=32 only)", "--"),
    ("mechinterp_locus.csv",      "s2_FULL",             "kfull", MOE_COARSE, "1e17", "full MoE 6/64",   "-"),
    ("mechinterp_locus_1e19.csv", "moe_coarse_1e19",     "base",  MOE_COARSE, "1e19", "full MoE 6/64",   "-"),
]


def per_layer(fname, label, variant):
    """-> {layer: median over experts of (context_AUC - token_AUC)}, dropping non-finite probes."""
    g = defaultdict(list)
    with open(os.path.join(DATA, fname)) as f:
        for r in csv.DictReader(f):
            if r["label"] != label or r["variant"] != variant:
                continue
            try:
                d = float(r["context_minus_token"])
            except (ValueError, TypeError):
                continue
            if not math.isnan(d):
                g[int(r["layer"])].append(d)
    return {ln: st.median(v) for ln, v in g.items()}, {ln: len(v) for ln, v in g.items()}


if PAPER:
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "legend.fontsize": 9,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})

fig, (hi, lo) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.2) if PAPER else (8.0, 7.2),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})

counts = []
for fname, label, variant, color, budget, legend, ls in SERIES:
    med, n = per_layer(fname, label, variant)
    if not med:
        print(f"[warn] no rows for {label}/{variant}", file=sys.stderr)
        continue
    xs = sorted(med)
    ys = [med[x] for x in xs]
    ax = hi if ys[0] > 0 else lo
    ax.plot(xs, ys, ls, color=color, marker=BUDGET_MARKER[budget], markersize=7,
            linewidth=1.8, markeredgecolor="white", markeredgewidth=0.8,
            label=f"{legend} @ {budget}")
    counts.append(f"{label}: n={min(n.values())}-{max(n.values())}/layer")

hi.axhline(0, color="#888", linewidth=1.0, linestyle=":")
hi.set_ylim(-0.02, 0.21)
lo.set_ylim(-0.36, -0.15)
lo.set_yticks([-0.35, -0.30, -0.25, -0.20, -0.15])
hi.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
hi.spines["bottom"].set_visible(False)
lo.spines["top"].set_visible(False)
hi.tick_params(labeltop=False, bottom=False)

# diagonal break marks
kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle="none",
          color="k", mec="k", mew=1, clip_on=False)
hi.plot([0, 1], [0, 0], transform=hi.transAxes, **kw)
lo.plot([0, 1], [1, 1], transform=lo.transAxes, **kw)

for ax in (hi, lo):
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xticks([2, 3, 4, 5, 6])
lo.set_xlabel("MoE layer index  (layer 1 is dense in every config)")
fig.supylabel("median over experts:  context AUC $-$ token AUC", x=0.035, fontsize=13)
hi.text(0.012, 0.90, "context-dominated (temporal)", transform=hi.transAxes,
        fontsize=10, color="#145a14", weight="bold")
lo.text(0.012, 0.08, "token-dominated (unconstrained MoE)", transform=lo.transAxes,
        fontsize=10, color="#0d3b66", weight="bold")

hh, ll = hi.get_legend_handles_labels()
h2, l2 = lo.get_legend_handles_labels()
lo.legend(hh + h2, ll + l2, loc="upper center", bbox_to_anchor=(0.5, -0.28),
          framealpha=0.95, ncol=2, handlelength=2.6, columnspacing=1.4)

if PAPER:
    out = os.path.join(OUT, "locus_by_layer_nocaption.png")
else:
    lo.text(-0.14, -0.72,
             "Locus of routing specialization by depth. Per (layer, expert) logistic probes predict\n"
             "whether expert e serves token t, from either the current token embedding E[x_t] or the\n"
             "excluded-context mean over +-w neighbours; AUC is held-out (fit 70%, score 30%), chance\n"
             "floor 0.500+-0.002. Points are medians over that layer's experts. Colour = setup (blue =\n"
             "unconstrained MoE, green = temporal; dark = fine 18/192, light = coarse 6/64), marker =\n"
             "compute budget (circle 1e16, square 1e17, triangle 1e19). Note the broken y-axis. Only\n"
             "MoE layers 2-6 were probed: the 1e18/1e19 models have 9 layers, so layers 7-9 are missing.",
             transform=lo.transAxes, fontsize=8.6, va="top", ha="left",
             family="monospace", color="#333")
    out = os.path.join(OUT, "locus_by_layer.png")

os.makedirs(OUT, exist_ok=True)
fig.savefig(out, dpi=190, bbox_inches="tight")
print("wrote", out)
print("  " + "; ".join(counts))
