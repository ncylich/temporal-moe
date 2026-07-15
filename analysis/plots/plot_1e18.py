#!/usr/bin/env python3
"""1e18 (FLAME-MoE-38M-100M scale): dense floor vs full MoE (coarse & fine-grained) vs temporal.
Test-set cross-entropy (nats, lower better), pythia-50k tokenizer, dclm. Single axes.
Measured locally on one split: MoE coarse (6/64, 2 seeds), temporal coarse (6/64, 2 seeds),
MoE fine (18/192), temporal fine (18/192). Coarse bars show the two-seed mean of the END-OF-TRAINING
TEST evals (canonical series, train.log-verified; see results/ablations/FINDINGS.md section 7) with
individual seed dots. Dense floor reused from the A6000 baseline (test 4.1373; same corpus/tokenizer,
cross-data <~0.01 nats). Seed finals: results/ablations/t18_1e18_curves.csv (final_test_ce @2121)
and flame38m_1e18_cells.csv.
Output: results/phase0/figures/temporal_vs_dense_and_moe_1e18_crossentropy.png
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --no-caption = paper mode: drop the baked-in caption + in-plot finding annotations (they move to
# the LaTeX caption), use a compact figsize + short title/labels + large fonts so the bars stay
# legible after downscaling into a paper column. Default mode is unchanged.
PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PAPER:
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 12})

# worst -> best CE (left -> right). Colors match the left isoFLOP figure: hue = method
# (gray dense, blue MoE, green temporal), shade = granularity (coarse normal / fine dark).
# Only dense is A6000 cross-data (different split, within ~0.01 nats) — noted in the caption;
# the other four are local, one shared split. All values = end-of-training TEST evals. Coarse arms:
# two seeds each, bar = seed mean (moe (3.918414+3.930210)/2, temporal (3.909421+3.904324)/2).
# Error bars ONLY where measured: two-seed half-ranges on the coarse bars (MoE 0.0059,
# temporal 0.0025). Dense and the fine bars are single measurements — NO error bar drawn
# (dense's ~0.01-nat cross-data comparability bound is stated in the caption instead).
NAN = float("nan")
ce     = [4.1373, 4.0087, 3.9768, 3.9243, 3.9069]     # coarse bars = two-seed means (test)
yerr   = [NAN, NAN, NAN, 0.0059, 0.0025]              # measured coarse half-ranges only
colors = ["#7f7f7f", "#0d3b66", "#145a14", "#5aa0dd", "#5cc85c"]
labels = (["dense", "full MoE\nfine", "temporal\nfine", "full MoE\ncoarse", "temporal\ncoarse"] if PAPER else
          ["dense\nbaseline", "full MoE\nfine (18 of 192)", "temporal\nfine (18 of 192)",
           "full MoE\ncoarse (6 of 64)", "temporal\ncoarse (6 of 64)"])

fig, ax = plt.subplots(figsize=(4.6, 3.6) if PAPER else (8.6, 5.8))
bars = ax.bar(labels, ce, color=colors, width=0.66, edgecolor="k", linewidth=0.6,
              yerr=yerr, capsize=(4 if PAPER else 5), error_kw=dict(lw=1.3, ecolor="k", zorder=5))
for b, v, e in zip(bars, ce, yerr):
    off = (e if e == e else 0)  # nan -> no bar drawn
    ax.text(b.get_x()+b.get_width()/2, v + off + 0.005, f"{v:.3f}", ha="center",
            fontsize=(11 if PAPER else 10.5), fontweight="bold")
# individual seed finals (test) on the two-seed coarse bars
SEEDS = {3: [3.918414, 3.930210], 4: [3.909421, 3.904324]}   # moe coarse, temporal coarse
for i, vals in SEEDS.items():
    x = bars[i].get_x() + bars[i].get_width()/2
    ax.scatter([x]*len(vals), vals, s=16, color="k", zorder=6)
ax.grid(True, axis="y", ls=":", alpha=0.4)

if PAPER:
    ax.set_ylim(3.6, 4.20)
    ax.set_ylabel("test CE (nats)")
    ax.set_title("Quality at $10^{18}$ FLOPs")
    fig.tight_layout()
    out = f"{REPO}/results/phase0/figures/temporal_vs_dense_and_moe_1e18_crossentropy_nocaption.png"
else:
    # finding 1: fine-graining hurts the full MoE (fine 4.009 -> coarse 3.924 mean)
    ax.annotate("", xy=(3, 3.9243), xytext=(1, 4.0087),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.4))
    ax.text(2.0, 4.15, "fine-graining hurts the full MoE  (−0.084 nats)", ha="center", fontsize=8.5, color="darkred")
    # finding 2: temporal (fine) beats its own-granularity full MoE
    ax.annotate("", xy=(2, 3.9768), xytext=(1, 4.0087),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.2))
    ax.text(2.55, 4.05, "temporal beats its own-granularity full MoE", ha="left", fontsize=8, color="green")
    rec = (4.1373 - 3.9768) / (4.1373 - 3.9243) * 100
    ax.text(0.985, 0.06, f"temporal (fine) recovers ~{rec:.0f}% of the\ndense→(coarse full-MoE) gap; coarse temporal\nbeats coarse MoE on BOTH seeds (dots)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3, color="dimgray")
    ax.set_ylim(3.6, 4.25)
    ax.set_ylabel("test cross-entropy (nats, lower better)")
    ax.set_title("At 10^18 FLOPs (FLAME-MoE-38M scale): fine-graining hurts the full MoE,\n"
                 "but temporal routing is robust — and stays inside the dense↔MoE band")
    fig.text(0.5, 0.005,
             "Test-set cross-entropy (CE, nats, lower better) at 10^18 FLOPs (~38M-active model, pythia-50k, "
             "dclm). Hue = method (dense gray, MoE blue, temporal green); shade = granularity (coarse normal, "
             "fine-grained dark). 'temporal' = rolling residency (keep top-k resident, swap 1/token). Coarse "
             "bars = two-seed means with individual seeds as dots and measured half-range error bars; fine bars "
             "and dense are single runs (no error bar); dense is A6000 cross-data (different val "
             "split, within ~0.01 nats); the other four share one local split.", ha="center", fontsize=7.8, wrap=True)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    out = f"{REPO}/results/phase0/figures/temporal_vs_dense_and_moe_1e18_crossentropy.png"
fig.savefig(out, dpi=200)
print("wrote", out)
