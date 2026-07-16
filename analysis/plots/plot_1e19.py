#!/usr/bin/env python3
"""1e19 FLOPs: dense floor vs full MoE (coarse) vs temporal (coarse & fine-grained).
Test-set cross-entropy (nats, lower better), pythia-50k tokenizer, dclm; all four cells trained on
the SAME box (H100) and corpus/split — no cross-data caveat. Single seed (1234) per cell.
Per-point data: results/ablations/t19_1e19_curves.csv (final_test_ce @ the last iteration).
No fine-grained full-MoE cell was trained at 1e19 (the 1e18 program already established
fine-graining hurts the full MoE at scale; temporal is the fine-grained contender here).
Output: results/phase0/figures/temporal_vs_dense_and_moe_1e19_crossentropy.png
(--no-caption -> _nocaption variant for the paper).
"""
import os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PAPER:
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 12})

# worst -> best CE (left -> right). Colors match the isoFLOP figure: hue = method
# (gray dense, blue MoE, green temporal), shade = granularity (coarse normal / fine dark).
# End-of-training TEST evals, single seed each, one shared corpus/split (all H100-local).
ce     = [3.3532, 3.1806, 3.1732, 3.1310]
colors = ["#7f7f7f", "#5cc85c", "#145a14", "#5aa0dd"]
labels = (["dense", "temporal\ncoarse", "temporal\nfine", "full MoE\ncoarse"] if PAPER else
          ["dense\nbaseline", "temporal\ncoarse (6 of 64)", "temporal\nfine (18 of 192)",
           "full MoE\ncoarse (6 of 64)"])

fig, ax = plt.subplots(figsize=(4.6, 3.6) if PAPER else (8.6, 5.8))
bars = ax.bar(labels, ce, color=colors, width=0.66, edgecolor="k", linewidth=0.6)
for b, v in zip(bars, ce):
    ax.text(b.get_x()+b.get_width()/2, v + 0.007, f"{v:.3f}", ha="center",
            fontsize=(11 if PAPER else 10.5), fontweight="bold")
ax.grid(True, axis="y", ls=":", alpha=0.4)

if PAPER:
    ax.set_ylabel("test CE (nats)")
    ax.set_ylim(3.0, 3.45)
    ax.set_title("Quality at $10^{19}$ FLOPs")
    fig.tight_layout()
    out = f"{REPO}/results/phase0/figures/temporal_vs_dense_and_moe_1e19_crossentropy_nocaption.png"
else:
    rec_coarse = (3.3532 - 3.1806) / (3.3532 - 3.1310) * 100
    rec_fine   = (3.3532 - 3.1732) / (3.3532 - 3.1310) * 100
    ax.text(0.985, 0.06, f"temporal recovers ~{rec_coarse:.0f}% (coarse) / ~{rec_fine:.0f}% (fine) of the\n"
            "dense→MoE gap; fine-grained temporal now BEATS coarse\n"
            "temporal (crossover — reversed from 10^18)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3, color="dimgray")
    ax.set_ylim(3.0, 3.45)
    ax.set_ylabel("test cross-entropy (nats, lower better)")
    ax.set_title("At $10^{19}$ FLOPs temporal keeps ~78–81% of the MoE gain —\n"
                 "and fine-graining now helps the temporal model")
    fig.text(0.5, 0.005,
             "Test-set cross-entropy (CE, nats, lower better) at 10^19 FLOPs (pythia-50k, dclm; all four cells "
             "share one box/corpus/split, single seed). Hue = method (dense gray, MoE blue, temporal green); "
             "shade = granularity (coarse normal, fine-grained dark). 'temporal' = rolling residency "
             "(keep top-k resident, swap 1/token).",
             ha="center", fontsize=8, color="dimgray", wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = f"{REPO}/results/phase0/figures/temporal_vs_dense_and_moe_1e19_crossentropy.png"

fig.savefig(out, dpi=200)
print("wrote", out)
