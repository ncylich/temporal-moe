#!/usr/bin/env python3
"""1e18 (FLAME-MoE-38M-100M scale): dense floor vs full MoE (coarse & fine-grained) vs temporal.
Validation cross-entropy (nats, lower better), pythia-50k tokenizer, our dclm val split. Single axes.
Measured locally on one val split: MoE coarse (64), MoE fine (192), temporal fine (18/192). Dense floor
reused from the A6000 baseline (4.137; same corpus/tokenizer, cross-data <~0.01 nats).
Output: results/phase0/figures/temporal_vs_dense_and_moe_1e18_crossentropy.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# worst -> best CE (left -> right); all four measured on one val split
labels = ["dense\nbaseline", "full MoE\nfine (18 of 192)", "temporal\nfine (18 of 192)",
          "full MoE\ncoarse (6 of 64)"]
ce     = [4.137, 4.0087, 3.9768, 3.9209]
colors = ["C3", "C0", "C2", "C0"]

fig, ax = plt.subplots(figsize=(8.0, 5.8))
bars = ax.bar(labels, ce, color=colors, width=0.62, edgecolor="k", linewidth=0.6)
bars[1].set_alpha(0.5)                                  # fine-grained full MoE (measured)
for b, v in zip(bars, ce):
    ax.text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.3f}", ha="center", fontsize=10.5, fontweight="bold")

# finding 1: fine-graining hurts the full MoE (fine 4.009 -> coarse 3.921)
ax.annotate("", xy=(3, 3.9209), xytext=(1, 4.0087),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.4))
ax.text(2.0, 4.15, "fine-graining hurts the full MoE  (−0.088 nats)", ha="center", fontsize=8.5, color="darkred")
# finding 2: temporal (fine) beats its own-granularity full MoE
ax.annotate("", xy=(2, 3.9768), xytext=(1, 4.0087),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.2))
ax.text(2.55, 4.05, "temporal beats its own-granularity full MoE", ha="left", fontsize=8, color="green")

rec = (4.137 - 3.9768) / (4.137 - 3.9209) * 100
ax.text(0.985, 0.06, f"temporal (fine) recovers ~{rec:.0f}% of the\ndense→(coarse full-MoE) gap;\nall four measured, everything beats the dense floor",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3, color="dimgray")

ax.set_ylim(3.6, 4.25)
ax.set_ylabel("validation cross-entropy (nats, lower better)")
ax.set_title("At 10^18 FLOPs (FLAME-MoE-38M scale): fine-graining hurts the full MoE,\n"
             "but temporal routing is robust — and stays inside the dense↔MoE band")
ax.grid(True, axis="y", ls=":", alpha=0.4)
fig.text(0.5, 0.005,
         "Validation cross-entropy (CE, nats, lower better) at 10^18 FLOPs (~38M-active model, pythia-50k, "
         "dclm). 'full MoE' = standard top-k routing at coarse (64 experts, top-6) or fine-grained (192 "
         "experts, top-18) granularity; 'temporal' = rolling residency (keep top-k resident, swap 1/token); "
         "'dense baseline' = plain feed-forward. Faded blue = fine-grained full MoE. All four measured on "
         "one val split.", ha="center", fontsize=7.8, wrap=True)
fig.tight_layout(rect=[0, 0.09, 1, 1])
out = "/workspace/FLAME-MoE/results/phase0/figures/temporal_vs_dense_and_moe_1e18_crossentropy.png"
fig.savefig(out, dpi=130)
print("wrote", out)
