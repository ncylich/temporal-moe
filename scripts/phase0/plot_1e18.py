#!/usr/bin/env python3
"""1e18 (FLAME-MoE-38M-100M scale): where temporal lands between the dense floor and full MoE.
Validation cross-entropy (nats, lower better), pythia-50k tokenizer, our dclm val split.
Dense + temporal are measured on our setup; MoE is the paper's compute-optimal scaling-law value.
Output: results/phase0/temporal_1e18_bars.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["dense floor\n(measured)", "temporal\n(measured)", "MoE\n(paper law)"]
ce     = [4.137, 3.906, 3.78]
colors = ["C3", "C2", "C0"]

fig, ax = plt.subplots(figsize=(7, 5.5))
bars = ax.bar(labels, ce, color=colors, width=0.6, edgecolor="k", linewidth=0.6)
bars[2].set_hatch("//"); bars[2].set_alpha(0.75)   # MoE is a law estimate, not measured here
for b, v in zip(bars, ce):
    ax.text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

# annotate the measured temporal-vs-dense gain and the recovery fraction
ax.annotate("", xy=(1, 3.906), xytext=(0, 4.137),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
ax.text(0.5, 4.03, "−0.231 nats\n(beats dense)", ha="center", fontsize=9, color="gray")
rec = (4.137-3.906)/(4.137-3.78)*100
ax.text(0.5, 0.04, f"temporal recovers ~{rec:.0f}% of the dense→MoE gap\n"
        "(dense & temporal measured on our setup; MoE = paper scaling law)",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="dimgray")

ax.set_ylim(3.6, 4.25)
ax.set_ylabel("validation CE (nats, lower better)")
ax.set_title("1e18 FLOPs (FLAME-MoE-38M-100M scale): temporal between dense and MoE")
ax.grid(True, axis="y", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig("results/phase0/temporal_1e18_bars.png", dpi=130)
print("wrote results/phase0/temporal_1e18_bars.png")
