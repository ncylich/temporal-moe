#!/usr/bin/env python3
"""Qwen3.5 D12-recipe replication figure: per-dataset deltas vs BASE-FREE.
Mirror of d12_final_figure.py for the qwen records (screening-class SINGLE runs,
think-off, same-batch refs: qwen35_val_base / qwen35_ce_d12r + duals)."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AB = "/workspace/temporal-moe/results/ablations"
V = {}
for r in csv.reader(open(f"{AB}/screening_genbench.csv")):
    if len(r) > 7 and r[0].startswith("qwen35_"):
        met = {"exact_match,flexible-extract": "GSM8K",
               "prompt_level_strict_acc,none": "IFEval",
               "pass@1,create_test": "HumanEval",
               "acc,relaxed-extract": "MMLU"}.get(r[6])
        if met: V[(r[0].replace("_dual", ""), r[3], met)] = 100 * float(r[7])

DS = ["GSM8K", "IFEval", "HumanEval", "MMLU"]
bf = {ds: V[("qwen35_val_base", "free", ds)] for ds in DS}
series = [("base under R8 constraint", "qwen35_val_base", "R8", "#b0b0b0"),
          ("r2 adapted (clean pool, KL 0.1), free", "qwen35_ce_d12r2", "free", "#a9dfbf"),
          ("r2 adapted (clean pool, KL 0.1), under R8", "qwen35_ce_d12r2", "R8", "#1e8449")]

fig, ax = plt.subplots(figsize=(9.5, 5.2))
W = 0.26
for i, (label, rec, arm, color) in enumerate(series):
    xs = [j + (i - 1) * W for j in range(len(DS))]
    ys = [V[(rec, arm, ds)] - bf[ds] for ds in DS]
    bars = ax.bar(xs, ys, W, label=label, color=color, edgecolor="black", linewidth=0.4)
    for b, y in zip(bars, ys):
        ax.annotate(f"{y:+.1f}", (b.get_x() + b.get_width() / 2, y),
                    ha="center", va="bottom" if y >= 0 else "top", fontsize=9,
                    xytext=(0, 2 if y >= 0 else -2), textcoords="offset points")
ax.axhline(0, color="black", linewidth=1)
ax.text(len(DS) - 0.42, 0.15, "unconstrained base = 0", fontsize=8.5,
        ha="right", color="#333333")
ax.set_xticks(range(len(DS)))
ax.set_xticklabels([f"{ds}\n(base free {bf[ds]:.1f})" for ds in DS])
ax.set_ylabel("accuracy delta vs unconstrained base (pts)")
ax.set_title("Qwen3.5-35B-A3B: adaptation (r2, max-min winner of the 2x2) vs the R8 constraint\n"
             "(200-item screens, think-off, same-batch refs; SINGLE runs, noise ±2 pts)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
ax.spines[["top", "right"]].set_visible(False)
ax.margins(y=0.12)
plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen_d12r_v1.png"
plt.savefig(out, dpi=160)
print("wrote", out)
for ds in DS:
    print(ds, {lbl: round(V[(rec, arm, ds)] - bf[ds], 1) for lbl, rec, arm, _ in series})
