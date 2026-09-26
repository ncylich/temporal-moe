#!/usr/bin/env python3
"""Qwen attribution square figure: free and R8 deltas vs base-free for the four
pool x KL configs plus the r5 unification run. Reads screening_genbench.csv."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AB = "/workspace/temporal-moe/results/ablations"
V = {}
for r in csv.reader(open(f"{AB}/screening_genbench.csv")):
    if len(r) > 7 and r[0].startswith("qwen35_"):
        met = {"exact_match,flexible-extract": "GSM8K", "prompt_level_strict_acc,none": "IFEval",
               "pass@1,create_test": "HumanEval", "acc,relaxed-extract": "MMLU"}.get(r[6])
        if met:
            rec = r[0].replace("_dual", "")
            if rec.endswith("_b"): rec = rec[:-2]
            V.setdefault((rec, r[3], met), []).append(100 * float(r[7]))
m = lambda k: sum(V[k]) / len(V[k])
bf = lambda t: m(("qwen35_val_base", "free", t))

DS = ["GSM8K", "IFEval", "HumanEval", "MMLU"]
CFGS = [("qwen35_ce_d12r",  "r1: pool w/ truncation, KL .05", "#c9c0e8"),
        ("qwen35_ce_d12r3", "r3: clean pool, KL .05",         "#9db8d9"),
        ("qwen35_ce_d12r4", "r4: pool w/ truncation, KL .1",  "#e8b7a5"),
        ("qwen35_ce_d12r2", "r2: clean pool, KL .1 (committed)", "#1e8449"),
        ("qwen35_ce_d12r5", "r5: all lengths, r16, KL .1",    "#8d5fb0")]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
for ax, arm, title in ((axes[0], "free", "free arm (no constraint)"),
                       (axes[1], "R8", "R8 arm (8 of 256 resident)")):
    W = 0.16
    for i, (rec, label, color) in enumerate(CFGS):
        xs = [j + (i - 2) * W for j in range(len(DS))]
        ys = [m((rec, arm, ds)) - bf(ds) for ds in DS]
        ax.bar(xs, ys, W, label=label if arm == "free" else None,
               color=color, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels(DS)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("accuracy delta vs unconstrained base (pts)")
fig.suptitle("Qwen3.5 adaptation variants: training pool x KL anchor (single 200-item screens, noise ±2 pts)",
             fontsize=12)
fig.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=(0, 0.04, 1, 1))
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen_square_v1.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
