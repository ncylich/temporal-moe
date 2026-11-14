#!/usr/bin/env python3
"""WritingBench fluency figure: critic score (1-10) per model and arm, mean over
three disjoint 50-query subsets, error bars = SD across subsets. Reads
results/ablations/writingbench/cell_stats.csv."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/workspace/temporal-moe/results/ablations/writingbench/cell_stats.csv"
V = {}
for r in csv.DictReader(open(SRC)):
    V[r["cell"]] = (float(r["mean"]), float(r["sd_across_subsets"]))

GROUPS = [  # (label, record stem, arms present)
    ("gpt-oss-120b", "oss120", ["free", "R4", "R16"]),
    ("Qwen3.5-35B", "qwen35_base", ["free", "R8", "R32"]),
    ("Qwen3.5 r2 (adapted)", "qwen35_r2", ["free", "R8", "R32"]),
    ("gpt-oss-20b", "oss20", ["free", "R4"]),
    ("gemma4-26B-IT", "gemma4_base", ["free", "R8", "R16"]),
    ("gemma4 D12 (adapted)", "gemma4_d12", ["free", "R8", "R16"]),
    ("LFM2.5-A1B", "lfm25", ["free", "R4"]),
]
ARMCOLOR = {"free": "#8c9dab", "k": "#1e618d", "pct": "#7fb3d5"}

ARMNAME = {"free": "free (unconstrained)", "k": "R = k (tightest)",
           "pct": "R = 12.5% of experts"}

fig, ax = plt.subplots(figsize=(11, 5))
W = 0.26
seen = set()
for gi, (label, stem, arms) in enumerate(GROUPS):
    if "adapted" in label:
        ax.axvspan(gi - 0.45, gi + 0.45, color="#f2e8d5", zorder=0)
    for ai, arm in enumerate(arms):
        m, sd = V[f"{stem}_{arm}"]
        kind = "free" if arm == "free" else ("k" if ai == 1 else "pct")
        x = gi + (ai - 1) * W
        lab = ARMNAME[kind] if kind not in seen else None
        seen.add(kind)
        ax.bar(x, m, W, yerr=sd, capsize=3, color=ARMCOLOR[kind],
               edgecolor="black", linewidth=0.4, label=lab, zorder=2)
        ax.annotate(f"{m:.2f}", (x, m + sd + 0.03), ha="center", fontsize=8)
ax.set_ylim(6.4, 9.0)
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([g[0] for g in GROUPS], fontsize=9)
ax.set_ylabel("WritingBench critic score (1-10, higher better)")
ax.set_title("Writing quality under decode-time residency\n"
             "(official critic model, 3 disjoint 50-query subsets; error bars = SD across subsets)")
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.text(0.0, -0.14, "gpt-oss-20b and LFM2.5 have 32 experts, so R = k (4) is also 12.5%: two arms only. "
        "Shaded groups are residency-adapted checkpoints (gemma D12, qwen r2).",
        transform=ax.transAxes, fontsize=8, color="#555555")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wb_fig_v1.png"
plt.savefig(out, dpi=160)
print("wrote", out)
