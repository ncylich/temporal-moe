#!/usr/bin/env python3
"""Response length against residency damage, era-E4 protocol.

Reads think_ablation_summary.csv (produced by think_analysis.py from corrected rows and
token dumps): x = mean generated tokens on the FREE arm (exact, token-counted -- no
throughput reconstruction), y = damage at the model's default mode and tightest arm.
Writes figures/length_vs_damage.png and prints the pooled Spearman.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

DEFAULT_MODE = {"gemma4-26B-IT": "off", "Qwen3.5-35B": "on", "gpt-oss-20b": "medium",
                "gpt-oss-120b": "medium", "LFM2.5-A1B": "on"}
TIGHT = {"gemma4-26B-IT": "R8", "Qwen3.5-35B": "R8", "gpt-oss-20b": "R4",
         "gpt-oss-120b": "R4", "LFM2.5-A1B": "R4"}
MARK = {"GSM8K": "o", "IFEval": "s", "HumanEval": "^", "MMLU": "D"}

pts = []
for r in csv.DictReader(l for l in open(f"{ABLATIONS}/think_ablation_summary.csv")
                        if not l.startswith('"#')):
    m = r["model"]
    if r["mode"] != DEFAULT_MODE.get(m) or r["arm"] != TIGHT.get(m):
        continue
    if not r["gen_free"]:
        continue
    if m == "LFM2.5-A1B" and r["task"] == "MMLU":
        continue                                    # extraction floor, censored
    pts.append((m, r["task"], float(r["gen_free"]), float(r["damage"])))

xs = [p[2] for p in pts]
ys = [p[3] for p in pts]
rho, p = spearmanr(xs, ys)
print(f"Spearman(length, damage) = {rho:+.2f} (p = {p:.3f}, n = {len(pts)})")

fig, ax = plt.subplots(figsize=(6.5, 5.2))
cols = {m: c for m, c in zip(DEFAULT_MODE, plt.cm.tab10.colors)}
for m, task, x, y in pts:
    ax.scatter(x, y, color=cols[m], marker=MARK[task], s=55)
handles = [plt.Line2D([], [], marker="o", ls="", color=cols[m], label=m)
           for m in DEFAULT_MODE if any(p[0] == m for p in pts)]
handles += [plt.Line2D([], [], marker=mk, ls="", color="grey", label=t)
            for t, mk in MARK.items()]
ax.set_xscale("log")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("mean generated tokens, free arm (exact token counts)")
ax.set_ylabel("accuracy change at tightest residency, points")
ax.set_title("Damage against response length, corrected protocol\n"
             f"(default mode per model; Spearman {rho:+.2f}, p={p:.3f})", fontsize=10)
ax.legend(handles=handles, fontsize=7, ncol=2)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{ABLATIONS}/figures/length_vs_damage.png", dpi=150)
print("wrote length_vs_damage.png")
