#!/usr/bin/env python3
"""Combined adaptation figure: gemma4 D12 (authoritative cells) and Qwen3.5 r2 (screening
cells, same-batch refs) side by side, per-dataset deltas vs each model's unconstrained
base, now including WritingBench (critic deltas x10, cells from writingbench/cell_stats:
gemma4_d12 / qwen35_r2; the qwen R16 series has no WritingBench cell). The qwen panel adds
the fraction-matched R16 arm (6.25 percent resident, gemma's R8 fraction).
Writes figures/adapt_combined.png; --no-caption writes the paper variant."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

FIG = os.path.join(ABLATIONS, "figures")
PAPER = "--no-caption" in sys.argv
if PAPER:
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 12.5, "xtick.labelsize": 10,
                         "ytick.labelsize": 11, "legend.fontsize": 9.5,
                         "axes.titlesize": 13})

# ---- gemma (authoritative instrument, MMLU multi-run means) ----
A, dualm, S = {}, {}, {}
for r in csv.reader(open(f"{ABLATIONS}/instruct_genbench_vllm.csv")):
    if len(r) > 7 and r[0] in ("gemma4_instruct", "gemma4_ce_d12"):
        met = {"exact_match,flexible-extract": "GSM8K",
               "prompt_level_strict_acc,none": "IFEval",
               "pass@1,channel-aware": "HumanEval"}.get(r[6])
        if met:
            A[(r[0], r[3], met)] = 100 * float(r[7])
    if len(r) > 7 and r[6] == "acc,relaxed-extract" and r[0].startswith("gemma4_ce_d12_dual"):
        dualm.setdefault(r[3], []).append(100 * float(r[7]))
for r in csv.reader(open(f"{ABLATIONS}/screening_genbench.csv")):
    if len(r) > 7 and r[6] == "acc,relaxed-extract":
        if r[0] in ("dual_base", "pair_base"):
            S.setdefault(r[3], []).append(100 * float(r[7]))
        if r[0] == "scr_d12_dual":
            dualm.setdefault(r[3], []).append(100 * float(r[7]))
mean = lambda xs: sum(xs) / len(xs)
wb = {r[0]: float(r[1]) for r in csv.reader(open(f"{ABLATIONS}/writingbench/cell_stats.csv"))
      if r and r[0] != "cell"}


def gval(model, arm, ds):
    if ds == "WB":
        c = {("gemma4_instruct", "free"): "gemma4_base_free",
             ("gemma4_instruct", "R8"): "gemma4_base_R8",
             ("gemma4_ce_d12", "free"): "gemma4_d12_free",
             ("gemma4_ce_d12", "R8"): "gemma4_d12_R8"}.get((model, arm))
        return 10 * wb[c] if c in wb else None
    if ds == "MMLU":
        return mean(dualm[arm]) if model == "gemma4_ce_d12" else mean(S[arm])
    return A[(model, arm, ds)]


# ---- qwen (screening cells, same-batch refs) ----
V = {}
for r in csv.reader(open(f"{ABLATIONS}/screening_genbench.csv")):
    if len(r) > 7 and r[0].startswith("qwen35_"):
        met = {"exact_match,flexible-extract": "GSM8K",
               "prompt_level_strict_acc,none": "IFEval",
               "pass@1,create_test": "HumanEval",
               "acc,relaxed-extract": "MMLU"}.get(r[6])
        if met:
            V[(r[0].replace("_dual", ""), r[3], met)] = 100 * float(r[7])


def qval(rec, arm, ds):
    if ds == "WB":
        c = {("qwen35_val_base", "free"): "qwen35_base_free",
             ("qwen35_val_base", "R8"): "qwen35_base_R8",
             ("qwen35_ce_d12r2", "free"): "qwen35_r2_free",
             ("qwen35_ce_d12r2", "R8"): "qwen35_r2_R8"}.get((rec, arm))
        return 10 * wb[c] if c in wb else None
    return V.get((rec, arm, ds))


DS = ["GSM8K", "IFEval", "HumanEval", "MMLU", "WB"]
DLAB = {"WB": "WritingB.\n(pts x10)"}
PANELS = [
    ("gemma4-26B-IT", gval, "gemma4_instruct", "gemma4_ce_d12",
     [("base under R=8", "R8", None, "#b0b0b0"),
      ("adapted, free", "free", "adapted", "#7fb3d5"),
      ("adapted, under R=8", "R8", "adapted", "#1f618d")]),
    ("Qwen3.5-35B", qval, "qwen35_val_base", "qwen35_ce_d12r2",
     [("base under R=8", "R8", None, "#b0b0b0"),
      ("adapted, free", "free", "adapted", "#a9dfbf"),
      ("adapted, under R=8", "R8", "adapted", "#1e8449"),
      ("adapted, under R=16 (6.25%)", "R16", "adapted", "#145a32")]),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6) if PAPER else (13, 5.2), sharey=True)
for ax, (title, val, baserec, adrec, series) in zip(axes, PANELS):
    bf = {ds: val(baserec, "free", ds) for ds in DS}
    n = len(series)
    W = 0.8 / n
    for i, (label, arm, which, color) in enumerate(series):
        rec = adrec if which else baserec
        xs, ys = [], []
        for j, ds in enumerate(DS):
            v = val(rec, arm, ds)
            if v is None or bf[ds] is None:
                continue
            xs.append(j + (i - (n - 1) / 2) * W)
            ys.append(v - bf[ds])
        bars = ax.bar(xs, ys, W * 0.92, label=label, color=color, edgecolor="black",
                      linewidth=0.4)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:+.1f}", (b.get_x() + b.get_width() / 2, y),
                        ha="center", va="bottom" if y >= 0 else "top",
                        fontsize=8 if PAPER else 8.5,
                        xytext=(0, 1.5 if y >= 0 else -1.5), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels([DLAB.get(ds, ds) for ds in DS])
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right", fontsize=9 if PAPER else 8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.22, axis="y")
axes[0].set_ylabel("delta vs unconstrained base, points")
axes[0].margins(y=0.15)
if not PAPER:
    fig.suptitle("Constraint-aware adaptation vs the R=k constraint: gemma authoritative "
                 "cells (MMLU multi-run means), qwen 200-item screens (same-batch refs, "
                 "noise ±2 pts); WritingBench critic deltas x10", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96) if not PAPER else None)
out = f"{FIG}/adapt_combined{'_nocaption' if PAPER else ''}.png"
fig.savefig(out, dpi=170)
print(f"wrote {out}")
