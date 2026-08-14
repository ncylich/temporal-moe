#!/usr/bin/env python3
"""Headline thinking-tax figure: per model, mean benchmark damage at R=k with
thinking off/low vs on/high, per-dataset dots overlaid.

Reads think_ablation_summary.csv only. Writes figures/think_tax.png.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

# (model, low-mode, high-mode, R=k arm, 12.5% arm)
SPEC = [("gemma4-26B-IT", "off", "on", "R8", "R16"),
        ("Qwen3.5-35B", "off", "on", "R8", "R32"),
        ("gpt-oss-20b", "low", "high", "R4", "R4"),
        ("gpt-oss-120b", "low", "high", "R4", "R16")]
TASKS = ["GSM8K", "IFEval", "HumanEval", "MMLU"]
TCOL = dict(zip(TASKS, plt.cm.Set2(np.linspace(0, 0.6, 4))))


def main():
    cells = {}
    for r in csv.reader(open(f"{ABLATIONS}/think_ablation_summary.csv")):
        if len(r) > 7 and r[0] != "model" and not r[0].startswith("#"):
            cells[(r[0], r[1], r[2], r[3])] = (float(r[6]), float(r[7]))

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.19
    for i, (model, lo, hi, armk, arm125) in enumerate(SPEC):
        for s, mode in enumerate((lo, hi)):
            for a, arm in enumerate((armk, arm125)):
                got = [(t,) + cells[(model, mode, arm, t)] for t in TASKS
                       if (model, mode, arm, t) in cells]
                if not got:
                    continue
                ds = [d for _, d, _ in got]
                mean = sum(ds) / len(ds)
                se = (sum(e * e for _, _, e in got) ** 0.5) / len(got)
                x = i + (2 * s + a - 1.5) * w
                ax.bar(x, mean, width=w * 0.9, yerr=se, capsize=3,
                       color="#4878b0" if s == 0 else "#d1605e",
                       alpha=1.0 if a == 0 else 0.5,
                       edgecolor="black", lw=0.5,
                       label=("thinking off / low effort" if s == 0 else
                              "thinking on / high effort")
                       if i == 0 and a == 0 else None)
                for t, d, _ in got:
                    ax.scatter(x, d, s=34, color=TCOL[t], edgecolor="black",
                               lw=0.5, zorder=3)
    for t in TASKS:
        ax.scatter([], [], s=42, color=TCOL[t], edgecolor="black", lw=0.5, label=t)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(SPEC)))
    ax.set_xticklabels([f"{s[0]}\n(k: {s[3]}, 12.5%: {s[4]})" for s in SPEC])
    ax.set_ylabel("accuracy change under residency, points")
    ax.set_title("Does thinking amplify constraint damage?  "
                 "bar = mean over benchmarks (whisker = SE of mean), "
                 "dark = R=k, light = R=12.5% of experts,\n"
                 "dots = individual benchmarks (constrained − free per mode; "
                 "single runs, per-cell SE 2-4 pts; gpt-oss-20b: k = 12.5%, "
                 "one cell shown twice)", fontsize=8.5)
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="0.3", edgecolor="black", label="dark: R = k"),
          Patch(facecolor="0.85", edgecolor="black", label="light: R = 12.5%")]
    ax.legend(h, l + ["dark: R = k", "light: R = 12.5%"], fontsize=8, ncol=2)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"{ABLATIONS}/figures/think_tax.png", dpi=150)
    print("wrote think_tax.png")


if __name__ == "__main__":
    main()
