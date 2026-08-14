#!/usr/bin/env python3
"""Instruct-program figures: self-CE damage vs residency fraction, and per-benchmark
damage at matched fraction. Reads instruct_selfce.csv and instruct_genbench*.csv only.
Writes results/ablations/figures/{instruct_selfce_damage,instruct_bench_damage}.png."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

FIG = os.path.join(ABLATIONS, "figures")
NAMES = {"olmoe_instruct": "OLMoE-Instruct (64 experts)",
         "lfm25_instruct": "LFM2.5-A1B (32 experts)",
         "gemma4_instruct": "gemma4-26B-IT (128 experts)",
         "qwen35_instruct": "Qwen3.5-35B (256 experts)"}


def selfce():
    rows = [r for r in csv.reader(open(f"{ABLATIONS}/instruct_selfce.csv"))
            if r and r[0] in NAMES]
    per = {}
    for r in rows:
        per.setdefault(r[0], {})[r[3]] = float(r[6])
    fig, ax = plt.subplots(figsize=(7, 5))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (m, d) in enumerate(per.items()):
        E = int(rows[0][1]) if False else int([r[1] for r in rows if r[0] == m][0])
        k = int([r[2] for r in rows if r[0] == m][0])
        free = d["free"]
        pts = sorted((100 * int(a[1:]) / E, d[a] - free) for a in d
                     if a.startswith("R") and not a.endswith("cold"))
        x, y = zip(*pts)
        ax.plot(x, y, "o-", color=cols[i], lw=2, ms=9,
                label=f"{NAMES[m]}, k={k}")
        ax.annotate("R=k", (x[0], y[0]), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=cols[i])
    ax.set_xscale("log")
    ax.set_xticks([3.125, 6.25, 12.5])
    ax.set_xticklabels(["3.1%", "6.25%", "12.5%"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("resident experts, % of total experts")
    ax.set_ylabel("self-CE damage, nats/token (constrained − free)")
    ax.set_title("Instruct models: damage on their own responses\n"
                 "(prefill free, rule on generated tokens; lower is better)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIG}/instruct_selfce_damage.png", dpi=150)
    print("wrote instruct_selfce_damage.png")


def bench():
    # deltas at R = 12.5% of E, per benchmark; floor cells excluded and marked
    want = {"exact_match,flexible-extract": "GSM8K", "prompt_level_strict_acc,none": "IFEval",
            "pass@1,create_test": "HumanEval", "exact_match,get-answer": "MMLU"}
    arm125 = {"olmoe_instruct": "R8", "lfm25_instruct": "R4",
              "gemma4_instruct": "R16", "qwen35_instruct": "R32"}
    armk = {"olmoe_instruct": "R8", "lfm25_instruct": "R4",
            "gemma4_instruct": "R8", "qwen35_instruct": "R8"}
    floor = {("lfm25_instruct", "MMLU")}
    vals = {}
    for f in ("instruct_genbench_vllm.csv",):   # live authoritative file only
        for r in csv.reader(open(f"{ABLATIONS}/{f}")):
            if len(r) > 8 and r[0] in NAMES and r[6] in want and r[8] != "10":
                vals[(r[0], r[3], want[r[6]])] = float(r[7])
            # channel-native humaneval variants are authoritative for think-in-text models
            if len(r) > 7 and r[0] == "gemma4_instruct" and r[5] == "humaneval_gemma_fixed":
                vals[(r[0], r[3], "HumanEval")] = float(r[7])
            if len(r) > 7 and r[0] in ("lfm25_instruct", "qwen35_instruct") \
                    and r[5] == "humaneval_think":
                vals[(r[0], r[3], "HumanEval")] = float(r[7])
    benches = ["GSM8K", "IFEval", "HumanEval", "MMLU"]
    models = list(NAMES)
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    w = 0.105
    GX = 1.45                                    # group spacing factor
    for i, m in enumerate(models):
        for shade, (arms, tag) in enumerate((( armk, "R=k"), (arm125, "12.5%"))):
            xs, ys = [], []
            for j, b in enumerate(benches):
                x = j * GX + (i - 1.5) * 2.1 * w + (shade - 0.5) * w
                if (m, b) in floor:
                    if shade == 0:
                        ax.annotate("floor", (j * GX + (i - 1.5) * 2.1 * w, 0.3),
                                    fontsize=7, rotation=90, ha="center", color="grey")
                    continue
                fr, cn = vals.get((m, "free", b)), vals.get((m, arms[m], b))
                if fr is None or cn is None:
                    continue
                d = 100 * (cn - fr)
                if abs(d) < 0.05:                # zero-damage results must stay visible
                    ax.annotate("0", (x, 0.35), fontsize=7, ha="center", color=cols[i],
                                fontweight="bold")
                xs.append(x)
                ys.append(d)
            ax.bar(xs, ys, width=w, color=cols[i], edgecolor="black", lw=0.4,
                   alpha=1.0 if shade == 0 else 0.55,
                   label=NAMES[m] if shade == 0 else None)
    for j in range(len(benches) - 1):
        ax.axvline((j + 0.5) * GX, color="grey", lw=0.6, alpha=0.4)
    from matplotlib.patches import Patch
    shade_handles = [Patch(facecolor="0.25", edgecolor="black",
                           label="dark: R = k (active params)"),
                     Patch(facecolor="0.85", edgecolor="black",
                           label="light: R = 12.5% of total experts")]
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([j * GX for j in range(len(benches))])
    ax.set_xticklabels(benches)
    ax.set_ylabel("accuracy change under residency, points")
    ax.set_title("Generative benchmarks under decode-time residency\n"
                 "(constrained − free, same items and stack per pair; "
                 "OLMoE and LFM: k = 12.5%, one cell)", fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + shade_handles, labels + [h.get_label() for h in shade_handles],
              fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIG}/instruct_bench_damage.png", dpi=150)
    print("wrote instruct_bench_damage.png")


if __name__ == "__main__":
    selfce()
    bench()
