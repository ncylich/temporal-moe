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
    # per-benchmark damage, all six models, no/low-thinking modes, both R levels
    # (record, label, R=k arm, 12.5% arm, humaneval task, mmlu task; None = floor)
    SPEC = [
        ("olmoe_instruct", "OLMoE-Instruct (64E, k=8)", "R8", "R8",
         "humaneval_instruct", "mmlu_flan_cot_fewshot"),
        ("lfm25_instruct", "LFM2.5-A1B, native mode -- no toggle (32E, k=4)", "R4", "R4",
         "humaneval_think", None),
        ("qwen35_think_off", "Qwen3.5-35B, think off (256E, k=8)", "R8", "R32",
         "humaneval_instruct", "mmlu_flan_cot_fewshot"),
        ("gemma4_instruct", "gemma4-26B-IT, think off (128E, k=8)", "R8", "R16",
         "humaneval_gemma_fixed", "mmlu_flan_cot_fewshot"),
        ("gptoss_20b_low", "gpt-oss-20b, low effort (32E, k=4)", "R4", "R4",
         "humaneval_gptoss", "mmlu_gptoss_relaxed"),
        ("gptoss_120b_low", "gpt-oss-120b, low effort (128E, k=4)", "R4", "R16",
         "humaneval_gptoss", "mmlu_gptoss_relaxed"),
    ]
    METRIC = {"gsm8k_cot_zeroshot": ("exact_match,flexible-extract",),
              "ifeval": ("prompt_level_strict_acc,none",),
              "mmlu_flan_cot_fewshot": ("exact_match,get-answer",),
              "mmlu_gptoss_relaxed": ("acc,relaxed-extract",)}
    vals = {}
    for r in csv.reader(open(f"{ABLATIONS}/instruct_genbench_vllm.csv")):
        if len(r) < 10 or r[0].startswith("#") or r[0] == "model":
            continue
        rec, arm, task, met = r[0], r[3], r[5], r[6]
        ok = met in METRIC.get(task, ()) or \
            (task.startswith("humaneval") and met.startswith("pass@1"))
        if ok:
            vals[(rec, arm, task)] = float(r[7])

    benches = ["GSM8K", "IFEval", "HumanEval", "MMLU"]

    def task_for(spec, b):
        return {"GSM8K": "gsm8k_cot_zeroshot", "IFEval": "ifeval",
                "HumanEval": spec[4], "MMLU": spec[5]}[b]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    w = 0.08
    GX = 1.45                                    # group spacing factor
    for i, spec in enumerate(SPEC):
        rec, label = spec[0], spec[1]
        for shade, (arm, tag) in enumerate(((spec[2], "R=k"), (spec[3], "12.5%"))):
            xs, ys = [], []
            for j, b in enumerate(benches):
                x = j * GX + (i - 2.5) * 2.1 * w + (shade - 0.5) * w
                task = task_for(spec, b)
                if task is None:
                    if shade == 0:
                        ax.annotate("floor", (j * GX + (i - 2.5) * 2.1 * w, 0.3),
                                    fontsize=7, rotation=90, ha="center", color="grey")
                    continue
                fr, cn = vals.get((rec, "free", task)), vals.get((rec, arm, task))
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
                   label=label if shade == 0 else None)
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
    ax.set_title("Generative benchmarks under decode-time residency, no/low-thinking modes\n"
                 "(constrained − free, same items and stack per pair; single runs, "
                 "binomial SE 2-4 pts;\nOLMoE, LFM and gpt-oss-20b: k = 12.5%, one cell)",
                 fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + shade_handles, labels + [h.get_label() for h in shade_handles],
              fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIG}/instruct_bench_damage.png", dpi=150)
    print("wrote instruct_bench_damage.png")


if __name__ == "__main__":
    selfce()
    bench()
