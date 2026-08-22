#!/usr/bin/env python3
"""Figure for the functional-displacement measurement: per-layer relative output
error under residency for all six IT models, plus damage-vs-displacement
scatters contrasting router-space W1 (no relation) with function-space rel_out.

Damage = mean over the four benchmarks (GSM8K, IFEval, HumanEval, MMLU) at the
tight arm (R=k) minus own free arm, points, thinking off/low
(instruct_genbench_vllm.csv / report.md grid; LFM has no off mode, its
thinking-on value is used and flagged). Reads functional_displacement.csv and
router_wasserstein.csv."""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ABL = "/workspace/temporal-moe/results/ablations"

STYLE = {  # model key -> (label, color)
    "gpt-oss-120b": ("gpt-oss-120b (R4 = 3.1%)", "#1e618d"),
    "qwen35-35b-a3b-instruct": ("Qwen3.5-35B (R8 = 3.1%)", "#c0392b"),
    "gemma4-26b-it": ("gemma4-26B-IT (R8 = 6.25%)", "#27ae60"),
    "olmoe-0125-instruct": ("OLMoE-1B-7B (R8 = 12.5%)", "#8e44ad"),
    "gpt-oss-20b": ("gpt-oss-20b (R4 = 12.5%)", "#7fb3d5"),
    "lfm25-8b-a1b": ("LFM2.5-A1B (R4 = 12.5%)", "#e67e22"),
}
# mean 4-benchmark damage at the tight arm vs own free, think off/low
# (report.md grid; olmoe from instruct_genbench_vllm.csv; lfm think-on only)
DAMAGE = {"gpt-oss-120b": 0.2, "qwen35-35b-a3b-instruct": -7.2,
          "gemma4-26b-it": -2.5, "olmoe-0125-instruct": -14.8,
          "gpt-oss-20b": -1.2, "lfm25-8b-a1b": -8.5}

def read(path, valcol):
    per = defaultdict(list)
    fh = open(path); fh.readline()
    for r in csv.DictReader(fh):
        per[r["model"]].append((int(r["layer"]), float(r[valcol])))
    return {m: [v for _, v in sorted(xs)] for m, xs in per.items()}

rel = read(f"{ABL}/functional_displacement.csv", "rel_out")
w1 = read(f"{ABL}/router_wasserstein.csv", "w1_imposed")

FRACMARK = {"3.1%": "o", "6.25%": "s", "12.5%": "^"}

def frac_of(m):
    return STYLE[m][0].split("= ")[1].rstrip(")")

def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))

PAPER = "--no-caption" in sys.argv
if PAPER:   # paper variant: the two correlation panels only, larger fonts
    plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "axes.titlesize": 12.5})
    fig, saxes = plt.subplots(1, 2, figsize=(10.5, 4.3))
else:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axes[0]
    for m, (lab, c) in STYLE.items():
        ys = rel[m]
        ax.plot(range(len(ys)), ys, color=c, label=lab, lw=1.8)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("relative output error of routed experts\n(‖y_masked − y_free‖ / ‖y_free‖, same inputs)")
    ax.set_title("Per-layer functional displacement under R=k")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    saxes = axes[1:]

for ax, data, xlabel, name in (
        (saxes[0], w1, "mean router W1 (probability displacement)", "router W1"),
        (saxes[1], rel, "mean relative output error", "functional error")):
    xs = [sum(data[m]) / len(data[m]) for m in STYLE]
    ys = [DAMAGE[m] for m in STYLE]
    rho = spearman(xs, [-y for y in ys])   # vs damage magnitude
    for m, x, y in zip(STYLE, xs, ys):
        lab, c = STYLE[m]
        ax.scatter(x, y, color=c, s=60, zorder=3, marker=FRACMARK[frac_of(m)])
        short = lab.split(" (")[0] + ("*" if m == "lfm25-8b-a1b" else "")
        ax.annotate(short, (x, y), textcoords="offset points",
                    xytext=(6, -3), fontsize=9 if PAPER else 8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean benchmark damage at R=k (points)")
    ax.set_title(f"{name}: Spearman ρ = {rho:+.2f}" if PAPER else
                 f"{name} vs damage magnitude: Spearman ρ = {rho:+.2f}", fontsize=None if PAPER else 10)
    ax.axhline(0, color="#bbbbbb", lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
saxes[1].set_xlim(left=0)
if not PAPER:
    fig.text(0.995, 0.01, "marker = residency fraction (○ 3.1%  □ 6.25%  △ 12.5%); "
             "within each fraction, functional error orders damage exactly. "
             "*LFM: thinking-on damage (no off mode)",
             ha="right", fontsize=7, color="#666666")
plt.tight_layout()
out = (f"{ABL}/figures/functional_displacement_nocaption.png" if PAPER else
       (sys.argv[1] if len(sys.argv) > 1 else f"{ABL}/figures/functional_displacement.png"))
plt.savefig(out, dpi=160)
print("wrote", out)
