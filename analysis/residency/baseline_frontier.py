#!/usr/bin/env python3
"""Serving-cache baselines against rolling residency, on the axes that matter at
serving: measured expert loads per token-layer against GSM8K, at OUR resident
memory (gemma4 C=8 of 128, 6.25%; Qwen3.5 C=8 of 256, 3.1%), plus Skliar's cache
at its own 50% setting, ReMoE's best pick, and the per-token-layer transfer
distribution at Skliar's matched-traffic point.

Accuracies come from instruct_genbench_vllm.csv by record name. Loads per
token-layer were measured on the eval generations (TEMPORAL_COUNT_SWAPS /
cache_bias counters) and are recorded per lambda in
BASELINE_METHODS_COMPARISON.md; they are carried here as literals keyed by the
same record names. The histogram is skliar_c8_lam05_hist.json (9.2M token-layers).

Writes results/ablations/figures/baseline_frontier.png (the paper figure).
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

FIG = os.path.join(ABLATIONS, "figures")
GRID = os.path.join(ABLATIONS, "instruct_genbench_vllm.csv")
HIST = os.path.join(ABLATIONS, "skliar_c8_lam05_hist.json")

OURS = "#0C6B66"
SKLIAR = "#B4532A"
REMOE = "#64748B"
REF = "#9AA1AB"

# loads per token-layer per record, measured on the eval generations
# (BASELINE_METHODS_COMPARISON.md, C=8 lambda curves and 50% sweeps)
LOADS = {
    "gemma4_skliar_C8_lam0_n1319": 4.67, "gemma4_skliar_C8_lam0p4_n1319": 1.33,
    "gemma4_skliar_C8_lam0p5_n1319": 0.94, "gemma4_skliar_C8_lam0p6_n1319": 0.70,
    "gemma4_skliar_C8_lam0p8_n1319": 0.52, "gemma4_skliar_C8_lam1p2_n1319": 0.49,
    "qwen35_skliar_C8_lam0_n1319": 5.63, "qwen35_skliar_C8_lam0p1_n1319": 3.44,
    "qwen35_skliar_C8_lam0p2_n1319": 2.13, "qwen35_skliar_C8_lam0p3_n1319": 1.41,
    "qwen35_skliar_C8_lam0p4_n1319": 1.01, "qwen35_skliar_C8_lam0p5_n1319": 0.81,
    "qwen35_skliar_C8_lam0p7_n1319": 0.66,
    "gemma4_skliar_C64_lam0_n1319": 0.307, "gemma4_skliar_C64_lam0p4_n1319": 0.032,
    "qwen35_skliar_C128_lam0_n1319": 0.837, "qwen35_skliar_C128_lam0p4_n1319": 0.069,
}

PANELS = [
    ("gemma4, 16$\\times$ memory reduction", 87.8,
     "gemma4_ce_online_fullpool_full_rho0_n1319",      # ours, adapted (R8)
     "gemma4_instruct_n1319",                          # released under the cap
     "gemma4_remoe_lr3e-4_n1319",                      # ReMoE best lr
     [r for r in LOADS if r.startswith("gemma4_skliar_C8")],
     [r for r in LOADS if r.startswith("gemma4_skliar_C64")]),
    ("Qwen3.5, 32$\\times$ memory reduction", 85.9,
     "qwen35_ce_online_fullpool_full_rho0_n1319",
     "qwen35_think_off_n1319",
     "qwen35_remoe_lr1e-4_n1319",
     [r for r in LOADS if r.startswith("qwen35_skliar_C8")],
     [r for r in LOADS if r.startswith("qwen35_skliar_C128")]),
]


def acc(record, arm="R8"):
    for r in csv.reader(open(GRID)):
        if len(r) > 7 and r[0] == record and r[3] == arm and \
                r[6] == "exact_match,flexible-extract":
            return 100 * float(r[7])
    raise SystemExit(f"no GSM8K row for {record} {arm}")


def draw():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.1))
    for ax, (title, free, ours, base, remoe, curve, own50) in zip(axes, PANELS):
        ax.set_xscale("log")
        ax.set_xlim(0.022, 8.5)
        ax.set_ylim(40, 93)
        ax.axhline(free, color=REF, lw=1.1, ls=(0, (5, 4)), zorder=1)
        ax.text(0.024, free + 1.4, f"free model {free:.1f}", ha="left",
                fontsize=7.5, color=REF)
        ax.axvline(1.0, color=OURS, lw=0.9, ls=(0, (4, 3)), alpha=0.5, zorder=1)
        pts = sorted((LOADS[r], acc(r)) for r in curve)
        ax.plot(*zip(*pts), "-o", color=SKLIAR, ms=4.5, lw=1.4,
                label="Skliar cache at our memory ($\\lambda$ sweep)", zorder=3)
        for r in own50:
            ax.plot(LOADS[r], acc(r), "o", mfc="none", mec=SKLIAR, mew=1.7,
                    ms=5.2, zorder=3,
                    label="Skliar at 50% resident" if r == own50[0] else None)
        ax.plot(1.0, acc(remoe), "s", color=REMOE, ms=5.5,
                label="ReMoE, best learning rate", zorder=3)
        ax.plot(1.0, acc(base), "D", mfc="none", mec=OURS, mew=1.7, ms=5.5,
                label="released model, hard 1-swap cap", zorder=4)
        a = acc(ours)
        ax.plot(1.0, a, "D", color=OURS, ms=6.5, zorder=5,
                label="ours, adapted (hard 1-swap cap)")
        ax.annotate(f"{a:.1f}", (1.0, a), textcoords="offset points",
                    xytext=(7, 4), fontsize=8, color=OURS)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("expert loads / token-layer (log)", fontsize=8.5)
        ax.set_xticks([0.03, 0.1, 0.3, 1, 3])
        ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("GSM8K", fontsize=9)
    axes[0].legend(fontsize=6.8, loc="lower right", framealpha=0.9)

    ax = axes[2]
    h = json.load(open(HIST))["hist"]
    tot = sum(h)
    xs = list(range(8))
    fr = [100 * h[i] / tot for i in xs]
    ax.bar(xs, fr, width=0.62, color=SKLIAR, alpha=0.88,
           label="Skliar, $\\lambda{=}0.5$ (mean 0.94)")
    ax.bar([1], [100], width=0.16, color=OURS, label="ours (every token-layer)")
    for i, f in zip(xs, fr):
        if f > 0.5:
            ax.annotate(f"{f:.0f}%", (i, f), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.5)
    ax.annotate("20% need $\\geq$2:\neach stalls the layer\nin proportion",
                (2.4, 40), fontsize=8, color=SKLIAR)
    ax.set_title("transfers in one token-layer (gemma4)", fontsize=9.5)
    ax.set_xlabel("expert transfers", fontsize=8.5)
    ax.set_ylabel("share of token-layers (%)", fontsize=9)
    ax.set_xticks(xs)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.8, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    p = f"{FIG}/baseline_frontier.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", p)
    for _, _, ours, base, remoe, curve, own50 in PANELS:
        for r in [ours, base, remoe] + curve + own50:
            arm = "R8"
            print(f"  {r:42s} loads={LOADS.get(r, 1.0):5.2f}  GSM8K={acc(r, arm):.1f}")


if __name__ == "__main__":
    draw()
