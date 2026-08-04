#!/usr/bin/env python3
"""Does the per-layer residency damage profile keep its shape across model scale?

OLMoE (16 layers, 64 experts) and Qwen3.5-35B (40 layers, 256 experts) are plotted against RELATIVE
depth, and each profile is normalised by its own mean, because the absolute damages differ by ~70x
and the question here is shape, not magnitude. Normalising also makes the uniform-damage null a flat
line at 1.0, so "which layers are worth freeing" is read directly off distance from that line.

The answer is that it does not transfer. OLMoE is U-shaped, elevated at both ends with layer 1 the
single worst. Qwen3.5 is late-heavy: its first layers sit at the middle-third baseline and the cost
piles into the final eighth, with the last layer alone an order of magnitude above uniform.

    plot_profile_transfer.py
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS, ROOT                                  # noqa: E402
FIGURES = os.path.join(ROOT, "results/phase0/figures")

# OLMoE per-layer damage, from results/ablations/layer_freeing_RESULTS.md section 2 (base model,
# one layer constrained at a time, no training). Transcribed rather than recomputed because the
# producing run is on a different framework; the source table is the audited one.
OLMOE = [0.2178, 0.2588, 0.1408, 0.1153, 0.1168, 0.0986, 0.1064, 0.0792,
         0.0810, 0.0822, 0.0837, 0.0698, 0.0729, 0.0748, 0.1225, 0.1408]


def qwen_profile(R=8):
    p = os.path.join(ABLATIONS, "qwen35_residency_suite.csv")
    rows = list(csv.DictReader([l for l in open(p) if not l.lstrip().lstrip('"').startswith("#")]))
    d = {r["cell"]: float(r["bpb"]) for r in rows}
    free = d["free_baseline"]
    solo = sorted((int(k[6:8]), v - free) for k, v in d.items()
                  if k.startswith("solo_") and k.endswith(f"_R{R}"))
    return [v for _, v in solo]


def main():
    q8, q32 = qwen_profile(8), qwen_profile(32)
    fig, ax = plt.subplots(figsize=(9, 5))
    for prof, lbl, style in ((OLMOE, "OLMoE 1B-7B — 16 layers, 64 experts, R=8 (12.5% resident)", "-o"),
                             (q8, "Qwen3.5-35B-A3B — 40 layers, 256 experts, R=8 (3.1% resident)", "-s"),
                             (q32, "Qwen3.5-35B-A3B — R=32 (12.5% resident, matched fraction)", "--^")):
        n = len(prof)
        mean = sum(prof) / n
        x = [i / (n - 1) for i in range(n)]
        ax.plot(x, [p / mean for p in prof], style, ms=4, lw=1.6, label=lbl)
    ax.axhline(1.0, color="0.4", lw=1, ls=":", label="uniform damage (no layer worth freeing)")
    ax.set_xlabel("relative depth   (0 = first MoE layer, 1 = last)")
    ax.set_ylabel("per-layer damage / that model's mean")
    ax.set_title("Residency damage profile does not keep its shape across scale\n"
                 "OLMoE is U-shaped; Qwen3.5 is late-heavy", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper center")
    ax.grid(alpha=0.25)
    out = os.path.join(FIGURES, "residency_profile_transfer.png")
    os.makedirs(FIGURES, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  OLMoE   ends/middle {(OLMOE[0]+OLMOE[1]+OLMOE[-2]+OLMOE[-1])/4 / (sum(OLMOE[5:11])/6):.2f}x")
    print(f"  Qwen R8 ends/middle {(q8[0]+q8[1]+q8[-2]+q8[-1])/4 / (sum(q8[13:27])/14):.2f}x")
    print(f"  Qwen R8 first2/middle {(q8[0]+q8[1])/2 / (sum(q8[13:27])/14):.2f}x  "
          f"last2/middle {(q8[-2]+q8[-1])/2 / (sum(q8[13:27])/14):.2f}x")
    print(f"  OLMoE   first2/middle {(OLMOE[0]+OLMOE[1])/2 / (sum(OLMOE[5:11])/6):.2f}x  "
          f"last2/middle {(OLMOE[-2]+OLMOE[-1])/2 / (sum(OLMOE[5:11])/6):.2f}x")
    print(f"\n[write] {out}", flush=True)


if __name__ == "__main__":
    main()
