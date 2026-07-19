#!/usr/bin/env python3
"""Vanilla-offload floor curve: decode tok/s vs pinned cold expert fetches per layer per token.
Data: results/ablations/serving_benchmarks.csv vanilla_offload rows (a6000@50af4263), measured on
the depth-re-pinned comparable model (L=45; fine ceiling 200.8 vs orig 200). Deploy references are
the original Table-2 rows (165 fine / 128 coarse). Low-N rows (n<=2) are compute/launch-bound and
carry GPU-boost variance (hollow markers); the bandwidth-bound floor targets are stable.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTD = f"{REPO}/results/phase0/figures"

MOE_C, MOE_F = "#5aa0dd", "#0d3b66"
TMP_C, TMP_F = "#5cc85c", "#145a14"

FINE = {0: 186.6, 1: 121.3, 2: 135.5, 4: 99.5, 8: 65.5, 14: 43.1, 16: 38.7, 18: 35.1}
COARSE = {0: 232.4, 1: 127.5, 2: 82.9, 4: 50.6, 5: 42.0, 6: 36.1}
FINE_TARGETS, COARSE_TARGETS = [14, 16], [5]
DEPLOY_FINE, DEPLOY_COARSE = 165, 128

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10})
fig, ax = plt.subplots(figsize=(3.9, 2.6))
for data, color, marker, label, targets in [
    (FINE, MOE_F, "o", "fine 18-of-192", FINE_TARGETS),
    (COARSE, MOE_C, "s", "coarse 6-of-64", COARSE_TARGETS),
]:
    xs = sorted(data)
    ax.plot(xs, [data[x] for x in xs], "-", color=color, lw=1.8, label=label)
    for x in xs:
        launchbound = x <= 2
        ax.plot(x, data[x], marker, color=color, ms=7 if x in targets else 5,
                mfc="white" if launchbound else color, mec=color, mew=1.3, zorder=4)
ax.axhline(DEPLOY_FINE, color=TMP_F, ls="--", lw=1.4)
ax.axhline(DEPLOY_COARSE, color=TMP_C, ls="--", lw=1.4)
ax.text(17.9, DEPLOY_FINE + 4, "deploy, fine", color=TMP_F, fontsize=8, ha="right")
ax.text(17.9, DEPLOY_COARSE + 4, "deploy, coarse", color=TMP_C, fontsize=8, ha="right")
ax.set_xlabel("cold expert fetches per layer per token")
ax.set_ylabel("decode tok/s")
ax.set_xticks([0, 2, 4, 8, 14, 16, 18])
ax.set_ylim(0, 245)
ax.grid(True, ls=":", alpha=0.4)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
fig.tight_layout()
out = f"{OUTD}/vanilla_floor_curve_nocaption.png"
fig.savefig(out, dpi=200)
print("wrote", out)
