#!/usr/bin/env python3
"""How much a better eviction rule can buy, against the offline optimum.

The seven-panel version of this figure was unreadable: seven panels across fourteen inches gave each
title about two inches of width for a label like "temporal @1e17, fine-grained (18 of 192 experts)",
so every title overran into its neighbours. Panels also repeated the same ordering seven times to make
one point.

This draws the shipped configurations only, one bar group per policy, with each configuration as a
separate bar. The comparison the figure exists to make is Belady minus min_logit: a small gap means a
smarter eviction rule has little left to win.

`probe_replay.py` also emits a version of this figure, but only on a machine holding the router logs.
This script reads the committed CSV instead, so the figure is reproducible anywhere. If the two ever
disagree, this one is derived from the committed numbers and wins.

  $PY analysis/plots/plot_eviction_headroom.py [--no-caption]

Reads   results/ablations/e5_eviction_policy_headroom.csv
Writes  results/phase0/figures/eviction_policy_headroom_belady_bound[_nocaption].png
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "results", "ablations")
OUT = os.path.join(REPO, "results", "phase0", "figures")

# The variant screens (anti-collapse penalties, bursty routing, head and momentum families) exist to
# distort the expert distribution, so pooling them with the shipped cells would misstate the headroom.
SCREENS = ("ant0p", "bursty", "_head", "_mom", "karen", "auxfree")

# Worst to best, so the bars read upward toward the bound.
ORDER = ["LRU", "min_logit", "Belady", "Belady+prefetch(h=16)", "Belady+prefetch(h=4)",
         "Belady+prefetch(h=1)", "discounted-oracle(g=0.95)", "discounted-oracle(g=0.9)",
         "discounted-oracle(g=0.5)"]
COLOR = {"LRU": "#9e9e9e", "min_logit": "#2e7d32"}


def load():
    per = defaultdict(lambda: defaultdict(list))
    label = {}
    for r in csv.DictReader(open(os.path.join(DATA, "e5_eviction_policy_headroom.csv"))):
        run = r["run"]
        if any(s in run for s in SCREENS):
            continue
        try:
            per[run][r["policy"]].append(float(r["set_coverage"]) * 100)
        except (TypeError, ValueError):
            continue
        label[run] = f"{'temporal' if r['regime'] != 'full' else 'unconstrained'} {r['grain']} @{r['budget']}"
    return {run: {p: st.median(v) for p, v in d.items()} for run, d in per.items()}, label


tbl, label = load()
runs = sorted(tbl, key=lambda r: label[r])
pol = [p for p in ORDER if any(p in tbl[r] for r in runs)]

fig, ax = plt.subplots(figsize=(9.2, 5.4) if PAPER else (9.6, 6.6))
h = 0.8 / len(runs)
for i, run in enumerate(runs):
    vals = [tbl[run].get(p, np.nan) for p in pol]
    ax.barh(np.arange(len(pol)) + i * h - 0.4 + h / 2, vals, height=h * 0.92,
            label=label[run], zorder=2)

gaps = [tbl[r]["Belady"] - tbl[r]["min_logit"] for r in runs if "Belady" in tbl[r] and "min_logit" in tbl[r]]
ax.set_yticks(np.arange(len(pol)))
ax.set_yticklabels(pol, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("set coverage, % of the top-k demand already resident   (higher is better)")
ax.grid(True, axis="x", ls=":", alpha=0.35, zorder=0)
ax.set_xlim(0, max(v for d in tbl.values() for v in d.values()) * 1.28)
ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
ax.set_title("Eviction has little headroom; using future demand is what buys coverage", fontsize=11)

if PAPER:
    out = os.path.join(OUT, "eviction_policy_headroom_belady_bound_nocaption.png")
else:
    fig.subplots_adjust(bottom=0.26)
    fig.text(0.02, 0.015,
             "Set coverage under each eviction policy, replayed on the same logged demand at K=k with\n"
             "one swap per token, median over layers. One bar per shipped configuration; the sixteen\n"
             "diversity-suppression screens are excluded because they distort the expert distribution\n"
             "by design. min_logit is the shipped rule and Belady is the offline optimum for a pure\n"
             f"eviction rule, so their gap bounds what a smarter rule can win: {min(gaps):.1f} to\n"
             f"{max(gaps):.1f} points here. The discounted-oracle rows clear that bound only because\n"
             "they are allowed to see future demand, which is a different lever.",
             fontsize=8.6, va="bottom", family="monospace", color="#333")
    out = os.path.join(OUT, "eviction_policy_headroom_belady_bound.png")

os.makedirs(OUT, exist_ok=True)
fig.savefig(out, dpi=190, bbox_inches="tight")
print(f"wrote {out}  ({len(runs)} shipped configs, {len(pol)} policies; "
      f"Belady minus min_logit {min(gaps):.1f} to {max(gaps):.1f} points)")
