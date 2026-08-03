#!/usr/bin/env python3
"""Cache hit rate against MoE layer -- the figure behind 01-findings.md section 1.1.

Hit rate is the fraction of a token's UNCONSTRAINED top-k already resident when it arrives, scored
before the swap. Higher is better; a random resident set scores k/E, which is 0.094 at both 6-of-64
and 18-of-192, so the granularities are directly comparable.

**Why this is a fair regime comparison.** Every arm is the same measurement: take that model's raw
router logits, replay the identical rolling-residency policy over them, and ask how often demand is
already resident. Neither regime is handed residency for free -- the constrained model's logits merely
come from a model that was trained under the policy, and the unconstrained model's from one that was
not. That is the difference being measured, and it is the only difference.

What is NOT symmetric is the sample. Twenty-one constrained arms have a preserved router log; exactly
one unconstrained run does. The matched pair at 1e19 coarse is therefore drawn heavy and everything
else is context.

  $PY analysis/plots/plot_hitrate_by_layer.py [--no-caption]

Reads   results/ablations/e6_per_layer_ranking.csv
Writes  results/phase0/figures/hitrate_by_layer[_nocaption].png
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "phase0", "figures")

MOE, TMP = "#0d3b66", "#145a14"
RANDOM_FLOOR = 6 / 64          # == 18/192; identical at both granularities

# The one budget/granularity cell where both regimes have a preserved router log.
PAIR = {"moe_coarse_1e19": ("unconstrained MoE", MOE),
        "g1_tmoe_coarse_1e19": ("temporal", TMP)}
# Plain temporal recipes at other cells. The 1e16 router-recipe variants (ant/bursty/head/mom/momr)
# are selection-shaping experiments, not the shipped recipe, and are excluded rather than dimmed --
# fourteen of them would swamp the comparison this figure exists to show.
CONTEXT = ("flame38m_g1_temporal", "flame38m_g3_temporal", "temporal_fine_g3_1e19",
           "g3_tmoe_s2_1e17")

rows = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(REPO, "results", "ablations",
                                          "e6_per_layer_ranking.csv"))):
    rows[r["run"]][int(r["layer"])] = float(r["hit_rate"])

fig, ax = plt.subplots(figsize=(6.4, 4.8) if PAPER else (7.8, 6.4))
ax.axhline(RANDOM_FLOOR, color="#b03030", lw=1.2, ls="--", zorder=1)
ax.text(2.15, RANDOM_FLOOR + 0.008, "random resident set (k/E = 0.094)",
        color="#b03030", fontsize=9, va="bottom")

for run in CONTEXT:
    if run not in rows:
        print(f"[warn] no rows for {run}", file=sys.stderr)
        continue
    ls = sorted(rows[run])
    ax.plot(ls, [rows[run][l] for l in ls], color=TMP, alpha=0.32, lw=1.4,
            marker="o", ms=3.5, zorder=2)

for run, (lab, col) in PAIR.items():
    ls = sorted(rows[run])
    ax.plot(ls, [rows[run][l] for l in ls], color=col, lw=2.6, marker="o", ms=7,
            markeredgecolor="white", markeredgewidth=0.9, zorder=4,
            label=f"{lab} — 1e19, 6 of 64 (matched pair)")
ax.plot([], [], color=TMP, alpha=0.32, lw=1.4, marker="o", ms=3.5,
        label="other temporal arms (1e17, 1e18, 1e19)")

ax.set_xlabel("MoE layer  (layer 1 is a dense FFN in every config)")
ax.set_ylabel("cache hit rate  (share of demanded top-k already resident)")
ax.set_xticks(range(2, 15))
ax.set_ylim(0.05, 0.50)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

if PAPER:
    out = os.path.join(OUT, "hitrate_by_layer_nocaption.png")
else:
    fig.subplots_adjust(bottom=0.32)
    fig.text(0.02, 0.015,
             "Routing demand becomes more cacheable with depth, in both regimes. Hit rate is the share\n"
             "of a token's unconstrained top-k already resident on arrival, scored pre-swap; every arm\n"
             "is the same replay of the same policy over that model's own router logits, so the only\n"
             "difference between regimes is whether the model was TRAINED under the constraint. The\n"
             "heavy pair is the one cell where both regimes have a preserved router log. Note the\n"
             "sample asymmetry: 21 constrained arms exist against 1 unconstrained, so the vertical gap\n"
             "is indicative rather than estimated.",
             fontsize=8.6, va="bottom", family="monospace", color="#333")
    out = os.path.join(OUT, "hitrate_by_layer.png")

os.makedirs(OUT, exist_ok=True)
fig.savefig(out, dpi=190, bbox_inches="tight")
print(f"wrote {out}")
