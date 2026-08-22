#!/usr/bin/env python3
"""The regime separation, one point per model arm -- the figure behind 01-findings.md section 1. (rewrite pending)

Each point is one trained model: the median over its experts of the token-probe AUC (x) against the
context-probe AUC (y). Colour is regime, marker is compute budget, size is granularity. The diagonal
is where the two probes tie; above it context wins.

The figure exists to show what separates the regimes and what does not, because the prose claim was
wrong once in exactly that way. Token AUC separates the arms completely, with a gap. Context AUC does
not -- several constrained arms sit below unconstrained ones on the y axis alone. What has no overlap
at all is which side of the diagonal an arm falls on.

  $PY analysis/plots/plot_arm_separation.py [--no-caption]

Reads   results/ablations/mechinterp_locus{,_1e19}.csv   (variant kfull = w=k, held-out documents)
Writes  results/phase0/figures/arm_separation[_nocaption].png
"""
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "results", "ablations")
OUT = os.path.join(REPO, "results", "phase0", "figures")

MOE, TMP = "#0d3b66", "#145a14"
MARKER = {"1e16": "o", "1e17": "s", "1e18": "D", "1e19": "^"}


def arms():
    """-> {run: (regime, budget, median token AUC, median context AUC, % context-dominated)}."""
    g = defaultdict(list)
    meta = {}
    for name in ("mechinterp_locus.csv", "mechinterp_locus_1e19.csv"):
        with open(os.path.join(DATA, name)) as f:
            rdr = csv.DictReader(f)
            has_split = "split" in (rdr.fieldnames or [])
            for r in rdr:
                if r["variant"] != "kfull":
                    continue
                if has_split and r["split"] != "sequence":
                    continue
                try:
                    t, c = float(r["token_AUC"]), float(r["context_AUC"])
                except (ValueError, TypeError):
                    continue
                if math.isnan(t) or math.isnan(c):
                    continue
                g[r["run"]].append((t, c))
                # regime and budget are blank on the legacy 1e16/1e17 rows; derive from the run name,
                # which is what miscounting these four cost the findings document once already.
                reg = r.get("regime") or ("temporal" if "tmoe" in r["run"] else "full")
                bud = r.get("budget") or ("1e17" if "1e17" in r["run"] else "1e16")
                meta[r["run"]] = (reg, bud)
    out = {}
    for run, v in g.items():
        reg, bud = meta[run]
        out[run] = (reg, bud, st.median([a for a, _ in v]), st.median([b for _, b in v]),
                    sum(1 for a, b in v if b > a) / len(v) * 100)
    return out


A = arms()
fig, ax = plt.subplots(figsize=(6.0, 5.6) if PAPER else (7.6, 6.8))
lo, hi = 0.50, 1.00
ax.plot([lo, hi], [lo, hi], color="#999", lw=1.0, ls=":", zorder=1)
ax.text(0.735, 0.745, "probes tie", color="#777", fontsize=9, rotation=38,
        ha="center", va="center", backgroundcolor="white")

def grain_of(run):
    """Granularity from the run name. The locus CSVs carry no grain column."""
    for tag, g in (("_g1", "coarse"), ("_g3", "fine"), ("_g5", "wide"),
                   ("coarse", "coarse"), ("fine", "fine")):
        if tag in run:
            return g
    return "fine" if "192" in run else "coarse"


# Join each series across budgets, so the reader can see which way a configuration moves with scale
# rather than only where the two clouds sit. One line per (regime, granularity).
BUD_ORDER = {"1e16": 0, "1e17": 1, "1e18": 2, "1e19": 3}
cell = defaultdict(list)
for run, (reg, bud, t, c, dom) in A.items():
    cell[(reg, grain_of(run), bud)].append((t, c))
series = defaultdict(list)
for (reg, g, bud), pts in cell.items():
    series[(reg, g)].append((BUD_ORDER.get(bud, 9),
                             st.median([p[0] for p in pts]), st.median([p[1] for p in pts])))
for (reg, g), pts in series.items():
    if len(pts) < 2:
        continue
    pts.sort()
    ax.plot([p[1] for p in pts], [p[2] for p in pts],
            color=TMP if reg == "temporal" else MOE, lw=1.4, alpha=0.55,
            ls="-" if g == "fine" else "--", zorder=2)

seen = set()
for run, (reg, bud, t, c, dom) in sorted(A.items()):
    col = TMP if reg == "temporal" else MOE
    lab = f"{'temporal' if reg=='temporal' else 'unconstrained'} at $10^{{{bud[2:]}}}$"
    ax.scatter(t, c, s=70, color=col, marker=MARKER.get(bud, "o"), alpha=0.85,
               edgecolor="white", linewidth=0.8, zorder=3,
               label=lab if lab not in seen else None)
    seen.add(lab)

tok_t = [v[2] for v in A.values() if v[0] == "temporal"]
tok_f = [v[2] for v in A.values() if v[0] != "temporal"]
gap_lo, gap_hi = max(tok_t), min(tok_f)
ax.axvspan(gap_lo, gap_hi, color="#f2c14e", alpha=0.18, zorder=0)
ax.text((gap_lo + gap_hi) / 2, 0.545, f"no model here\n{gap_hi - gap_lo:.3f} wide",
        ha="center", va="bottom", fontsize=9, color="#8a6d1f")

ax.set_xlim(lo, hi)
ax.set_ylim(0.52, 0.82)
ax.set_xlabel("token probe AUC  (current token alone)")
ax.set_ylabel("context probe AUC  (neighbours, token excluded)")
ax.grid(alpha=0.25, lw=0.6)
h, l = ax.get_legend_handles_labels()
order = sorted(range(len(l)), key=lambda i: (l[i].split("10^{")[1], l[i]))
ax.legend([h[i] for i in order], [l[i] for i in order],
          loc="upper left", fontsize=9, framealpha=0.95, ncol=2)

if PAPER:
    out = os.path.join(OUT, "arm_separation_nocaption.png")
else:
    fig.subplots_adjust(bottom=0.30)
    fig.text(0.02, 0.015,
             "Regime separation, one point per trained model (34 arms, four budgets, three\n"
             "granularities). x = median token-probe AUC, y = median context-probe AUC, both held out\n"
             "on unseen documents at window w=k; chance is 0.500. Colour = regime, marker = budget.\n"
             "Lines join one granularity across budgets (solid fine, dashed coarse), through the\n"
             "median of the cells sharing a budget, so a line shows where a configuration moves as\n"
             "it scales. The token axis separates the arms completely: the shaded band holds no model\n"
             "of either regime. The context axis does NOT separate them: several constrained arms sit\n"
             "below unconstrained ones. What never overlaps is which side of the dotted diagonal an\n"
             "arm falls on, i.e. which of the two features wins.",
             fontsize=8.6, va="bottom", family="monospace", color="#333")
    out = os.path.join(OUT, "arm_separation.png")

os.makedirs(OUT, exist_ok=True)
fig.savefig(out, dpi=190, bbox_inches="tight")
print(f"wrote {out}  ({len(A)} arms: "
      f"{sum(1 for v in A.values() if v[0]=='temporal')} temporal, "
      f"{sum(1 for v in A.values() if v[0]!='temporal')} unconstrained; "
      f"token-AUC gap {gap_lo:.3f}-{gap_hi:.3f})")
