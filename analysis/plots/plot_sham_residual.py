#!/usr/bin/env python3
"""What the magnitude-matched sham does and does not reproduce -- behind 01-findings.md section 3.2.

Top: per-layer cost of imposing the residency constraint on one layer of a trained unconstrained
checkpoint, against a sham of matched mean magnitude that carries no lexical information (router-logit
noise, sigma calibrated so the mean cost matches to 0.2%).

Bottom: the residual, real minus sham. This is the panel that matters. Reporting only "the sham
reproduces 58% of the endpoint excess" invites the reading that the remaining 42% is diffuse noise.
It is not: the residual is near zero or negative across the interior and positive at both ends, and
the last layer's is four times any other. Whatever the sham fails to explain is itself an endpoint
effect.

  $PY analysis/plots/plot_sham_residual.py [--no-caption]

Reads   results/ablations/swap_sweep.csv, results/ablations/sham_magnitude_matched.csv
Writes  results/phase0/figures/sham_residual[_nocaption].png
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
DATA = os.path.join(REPO, "results", "ablations")
OUT = os.path.join(REPO, "results", "phase0", "figures")
RUN = "flame38m_g1_moe"          # the arm with both a real profile and a matched sham

real, native = {}, None
for r in csv.DictReader(open(os.path.join(DATA, "swap_sweep.csv"))):
    if r["run"] != RUN or r.get("perturbation") != "real":
        continue
    if r["arm"] == "native":
        native = float(r["test_CE"])
    elif r["arm"] == "impose_one":
        real[int(r["layer"])] = float(r["test_CE"])
real = {l: ce - native for l, ce in real.items()}

sham, base = defaultdict(dict), {}
for r in csv.DictReader(open(os.path.join(DATA, "sham_magnitude_matched.csv"))):
    if r["run"] != RUN:
        continue
    sham[r["sigma"]][int(r["layer"])] = float(r["cost"])
    base[r["sigma"]] = float(r["baseline_CE"])
# the sham calibrated closest to the real mean
mean_real = sum(real.values()) / len(real)
sig = min(sham, key=lambda s: abs(sum(sham[s].values()) / len(sham[s]) - mean_real))
sh = sham[sig]

layers = sorted(real)
fig, (hi, lo) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.2) if PAPER else (7.6, 7.4),
                             gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.10})

hi.plot(layers, [real[l] for l in layers], color="#0d3b66", lw=2.4, marker="o", ms=7,
        markeredgecolor="white", label="real constraint imposed on one layer")
hi.plot(layers, [sh[l] for l in layers], color="#c26a2a", lw=2.4, marker="s", ms=6,
        markeredgecolor="white", ls="--",
        label=f"magnitude-matched sham (logit noise, σ={sig})")
hi.set_ylabel("cost of constraining that\nlayer alone (Δ test CE)")
hi.grid(alpha=0.25, lw=0.6)
hi.legend(loc="upper center", fontsize=9, framealpha=0.95)

resid = [real[l] - sh[l] for l in layers]
lo.axhline(0, color="#888", lw=1.0, ls=":")
lo.bar(layers, resid, color=["#b03030" if r > 0 else "#5aa0dd" for r in resid],
       width=0.62, edgecolor="white", linewidth=0.8)
for l, r in zip(layers, resid):
    lo.annotate(f"{r:+.3f}", (l, r), textcoords="offset points",
                xytext=(0, 4 if r > 0 else -12), ha="center", fontsize=8)
lo.set_ylabel("residual\n(real − sham)")
lo.set_xlabel("MoE layer  (layer 1 is a dense FFN; 9 is the last, feeding the unembedding)")
lo.set_xticks(layers)
lo.grid(alpha=0.25, lw=0.6, axis="y")

if PAPER:
    out = os.path.join(OUT, "sham_residual_nocaption.png")
else:
    fig.subplots_adjust(bottom=0.26)
    fig.text(0.02, 0.015,
             "A lexicality-free perturbation of matched average size reproduces most of the endpoint\n"
             "cost, but not all of it, and the part it misses is not spread across the network. The\n"
             "residual is near zero or negative at every interior layer and positive at both ends, with\n"
             "the last layer's four times any other. So the 58-85% positional figure does not license\n"
             "ignoring the remainder: what a generic perturbation fails to explain is itself an\n"
             "endpoint effect, and it is unexplained.",
             fontsize=8.6, va="bottom", family="monospace", color="#333")
    out = os.path.join(OUT, "sham_residual.png")

os.makedirs(OUT, exist_ok=True)
fig.savefig(out, dpi=190, bbox_inches="tight")
print(f"wrote {out}  (sham σ={sig}; mean real {mean_real:+.4f} vs sham "
      f"{sum(sh.values())/len(sh):+.4f}; residual at last layer {resid[-1]:+.4f})")
