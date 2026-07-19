#!/usr/bin/env python3
"""Training-loss curves at 1e19 (the paper's largest budget): test that stability reads the way
people expect it to, as smooth loss descent. Data: results/ablations/t19_1e19_curves.csv
(train_lm_loss_bpb every 10 iterations; run names per that file). Grad-norm census lives in
stability_gradnorms.csv; this figure is the loss-side companion."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTD = f"{REPO}/results/phase0/figures"

DENSE_C, MOE_C = "#7f7f7f", "#5aa0dd"
TMP_C, TMP_F = "#5cc85c", "#145a14"
RUNS = [
    ("dense_1e19", DENSE_C, "dense"),
    ("moe_coarse_1e19", MOE_C, "MoE · coarse"),
    ("g1_tmoe_coarse_1e19", TMP_C, "temporal · coarse"),
    ("temporal_fine_g3_1e19", TMP_F, "temporal · fine"),
]

series = {r: ([], []) for r, _, _ in RUNS}
with open(f"{REPO}/results/ablations/t19_1e19_curves.csv") as f:
    for row in csv.DictReader(f):
        if row["run"] in series and row["train_lm_loss_bpb"]:
            series[row["run"]][0].append(int(row["iteration"]))
            series[row["run"]][1].append(float(row["train_lm_loss_bpb"]))

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10})
fig, ax = plt.subplots(figsize=(3.9, 2.6))
for run, color, label in RUNS:
    xs, ys = series[run]
    ax.plot(xs, ys, "-", color=color, lw=1.5, label=label)
ax.set_xlabel("iteration")
ax.set_ylabel("train loss (BPB)")
ax.set_ylim(1.0, 2.6)
ax.grid(True, ls=":", alpha=0.4)
ax.legend(fontsize=8, framealpha=0.9)
fig.tight_layout()
out = f"{OUTD}/loss_curves_1e19_nocaption.png"
fig.savefig(out, dpi=200)
print("wrote", out)
