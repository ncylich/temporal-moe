#!/usr/bin/env python3
"""De-lexicalization figure pair (paper Section 4).

Left: per-expert locus scatter -- token-only AUC (x) vs context-excluding-token AUC (y) at the
residency-lifetime window (w = k). One point per expert, four cells:
All four cells come from results/ablations/mechinterp_locus.csv, selected on its `label` column:
  full MoE 18/192 @1e16  label s0_SOFTMAX_BASELINE (retrained softmax-aux baseline, w=18)
  temporal 18/192 @1e16  label s0_TEMPORAL (w=18)
  full MoE 6/64  @1e17   label s2_FULL (w=6, softmax-aux baseline)
  temporal 6/64  @1e17   label s2_TEMPORAL (w=6)
Chance floors (permutation + circular-shift nulls) all 0.500 +/- 0.002:
results/ablations/mechinterp_floors.csv, `model` column, same five labels.

These were once separate files under results/phase0/figure_data/ (mechinterp_softmax_locus.csv,
mechinterp_locus_kfull.csv, mechinterp_softmax_floors.csv). They were consolidated into the two
files above and that directory no longer exists.

Right: residency dose -- held-out BPB vs cache size R at fixed FLOPs (192E, k=18, 1e16), from
rsweep.csv; R=192 endpoint is the retrained softmax baseline (seed 1234, definitive 1.4519).

Colors follow the isoFLOP standard: hue = method (MoE blue, temporal green), shade =
granularity (fine 18/192 dark, coarse 6/64 normal).
Output: results/phase0/figures/delexicalization_locus_scatter[_nocaption].png,
        results/phase0/figures/residency_dose_curve[_nocaption].png
"""
import csv, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = "--no-caption" in sys.argv
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "results", "ablations")
OUT = os.path.join(REPO, "results", "phase0", "figures")

if PAPER:
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10})


def read(name, label):
    pts = []
    with open(os.path.join(DATA, name)) as f:
        for r in csv.DictReader(f):
            if r["label"] != label:
                continue
            t, c = float(r["token_AUC"] or "nan"), float(r["context_AUC"] or "nan")
            if not (math.isnan(t) or math.isnan(c)):
                pts.append((t, c))
    return pts


# ---------------- left: locus scatter ----------------
SERIES = [  # (points, color, label)
    (read("mechinterp_locus.csv", "s0_SOFTMAX_BASELINE"), "#0d3b66", "full MoE 18/192"),
    (read("mechinterp_locus.csv", "s2_FULL"), "#5aa0dd", "full MoE 6/64"),
    (read("mechinterp_locus.csv", "s0_TEMPORAL"), "#145a14", "temporal 18/192"),
    (read("mechinterp_locus.csv", "s2_TEMPORAL"), "#5cc85c", "temporal 6/64"),
]
fig, ax = plt.subplots(figsize=(4.5, 4.15) if PAPER else (7.2, 6.6))
lo, hi = 0.35, 1.0
ax.plot([lo, hi], [lo, hi], ls="--", color="0.45", lw=1.2, zorder=1)
ax.axhline(0.5, ls=":", color="0.75", lw=0.9, zorder=0)
ax.axvline(0.5, ls=":", color="0.75", lw=0.9, zorder=0)
for pts, color, label in SERIES:
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, s=11 if PAPER else 13, color=color, alpha=0.45, lw=0, label=label, zorder=2)
ax.text(0.965, 0.44, "token-dominated", ha="right", fontsize=11.5 if PAPER else 11,
        style="italic", color="0.35")
ax.text(0.385, 0.93, "context-dominated", ha="left", fontsize=11.5 if PAPER else 11,
        style="italic", color="0.35")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.set_xlabel("token AUC")
ax.set_ylabel("context AUC (token excluded)")
ax.set_title("What predicts expert firing")
leg = ax.legend(loc="lower left", framealpha=0.9, borderpad=0.4, handletextpad=0.2)
for h in leg.legend_handles:
    h.set_alpha(0.9)
if PAPER:
    fig.tight_layout()
    out = os.path.join(OUT, "delexicalization_locus_scatter_nocaption.png")
else:
    fig.text(0.5, 0.005,
             "Per-expert logistic probes at the residency-lifetime window (w=k): x = held-out AUC of the "
             "current token alone, y = AUC of the surrounding context with the token excluded. Chance "
             "floors 0.500+/-0.002. Full-MoE experts (blue) are token lookups; temporal experts (green) "
             "are context-dominated (85-91%).", ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUT, "delexicalization_locus_scatter.png")
fig.savefig(out, dpi=200)
print("wrote", out)

# ---------------- right: residency dose ----------------
R, bpb = [], []
with open(os.path.join(DATA, "rsweep.csv")) as f:
    for r in csv.DictReader(f):
        R.append(int(r["R"])); bpb.append(float(r["test_bpb"]))
# Wider than tall: five monotone points do not need a square, and the figure now stands
# in its own half-width column beside the locus panel.
fig, ax = plt.subplots(figsize=(4.9, 3.05) if PAPER else (6.6, 5.2))
ax.plot(R, bpb, "-", color="#145a14", lw=2.0, zorder=2)
ax.scatter(R[:-1], bpb[:-1], s=64, color="#145a14", zorder=3, label="temporal, cache $R$")
ax.scatter(R[-1:], bpb[-1:], s=74, color="#0d3b66", marker="s", zorder=3, label="full MoE ($R{=}E$)")
ax.annotate("$R{=}k$", (R[0], bpb[0]), textcoords="offset points", xytext=(9, -5),
            fontsize=12 if PAPER else 11)
ax.grid(True, ls=":", alpha=0.4)
ax.set_xticks(R)
ax.set_xlim(4, 206)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.005))   # 0.023 BPB total range
ax.set_xlabel("resident experts $R$ (of 192)")
ax.set_ylabel("held-out BPB")
ax.set_title("Residency dose")
ax.legend(loc="upper right", framealpha=0.9, borderpad=0.4)
if PAPER:
    fig.tight_layout()
    out = os.path.join(OUT, "residency_dose_curve_nocaption.png")
else:
    fig.text(0.5, 0.005,
             "Trained-from-scratch cells at 1e16 FLOPs (192 experts, k=18, FLOPs identical at every R): "
             "loss falls monotonically as the cache grows, to the retrained softmax-baseline endpoint "
             "1.4519. R is the resident-expert memory, so this is the serving memory-quality frontier "
             "(+0.023 BPB at ~1/10 routed-expert memory).", ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(OUT, "residency_dose_curve.png")
fig.savefig(out, dpi=200)
print("wrote", out)
