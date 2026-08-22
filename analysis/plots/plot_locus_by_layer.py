#!/usr/bin/env python3
"""C1 -- locus of routing specialization against NORMALIZED depth, with bootstrap intervals.

y = median over that layer's experts of (context AUC - token AUC). Above 0 = routing at that layer is
better predicted by the excluded-context mean than by the current token; below 0 = lexical.

x = l/L, the layer's index over the model's transformer depth. This is the change that makes the
figure readable: the previous version plotted the raw MoE layer index, which is not comparable across
models of different depth -- the 1e16 model's layer 4 is its last, while the 1e19 model's layer 4 is
under a third of the way through a 14-layer stack. Every H1 claim is a claim about depth, so the axis
has to be depth.

Bands are 95% bootstrap intervals on the per-layer median, 2000 resamples of that layer's experts. A
slope without an interval is not testable, and the slope table this writes
(`mechinterp_locus_slopes.csv`) is what report item 2 -- do the temporal and baseline depth slopes
differ? -- is answered from.

Sources:
  mechinterp_locus.csv       1e16/1e17 cells, MoE layers 2-6, variant kfull = w=k, position split.
                             These runs are absent from MANIFEST.csv, so no capture or checkpoint
                             survives and they cannot be extended past layer 6 or re-split.
  mechinterp_locus_1e19.csv  1e19 cells, MoE layers 2-14, all three windows, both splits.

Both files carry a `window` column now; w=k is selected everywhere via `kfull`, which also fixes the
collision whereby the old 1e19 file wrote w=k under the name `base` while `base` means w=32 in the
1e16/1e17 file.

The 1e19 rows are read at split=sequence (documents held out). The published split cuts the flattened
[S*B] stream at 70%, which is a sequence *position*, so every document appears in both the fit and
score halves; `--split position` reproduces it and the slope CSV carries both.

  python3 analysis/plots/plot_locus_by_layer.py [--no-caption] [--split sequence|position]

Output: results/phase0/figures/locus_by_layer[_nocaption].png
        results/ablations/mechinterp_locus_slopes.csv
"""
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib
import hashlib

import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = "--no-caption" in sys.argv
SPLIT = "position" if "--split" in sys.argv and sys.argv[sys.argv.index("--split") + 1] == "position" \
    else "sequence"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "results", "ablations")
OUT = os.path.join(REPO, "results", "phase0", "figures")
sys.path.insert(0, os.path.join(REPO, "analysis", "probes"))
import registry                                                    # noqa: E402  (needs REPO first)

BOOT = 2000


def _rng_for(vals):
    """A generator seeded from the data itself, so a bootstrap interval depends only on its own
    inputs.

    A single module-level RNG shared across every call looks seeded but is not reproducible: each
    draw advances the shared state, so every interval depends on how many were computed before it.
    Adding one run or one layer silently shifts every subsequent CI -- which is what happened here.
    The committed CSV stopped matching what the script produced, and both the intervals and the point
    estimates moved (slope 0.0408 -> 0.0397 on one row).

    Seeding from a hash of the values makes each interval a pure function of its own input, so the
    file regenerates byte-identically regardless of what else is in the run set or what order it is
    processed in.
    """
    h = hashlib.blake2b(np.ascontiguousarray(np.asarray(vals, dtype=np.float64)).tobytes(),
                        digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))

MOE_FINE, MOE_COARSE = "#0d3b66", "#5aa0dd"
TMP_FINE, TMP_COARSE = "#145a14", "#5cc85c"
BUDGET_MARKER = {"1e16": "o", "1e17": "s", "1e18": "D", "1e19": "^"}

# (file, label, variant, colour, budget, legend, linestyle). variant kfull = w=k everywhere.
SERIES = [
    ("mechinterp_locus.csv",      "s0_TEMPORAL",          "kfull", TMP_FINE,   "1e16", "temporal 18/192", "-"),
    ("mechinterp_locus_1e19.csv", "temporal_fine_1e19",   "kfull", TMP_FINE,   "1e19", "temporal 18/192", "-"),
    ("mechinterp_locus.csv",      "s2_TEMPORAL",          "kfull", TMP_COARSE, "1e17", "temporal 6/64",   "-"),
    ("mechinterp_locus_1e19.csv", "temporal_coarse_1e19", "kfull", TMP_COARSE, "1e19", "temporal 6/64",   "-"),
    ("mechinterp_locus.csv",      "s0_FULL",              "kfull", MOE_FINE,   "1e16", "full MoE 18/192 (sigmoid router)", "-"),
    ("mechinterp_locus.csv",      "s0_SOFTMAX_BASELINE",  "base",  MOE_FINE,   "1e16", "full MoE 18/192 (widest window only)", "--"),
    ("mechinterp_locus.csv",      "s2_FULL",              "kfull", MOE_COARSE, "1e17", "full MoE 6/64",   "-"),
    ("mechinterp_locus_1e19.csv", "moe_coarse_1e19",      "kfull", MOE_COARSE, "1e19", "full MoE 6/64",   "-"),
    # 1e18, the budget at which the temporal model wins and where no capture-based measurement existed
    # before the Step 3 sweep. Matched temporal/unconstrained pairs at both granularities.
    ("mechinterp_locus_1e19.csv", "flame38m_g3_temporal", "kfull", TMP_FINE,   "1e18", "temporal 18/192", "-"),
    ("mechinterp_locus_1e19.csv", "flame38m_g1_temporal", "kfull", TMP_COARSE, "1e18", "temporal 6/64",   "-"),
    ("mechinterp_locus_1e19.csv", "flame38m_g3_moe",      "kfull", MOE_FINE,   "1e18", "full MoE 18/192", "-"),
    ("mechinterp_locus_1e19.csv", "flame38m_g1_moe",      "kfull", MOE_COARSE, "1e18", "full MoE 6/64",   "-"),
]


def per_layer(fname, label, variant, split):
    """-> {layer: [context_minus_token per expert]}, run name, window. Drops non-finite probes."""
    g = defaultdict(list)
    run = window = None
    path = os.path.join(DATA, fname)
    with open(path) as f:
        rdr = csv.DictReader(f)
        has_split = "split" in (rdr.fieldnames or [])
        for r in rdr:
            if r["label"] != label or r["variant"] != variant:
                continue
            if has_split and r["split"] != split:
                continue
            try:
                d = float(r["context_minus_token"])
            except (ValueError, TypeError):
                continue
            if not math.isnan(d):
                g[int(r["layer"])].append(d)
                run = r["run"]
                window = r.get("window") or ""
    return g, run, window


def boot_median_ci(vals, n=BOOT):
    """95% bootstrap interval on the median of one layer's experts."""
    a = np.asarray(vals, float)
    if a.size < 3:
        return float(np.median(a)), float(np.median(a)), float(np.median(a))
    draws = np.median(_rng_for(a).choice(a, size=(n, a.size), replace=True), axis=1)
    return float(np.median(a)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_stat(g, xs_by_layer, stat, n=BOOT):
    """Bootstrap a statistic of the per-layer median curve. Returns (point, lo95, hi95).

    Experts are resampled within each layer independently, mirroring how the medians were formed, so
    the interval reflects per-expert sampling noise rather than treating the medians as exact.
    """
    layers = sorted(g)
    if len(layers) < 3:
        return (float("nan"),) * 3
    x = np.array([xs_by_layer[l] for l in layers], float)
    arrs = [np.asarray(g[l], float) for l in layers]
    point = stat(x, np.array([np.median(a) for a in arrs]))
    # One generator per array, built ONCE and then drawn from n times. Constructing it inside the
    # resampling loop reseeds it identically on every iteration, so all n draws come out the same and
    # the percentiles collapse to a point -- a "confidence interval" of zero width that still passes
    # an idempotence check, because two runs agree perfectly on the same degenerate number.
    rngs = [_rng_for(a) for a in arrs]
    resampled = [np.median(r.choice(a, size=(n, a.size), replace=True), axis=1)
                 for r, a in zip(rngs, arrs)]        # [n] medians per layer
    draws = np.array([stat(x, np.array([col[i] for col in resampled])) for i in range(n)])
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))



def _vertex_ci(vx, depth):
    """Report the vertex interval only when it locates the vertex inside the model.

    A quadratic fitted to a nearly straight per-layer profile fits well and still says nothing about
    where its vertex is: `s0_FULL` has quadratic_r2 = 1.00 and a vertex interval of +-25 layers on a
    4-layer model, and one 14-layer run spans 550 layers. So goodness of fit is the wrong gate --
    r2 measures whether a parabola describes the points, not whether the data pin down its turning
    point.

    The gate used instead is identifiability: if the interval is wider than the stack it is supposed
    to locate a layer within, it excludes nothing and is not reported. The point estimate is kept,
    being a deterministic property of the fit, and `vertex_identified` records the verdict so a reader
    sees a blank was deliberate rather than missing.
    """
    lo, hi = vx[1], vx[2]
    if not (np.isfinite(lo) and np.isfinite(hi)) or (hi - lo) > depth:
        return "", "", 0
    return round(lo, 2), round(hi, 2), 1


def _r2(p, x, y):
    resid = ((y - np.polyval(p, x)) ** 2).sum()
    tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - resid / tot if tot > 0 else float("nan")


# A straight line is the wrong summary for these curves and reporting only its slope actively misleads.
# The contextual share rises with depth and then turns over, so a full-range OLS slope mixes the rising
# and falling halves: on the coarse temporal arm the linear fit explains R2=0.43 while a quadratic
# explains 0.94, and the linear slope (+0.059) is dragged so far below the rising-region slope (+0.191)
# that it reverses the comparison against the unconstrained baseline. Curvature and vertex are therefore
# reported alongside, plus the slope restricted to layers at or above the vertex region.
SLOPE = lambda x, y: np.polyfit(x, y, 1)[0]
CURV = lambda x, y: np.polyfit(x, y, 2)[0]


def vertex_in_layers(depth):
    """Quadratic vertex, expressed as a layer index rather than in normalized-depth units."""
    def f(x, y):
        q = np.polyfit(x, y, 2)
        return -q[1] / (2 * q[0]) * depth if q[0] != 0 else float("nan")
    return f


def rising_slope(cut_x):
    """OLS slope over the rising portion only, i.e. x <= cut_x."""
    def f(x, y):
        m = x <= cut_x
        return np.polyfit(x[m], y[m], 1)[0] if m.sum() >= 2 else float("nan")
    return f


if PAPER:
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "legend.fontsize": 9,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})

fig, (hi, lo) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 5.0) if PAPER else (8.0, 7.4),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})

slope_rows, counts, missing = [], [], []
for fname, label, variant, color, budget, legend, ls in SERIES:
    g, run, window = per_layer(fname, label, variant, SPLIT)
    if not g:
        print(f"[warn] no rows for {label}/{variant} at split={SPLIT}", file=sys.stderr)
        missing.append(f"{label}/{variant}")
        continue
    depth = registry.depth_of(run) if run else None
    if not depth:
        print(f"[warn] unknown depth for run {run}; cannot place {label} on a normalized axis",
              file=sys.stderr)
        missing.append(f"{label} (depth unknown)")
        continue
    layers = sorted(g)
    xs = {l: l / depth for l in layers}
    med, ylo, yhi = zip(*(boot_median_ci(g[l]) for l in layers))
    x = [xs[l] for l in layers]
    ax = hi if med[0] > 0 else lo
    ax.fill_between(x, ylo, yhi, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, med, ls, color=color, marker=BUDGET_MARKER[budget], markersize=7,
            linewidth=1.8, markeredgecolor="white", markeredgewidth=0.8,
            label=f"{legend} at $10^{{{budget[2:]}}}$")
    # slopes in both units: per unit normalized depth (comparable across models) and per layer
    # index (comparable with the published table), then the shape statistics that a slope hides
    s_nd = boot_stat(g, xs, SLOPE)
    s_ix = boot_stat(g, {l: float(l) for l in layers}, SLOPE)
    cu = boot_stat(g, xs, CURV)
    vx = boot_stat(g, xs, vertex_in_layers(depth))
    ymed = np.array([np.median(g[l]) for l in layers])
    xv = np.array([xs[l] for l in layers])
    lin_r2 = _r2(np.polyfit(xv, ymed, 1), xv, ymed)
    quad_r2 = _r2(np.polyfit(xv, ymed, 2), xv, ymed) if len(layers) >= 3 else float("nan")
    # Rising region: up to the vertex where one falls inside the stack, else the whole range.
    cut = min(max(vx[0], layers[0] + 1), layers[-1]) / depth if np.isfinite(vx[0]) else 1.0
    rs = boot_stat(g, xs, rising_slope(cut))
    slope_rows.append([label, run, budget, "temporal" if med[0] > 0 else "full", variant,
                       window, SPLIT, depth,
                       layers[0], layers[-1], len(layers),
                       round(s_nd[0], 4), round(s_nd[1], 4), round(s_nd[2], 4),
                       round(s_ix[0], 5), round(s_ix[1], 5), round(s_ix[2], 5),
                       round(lin_r2, 3), round(quad_r2, 3),
                       round(cu[0], 4), round(cu[1], 4), round(cu[2], 4),
                       round(vx[0], 2), *_vertex_ci(vx, depth),
                       round(rs[0], 4), round(rs[1], 4), round(rs[2], 4),
                       round(med[0], 4), round(med[-1], 4)])
    counts.append(f"{label}: n={min(len(v) for v in g.values())}-{max(len(v) for v in g.values())}"
                  f"/layer, layers {layers[0]}-{layers[-1]} of {depth}")

hi.axhline(0, color="#888", linewidth=1.0, linestyle=":")
hi.set_ylim(-0.02, 0.24)
lo.set_ylim(-0.38, -0.13)
hi.spines["bottom"].set_visible(False)
lo.spines["top"].set_visible(False)
hi.tick_params(labeltop=False, bottom=False)

kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle="none",
          color="k", mec="k", mew=1, clip_on=False)
hi.plot([0, 1], [0, 0], transform=hi.transAxes, **kw)
lo.plot([0, 1], [1, 1], transform=lo.transAxes, **kw)

for ax in (hi, lo):
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(0.0, 1.05)
lo.set_xlabel("normalized depth  $l/L$   (layer 1 is a dense FFN in every config)")
fig.supylabel("median over experts:  context AUC $-$ token AUC",
              x=0.012 if PAPER else 0.035, fontsize=13)
if PAPER:
    fig.subplots_adjust(left=0.17)
hi.text(0.012, 0.90, "context-dominated (temporal)", transform=hi.transAxes,
        fontsize=10, color="#145a14", weight="bold")
lo.text(0.012, 0.08, "token-dominated (unconstrained MoE)", transform=lo.transAxes,
        fontsize=10, color="#0d3b66", weight="bold")

hh, ll = hi.get_legend_handles_labels()
h2, l2 = lo.get_legend_handles_labels()
LEGEND_NCOL, LEGEND_Y = 2, -0.28
if PAPER:
    # colour/shade/marker encoding is carried by the caption (isoFLOP standard);
    # only the two control series need naming in-figure
    keep = [(h, l) for h, l in zip(hh + h2, ll + l2)
            if "sigmoid" in l or "widest" in l]
    lo.legend([h for h, _ in keep], [l for _, l in keep], loc="lower right",
              fontsize=10, framealpha=0.95)
else:
    lo.legend(hh + h2, ll + l2, loc="upper center", bbox_to_anchor=(0.5, LEGEND_Y),
              framealpha=0.95, ncol=LEGEND_NCOL, handlelength=2.6, columnspacing=1.4)

# The caption sits below the legend, so its offset has to follow the legend's height rather than be a
# constant. It was a constant, tuned when SERIES held 8 entries (4 rows); adding the four 1e18 arms
# grew the legend to 6 rows and it landed on top of the caption. Deriving the offset from the row
# count reproduces the known-good 4-row spacing (-0.72) and keeps working as series are added.
_legend_rows = -(-len(hh + h2) // LEGEND_NCOL)          # ceil division
CAPTION_Y = LEGEND_Y - 0.095 * _legend_rows - 0.06

if PAPER:
    out = os.path.join(OUT, "locus_by_layer_nocaption.png")
else:
    lo.text(-0.14, CAPTION_Y,
            "Locus of routing specialization by normalized depth. Per (layer, expert) ridge probes\n"
            "predict whether expert e serves token t, from either the current token embedding E[x_t]\n"
            "or the excluded-context mean over +-w=k neighbours; AUC is held out on unseen documents,\n"
            "measured chance floor 0.500+-0.002 under iid permutation. Points are medians over that\n"
            "layer's experts, bands are 95% bootstrap intervals (2000 resamples). Colour = setup\n"
            "(blue = unconstrained MoE, green = temporal; dark = fine 18/192, light = coarse 6/64),\n"
            "marker = compute budget. Note the broken y-axis: the regime gap (~0.3) dwarfs every\n"
            "depth effect (~0.05). The 1e19 models are 14 layers deep and are probed at every MoE\n"
            "layer, 2-14; the 1e16/1e17 cells stop at 6 because their captures were not preserved.",
            transform=lo.transAxes, fontsize=8.6, va="top", ha="left",
            family="monospace", color="#333")
    out = os.path.join(OUT, "locus_by_layer.png")

os.makedirs(OUT, exist_ok=True)

# A regenerated, committed artifact may not silently narrow -- and that applies to the figure, not
# only to the CSV beside it. The slopes CSV is merged per series so a run missing some inputs cannot
# shrink it; the PNG is rewritten wholesale by whichever split ran last, so the documented
# `--split position` invocation replaced a 12-series figure with a 5-series one (384909 -> 252674
# bytes) while exiting 0, leaving a caption describing curves the image no longer contained.
#
# `missing` already lists every series that could not be plotted this run, so the check is just:
# refuse to overwrite a committed figure with a partial one.
if missing and "--replace" not in sys.argv and os.path.exists(out):
    sys.exit(
        f"[abort] refusing to overwrite {os.path.basename(out)} with a partial figure: "
        f"{len(SERIES) - len(missing)} of {len(SERIES)} series plotted.\n"
        + "".join(f"         omitted: {m}\n" for m in missing)
        + f"         The committed figure has all {len(SERIES)}. This usually means the requested "
          f"split\n         lacks rows for those series -- check the locus driver ran with "
          f"--both-splits.\n         Pass --replace if a narrower figure is intended.")

fig.savefig(out, dpi=190, bbox_inches="tight")
print("wrote", out)
for c in counts:
    print("  " + c)
if missing:
    print("  [omitted] " + "; ".join(missing))

HEADER = ["label", "run", "budget", "regime", "variant", "window", "split", "depth_L",
          "first_layer", "last_layer", "n_layers",
          "slope_per_normdepth", "slope_nd_lo95", "slope_nd_hi95",
          "slope_per_layer", "slope_ix_lo95", "slope_ix_hi95",
          "linear_r2", "quadratic_r2",
          "curvature", "curvature_lo95", "curvature_hi95",
          "vertex_layer", "vertex_lo95", "vertex_hi95", "vertex_identified",
          "rising_slope", "rising_lo95", "rising_hi95",
          "median_at_first_layer", "median_at_last_layer"]
sp = os.path.join(DATA, "mechinterp_locus_slopes.csv")
KEY = ("label", "variant", "split")          # one row per series per split


def _key_of_row(row):
    return (row[HEADER.index("label")], row[HEADER.index("variant")], row[HEADER.index("split")])


prior = []
if os.path.exists(sp):
    with open(sp) as f:
        prior = list(csv.DictReader(f))

# Merge per SERIES, not per split. Replacing a whole split wholesale creates a dilemma with no good
# answer: allow it and a run whose inputs are missing silently deletes series from a committed file
# (24 rows -> 17); forbid it and the series that CAN be recomputed are frozen at stale values, because
# the only command that would refresh them aborts. Both happened here in turn.
#
# Keying on (label, variant, split) dissolves it. A series computed this run is updated; a series not
# computed keeps whatever it had. Nothing is lost and nothing is stranded, so no --replace escape and
# no shrink check are needed for the normal path.
fresh = {_key_of_row(r): r for r in slope_rows}
merged, updated, carried = [], 0, 0
for e in prior:
    k = (e.get("label"), e.get("variant"), e.get("split"))
    if k in fresh:
        merged.append(fresh.pop(k)); updated += 1
    else:
        # May predate a schema change; fill absent columns rather than dropping the row.
        merged.append([e.get(h, "") for h in HEADER]); carried += 1
added = len(fresh)
merged.extend(fresh.values())                # series that did not exist in the file before

with open(sp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(HEADER)
    w.writerows(merged)
print(f"wrote {sp}: {len(merged)} rows total at split={SPLIT} "
      f"({updated} updated, {added} added, {carried} carried over untouched)")
print(f"\n{'label':22} {'R2 lin/quad':>12} {'full slope':>20} {'rising slope':>20} "
      f"{'curvature':>20} {'vertex':>16}")
# Index by name rather than by position: adding vertex_identified shifted every column after it and
# silently broke this line, which is the kind of breakage a positional index invites.
_i = {h: n for n, h in enumerate(HEADER)}
for r in slope_rows:
    _v = (f"L{r[_i['vertex_layer']]:.1f} [{r[_i['vertex_lo95']]:.1f},{r[_i['vertex_hi95']]:.1f}]"
          if r[_i["vertex_identified"]] else
          f"L{r[_i['vertex_layer']]:.1f} (CI wider than the stack; not identified)")
    print(f"{r[0]:22} {r[_i['linear_r2']]:>5.2f}/{r[_i['quadratic_r2']]:<5.2f} "
          f"{r[_i['slope_per_normdepth']]:+.4f} "
          f"[{r[_i['slope_nd_lo95']]:+.3f},{r[_i['slope_nd_hi95']]:+.3f}] "
          f"{r[_i['rising_slope']]:+.4f} "
          f"[{r[_i['rising_lo95']]:+.3f},{r[_i['rising_hi95']]:+.3f}] "
          f"{r[_i['curvature']]:+.4f} "
          f"[{r[_i['curvature_lo95']]:+.3f},{r[_i['curvature_hi95']]:+.3f}] "
          f"{_v}")
