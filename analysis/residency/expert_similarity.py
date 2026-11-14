#!/usr/bin/env python3
"""Aggregate results/ablations/expert_similarity/<run>.npz (analysis/probes/expert_similarity.py)
into results/ablations/expert_similarity.csv (one row per run and MoE layer, every metric), draw
results/ablations/figures/expert_similarity_depth.png (selected-vs-next-best output cosine and
the gated-layer change from a next-best swap, against depth, temporal vs full MoE per grain), and
run the prediction test: per layer and matched pair, does the regime gap in output similarity
line up with the regime gap in substitution cost (substitution_tolerance.csv, random / own /
matched, per-layer rows)? Prints Spearman correlations and writes them to the CSV header.

    $PY analysis/residency/expert_similarity.py [--no-caption]
"""
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS  # noqa: E402

SRC = os.path.join(ABLATIONS, "expert_similarity")
CSV = os.path.join(ABLATIONS, "expert_similarity.csv")
SUBST = os.path.join(ABLATIONS, "substitution_tolerance.csv")
FIG = os.path.join(ABLATIONS, "figures")
FULL, TEMPORAL, INIT = "#2F6DB5", "#2E8B57", "#9AA1AB"
PAIRS = [("flame38m_g1_temporal", "flame38m_g1_moe"), ("flame38m_g1_temporal_s2", "flame38m_g1_moe_s2"),
         ("flame38m_g1_temporal_s3", "flame38m_g1_moe_s3"), ("flame38m_g3_temporal", "flame38m_g3_moe"),
         ("flame38m_g3_temporal_s2", "flame38m_g3_moe_s2"), ("flame38m_g3_temporal_s3", "flame38m_g3_moe_s3"),
         ("g3_tmoe_s2_1e17", "g3_moe_s2_1e17"), ("g1_tmoe_s2_1e17", "g1_moe_s2_1e17"),
         ("g1_tmoe_coarse_1e19", "moe_coarse_1e19"), ("temporal_fine_g3_1e19", "moe_fine_g3_1e19")]


def meta(run):
    if run.startswith("init_"):
        shape, g = run.split("_")[1], int(run.split("_g")[-1])
        return {"s2": "1e17", "s38m": "1e18", "s19opt": "1e19"}[shape], g, 0
    if run.startswith("flame38m_"):
        s = re.search(r"_s(\d)$", run)
        return "1e18", int(re.search(r"_g(\d)_", run).group(1)), (int(s.group(1)) if s else 1)
    if run.endswith("_1e17"):
        return "1e17", (1 if run.startswith("g1_") else 3), 1
    if run.endswith("_1e19"):
        return "1e19", (3 if ("fine" in run or "g3" in run) else 1), 1
    raise ValueError(run)


def load_all():
    rows = []
    for p in sorted(glob.glob(os.path.join(SRC, "*.npz"))):
        z = np.load(p, allow_pickle=False)
        run = os.path.basename(p)[:-4]
        budget, grain, seed = meta(run)
        keys = [str(k) for k in z["metrics"]]
        for i, ln in enumerate(z["layers"]):
            r = {"run": run, "budget": budget, "regime": str(z["regime"]), "grain": grain, "seed": seed,
                 "experts": int(z["E"]), "topk": int(z["k"]), "n_tokens": int(z["N"]), "layer": int(ln),
                 "tokens_sha256": str(z["tokens_sha256"])}
            for j, k in enumerate(keys):
                r[k] = float(z["values"][i, j])
            rows.append(r)
    return rows


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def prediction_test(rows):
    """Per matched pair and layer: (temporal - full) similarity gap vs (temporal - full) substitution
    delta gap. Similarity gap uses the gated-layer change from a random swap (the quantity the CE
    cost should track) and the selected-vs-random output cosine."""
    if not os.path.exists(SUBST):
        return {}
    sub = {}
    for r in csv.DictReader(l for l in open(SUBST) if not l.startswith("#")):
        if r["condition"] == "random" and r["gate"] == "own" and r["fraction"] == "matched" and r["layer"].startswith("L"):
            sub[(r["run"], int(r["layer"][1:]))] = float(r["delta_bpb"])
    by = {(r["run"], r["layer"]): r for r in rows}
    xs_change, xs_cos, ys, tags = [], [], [], []
    for t, f in PAIRS:
        for (run, ln), rt in by.items():
            if run != t or (f, ln) not in by or (t, ln) not in sub or (f, ln) not in sub:
                continue
            rf = by[(f, ln)]
            xs_change.append(rt["layer_relchange_random"] - rf["layer_relchange_random"])
            xs_cos.append(rt["cos_sel_random"] - rf["cos_sel_random"])
            ys.append(sub[(t, ln)] - sub[(f, ln)])
            tags.append((t, ln))
    out = {"n_pair_layers": len(ys),
           "spearman_layerchange_gap_vs_subst_gap": spearman(xs_change, ys),
           "spearman_cos_gap_vs_subst_gap": spearman(-np.asarray(xs_cos), ys)}
    # within-model: does the per-layer swap change predict the per-layer substitution cost?
    xs, yy = [], []
    for (run, ln), r in by.items():
        if (run, ln) in sub:
            xs.append(r["layer_relchange_random"]); yy.append(sub[(run, ln)])
    out["n_model_layers"] = len(yy)
    out["spearman_layerchange_vs_subst_within_models"] = spearman(xs, yy)
    return out


def write_csv(rows, test):
    keys = [k for k in rows[0] if k not in ("run", "budget", "regime", "grain", "seed", "experts", "topk", "n_tokens", "layer", "tokens_sha256")]
    cols = ["run", "budget", "regime", "grain", "seed", "experts", "topk", "n_tokens", "layer"] + keys + ["tokens_sha256"]
    with open(CSV, "w", newline="") as fh:
        fh.write("# expert output similarity per run and MoE layer (analysis/probes/expert_similarity.py, "
                 "N sampled tokens from one cached test micro-batch, every routed expert evaluated on the same "
                 "inputs, ungated, shared expert excluded); cos_* are mean cosines between expert outputs, "
                 "relerr_* relative output differences, layer_relchange_* the relative change of the gated "
                 "layer output when the displaced expert's output is swapped for the substitute's at the same "
                 "gate, cos_weights the mean pairwise cosine of flattened expert weights; regime init = random "
                 "initialisation of the same shape. Prediction test vs substitution_tolerance.csv: "
                 + "; ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in test.items())
                 + ". Producer analysis/residency/expert_similarity.py\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.5f}" if isinstance(r[c], float) else r[c]) for c in cols})
    print(f"wrote {CSV} ({len(rows)} rows)")


def figure(rows, paper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if paper:
        plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 10,
                             "ytick.labelsize": 10.5, "legend.fontsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    marks = {"1e17": ("s", ":"), "1e18": ("o", "-"), "1e19": ("^", "--")}
    panels = [("cos_sel_nextbest", "cosine, selected vs next-best expert"),
              ("layer_relchange_random", "gated-layer change, random swap")]
    for row_i, (metric, ylab) in enumerate(panels):
        for col_i, (grain, title) in enumerate(((1, "coarse, 6 of 64"), (3, "fine, 18 of 192"))):
            ax = axes[row_i, col_i]
            for regime, color in (("full", FULL), ("temporal", TEMPORAL), ("init", INIT)):
                for budget, (mk, ls) in marks.items():
                    sel = [r for r in rows if r["regime"] == regime and r["grain"] == grain and r["budget"] == budget]
                    if not sel:
                        continue
                    layers = sorted({r["layer"] for r in sel}); L = max(layers)
                    ys = [(ln / L, np.mean([r[metric] for r in sel if r["layer"] == ln]),
                           np.min([r[metric] for r in sel if r["layer"] == ln]),
                           np.max([r[metric] for r in sel if r["layer"] == ln]),
                           len([r for r in sel if r["layer"] == ln])) for ln in layers]
                    x = [y[0] for y in ys]
                    label = f"{'full MoE' if regime == 'full' else 'temporal' if regime == 'temporal' else 'random init'}, {budget}"
                    ax.plot(x, [y[1] for y in ys], marker=mk, ls=ls, color=color, ms=4, lw=1.4 if regime != "init" else 1.0,
                            alpha=1.0 if regime != "init" else 0.8, label=label + (f" ({ys[0][4]} seeds)" if ys[0][4] > 1 else ""))
                    if ys[0][4] > 1:
                        ax.fill_between(x, [y[2] for y in ys], [y[3] for y in ys], color=color, alpha=0.15, lw=0)
            if row_i == 0:
                ax.set_title(title)
            if row_i == 1:
                ax.set_xlabel("relative depth of the MoE layer")
            if col_i == 0:
                ax.set_ylabel(ylab, fontsize=10)
            ax.grid(alpha=0.25); ax.set_axisbelow(True)
    handles, labels = {}, []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h; labels.append(l)
    order = sorted(labels, key=lambda l: (l.startswith("temporal"), l.startswith("random"), l))
    fig.legend([handles[l] for l in order], order, frameon=False, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.035))
    if not paper:
        fig.suptitle("Expert output similarity by depth. Top: cosine between the outputs of a selected expert "
                     "and the router's next-best unselected expert on the same input.\nBottom: relative change "
                     "of the gated layer output when one selected expert is swapped for a random other expert "
                     "at the same gate. Bands span the seeds.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    os.makedirs(FIG, exist_ok=True)
    name = f"expert_similarity_depth{'_nocaption' if paper else ''}.png"
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight")
    print(f"wrote {name}")


def main():
    paper = "--no-caption" in sys.argv
    rows = load_all()
    if not rows:
        sys.exit(f"no records in {SRC}")
    test = prediction_test(rows)
    for k, v in test.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    write_csv(rows, test)
    figure(rows, paper)


if __name__ == "__main__":
    main()
