#!/usr/bin/env python3
"""Regenerate the granularity-program figures from the committed CSVs.

Figure 1  damage_law.png          % BPB degradation vs R/k, per-model fits sharing one
                                  slope (fixed-effects), base checkpoints only.
Figure 2  downstream_scaling.png  % downstream degradation vs E/k at fixed memory
                                  fractions, all 7 models, bootstrap 68% bands.

    plot_scaling.py               # writes both PNGs into results/ablations/figures/
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

OUT = os.path.join(ABLATIONS, "figures")
E = {"LFM2.5-8B": 32, "OLMoE": 64, "gpt-oss-20b": 32, "Qwen3-30B": 128,
     "Gemma4-26B": 128, "gpt-oss-120b": 128, "Qwen3.5": 256}
K = {"LFM2.5-8B": 4, "OLMoE": 8, "gpt-oss-20b": 4, "Qwen3-30B": 8,
     "Gemma4-26B": 8, "gpt-oss-120b": 4, "Qwen3.5": 8}
SHARED = {"Gemma4-26B", "Qwen3.5"}
MARK = {"LFM2.5-8B": "o", "OLMoE": "s", "Qwen3-30B": "^", "Gemma4-26B": "D",
        "Qwen3.5": "v", "gpt-oss-20b": "P", "gpt-oss-120b": "X"}


def rows(name):
    with open(os.path.join(ABLATIONS, name)) as f:
        return [r for r in csv.DictReader(l for l in f if not l.startswith('"#'))]


def bpb_curves():
    """model -> (free_bpb, {R: damage}), base surfaces, s=1."""
    out = {}
    lad = rows("granularity_ladder.csv")
    for tag, name in (("lfm25", "LFM2.5-8B"), ("gemma4", "Gemma4-26B")):
        free = next(float(r["bpb"]) for r in lad if r["model"] == tag and r["cell"] == "free")
        cur = {int(r["R"]): float(r["bpb"]) - free for r in lad
               if r["model"] == tag and r["R"]}
        out[name] = (free, cur)
    fo = rows("frontier_olmoe.csv")
    free = next(float(r["bpb"]) for r in fo if r["cell"] == "free")
    out["OLMoE"] = (free, {int(r["R"]): float(r["bpb"]) - free for r in fo
                           if r["stage"] == "grid" and r["surface"] == "base"
                           and r["swaps"] == "1"})
    for fam, name, fb in (("qwen3", "Qwen3-30B", 0.615392), ("qwen3_5", "Qwen3.5", 0.625152)):
        fr = rows(f"frontier_{fam}.csv")
        cur = {}
        for r in fr:
            if r["surface"] == "base" and r["swaps"] == "1":
                cur[int(r["R"])] = float(r["bpb"]) - fb
        out[name] = (fb, cur)
    return out


def figure1():
    curves = bpb_curves()
    models = list(curves)
    pts = [(m, np.log10(R / K[m]), np.log10(100 * d / curves[m][0]))
           for m in models for R, d in curves[m][1].items() if d > 0]
    y = np.array([v for *_, v in pts])
    X = np.zeros((len(pts), len(models) + 1))
    for i, (m, r, _) in enumerate(pts):
        X[i, models.index(m)] = 1
        X[i, -1] = r
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, m in enumerate(models):
        free, cur = curves[m]
        x = [R / K[m] for R in cur]
        v = [100 * d / free for d in cur.values()]
        xs = np.logspace(np.log10(min(x)), np.log10(max(x)), 30)
        ax.plot(xs, 10 ** (b[i] + b[-1] * np.log10(xs)), "--" if m in SHARED else "-",
                color=cols[i], lw=1.8, alpha=0.85)
        ax.scatter(x, v, color=cols[i], s=55, edgecolor="black", linewidth=0.5, zorder=5,
                   label=f"{m}{' (shared)' if m in SHARED else ''}  C={10 ** b[i]:.1f}%")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_yticks([1, 2, 5, 10, 25]); ax.set_yticklabels(["1%", "2%", "5%", "10%", "25%"])
    ax.set_xlabel("R / k  (resident slots per active expert)")
    ax.set_ylabel("BPB degradation (% over free routing)")
    ax.set_title(f"degradation = C * (k/R)^{-b[-1]:.2f}   fixed-effects R2 = {r2:.2f}",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "damage_law.png"), dpi=150)


def ds_cells():
    """(fraction, model) -> % downstream degradation, same-protocol cells."""
    free = {}
    cons = {}
    for r in rows("gptoss_downstream_deltas.csv"):
        m = "gpt-oss-20b" if r["model"].startswith("gptoss20b") else "gpt-oss-120b"
        if r["arm"] == "free":
            free[m] = float(r["mean_acc"])
        else:
            cons[(m, int(r["arm"][1:]))] = float(r["mean_acc"])
    lad_free = {"LFM2.5-8B": .6714, "OLMoE": .6820, "Qwen3-30B": .7267,
                "Gemma4-26B": .7550, "Qwen3.5": .7501}
    name = {"lfm25": "LFM2.5-8B", "olmoe": "OLMoE", "qwen3": "Qwen3-30B",
            "gemma4": "Gemma4-26B", "qwen3_5": "Qwen3.5"}
    for r in rows("downstream_ladder.csv"):
        if r["cell"] != "free":
            cons[(name[r["stack"]], int(r["R"]))] = float(r["mean_acc"])
    cons[("OLMoE", 8)] = 0.5723
    cons[("Qwen3-30B", 8)] = 0.6311
    cons[("Qwen3.5", 8)] = 0.7030
    free.update(lad_free)
    out = {}
    for (m, R), acc in cons.items():
        f = R / E[m]
        if R >= K[m] and abs(f - round(f * 64) / 64) < 1e-9:
            out.setdefault(round(f, 6), {})[m] = 100 * (free[m] - acc) / free[m]
    return out, free


def figure2():
    cells, _ = ds_cells()
    rng = np.random.default_rng(0)
    colors = {0.25: "#2a9d8f", 0.125: "#e9c46a", 0.0625: "#e76f51"}
    labels = {0.25: "25% memory", 0.125: "12.5% memory", 0.0625: "6.25% memory"}
    fig, ax = plt.subplots(figsize=(8, 5.8))
    xs = np.logspace(np.log10(7), np.log10(40), 60)
    n = 0
    for f in (0.25, 0.125, 0.0625):
        pts = cells.get(f, {})
        lx = np.log10([E[m] / K[m] for m in pts]); ly = np.log10(list(pts.values()))
        B, A = np.polyfit(lx, ly, 1)
        ax.plot(xs, 10 ** (A + B * np.log10(xs)), color=colors[f], lw=2.5,
                label=f"{labels[f]}: slope {B:+.2f} (n={len(pts)})")
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(lx), len(lx))
            if len(set(lx[idx])) < 2:
                continue
            bb, aa = np.polyfit(lx[idx], ly[idx], 1)
            boots.append(aa + bb * np.log10(xs))
        lo, hi = np.percentile(10 ** np.array(boots), [16, 84], axis=0)
        ax.fill_between(xs, lo, hi, color=colors[f], alpha=0.16, linewidth=0)
        for m, v in pts.items():
            ax.scatter(E[m] / K[m], v, color=colors[f], marker=MARK[m], s=85,
                       edgecolor="black", linewidth=0.6, zorder=5)
            n += 1
    h = [plt.Line2D([], [], color="gray", marker=mk, ls="", markersize=8,
                    label=m + (" (shared)" if m in SHARED else ""))
         for m, mk in MARK.items()]
    leg1 = ax.legend(fontsize=9, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=h, fontsize=7.5, loc="lower left", title="models", ncol=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([8, 16, 32]); ax.set_xticklabels(["8", "16", "32"])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.5%", "1%", "2%", "5%", "10%", "20%"])
    ax.set_xlabel("expert sparsity  E / k")
    ax.set_ylabel("downstream accuracy degradation (% of free)")
    ax.set_title(f"Downstream cost of residency at fixed memory budgets - {n} cells, "
                 f"7 models, 5 labs", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "downstream_scaling.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure1()
    figure2()
    print("wrote", OUT)
