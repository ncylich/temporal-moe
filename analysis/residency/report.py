#!/usr/bin/env python3
"""Reporting for the PLE program: ladder gates, and the figure.

    report.py gate r32       -> RUN / SKIP / AMBIGUOUS, per PLE_PLAN.md §5's r=32 rule
    report.py gate winners   -> the top two tying ranks, for the CE stage
    report.py figure         -> results/phase0/figures/layer_freeing_damage.png
                                from layer_freeing_results.csv

Cell results are NOT assembled here; consolidate.py writes the single results CSV from the per-cell
JSONs the trainer emits. This file holds only the decision rules and the plot, so there is one
place that decides and one place that draws.

The figure is rendered from the CSV rather than inside layer_ablation.py because matplotlib is
absent from the venv that runs the model.
"""
import csv, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmoe_paths import DATA_DIR
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS

TWO_SIGMA = 0.012


def cells():
    out = {}
    for p in glob.glob(os.path.join(DATA_DIR, "ple_ladder_*.json")):
        r = json.load(open(p))
        out[str(r["rank"])] = r["final_bpb"]
    return out


def gate(what):
    c = cells()
    if what == "r32":
        need = ("full", "512", "128")
        if not all(k in c for k in need):
            print("AMBIGUOUS missing " + ",".join(k for k in need if k not in c)); return
        f, r512, r128 = c["full"], c["512"], c["128"]
        if (r512 - f) > TWO_SIGMA and (r128 - r512) > TWO_SIGMA:
            print("SKIP rank binds: full < 512 < 128 each by >2sigma")
        elif (r128 - min(f, r512)) < TWO_SIGMA:
            print("RUN 128 is within 2sigma of the best of full/512")
        else:
            print("AMBIGUOUS neither monotone-degrading nor 128-competitive; §5 does not cover this")
    else:
        if not c:
            print(""); return
        best = min(c.values())
        # §5 carries "the top two ranks" when they tie, not every rank inside 2 sigma of the best.
        win = sorted([k for k, v in c.items() if (v - best) < TWO_SIGMA], key=lambda k: c[k])[:2]
        print(" ".join(win))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # layer damage lives in the layer-freeing table, not the PLE one: it is a property of the
    # constraint, measured by removing it, and has nothing to do with per-layer embeddings.
    dmg, full = {}, None
    src = os.path.join(ABLATIONS, "layer_freeing_results.csv")
    for r in csv.DictReader(open(src)):
        if r["group"] == "layer_damage" and r["metric"] == "damage_bpb":
            if r["name"].isdigit():
                dmg[int(r["name"])] = float(r["value"])
            elif r["name"] == "all constrained":
                full = float(r["value"])
    if full is None or not dmg:
        raise SystemExit(f"no layer_damage rows in {src}; run layer_ablation.py then consolidate.py")
    xs = sorted(dmg); ys = [dmg[i] for i in xs]; u = full / len(xs); s = sum(ys)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(xs, ys, color=["#0d3b66" if v >= u else "#5aa0dd" for v in ys], alpha=0.9)
    ax.axhline(u, ls="--", color="0.4", lw=1.3, label=f"uniform share of full damage ({u:.4f})")
    ax.set_xlabel("MoE layer index"); ax.set_ylabel("BPB increase vs free routing")
    ax.set_title("Residency damage per layer (R=8, one layer constrained at a time)")
    ax.set_xticks(xs); ax.grid(True, axis="y", ls=":", alpha=0.4); ax.legend()
    fig.text(0.5, 0.005,
             f"Base OLMoE, no training. BPB all-free 0.6727, all-constrained 2.7507; full damage "
             f"{full:.4f}. Sum of single-layer damage {s:.4f} = {s/full:.3f} of the full, so the "
             f"constraint is mildly SUPER-additive across the network.",
             ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(os.path.dirname(ABLATIONS), "phase0", "figures", "layer_freeing_damage.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if a == "figure":
        figure()
    else:
        gate(sys.argv[2] if len(sys.argv) > 2 else "winners")
