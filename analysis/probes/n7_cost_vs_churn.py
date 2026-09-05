#!/usr/bin/env python3
"""N7 -- does a layer cost what it costs because of how much it churns, or because of where it sits?

The per-layer sweeps give, for every MoE layer, the loss change from relaxing (or imposing) rolling
residency at that layer alone. The replay gives, for the same layers, the mean swap rate -- how often
the resident set actually turns over. If the expensive layers are the churny ones, per-layer cost is a
statement about traffic and a prefetcher could act on it. If instead cost tracks depth with churn flat,
the cost is about position in the stack and no amount of caching addresses it.

The plan calls this free because both inputs are already on disk: swap_sweep.csv and
e1_swap_rate_by_layer.csv. Nothing is recomputed here.

Two correlations are reported per run, both Spearman so a monotone-but-curved relation still registers:

  cost vs swap rate   the traffic explanation
  cost vs depth       the position explanation

Reporting both matters because they are not exclusive and, on this data, they disagree -- reporting
only the one that comes out larger would be picking a conclusion rather than measuring one. Depth is
scored as |layer - middle| so that a U-shaped profile, which is what the control shows, registers as
a relation instead of cancelling to zero the way a raw-depth correlation would.

Output: results/ablations/n7_cost_vs_churn.csv, one row per (run, layer), plus a per-run summary.
"""
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS

OUT = os.path.join(ABLATIONS, "n7_cost_vs_churn.csv")
HEADER = ["run", "budget", "regime", "arm", "layer", "depth_frac", "dist_from_middle",
          "test_CE", "native_CE", "cost", "mean_swap_rate"]


def _spearman(xs, ys):
    """Rank correlation, written out so this has no scipy dependency."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:                                  # average ranks within ties
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    sweep = list(csv.DictReader(open(os.path.join(ABLATIONS, "swap_sweep.csv"))))
    churn_path = os.path.join(ABLATIONS, "e1_swap_rate_by_layer.csv")
    churn = {}
    if os.path.exists(churn_path):
        for r in csv.DictReader(open(churn_path)):
            churn[(r["run"], r["layer"])] = float(r["mean_swap_rate"])
    else:
        print(f"[warn] {churn_path} missing; swap rates will be blank and only the depth "
              f"relation is measurable")

    # native baseline per run, taken from the real arm only -- the sham rows carry a different
    # perturbation and would silently shift every cost if pooled in.
    native = {r["run"]: float(r["test_CE"]) for r in sweep
              if r["arm"] == "native" and r.get("perturbation", "real") == "real"}

    rows, per_run = [], defaultdict(list)
    for r in sweep:
        if r["arm"] not in ("unmask_one", "impose_one") or r["layer"] in ("-", ""):
            continue
        if ";" in r["layer"]:                          # multi-layer set arms are not per-layer points
            continue
        if r.get("perturbation", "real") != "real" or r["run"] not in native:
            continue
        L = int(r["layer"])
        cost = float(r["test_CE"]) - native[r["run"]]
        rows.append([r["run"], r["budget"], r["regime"], r["arm"], L, "", "",
                     r["test_CE"], f"{native[r['run']]:.6f}", f"{cost:.6f}",
                     churn.get((r["run"], r["layer"]), "")])
        per_run[r["run"]].append((L, cost, churn.get((r["run"], r["layer"]))))

    # depth is only meaningful relative to that run's own layer range
    for run, pts in per_run.items():
        Ls = [p[0] for p in pts]
        lo, hi = min(Ls), max(Ls)
        mid = (lo + hi) / 2.0
        span = max(hi - lo, 1)
        for row in rows:
            if row[0] == run:
                row[5] = f"{(row[4] - lo) / span:.4f}"
                row[6] = f"{abs(row[4] - mid):.4f}"

    os.makedirs(ABLATIONS, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"[write] {OUT}: {len(rows)} rows\n")

    print(f"{'run':26} {'n':>3}  {'rho(cost,swap)':>15} {'rho(cost,|depth-mid|)':>22}  reading")
    print("-" * 100)
    for run in sorted(per_run):
        pts = sorted(per_run[run])
        Ls = [p[0] for p in pts]
        mid = (min(Ls) + max(Ls)) / 2.0
        costs = [p[1] for p in pts]
        withchurn = [(c, s) for _, c, s in pts if s is not None]
        r_sw = _spearman([c for c, _ in withchurn], [s for _, s in withchurn]) if len(withchurn) >= 3 \
            else float("nan")
        r_dp = _spearman(costs, [abs(L - mid) for L in Ls])
        if r_dp == r_dp and abs(r_dp) > 0.5 and not (r_sw == r_sw and abs(r_sw) > abs(r_dp)):
            reading = "position: cost rises toward the ends"
        elif r_sw == r_sw and abs(r_sw) > 0.5:
            reading = "traffic: cost tracks churn"
        else:
            reading = "neither dominates"
        sw = f"{r_sw:.3f}" if r_sw == r_sw else "  n/a"
        print(f"{run:26} {len(pts):3d}  {sw:>15} {r_dp:>22.3f}  {reading}")
    print("\nrho is Spearman, range -1..1. |depth-mid| is distance from the middle layer, so a U -- "
          "expensive at both ends -- shows up as a positive correlation rather than cancelling out.")


if __name__ == "__main__":
    main()
