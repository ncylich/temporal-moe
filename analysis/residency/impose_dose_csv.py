#!/usr/bin/env python3
"""Imposition dose curve at 1e19: rolling residency imposed on the two trained full MoEs at
R = k, 2k, 4k, 8k, E, from the sweep rows scripts/residency/orchestration/tmoe_impose_dose_1e19.sh
writes to results/ablations/sweep_eval.csv (tags imposeR<R>, full 20-iteration test split).

Writes results/ablations/impose_dose_1e19.csv (run, grain, E, k, R, R_over_k, test_CE, test_BPB,
delta_bpb_vs_unconstrained) and results/ablations/figures/impose_dose_1e19.png, test BPB against
R on a log axis with the matched temporal model's native BPB at R = k as a reference line. The
header records, per grain, the smallest R at which the imposed full MoE beats the temporal model
at R = k, and the linear-in-log-R interpolation of the R where they cross.

    $PY analysis/residency/impose_dose_csv.py [--no-caption]
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS  # noqa: E402

SWEEP = os.path.join(ABLATIONS, "sweep_eval.csv")
UNMASK = os.path.join(ABLATIONS, "unmask_eval_1e19.csv")
OUT = os.path.join(ABLATIONS, "impose_dose_1e19.csv")
FIG = os.path.join(ABLATIONS, "figures")
DIV = 2.9780
CELLS = [("moe_coarse_1e19", 1, 64, 6, "temporal_coarse_1e19"), ("moe_fine_g3_1e19", 3, 192, 18, "temporal_fine_1e19")]


def main():
    paper = "--no-caption" in sys.argv
    last = {}
    for r in csv.DictReader(open(SWEEP)):
        if r["tag"].startswith("imposeR") and "__" not in r["tag"]:      # skip the selftest repeat
            last[(r["run"], int(r["tag"][7:]))] = float(r["lm_loss"])
    temporal = {r["cell"]: float(r["native_value"]) for r in csv.DictReader(open(UNMASK)) if r["metric"] == "test_BPB"}
    rows, notes, curves = [], [], {}
    for run, grain, E, k, tcell in CELLS:
        Rs = sorted(R for (rr, R) in last if rr == run)
        if not Rs:
            print(f"[skip] {run}: no sweep rows"); continue
        ce = {R: last[(run, R)] for R in Rs}
        base = ce[E] if E in ce else None
        for R in Rs:
            rows.append([run, grain, E, k, R, f"{R / k:.2f}", f"{ce[R]:.6f}", f"{ce[R] / DIV:.4f}",
                         (f"{(ce[R] - base) / DIV:+.4f}" if base is not None else "")])
        bpb = {R: ce[R] / DIV for R in Rs}
        curves[run] = (Rs, bpb, k, tcell)
        tb = temporal.get(tcell)
        if tb is not None:
            beats = [R for R in Rs if bpb[R] <= tb]
            first = min(beats) if beats else None
            # crossing by linear interpolation in log2(R)
            cross = None
            for a, b in zip(Rs, Rs[1:]):
                if (bpb[a] - tb) * (bpb[b] - tb) <= 0 and bpb[a] != bpb[b]:
                    t = (bpb[a] - tb) / (bpb[a] - bpb[b])
                    cross = 2 ** (np.log2(a) + t * (np.log2(b) - np.log2(a))); break
            notes.append(f"{run}: temporal at R={k} reads {tb:.4f} BPB, imposed full MoE first beats it at "
                         + (f"R={first} ({first / k:.0f}k)" if first is not None else "no R below E")
                         + (f", crossing near R={cross:.0f} ({cross / k:.1f}k)" if cross else ""))
    with open(OUT, "w", newline="") as fh:
        fh.write("# imposition dose curve at 1e19 (full 20-iteration test split, BPB = CE/2.9780, "
                 "producer analysis/residency/impose_dose_csv.py from sweep_eval.csv imposeR<R> rows); "
                 + "; ".join(notes) + "\n")
        w = csv.writer(fh)
        w.writerow(["run", "grain", "E", "k", "R", "R_over_k", "test_CE", "test_BPB", "delta_bpb_vs_unconstrained"])
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")
    for n in notes:
        print("  " + n)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if paper:
        plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 10,
                             "ytick.labelsize": 10.5, "legend.fontsize": 10})
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    cols = {"moe_coarse_1e19": ("#5aa0dd", "#5cc85c", "6 of 64"), "moe_fine_g3_1e19": ("#0d3b66", "#145a14", "18 of 192")}
    for run, (Rs, bpb, k, tcell) in curves.items():
        cf, ct, lab = cols[run]
        ax.plot([R / k for R in Rs], [bpb[R] for R in Rs], "-o", color=cf, ms=5, label=f"full MoE, {lab}, residency imposed")
        if tcell in temporal:
            ax.axhline(temporal[tcell], color=ct, ls="--", lw=1.4, label=f"temporal, {lab}, trained at R = k")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 64 / 6])
    ax.set_xticklabels(["k", "2k", "4k", "8k", "E"])
    ax.set_xlabel("resident experts R per layer, as a multiple of k (rightmost point is R = E)")
    ax.set_ylabel("test BPB")
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5)
    if not paper:
        ax.set_title("Rolling residency imposed on trained full MoEs at 1e19", fontsize=10.5)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    name = f"impose_dose_1e19{'_nocaption' if paper else ''}.png"
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight")
    print(f"wrote {name}")


if __name__ == "__main__":
    main()
