#!/usr/bin/env python3
"""Summarise the temporal-to-free curriculum runs (results/ablations/CURRICULUM_PLAN.md) into
results/ablations/curriculum_1e17.csv: one row per run with the recipe (from the router banner in
its log), the final unconstrained test CE and BPB (pythia divisor 2.9780), the delta against the
recorded full-MoE baseline of the same grain, and the validation loss at every tenth of training
(the unmask shock and the recovery are read off these).

    $PY analysis/residency/curriculum_csv.py
"""
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS  # noqa: E402

RUNS = os.path.join(os.path.dirname(ABLATIONS), "phase0", "runs")
OUT = os.path.join(ABLATIONS, "curriculum_1e17.csv")
DIV = 2.9780
BASE = {"g3": ("g3_moe_s2_1e17", 3.507410), "g1": ("g1_moe_s2_1e17", 3.492985)}
TEMPORAL = {"g3": 3.553032, "g1": 3.548630}
VAL = re.compile(r"validation loss at iteration (\d+) \| lm loss value: ([0-9.E+-]+)")
TEST = re.compile(r"validation loss at iteration \d+ on test set \| lm loss value: ([0-9.E+-]+)")


def recipe(run):
    log = os.path.join(RUNS, run, "train.log")
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    m = re.search(r"\[temporal\][^\n]*", txt)
    return (m.group(0) if m else "").replace(",", ";")


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, "cur_g*_1e17_*"))):
        run = os.path.basename(d)
        grain = run.split("_")[1]
        log = os.path.join(d, "train.log")
        if not os.path.exists(log):
            continue
        txt = open(log, errors="ignore").read()
        vals = {int(i): float(v) for i, v in VAL.findall(txt)}
        tests = TEST.findall(txt)
        if not tests:
            print(f"[running] {run}: {len(vals)} evals so far"); continue
        ce = float(tests[-1])
        bname, bce = BASE[grain]
        tenths = [vals[i] for i in sorted(vals)]
        rows.append([run, run.split("_", 3)[3], grain, bname, f"{ce:.6f}", f"{ce / DIV:.4f}", f"{ce - bce:+.4f}",
                     f"{(ce - bce) / DIV:+.4f}", f"{ce - TEMPORAL[grain]:+.4f}", recipe(run)]
                    + [f"{v:.4f}" for v in tenths])
    with open(OUT, "w", newline="") as fh:
        fh.write("# temporal-to-free curriculum at 1e17 (CURRICULUM_PLAN.md): final unconstrained test CE per run, "
                 "delta vs the recorded full-MoE baseline of the same grain (g3 3.5074, g1 3.4930) and vs the "
                 "temporal-at-R=k cell (g3 3.5530, g1 3.5486); v<i> = validation loss at the i-th tenth of training; "
                 "BPB = CE/2.9780; seed sd at 1e18 is ~0.005 CE, win bar 0.010. Producer analysis/residency/curriculum_csv.py\n")
        w = csv.writer(fh)
        w.writerow(["run", "arm", "grain", "baseline", "test_CE", "test_BPB", "delta_CE_vs_moe", "delta_BPB_vs_moe",
                    "delta_CE_vs_temporal", "recipe"] + [f"v{i}" for i in range(1, 11)])
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")
    for r in rows:
        print("  " + ",".join(r[:9]))


if __name__ == "__main__":
    main()
