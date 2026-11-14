#!/usr/bin/env python3
"""Summarise the temporal-to-free curriculum runs (results/ablations/CURRICULUM_PLAN.md) into
results/ablations/curriculum_1e17.csv: one row per run with the recipe (from the router banner in
its log), the final unconstrained test CE and BPB (pythia-50k divisor 2.9780), the delta against
the C0 control of the same grain (the full MoE trained through the same router path on the same
corpus; the recorded g*_moe_s2_1e17 cells used the 16k tokenizer and are not comparable), and the
validation loss at every tenth of training (the unmask shock and the recovery are read off these).

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
VAL = re.compile(r"validation loss at iteration (\d+) \| lm loss value: ([0-9.E+-]+)")
TEST = re.compile(r"validation loss at iteration \d+ on test set \| lm loss value: ([0-9.E+-]+)")


def recipe(run):
    log = os.path.join(RUNS, run, "train.log")
    txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
    m = re.search(r"\[temporal\][^\n]*", txt)
    return (m.group(0) if m else "").replace(",", ";")


def main():
    rows, c0 = [], {}
    for d in glob.glob(os.path.join(RUNS, "cur_g*_1e17_C0")):
        t = TEST.findall(open(os.path.join(d, "train.log"), errors="ignore").read()) if os.path.exists(os.path.join(d, "train.log")) else []
        if t:
            c0[os.path.basename(d).split("_")[1]] = float(t[-1])
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
        arm = run.split("_", 3)[3]
        free_ce = ""
        sw = [r for r in csv.DictReader(open(os.path.join(ABLATIONS, "sweep_eval.csv"))) if r["run"] == run and r["tag"] == "cross"]
        if sw:
            free_ce = f"{float(sw[-1]['lm_loss']):.6f}"
        if arm.startswith("HET") and sw:
            ce = float(sw[-1]["lm_loss"])      # HET before the eval fix: the logged final eval was constrained; the free re-score is the score
        # WK arms (reuse-fraction sweep): the score is the logged final eval under the model's own policy; free_CE is informational
        ref = c0.get(grain)
        tenths = [vals[i] for i in sorted(vals)]
        rows.append([run, arm, grain, f"cur_{grain}_1e17_C0" if ref else "", f"{ce:.6f}", f"{ce / DIV:.4f}",
                     f"{ce - ref:+.4f}" if ref else "", f"{(ce - ref) / DIV:+.4f}" if ref else "", free_ce, recipe(run)]
                    + [f"{v:.4f}" for v in tenths])
    with open(OUT, "w", newline="") as fh:
        fh.write("# temporal-to-free curriculum at 1e17 (CURRICULUM_PLAN.md), pythia-50k tokenizer and the 1e18/1e19 DCLM "
                 "corpus: final unconstrained test CE per run and delta vs the C0 control of the same grain (full MoE through "
                 "the router path, same corpus); the recorded g*_moe_s2_1e17 cells used the 16k tokenizer and are not comparable; "
                 "v<i> = validation loss at the i-th tenth of training; BPB = CE/2.9780; seed sd at 1e18 is ~0.005 CE, "
                 "win bar 0.010; _16k runs are on the recorded cells' tokenizer (BPB divisor 2.7568 there, not applied) and compare to g1_moe_s2_1e17 3.4930 / g1_tmoe_s2_1e17 3.5486; test_CE is under the model's own policy, free_CE the re-score with every expert allowed. Producer analysis/residency/curriculum_csv.py\n")
        w = csv.writer(fh)
        w.writerow(["run", "arm", "grain", "reference", "test_CE", "test_BPB", "delta_CE_vs_C0", "delta_BPB_vs_C0",
                    "free_CE", "recipe"] + [f"v{i}" for i in range(1, 11)])
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")
    for r in rows:
        print("  " + ",".join(r[:8]))


if __name__ == "__main__":
    main()
