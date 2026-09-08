#!/usr/bin/env python3
"""Producer for results/ablations/unmask_eval_1e19.csv from the in-process sweep rows that
scripts/residency/orchestration/tmoe_unmask_1e19.sh writes to results/ablations/sweep_eval.csv.

Each 1e19 checkpoint is scored on the full 20-iteration test split (the canonical end-of-training
test eval) in its native regime and in the crossed one: a temporal model with every expert
resident (unmask), a full MoE under rolling residency at R = k (impose). BPB = CE / 2.9780.
Rows keep the July schema; the two coarse cells and the fine temporal cell reproduce the July
values to the third decimal, the fine full MoE is new (trained 2026-09-03).

    $PY analysis/residency/unmask_eval_csv.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS  # noqa: E402

SWEEP = os.path.join(ABLATIONS, "sweep_eval.csv")
OUT = os.path.join(ABLATIONS, "unmask_eval_1e19.csv")
DIV = 2.9780
CELLS = [  # run, cell label, paradigm, native regime label, cross regime label
    ("moe_coarse_1e19", "moe_coarse_1e19", "full_moe", "unconstrained", "imposed_R6"),
    ("g1_tmoe_coarse_1e19", "temporal_coarse_1e19", "temporal", "masked_R6", "unmasked_R64"),
    ("temporal_fine_g3_1e19", "temporal_fine_1e19", "temporal", "masked_R18", "unmasked_R192"),
    ("moe_fine_g3_1e19", "moe_fine_1e19", "full_moe", "unconstrained", "imposed_R18"),
]


def main():
    last = {}
    for r in csv.DictReader(open(SWEEP)):
        last[(r["run"], r["tag"])] = float(r["lm_loss"])          # latest row wins
    rows = []
    for run, cell, para, nat, cross in CELLS:
        if (run, "native") not in last or (run, "cross") not in last:
            print(f"[skip] {run}: sweep rows missing")
            continue
        n, c = last[(run, "native")], last[(run, "cross")]
        rows.append([cell, "1e19_50k", para, "test_CE", nat, f"{n:.4f}", cross, f"{c:.4f}", f"{c - n:+.4f}"])
        rows.append([cell, "1e19_50k", para, "test_BPB", nat, f"{n / DIV:.4f}", cross, f"{c / DIV:.4f}",
                     f"{(c - n) / DIV:+.4f}"])
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "scale", "trained_paradigm", "metric", "native_regime", "native_value",
                    "cross_regime", "cross_value", "delta"])
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")
    for r in rows:
        print("  " + ",".join(r))


if __name__ == "__main__":
    main()
