#!/usr/bin/env python3
"""Assemble the correct-convention OLMoE downstream reference table.

downstream.py used to join its cells onto olmoe_adapt_downstream.csv, whose impose_R8 and
CE_adapt_R8 columns were measured under the renorm-era gate convention (archived in
results/archive/olmoe_wrong_renorm/ — see its README; the renorm impose floor 0.3164 vs the
correct 0.5723 silently inflated every cell_gap_closed). This rebuilds the same three
reference columns from correct-convention, already-committed measurements:

  base_free / impose_R8 : olmoe_downstream_naive_preserve.csv (free, R8 at gate_mass=preserve)
  CE_adapt_R8           : the sweep_lr3e5_win rows of layer_freeing_downstream.csv
                          (the 15M CE winner, trained and scored under preserve)

    make_downstream_ref.py        # writes results/ablations/olmoe_downstream_ref.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

NAIVE = os.path.join(ABLATIONS, "olmoe_downstream_naive_preserve.csv")
CELLS = os.path.join(ABLATIONS, "layer_freeing_downstream.csv")
OUT = os.path.join(ABLATIONS, "olmoe_downstream_ref.csv")


def rows_of(path):
    with open(path) as f:
        rr = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return rr[0], rr[1:]


hdr, rr = rows_of(NAIVE)
ref = {(r[hdr.index("task")], r[hdr.index("metric")]):
       [r[hdr.index("free")], r[hdr.index("R8")]] for r in rr}

chdr, crr = rows_of(CELLS)
ci = {c: chdr.index(c) for c in ("task", "metric", "cell", "cell_acc")}
ce = {(r[ci["task"]], r[ci["metric"]]): r[ci["cell_acc"]]
      for r in crr if r[ci["cell"]] == "sweep_lr3e5_win"}

with open(OUT, "w", newline="") as f:
    f.write('"# OLMoE downstream reference, correct convention (gate_mass=preserve) throughout. '
            'base_free / impose_R8 from olmoe_downstream_naive_preserve.csv; CE_adapt_R8 = the 15M '
            'CE winner (sweep_lr3e5_win rows of layer_freeing_downstream.csv). Replaces the '
            'renorm-era olmoe_adapt_downstream.csv reference (archived, results/archive/'
            'olmoe_wrong_renorm) whose impose/CE columns measured the wrong intervention. '
            'Producer: analysis/ple/make_downstream_ref.py"\n')
    w = csv.writer(f)
    w.writerow(["task", "metric", "base_free", "impose_R8", "CE_adapt_R8"])
    for (t, m), (free, r8) in ref.items():
        w.writerow([t, m, free, r8, ce.get((t, m), "")])
print(f"[ref] wrote {OUT}: {len(ref)} (task, metric) rows, "
      f"{sum(1 for k in ref if k in ce)} with CE winner values")
