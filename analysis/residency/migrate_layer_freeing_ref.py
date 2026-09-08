#!/usr/bin/env python3
"""One-shot migration of layer_freeing_downstream.csv onto the correct-convention reference.

Rows written before 08-08 joined the renorm-era olmoe_adapt_downstream.csv (archived:
results/archive/olmoe_wrong_renorm), so their base_free / impose_R8 / CE_adapt_R8 columns —
and everything derived from them (cell_minus_base, cell_minus_CE, cell_gap_closed,
CE_gap_closed) — measured against the wrong intervention (impose mean 0.3164 vs the correct
0.5723). This rewrites exactly those seven columns for EVERY row from olmoe_downstream_ref.csv;
the measured columns (cell_acc, cell_se, cell, free_set, train_tokens, cell_bpb) are untouched.

    migrate_layer_freeing_ref.py           # verify + rewrite in place
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS                                          # noqa: E402

REF = os.path.join(ABLATIONS, "olmoe_downstream_ref.csv")
TARGET = os.path.join(ABLATIONS, "layer_freeing_downstream.csv")
NOTE = ("# Downstream 10-task 0-shot for free-set cells. base_free / impose_R8 / CE_adapt_R8 are "
        "reused verbatim from olmoe_downstream_ref.csv (correct convention, gate_mass=preserve). "
        "All rows migrated onto that reference on 08-08 by migrate_layer_freeing_ref.py; before "
        "then they carried the renorm-era reference (archive/olmoe_wrong_renorm). cell_gap_closed "
        "= (cell - impose)/(base_free - impose): 1.0 is free-routing quality, 0.0 is the "
        "untrained mask. Higher is better.")

with open(REF) as f:
    rr = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
rh = rr[0]
ref = {(r[rh.index("task")], r[rh.index("metric")]):
       (r[rh.index("base_free")], r[rh.index("impose_R8")], r[rh.index("CE_adapt_R8")])
       for r in rr[1:]}

with open(TARGET) as f:
    tr = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
hdr, rows = tr[0], tr[1:]
ix = {c: hdr.index(c) for c in hdr}

migrated, missing = 0, []
for r in rows:
    key = (r[ix["task"]], r[ix["metric"]])
    if key not in ref:
        missing.append((key, r[ix["cell"]]))
        continue
    base_s, imp_s, ce_s = ref[key]
    base, imp = float(base_s), float(imp_s)
    v = float(r[ix["cell_acc"]])
    ce = float(ce_s) if ce_s else None
    r[ix["base_free"]], r[ix["impose_R8"]] = f"{base:.4f}", f"{imp:.4f}"
    r[ix["CE_adapt_R8"]] = f"{ce:.4f}" if ce is not None else ""
    r[ix["cell_minus_base"]] = f"{v - base:+.4f}"
    r[ix["cell_minus_CE"]] = f"{v - ce:+.4f}" if ce is not None else ""
    denom = base - imp
    r[ix["cell_gap_closed"]] = f"{(v - imp) / denom:.4f}" if abs(denom) > 1e-6 else ""
    r[ix["CE_gap_closed"]] = f"{(ce - imp) / denom:.4f}" if (ce is not None
                                                            and abs(denom) > 1e-6) else ""
    migrated += 1

if missing:
    sys.exit(f"ABORT, nothing written: {len(missing)} rows have no (task, metric) in the "
             f"reference, e.g. {missing[:4]}")

with open(TARGET, "w", newline="") as f:
    f.write(f'"{NOTE}"\n')
    w = csv.writer(f)
    w.writerow(hdr)
    w.writerows(rows)
print(f"[migrate] {migrated} rows rewritten onto the correct-convention reference; "
      f"measured columns untouched")
