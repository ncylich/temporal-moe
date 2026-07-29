#!/usr/bin/env python3
"""Driver for the locus probes (A1-A3). Runs delex_locus.analyze on every capture the registry
reports, at every MoE layer the capture contains, all three context windows, and both fit/score
splits.

Writes four files, all keyed by (label, run, budget, layer, variant, split) so pooling is always a
reporting decision:

  mechinterp_locus_1e19.csv     per-expert AUCs
  mechinterp_floors_1e19.csv    measured chance floors, per layer, per feature, per null type
  mechinterp_locus_coverage.csv experts probed vs omitted per layer, so omissions are on the record

Medians, bootstrap intervals and depth slopes are NOT computed here: pooling is a reporting decision,
made in analysis/plots/plot_locus_by_layer.py, which needs no torch.

**The null-control gate is on the iid permutation only, and that is a change.** The acceptance
criteria ask for 0.500 +- 0.002 under iid permutation *and* circular shift. Measured at full depth
over every expert, the iid null holds everywhere (0.4996-0.5002) while the circular shift is inflated
by up to +0.017, monotonically in the context window width. delex_null_check.py establishes that this
is a defect of the shift, not of the models: the shift is applied to the flattened [S*B] stream whose
adjacent entries are adjacent *batch elements*, so it never shifted along the token axis, and what it
leaves intact is a document-level association between a label series and the feature. Gating on it
would withhold every number in the program on the strength of a mis-specified control. It is still
computed, written out, and reported here as a diagnostic.

The 1e16/1e17 cells in mechinterp_locus.csv are NOT regenerated here: their captures were never
preserved (MANIFEST.csv has 3 delex_capture.pt, all at 1e19), so those rows stay as the historical
record at layers 2-6 until Phase 2 recaptures them from checkpoints.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delex_locus
import registry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ABLATIONS

# Published labels for the three preserved captures, kept so the new rows join the existing figures.
LABELS = {"moe_coarse_1e19": "moe_coarse_1e19",
          "g1_tmoe_coarse_1e19": "temporal_coarse_1e19",
          "temporal_fine_g3_1e19": "temporal_fine_1e19"}
GATE = 0.002


def main():
    verify = "--verify" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    cells = [r for r in registry.runs(capture=True) if not only or r.name in only]
    if not cells:
        sys.exit("no runs with a preserved delex_capture.pt (see registry.py)")

    splits = ["sequence", "position"] if "--both-splits" in sys.argv else ["sequence"]
    rows, floors, coverage, gate_ok, headline = [], [], [], True, []
    for r in cells:
        cap = r.path("delex_capture.pt")
        if not os.path.exists(cap):
            print(f"[skip] {r.name}: capture preserved but not on disk "
                  f"(scripts/artifacts.py pull --run {r.name})")
            continue
        label = LABELS.get(r.name, r.name)
        for split in splits:
            print(f"[run] {label} ({r.name}, {r.regime}, {r.grain_label}, {r.budget}) "
                  f"split={split}", flush=True)
            rr, ff, cc, summary = delex_locus.analyze(cap, r.name, label=label, verify=verify,
                                                      split=split)
            rows += rr
            floors += ff
            coverage += cc
            for variant, s in summary.items():
                iid = {k: v for k, v in s["nulls"].items() if k.endswith("/iid")}
                shift = {k: v for k, v in s["nulls"].items() if k.endswith("/circular")}
                worst_iid = max(abs(v - 0.5) for v in iid.values() if np.isfinite(v))
                worst_sh = max(abs(v - 0.5) for v in shift.values() if np.isfinite(v))
                ok = worst_iid <= GATE
                gate_ok = gate_ok and ok
                headline.append((label, split, variant, s, ok, worst_iid, worst_sh))
                print(f"  {variant:6} layers {s['layers'][0]}-{s['layers'][-1]} n={s['n']:5d} "
                      f"med_tokAUC={s['tok']:.3f} med_ctxAUC={s['ctx']:.3f} "
                      f"ctx_dom={s['ctx_dom']*100:3.0f}%  iid_null_dev={worst_iid:.4f} "
                      f"(shift {worst_sh:+.4f}, diagnostic) gate={'PASS' if ok else 'FAIL'}",
                      flush=True)

    _write("mechinterp_locus_1e19.csv", delex_locus.ROW_HEADER, rows)
    _write("mechinterp_floors_1e19.csv", delex_locus.FLOOR_HEADER, floors)
    _write("mechinterp_locus_coverage.csv", delex_locus.COVERAGE_HEADER, coverage)

    omitted = sum(int(c[8]) for c in coverage)
    print(f"\ncoverage: {len(coverage)} (layer, variant, split) cells, {omitted} expert-probes "
          f"omitted for too few firings — all recorded in mechinterp_locus_coverage.csv, none "
          f"dropped silently")
    print("NULL-CONTROL GATE (iid permutation):",
          f"PASS (every model within 0.500+-{GATE})" if gate_ok
          else f"FAIL — an iid null median is off 0.500 by more than {GATE}; those numbers are "
               "suspect, STOP and report")
    print(f"\nHEADLINE  {'model':22} {'split':9} {'variant':8}  med_tok  med_ctx  ctx_dom   "
          f"iid_null  shift_null")
    for lab, split, variant, s, ok, wi, ws in headline:
        print(f"          {lab:22} {split:9} {variant:8}  {s['tok']:.3f}    {s['ctx']:.3f}    "
              f"{s['ctx_dom']*100:5.0f}%   {wi:+.4f}   {ws:+.4f}"
              f"{'' if ok else '  <-- GATE FAIL'}")
    return gate_ok


def _write(name, header, rows):
    os.makedirs(ABLATIONS, exist_ok=True)
    p = os.path.join(ABLATIONS, name)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[write] {p}: {len(rows)} rows")


if __name__ == "__main__":
    sys.exit(0 if main() else 2)
