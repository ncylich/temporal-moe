#!/usr/bin/env python3
"""PART E converter: parse fake-quant eval logs -> results/ablations/stability_fakequant.csv.

Reads results/phase0/runs/<run>/quanteval_b{16,8,4,3}.log, pulls the "on test set" lm loss, and
writes run,bits,test_ce,test_bpb.  BPB divisor = 2.9780 (50k pythia vocab; NOT the 16k 2.7568).
bits=16 is the un-quantized baseline (routed-expert weights untouched).
"""
import os, re, csv
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT   # canonical resolver: $TMOE_ROOT, then git, then file location
RUNS = os.path.join(ROOT, "results/phase0/runs")
OUT = os.path.join(ROOT, "results/ablations/stability_fakequant.csv")
DIV = 2.9780

CELLS = [("g1_moe", "flame38m_g1_moe"), ("g1_temporal", "flame38m_g1_temporal"),
         ("g3_moe", "flame38m_g3_moe"), ("g3_temporal", "flame38m_g3_temporal"),
         ("moe_coarse_1e19", "moe_coarse_1e19"), ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19"),
         ("temporal_fine_1e19", "temporal_fine_g3_1e19"),
         ("moe_fine_1e19", "moe_fine_g3_1e19")]   # fine full MoE trained 2026-09-03
TEST = re.compile(r"on test set \| lm loss value:\s*([0-9.]+(?:[eE][+-]?\d+)?)")


def main():
    # Upsert only the cells named on the command line and keep every other row from the existing
    # CSV. The hub copies of the three July 1e19 quanteval_b16.log files are NOT the 16-bit
    # baselines: the July constraint-swap evaluations reused that log name and overwrote them
    # before mirroring (they read 4.41 / 3.79, the imposed and unmasked values), so a regeneration
    # from disk would corrupt those rows. The committed rows are the record.
    todo = {run for run in sys.argv[1:]}
    if not todo:
        print("usage: stability_fakequant_to_csv.py <run_name> [...]  (cells:",
              ", ".join(r for _, r in CELLS), ")"); return
    rows = []
    for label, run in CELLS:
        if run not in todo:
            continue
        for bits in (16, 8, 4, 3):
            log = os.path.join(RUNS, run, f"quanteval_b{bits}.log")
            if not os.path.exists(log):
                print(f"[skip] {label} b{bits}: no log")
                continue
            m = None
            for line in open(log, errors="ignore"):
                mm = TEST.search(line)
                if mm:
                    m = mm
            if not m:
                print(f"[skip] {label} b{bits}: no test line")
                continue
            ce = float(m.group(1))
            rows.append([label, bits, f"{ce:.6f}", f"{ce / DIV:.6f}"])
            print(f"[ok] {label} b{bits}: test_ce {ce:.4f} bpb {ce / DIV:.4f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    labels = {label for label, run in CELLS if run in todo}
    kept = []
    if os.path.exists(OUT):
        with open(OUT, newline="") as f:
            kept = [r for r in list(csv.reader(f))[1:] if r and r[0] not in labels]
    rows = kept + rows
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "bits", "test_ce", "test_bpb"])
        w.writerows(rows)
    print(f"[write] {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
