#!/usr/bin/env python3
"""PART E converter: parse fake-quant eval logs -> results/ablations/stability_fakequant.csv.

Reads results/phase0/runs/<run>/quanteval_b{16,8,4,3}.log, pulls the "on test set" lm loss, and
writes run,bits,test_ce,test_bpb.  BPB divisor = 2.9780 (50k pythia vocab; NOT the 16k 2.7568).
bits=16 is the un-quantized baseline (routed-expert weights untouched).
"""
import os, re, csv

ROOT = os.environ.get("TEMPORAL_MOE_ROOT",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
RUNS = os.path.join(ROOT, "results/phase0/runs")
OUT = os.path.join(ROOT, "results/ablations/stability_fakequant.csv")
DIV = 2.9780

CELLS = [("g1_moe", "flame38m_g1_moe"), ("g1_temporal", "flame38m_g1_temporal"),
         ("g3_moe", "flame38m_g3_moe"), ("g3_temporal", "flame38m_g3_temporal"),
         ("moe_coarse_1e19", "moe_coarse_1e19"), ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19"),
         ("temporal_fine_1e19", "temporal_fine_g3_1e19")]
TEST = re.compile(r"on test set \| lm loss value:\s*([0-9.]+(?:[eE][+-]?\d+)?)")


def main():
    rows = []
    for label, run in CELLS:
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
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "bits", "test_ce", "test_bpb"])
        w.writerows(rows)
    print(f"[write] {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
