#!/usr/bin/env python3
"""PART D of the stability probe: grad-norm census from surviving train.logs.

Extract Megatron's per-iteration grad norm from each cell's train.log, subsample every 10 iters.
Output: results/ablations/stability_gradnorms.csv  columns: run,iteration,grad_norm
Also report per run: max, median, and count of iterations with grad_norm > 5x the running median
(trailing window of 100 logged iterations).
"""
import os, re, csv, statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT   # canonical resolver: $TMOE_ROOT, then git, then file location
RUNS = os.path.join(ROOT, "results/phase0/runs")
OUT = os.path.join(ROOT, "results/ablations/stability_gradnorms.csv")

CELLS = [
    ("dense_local", "flame38m_dense_local"), ("g1_moe", "flame38m_g1_moe"),
    ("g1_temporal", "flame38m_g1_temporal"), ("g3_moe", "flame38m_g3_moe"),
    ("g3_temporal", "flame38m_g3_temporal"), ("dense_1e19", "dense_1e19"),
    ("moe_coarse_1e19", "moe_coarse_1e19"), ("temporal_coarse_1e19", "g1_tmoe_coarse_1e19"),
    ("temporal_fine_1e19", "temporal_fine_g3_1e19"),
]
LINE = re.compile(r"iteration\s+(\d+)/.*?grad norm:\s*([0-9.]+(?:[eE][+-]?\d+)?)")


def parse(path):
    out = []
    with open(path, errors="ignore") as f:
        for ln in f:
            m = LINE.search(ln)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
    return out


def spikes(gn, k=5.0, win=100):
    n = 0
    for i in range(len(gn)):
        lo = max(0, i - win)
        med = st.median(gn[lo:i + 1])
        if med > 0 and gn[i] > k * med:
            n += 1
    return n


def main():
    rows, summary = [], []
    for label, run in CELLS:
        p = os.path.join(RUNS, run, "train.log")
        if not os.path.exists(p):
            print(f"[skip] {label}: no train.log")
            continue
        series = parse(p)
        if not series:
            print(f"[skip] {label}: no grad-norm lines")
            continue
        gn = [g for _, g in series]
        for it, g in series:
            if it % 10 == 0:                     # subsample every 10 iters
                rows.append([label, it, f"{g:.6g}"])
        summary.append((label, len(series), max(gn), st.median(gn), spikes(gn)))
        print(f"[ok] {label}: {len(series)} iters logged, max {max(gn):.3f}, "
              f"median {st.median(gn):.3f}, spikes(>5x med) {spikes(gn)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "iteration", "grad_norm"])
        w.writerows(rows)
    print(f"[write] {OUT}: {len(rows)} rows")
    print("\nSUMMARY run | logged_iters | max | median | spikes>5x-running-median(win100)")
    for lab, n, mx, md, sp in summary:
        print(f"  {lab:22} {n:6d} {mx:8.3f} {md:8.3f} {sp:5d}")


if __name__ == "__main__":
    main()
