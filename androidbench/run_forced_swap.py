#!/usr/bin/env python3
"""Forced expert turnover sweep -- the Android analogue of TEMPORAL_SWAP_PROB.

The random-weight router picks a near-fixed expert set, so natural page-fault traffic is
unrepresentative (the CUDA side hit this too and solved it by driving swaps at a
prescribed rate rather than trusting the router). Here a background thread evicts expert
slices with madvise(MADV_DONTNEED) at a set rate, so the next use of that expert is a real
fault from UFS.

Rate units: one expert = 3 matrices (gate/up/down), so "1 expert/layer/token" at 30 tok/s
over 45 layers = 45*30*3 = 4050 slices/s.
"""
import csv, json, time
from dataclasses import asdict
from bench import measure, to_row, battery, sh

MODEL = "qwen3moe-rand-fine-Q4_K_M.gguf"
RATES = [0, 1350, 4050, 8100, 16200]      # slices/s; 4050 = 1 expert/layer/token @30tok/s

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

for hz in RATES:
    while True:
        p, l = battery()
        if p and l >= 80: break
        log(f"  battery {l}% - waiting"); time.sleep(300)
    import bench
    orig = bench.measure.__globals__
    r = measure(MODEL, f"forced_swap_{hz}hz", "cpu-temporal2", "cold", 6, 64, 512, 0,
                1024, 128, 3, None, cool_first=True, evict_hz=hz)
    mib = (r.read_bytes if r.read_bytes > 0 else r.disk_sectors_read*512)/1048576
    with open("results/serving_benchmarks_android.csv","a",newline="") as f:
        csv.writer(f).writerow(to_row(r, MODEL))
    with open("results/runs.jsonl","a") as f:
        d=asdict(r); d["evict_hz"]=hz; f.write(json.dumps(d)+"\n")
    log(f"  -> {hz:>6} slices/s: decode={r.decode_tps} +-{r.decode_sd} "
        f"prefill={r.prefill_tps} read={mib:.0f}MiB")
log("FORCED SWAP SWEEP COMPLETE")
