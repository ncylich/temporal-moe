#!/usr/bin/env python3
"""Replicate the two claims that matter, because tonight repeatedly showed n=1 lying.

The fine model gave 28.49 and 2.32 tok/s for the SAME mode-2 configuration 40 min apart.
So before any claim goes in the report, the coarse mode-2 result and the coarse ceiling
each get an independent repeat, and the fine mode-2 config gets one more sample to
characterise its bimodality rather than pretending it has a single value.
"""
import csv, json, time
from dataclasses import asdict
from bench import measure, to_row, battery, meminfo, sh, DEV_DIR

FINE = "qwen3moe-rand-fine-Q4_K_M.gguf"
COARSE = "qwen3moe-rand-coarse-Q4_K_M.gguf"
OUT, JSONL = "results/serving_benchmarks_android.csv", "results/runs.jsonl"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def rec(r, model, extra=None):
    with open(OUT, "a", newline="") as f: csv.writer(f).writerow(to_row(r, model))
    with open(JSONL, "a") as f:
        d = asdict(r); d.update(extra or {}); f.write(json.dumps(d) + "\n")
    mib = (r.read_bytes if r.read_bytes > 0 else r.disk_sectors_read*512)/1048576
    log(f"  -> {r.label}: decode={r.decode_tps} +-{r.decode_sd} prefill={r.prefill_tps} read={mib:.0f}MiB")

PLAN = [
    (COARSE, "rep_coarse_temporal2_b", "cpu-temporal2", ""),
    (COARSE, "rep_coarse_mmap0_b",     "cpu",           "--mmap 0"),
    (COARSE, "rep_coarse_temporal2_c", "cpu-temporal2", ""),
    (FINE,   "rep_fine_temporal2_c",   "cpu-temporal2", ""),
    (FINE,   "rep_fine_mmap0_b",       "cpu",           "--mmap 0"),
]
for model, tag, arm, extra in PLAN:
    while True:
        p, l = battery()
        if p and l >= 80: break
        log(f"  battery {l}% - waiting"); time.sleep(300)
    sh("pkill -f balloon 2>/dev/null; true")
    r = measure(model, tag, arm, "cold", 6, 64, 512, 0, 1024, 128, 3, None,
                cool_first=True)
    rec(r, model, {"replication": True})
log("REPLICATION COMPLETE")
