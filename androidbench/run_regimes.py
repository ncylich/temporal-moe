#!/usr/bin/env python3
"""The three regimes, measured with the REAL protocol.

Today's first attempt used -p 0, so with a cold cache the entire ~6.5 GB load cost landed
inside the timed decode window and every regime read ~0.1 tok/s regardless of residency.
Here -p 512 runs first (warming the weights), so decode measures decode.

Hot (attn/norm/embed) weights are held resident in ALL regimes by the controller, so the
only variable is expert residency.
"""
import csv, json, time
from dataclasses import asdict
from bench import measure, to_row, battery, residency, DEV_DIR

MODEL = "qwen3moe-rand-fine-Q4_K_M.gguf"
REGIMES = [("streamed", 0), ("temporal_R18", 18), ("resident_R192", 192)]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

for name, R in REGIMES:
    while True:
        p, l = battery()
        if p and l >= 80: break
        log(f"battery {l}% - waiting"); time.sleep(300)
    r = measure(MODEL, f"regime_{name}", "cpu-temporal2", "cold", 6, 64, 512, 0,
                1024, 128, 3, None, prompt=512, cool_first=True, temporal_r=R)
    pct = residency(f"{DEV_DIR}/{MODEL}")[1]
    mib = (r.read_bytes if r.read_bytes > 0 else r.disk_sectors_read*512)/1048576
    with open("results/serving_benchmarks_android.csv","a",newline="") as f:
        csv.writer(f).writerow(to_row(r, MODEL))
    with open("results/runs.jsonl","a") as f:
        d=asdict(r); d.update({"regime":name,"temporal_R":R,"residency_after":pct}); f.write(json.dumps(d)+"\n")
    log(f"  -> {name} (R={R}): decode={r.decode_tps} +-{r.decode_sd} prefill={r.prefill_tps} "
        f"residency={r.resident_pct_after:.1f}% read={mib:.0f}MiB")
log("REGIMES COMPLETE")
