#!/usr/bin/env python3
"""Three-regime measurement on the explicit slot-pool (session 2 design, LEDGER S2-5).

All arms: same binary (llama-bench-temporal), --mmap 0, experts in plain CPU buffers
(-ot "_exps=CPU"), O_DIRECT fetches. Only the env differs:

  ceiling    R=192            all experts resident, hook idle
  streamed   R=0              nothing persists between ops; every use is a device read
  temporal   R=18 p in {0,.1,.3}  window = top_k; p = prescribed per-use eviction prob
                                  (the CUDA TEMPORAL_SWAP_PROB analogue; drives real
                                  turnover despite the degenerate router)

Arms are INTERLEAVED across 3 invocations (n=1 proves nothing), cooldown-gated on
scaling_max_freq (not Thermal Status), battery >= 80% on AC. Each run's temporal-pool
stderr stats (fetched bytes, avg fetch us) are captured and recorded.
"""
import csv, json, subprocess, sys, time
from dataclasses import asdict

import bench

MODEL = "qwen3moe-rand-fine-Q4_K_M.gguf"
DEV = bench.DEV_DIR
OUT_CSV = "results/serving_benchmarks_android.csv"
OUT_JSONL = "results/pool_regimes.jsonl"

ARMS = [
    ("pool-ceiling-R192",      192, ""),
    ("pool-streamed-R0",         0, ""),
    ("pool-temporal-R18-p0",    18, ""),
    ("pool-temporal-R18-p0.1",  18, "LLAMA_TEMPORAL_SWAP_PROB=0.1"),
    ("pool-temporal-R18-p0.3",  18, "LLAMA_TEMPORAL_SWAP_PROB=0.3"),
]

def pool_stats() -> str:
    out = bench.sh(f"grep temporal-pool {DEV}/pool_stderr.txt 2>/dev/null | tail -1")
    return out.strip()

def main():
    reps_outer = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    bench.preflight(MODEL)
    rows = []
    for it in range(reps_outer):
        for label, r, env_extra in ARMS:
            print(f"\n### iteration {it+1}/{reps_outer}: {label}", flush=True)
            bench.sh(f"rm -f {DEV}/pool_stderr.txt")
            run = bench.measure(
                MODEL, f"{label}-it{it+1}", "cpu", "cold",
                threads=6, ubatch=512, batch=2048, ngl=0, ctx=1024,
                ngen=64, reps=3, pin=None, prompt=128,
                extra='-mmp 0 -ot "_exps=CPU" 2>pool_stderr.txt',
                temporal_r=r,
                env_extra=("LLAMA_TEMPORAL_ODIRECT=1 " + env_extra).strip(),
            )
            stats = pool_stats()
            rec = asdict(run); rec["pool_stats"] = stats; rec["iteration"] = it + 1
            print(f"  prefill={run.prefill_tps} decode={run.decode_tps} status={run.status}")
            print(f"  {stats}")
            print(f"  clock_min={run.clock_min} read_bytes={run.read_bytes/1e6 if run.read_bytes>0 else -1:.0f}MB")
            with open(OUT_JSONL, "a") as f:
                f.write(json.dumps(rec) + "\n")
            with open(OUT_CSV, "a", newline="") as f:
                row = bench.to_row(run, MODEL)
                row[9] += f";pool={stats}"
                csv.writer(f).writerow(row)
            rows.append((run.label, run.decode_tps, run.status))
    print("\n=== summary (decode tok/s) ===")
    for label, d, s in rows:
        print(f"  {label:32s} {d if d else 0:8.3f}  [{s}]")

if __name__ == "__main__":
    main()
