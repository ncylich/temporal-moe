#!/usr/bin/env python3
"""Phases E/F/G, re-run with two contamination fixes found during the night.

FIX 1 -- memory settle. Phase C's 5000 MiB balloon left ~1.49 GB in zram swap, and the
run that followed it stalled on swap-in: 12.56 tok/s with a +-15.5 sd, and 20 minutes of
wall clock instead of 3. Thermal cooldown does not cover this. Every run here waits for
MemAvailable to recover AND for swap usage to stop shrinking before it measures.

FIX 2 -- arm order rotation. Phase E ran its arms in a fixed order (cpu, temporal,
temporal2) inside each repetition. Any effect that decays with time -- swap-in settling,
page cache re-warming -- would then systematically favour the arm that always runs last,
which happens to be the one whose hypothesis I am trying to confirm. The order is rotated
per repetition so that bias cancels instead of accumulating.

Phases A-C are already recorded and are not repeated.
"""
from __future__ import annotations
import csv, json, subprocess, time
from dataclasses import asdict

from bench import DEV_DIR, measure, to_row, battery, wait_until_cool, meminfo, residency, sh

MODEL = "qwen3moe-rand-fine-Q4_K_M.gguf"
MODEL_COARSE = "qwen3moe-rand-coarse-Q4_K_M.gguf"
OUT = "results/serving_benchmarks_android.csv"
JSONL = "results/runs.jsonl"
CTX, NGEN, REPS = 1024, 128, 5
MIN_BATT = 80
SWAP_TOTAL_KB = 12582908


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wait_for_power(min_batt=MIN_BATT):
    while True:
        powered, lvl = battery()
        if powered and lvl >= min_batt:
            return lvl
        log(f"  battery {lvl}% powered={powered} - waiting for >={min_batt}%")
        time.sleep(300)


def kill_balloons():
    sh("pkill -f balloon 2>/dev/null; true")


def wait_memory_settled(min_avail_kb=7_000_000, timeout_s=600):
    """Wait until RAM has actually recovered from ballooning.

    Two conditions: MemAvailable back above a floor, and swap usage no longer falling
    (i.e. the kernel has finished paging things back in). Without this, the next run
    measures swap-in latency and reports it as decode throughput.
    """
    kill_balloons()
    t0, prev_used = time.time(), None
    while time.time() - t0 < timeout_s:
        mi = meminfo()
        avail = mi.get("MemAvailable", 0)
        swap_used = SWAP_TOTAL_KB - mi.get("SwapFree", SWAP_TOTAL_KB)
        settled = prev_used is not None and abs(swap_used - prev_used) < 20_000
        if avail >= min_avail_kb and settled:
            log(f"    memory settled: avail={avail} kB swap_used={swap_used} kB")
            return True
        prev_used = swap_used
        time.sleep(15)
    log(f"    WARNING: memory did not settle in {timeout_s}s (avail={avail} kB)")
    return False


def record(r, extra=None, model=None):
    with open(OUT, "a", newline="") as f:
        csv.writer(f).writerow(to_row(r, model or MODEL))
    with open(JSONL, "a") as f:
        d = asdict(r)
        if extra:
            d.update(extra)
        f.write(json.dumps(d) + "\n")
    log(f"  -> {r.label}: decode={r.decode_tps} +-{r.decode_sd} prefill={r.prefill_tps} "
        f"status={r.status} resident={r.resident_pct_before:.1f}% "
        f"read={(r.read_bytes if r.read_bytes>0 else r.disk_sectors_read*512)/1048576:.0f}MiB")


def balloon_bg(mib, hold_s):
    return subprocess.Popen(["adb", "shell", f"{DEV_DIR}/balloon {mib} {hold_s}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


ARMS = ["cpu", "cpu-temporal", "cpu-temporal2"]


def main():
    log("=== PHASE E: temporal mmap A/B, interleaved + order-rotated ===")
    for mib in (0, 3000, 4000):
        for i in range(2):
            arms = ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]   # FIX 2: rotate
            log(f"  balloon={mib} MiB rep={i} order={arms}")
            for arm in arms:
                wait_for_power()
                wait_memory_settled()                            # FIX 1
                b = balloon_bg(mib, 600) if mib else None
                if b:
                    time.sleep(20)
                pct = residency(f"{DEV_DIR}/{MODEL}")[1]
                r = measure(MODEL, f"ab_{arm}_balloon{mib}_r{i}", arm, "cold", 6, 64, 512,
                            0, CTX, NGEN, REPS, None, cool_first=True)
                record(r, {"balloon_mib": mib, "residency_before_run": pct, "arm": arm,
                           "rep": i, "arm_order": arms})
                if b:
                    try:
                        b.terminate()
                    except Exception:
                        pass
                kill_balloons()

    log("=== PHASE F: expert granularity, fine vs coarse ===")
    for model in (MODEL, MODEL_COARSE):
        for arm in ("cpu", "cpu-temporal2"):
            wait_for_power()
            wait_memory_settled()
            tag = f"gran_{'fine' if model == MODEL else 'coarse'}_{arm}"
            r = measure(model, tag, arm, "cold", 6, 64, 512, 0, CTX, NGEN, REPS, None,
                        cool_first=True)
            record(r, {"variant": tag}, model=model)

    log("=== PHASE G: routing diversity via storage reads vs decode length ===")
    for n in (16, 64, 256):
        wait_for_power()
        wait_memory_settled()
        r = measure(MODEL, f"diversity_n{n}", "cpu-temporal2", "cold", 6, 64, 512, 0,
                    CTX, n, 1, None, prompt=0, cool_first=True)
        mib = (r.read_bytes if r.read_bytes > 0 else r.disk_sectors_read * 512) / 1048576
        record(r, {"decode_tokens": n, "read_mib": round(mib, 1)})
        log(f"  n={n}: read {mib:.0f} MiB from storage, resident_after={r.resident_pct_after:.1f}%")

    log("PHASES EFG COMPLETE")


if __name__ == "__main__":
    main()
