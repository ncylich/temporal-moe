#!/usr/bin/env python3
"""Time-boxed remainder: the honesty check first, then a compact A/B, then granularity.

Reprioritised at 04:10 after measuring ~30 min/run: the original Phase E (3 balloon levels
x 2 reps x 3 arms = 18 runs) would have taken ~9 hours and finished well after morning.
What gets cut and why:

  - Phase G runs FIRST. It decides whether the whole temporal result means anything
    (fixed expert set vs real turnover), so it must not be the thing that runs out of time.
  - Mode 1 (whole-file MADV_RANDOM) is dropped from the A/B. It was already measured at
    0.09 tok/s -- 13x worse than doing nothing -- and re-measuring a known-dead arm is not
    worth 30 minutes a run.
  - Balloon levels cut to {0, 4000}: enough to contrast "fits" against "does not fit".
  - reps 5 -> 3. The ceiling's noise floor is 2.2%, and the effects being tested here are
    multiples, not percentages.

Nothing about the gates is relaxed: power, eviction verification, memory settle, cooldown
and clock recording all still apply.
"""
from __future__ import annotations
import csv, json, subprocess, time
from dataclasses import asdict

from bench import DEV_DIR, measure, to_row, battery, meminfo, residency, sh

MODEL = "qwen3moe-rand-fine-Q4_K_M.gguf"
MODEL_COARSE = "qwen3moe-rand-coarse-Q4_K_M.gguf"
OUT = "results/serving_benchmarks_android.csv"
JSONL = "results/runs.jsonl"
CTX, REPS = 1024, 3
SWAP_TOTAL_KB = 12582908


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wait_for_power(min_batt=80):
    while True:
        powered, lvl = battery()
        if powered and lvl >= min_batt:
            return lvl
        log(f"  battery {lvl}% - waiting for >={min_batt}%")
        time.sleep(300)


def kill_balloons():
    sh("pkill -f balloon 2>/dev/null; true")


def wait_memory_settled(min_avail_kb=7_000_000, timeout_s=420):
    kill_balloons()
    t0, prev = time.time(), None
    while time.time() - t0 < timeout_s:
        mi = meminfo()
        avail = mi.get("MemAvailable", 0)
        used = SWAP_TOTAL_KB - mi.get("SwapFree", SWAP_TOTAL_KB)
        if avail >= min_avail_kb and prev is not None and abs(used - prev) < 20_000:
            log(f"    settled: avail={avail} kB swap_used={used} kB")
            return True
        prev = used
        time.sleep(15)
    log(f"    WARNING: memory not settled in {timeout_s}s")
    return False


def record(r, extra=None, model=None):
    with open(OUT, "a", newline="") as f:
        csv.writer(f).writerow(to_row(r, model or MODEL))
    with open(JSONL, "a") as f:
        d = asdict(r)
        if extra:
            d.update(extra)
        f.write(json.dumps(d) + "\n")
    mib = (r.read_bytes if r.read_bytes > 0 else r.disk_sectors_read * 512) / 1048576
    log(f"  -> {r.label}: decode={r.decode_tps} +-{r.decode_sd} prefill={r.prefill_tps} "
        f"status={r.status} res={r.resident_pct_before:.1f}->{r.resident_pct_after:.1f}% "
        f"read={mib:.0f}MiB")
    return mib


def balloon_bg(mib, hold_s):
    return subprocess.Popen(["adb", "shell", f"{DEV_DIR}/balloon {mib} {hold_s}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main():
    # ---- PHASE G FIRST: does the working set grow with decode length? ----------
    # Fixed expert set -> storage reads plateau. Real turnover -> reads grow with tokens.
    # Uses mode 2 (no MAP_POPULATE), so reads reflect what decode actually touches rather
    # than the loader's eager population of the whole file.
    log("=== PHASE G: routing diversity (runs first: it decides what everything means) ===")
    for n in (16, 64, 256):
        wait_for_power(); wait_memory_settled()
        r = measure(MODEL, f"diversity_n{n}", "cpu-temporal2", "cold", 6, 64, 512, 0,
                    CTX, n, 1, None, prompt=0, cool_first=True)
        mib = record(r, {"decode_tokens": n})
        log(f"  n={n}: {mib:.0f} MiB read, resident_after={r.resident_pct_after:.1f}%")

    # ---- COMPACT A/B: default mmap vs experts-only MADV_RANDOM ------------------
    log("=== PHASE E (compact): default mmap vs temporal mode 2 ===")
    for mib_balloon in (0, 4000):
        for i, arm in enumerate(("cpu", "cpu-temporal2") if mib_balloon == 0
                                else ("cpu-temporal2", "cpu")):     # rotate order
            wait_for_power(); wait_memory_settled()
            b = balloon_bg(mib_balloon, 900) if mib_balloon else None
            if b:
                time.sleep(20)
            pct = residency(f"{DEV_DIR}/{MODEL}")[1]
            r = measure(MODEL, f"ab2_{arm}_balloon{mib_balloon}", arm, "cold", 6, 64, 512,
                        0, CTX, 128, REPS, None, cool_first=True)
            record(r, {"balloon_mib": mib_balloon, "arm": arm, "residency_before": pct})
            if b:
                try:
                    b.terminate()
                except Exception:
                    pass
            kill_balloons()

    # ---- GRANULARITY: fine (216 KiB faults) vs coarse (648 KiB faults) ----------
    log("=== PHASE F: expert granularity ===")
    for model in (MODEL, MODEL_COARSE):
        wait_for_power(); wait_memory_settled()
        tag = f"gran_{'fine' if model == MODEL else 'coarse'}_temporal2"
        r = measure(model, tag, "cpu-temporal2", "cold", 6, 64, 512, 0, CTX, 128, REPS,
                    None, cool_first=True)
        record(r, {"variant": tag}, model=model)

    # ---- CONFIRM the anonymous-memory finding on the coarse model too ----------
    wait_for_power(); wait_memory_settled()
    r = measure(MODEL_COARSE, "gran_coarse_mmap0", "cpu", "cold", 6, 64, 512, 0, CTX, 128,
                REPS, None, extra="--mmap 0", cool_first=True)
    record(r, {"variant": "gran_coarse_mmap0"}, model=MODEL_COARSE)

    log("FINAL PHASES COMPLETE")


if __name__ == "__main__":
    main()
