#!/usr/bin/env python3
"""Overnight sweep: establish the all-resident ceiling, then measure how decode degrades
as page-cache residency falls.

Phases run in priority order and every row is appended as soon as it completes, so a
partial night still delivers the ceiling and the tiers.

  A  ceiling      -- --mmap 0 (weights in anonymous RAM, no page-cache dependency).
                     Repeated to establish the noise floor. Every later number is a
                     ratio to this.
  B  tiers        -- mmap warm / mmap cold / direct-io, i.e. resident -> streamed.
  C  pressure     -- balloon-forced residency sweep: how low can residency go before
                     decode falls under 75% of ceiling?
  D  levers       -- threads, poll, cpu-strict, ubatch. One change, one A/B, one result.

Battery is re-checked before every run; the sweep parks itself rather than producing a
power-limited number (M1).
"""
from __future__ import annotations
import csv, json, subprocess, sys, time
from dataclasses import asdict

import bench
from bench import (DEV_DIR, measure, to_row, battery, wait_until_cool, meminfo,
                   residency, sh)

MODEL = "qwen3moe-rand-fine-Q4_K_M.gguf"
MODEL_COARSE = "qwen3moe-rand-coarse-Q4_K_M.gguf"
OUT = "results/serving_benchmarks_android.csv"
JSONL = "results/runs.jsonl"
CTX, NGEN, REPS = 1024, 128, 5
MIN_BATT = 80

HEADER = ["phase", "model", "tier", "setup", "ubatch", "context", "prefill_ms",
          "decode_tok_s", "peak_vram_mib", "note", "decode_tok_s_std",
          "copied_bytes_per_token"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_power(min_batt=MIN_BATT):
    """Park until the device is charged enough. Never produce a power-limited number."""
    while True:
        powered, lvl = battery()
        if powered and lvl >= min_batt:
            return lvl
        log(f"  battery {lvl}% powered={powered} - waiting for >={min_batt}%")
        time.sleep(300)


def record(r, extra=None, model=None):
    with open(OUT, "a", newline="") as f:
        csv.writer(f).writerow(to_row(r, model or MODEL))
    with open(JSONL, "a") as f:
        d = asdict(r)
        if extra:
            d.update(extra)
        f.write(json.dumps(d) + "\n")
    ratio = ""
    if r.decode_tps and CEILING["decode"]:
        ratio = f"  ({100*r.decode_tps/CEILING['decode']:.1f}% of ceiling)"
    log(f"  -> {r.label}: decode={r.decode_tps} +-{r.decode_sd} "
        f"prefill={r.prefill_tps} status={r.status} resident={r.resident_pct_before:.1f}%{ratio}")


CEILING = {"decode": None, "prefill": None, "sd": None}


def run(label, **kw):
    wait_for_power()
    kw.setdefault("threads", 6)
    kw.setdefault("ubatch", 64)
    kw.setdefault("batch", 512)
    kw.setdefault("ngl", 0)
    kw.setdefault("ctx", CTX)
    kw.setdefault("ngen", NGEN)
    kw.setdefault("reps", REPS)
    kw.setdefault("pin", None)
    kw.setdefault("backend", "cpu")
    kw.setdefault("cache", "warm")
    log(f"RUN {label}: {kw}")
    r = measure(MODEL, label, **kw)
    record(r)
    return r


def balloon_bg(mib, hold_s):
    """Inflate a balloon in the background; returns the Popen so it can be reaped."""
    return subprocess.Popen(["adb", "shell", f"{DEV_DIR}/balloon {mib} {hold_s}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main():
    import os
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(OUT):
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)

    lvl = wait_for_power()
    log(f"starting sweep, battery={lvl}%")

    # ---- PHASE A: the ceiling ------------------------------------------------
    # --mmap 0 puts weights in anonymous RAM: no page cache involvement, nothing to
    # fault. Verified earlier not to touch swap, so "resident" is literally true.
    log("=== PHASE A: all-resident ceiling (--mmap 0) ===")
    ceil_runs = []
    # cache="cold" first: with --mmap 0 the weights are copied into anonymous RAM, so
    # page-cache state is irrelevant to the measurement -- but evicting first frees the
    # ~5 GB of cache that would otherwise compete with the 6.5 GB anonymous allocation
    # and risk pushing us into zram.
    for i in range(3):
        r = run(f"ceiling_mmap0_r{i}", cache="cold", extra="--mmap 0")
        # This phone cannot hold peak clock through a 5-rep run: the clock sags to
        # ~42-47% of rated within ~90 s on EVERY run, ceiling included. degraded_clock is
        # therefore the normal sustained-load state here, not a spoiled run, and rejecting
        # it would discard every measurement we can actually take. Only genuinely invalid
        # runs (zero metric, no output, failed eviction) are excluded.
        if r.decode_tps and not r.status.startswith(("error", "no_output", "evict_failed")):
            ceil_runs.append(r)
    use = ceil_runs
    if use:
        CEILING["decode"] = sum(r.decode_tps for r in use) / len(use)
        CEILING["prefill"] = sum(r.prefill_tps for r in use if r.prefill_tps) / max(
            1, len([r for r in use if r.prefill_tps]))
        spread = max(r.decode_tps for r in use) - min(r.decode_tps for r in use)
        CEILING["sd"] = spread
        log(f"CEILING decode={CEILING['decode']:.2f} tok/s "
            f"(inter-run spread {spread:.2f}), prefill={CEILING['prefill']:.1f}")
    else:
        log("FATAL: no usable ceiling run; aborting")
        return

    # ---- PHASE B: residency tiers -------------------------------------------
    log("=== PHASE B: tiers (resident -> streamed) ===")
    run("mmap_warm", cache="warm")
    run("mmap_cold", cache="cold")
    run("direct_io", cache="cold", extra="--direct-io 1")

    # ---- PHASE C: pressure sweep --------------------------------------------
    # How far can residency fall before decode drops below 75% of ceiling? The balloon
    # holds dirty anonymous pages, so the kernel must evict page cache to satisfy it.
    log("=== PHASE C: memory-pressure residency sweep ===")
    for mib in (2000, 3000, 4000, 5000):
        wait_for_power()
        wait_until_cool()
        b = balloon_bg(mib, 900)
        time.sleep(20)                       # let the kernel actually reclaim
        pct = residency(f"{DEV_DIR}/{MODEL}")[1]
        avail = meminfo().get("MemAvailable", 0)
        log(f"  balloon {mib} MiB -> residency {pct:.1f}%, MemAvailable {avail} kB")
        r = measure(MODEL, f"pressure_{mib}mib", "cpu", "warm", 6, 64, 512, 0,
                    CTX, NGEN, REPS, None, cool_first=False)
        record(r, {"balloon_mib": mib, "residency_at_start": pct})
        try:
            b.terminate()
        except Exception:
            pass
        sh("pkill -f balloon || true")

    # ---- PHASE D: levers -----------------------------------------------------
    log("=== PHASE D: levers ===")
    for t in (2, 4, 8):
        run(f"threads_{t}", threads=t)
    run("pin_prime_c0_t2", threads=2, pin="c0")
    run("cpu_strict", extra="--cpu-mask 0xC0 --cpu-strict 1", threads=2)
    run("poll_0", extra="--poll 0")
    run("ubatch_128", ubatch=128)
    run("ubatch_32", ubatch=32)

    # ---- PHASE E: the temporal lever, interleaved A/B under pressure ---------
    # Hypothesis: llama.cpp's default mmap policy (MAP_POPULATE + MADV_WILLNEED +
    # FADV_SEQUENTIAL over the whole file) is wrong for a MoE model larger than the
    # device's usable RAM -- it pulls in all 192 experts when only 18 are touched per
    # token. LLAMA_TEMPORAL_MMAP=1 switches to demand paging (no populate, MADV_RANDOM).
    # Verified to change behaviour: resident-after-load 36.06% -> 20.15%.
    #
    # Interleaved A/B (M29), because sequential runs would confound the arms with
    # thermal decay. Same binary both arms; only the env var differs.
    log("=== PHASE E: temporal mmap A/B (interleaved) ===")
    for mib in (0, 3000, 4000):
        for i in range(2):
            b = balloon_bg(mib, 1200) if mib else None
            if b:
                time.sleep(20)
            for arm in ("cpu", "cpu-temporal", "cpu-temporal2"):
                wait_for_power()
                tag = f"ab_{arm}_balloon{mib}_r{i}"
                pct = residency(f"{DEV_DIR}/{MODEL}")[1]
                r = measure(MODEL, tag, arm, "cold", 6, 64, 512, 0, CTX, NGEN, REPS,
                            None, cool_first=True)
                record(r, {"balloon_mib": mib, "residency_before": pct, "arm": arm})
            if b:
                try:
                    b.terminate()
                except Exception:
                    pass
                sh("pkill -f balloon || true")

    # ---- PHASE F: expert granularity ----------------------------------------
    # Same total params and active FLOPs, different expert size: fine = 18-of-192 at
    # moe_ff 384 (~840 KiB/expert), coarse = 6-of-64 at moe_ff 1152 (~2.5 MiB/expert).
    # Granularity should matter for demand paging specifically: a coarse expert is one
    # large contiguous fault, a fine expert is a small one, and UFS strongly prefers
    # large sequential reads. This is the paging analogue of the CUDA swap-size question.
    log("=== PHASE F: expert granularity, fine vs coarse ===")
    for model in (MODEL, MODEL_COARSE):
        for arm in ("cpu", "cpu-temporal", "cpu-temporal2"):
            wait_for_power()
            tag = f"gran_{'fine' if model == MODEL else 'coarse'}_{arm}"
            r = measure(model, tag, arm, "cold", 6, 64, 512, 0, CTX, NGEN, REPS, None,
                        cool_first=True)
            record(r, {"variant": tag}, model=model)

    # ---- PHASE G: routing diversity (the honesty check) ----------------------
    # If the random-weight router selects a near-fixed expert set every token, the
    # working set is ~513 MiB TOTAL rather than 513 MiB PER TOKEN, everything stays
    # resident after the first token, and any paging result looks wonderful for a reason
    # that has nothing to do with temporal residency working.
    #
    # Test: decode length vs bytes actually read from storage, cold start each time.
    # Fixed expert set  -> reads plateau (first token pays, the rest are free).
    # Diverse routing   -> reads keep growing with token count.
    log("=== PHASE G: routing diversity via storage reads vs decode length ===")
    for n in (16, 64, 256):
        wait_for_power()
        r = measure(MODEL, f"diversity_n{n}", "cpu-temporal2", "cold", 6, 64, 512, 0,
                    CTX, n, 1, None, prompt=0, cool_first=True)
        record(r, {"decode_tokens": n,
                   "read_mib": round(r.read_bytes / 1048576, 1) if r.read_bytes > 0 else None})
        log(f"  n={n}: read {r.read_bytes/1048576:.0f} MiB from storage, "
            f"resident_after={r.resident_pct_after:.1f}%")

    log("SWEEP COMPLETE")
    log(f"ceiling decode = {CEILING['decode']:.2f} tok/s")


if __name__ == "__main__":
    main()
