#!/usr/bin/env python3
"""A/B the async fetch pool for temporal R=18: QD scaling, sibling prefetch, poll mode.

Decode-only runs (-p 0 -n 64): with --mmap 0 the whole model is read during load
(untimed), the pool starts all-resident and trims to R on the first ops, so the decode
window measures steady-state temporal behaviour without prefill contamination.
Each arm: cooldown gate on scaling_max_freq, then -r 3 (internal reps).
"""
import subprocess, sys, time, re

DEV = "/data/local/tmp/tmoe"
import os
MODEL = os.environ.get("TMOE_MODEL", "qwen3moe-rand-fine-Q4pure.gguf")
RATED = {0: 1950000, 4: 2600000, 7: 2147000}   # Pixel 10a: little/mid/prime-CAPPED
# cpu7 hardware rating is 3105000, but Tensor G4 pins scaling_max_freq at 2147000
# in sustained/charging state (never observed higher this session). Gating on the
# unreachable rating would flag every arm; the cap is stable, so arms gated at the
# cap are mutually comparable. clock_min is still recorded per arm.

def sh(cmd, timeout=3600):
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.replace("\r", "")

def clocks():
    out = sh("for c in 0 4 7; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq; done")
    v = [int(x) for x in out.split() if x.isdigit()]
    return dict(zip([0, 4, 7], v)) if len(v) == 3 else {}

def wait_cool(timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        c = clocks()
        if c and all(c[k] >= RATED[k] * 0.98 for k in c):
            return True
        time.sleep(20)
    return False

def run_arm(label, env, extra="", threads=6):
    if not wait_cool():
        print(f"{label}: COOL TIMEOUT, running anyway (flagged)", flush=True)
    cmd = (f"cd {DEV} && echo 1000 > /proc/self/oom_score_adj; {env} ./llama-bench-temporal -m {MODEL} -t {threads} "
           f"-p 0 -n 64 -r 3 -mmp 0 -ot \"_exps=CPU\" {extra} -o csv 2>&1")
    t0 = time.time()
    out = sh(cmd, timeout=1800)
    dur = time.time() - t0
    dec = sd = None
    for line in out.splitlines():
        f = line.split(",")
        if len(f) > 40 and f[0] == '"0badc06a"':
            dec, sd = float(f[-2].strip('"')), float(f[-1].strip('"'))
    pool = ""
    m = re.search(r"temporal-pool: fetches=.*", out)
    if m: pool = m.group(0)
    cmin = min(clocks().values())
    print(f"{label:34s} decode={dec} sd={sd} wall={dur:.0f}s clock_after_min={cmin}", flush=True)
    print(f"    {pool}", flush=True)
    return dec, pool

BASE = "LLAMA_NO_REPACK=1 LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_R=18 LLAMA_TEMPORAL_SWAP_PROB=0"

if __name__ == "__main__":
    arms = sys.argv[1:] or ["qd1", "qd8", "qd8sib", "qd8sibpoll0"]
    if "ceiling" in arms:   # same-power-state ceiling: the denominator for every ratio
        run_arm("CEIL R=192", "LLAMA_NO_REPACK=1 LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_R=192 LLAMA_TEMPORAL_FETCH_THREADS=8")
    if "p01" in arms:
        run_arm("E qd8 sibling p=0.1",
                BASE.replace("SWAP_PROB=0", "SWAP_PROB=0.1") + " LLAMA_TEMPORAL_FETCH_THREADS=8")
    if "qd1" in arms:
        run_arm("A qd1 nosib (old behaviour)", BASE + " LLAMA_TEMPORAL_FETCH_THREADS=1 LLAMA_TEMPORAL_SIBLING_PREFETCH=0")
    if "qd8" in arms:
        run_arm("B qd8 nosib", BASE + " LLAMA_TEMPORAL_FETCH_THREADS=8 LLAMA_TEMPORAL_SIBLING_PREFETCH=0")
    if "qd8sib" in arms:
        run_arm("C qd8 sibling", BASE + " LLAMA_TEMPORAL_FETCH_THREADS=8")
    if "qd8sibpoll0" in arms:
        run_arm("D qd8 sibling poll0", BASE + " LLAMA_TEMPORAL_FETCH_THREADS=8", extra="--poll 0")
