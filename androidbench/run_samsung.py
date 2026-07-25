#!/usr/bin/env python3
"""Port the Pixel-tuned temporal-MoE config to the Samsung SM-S942U1 and re-tune it.

This is a DIFFERENT RIG and most of the Pixel's production constants do not
transfer. Recorded here so nobody copies a number across devices by accident:

  * NO ROOT. `scaling_min_freq` cannot be pinned, so every number from this device
    is STOCK GOVERNOR. Pitfall #7 measured that artifact at 1.25x on the Pixel.
    Never compare a Samsung number to a Pixel pinned number.
  * NO UFS driver monitor (root sysfs), so device-side concurrency -- the instrument
    that settled S3-36 -- is unavailable here. FETCHPROF is still in-engine.
  * NO io_uring (EPERM in the `shell` SELinux domain). Rejected anyway (S3-36).
  * CPU topology is 6x perf @3.63 GHz + 2x prime @4.74 GHz and NO little cores.
    The Pixel's `-t 4` exists solely to keep its A520 little cores out of the ggml
    barriers; that reason does not exist here, so thread count must be re-swept.
  * 11.4 GB RAM, not 7.75. Under BASELINE_POLICY the baseline is the largest E that
    fits FULLY RESIDENT, and E=192 (5.94 GB) may fit here -- in which case the
    ceiling arm and the temporal arm are the same model.

Arms report VmSwap: this device has 12.5 GB of zram, so a resident arm that does
not fit does not fail, it silently thrashes and reports a slow-but-plausible number.

    ./run_samsung.py ceiling temporal
    ./run_samsung.py t4 t6 t8            # thread sweep
    ./run_samsung.py split1 split2 split3
"""
import subprocess, sys, time, re

DEV   = "/data/local/tmp/tmoe"
MODEL = "qwen3moe-rand-fine-Q4pure.gguf"
SIDE  = "qwen3moe-rand-fine-Q4pure-repacked.bin"
NEXP  = 192
K     = 18
RATED = {0: 3628800, 6: 4742400}
P0    = "/sys/devices/system/cpu/cpufreq/policy0"
P6    = "/sys/devices/system/cpu/cpufreq/policy6"     # SM8850: policy0 (6x perf), policy6 (2x prime)

# the Pixel production config, minus the Pixel-specific knobs (-t, the fetch-shape
# pair) and minus TWOPASS, which each arm sets explicitly.
#
# TWOPASS is NOT in BASE on purpose. It is parsed as `getenv(...) != nullptr`, so
# `LLAMA_TEMPORAL_TWOPASS=0` ENABLES it. A no-two-pass arm must OMIT the variable.
BASE = (f"LLAMA_TEMPORAL_REPACK=1 LLAMA_TEMPORAL_REPACK_FILE={SIDE} "
        "LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_MADV_FREE=1 "
        "LLAMA_TEMPORAL_SPIN_US=5000 LLAMA_TEMPORAL_FETCHPROF=1")
TP    = "LLAMA_TEMPORAL_TWOPASS=1 "
SHAPE = "LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6"

# Two different ceilings, and they answer different questions -- keep them apart:
#   ceilplain  R=192 with TWOPASS OFF. What the model does fully resident with the
#              streaming machinery not running at all. This is the headline
#              denominator and it matches how the Pixel's ~30.5 was taken.
#   ceiling    R=192 with TWOPASS ON. Same two-pass graph, same enforced-swap policy,
#              same eviction bookkeeping as the temporal arm -- so temporal/ceiling
#              isolates the cost of STREAMING from the cost of the POLICY.
# NOMADV is NOT a ceiling: it keeps every state transition and every refetch, and
# only skips the page release.
#
# key -> (label, extra env, threads, R)
ARMS = {
    "ceilplain": ("CEILING R=192 plain (no 2pass)", SHAPE,      6, NEXP),
    "ceilplain4":("CEILING R=192 plain t4",         SHAPE,      4, NEXP),
    "ceiling":   ("CEILING R=192 same-policy",      TP + SHAPE, 6, NEXP),
    # decomposing the plain -> same-policy gap. NOMADV keeps every state transition and
    # every refetch and only skips the page release, so it isolates the madvise cost
    # (TLB shootdown across the compute threads + soft refault) from the graph cost.
    # Diagnostic only: residency is unbounded, never a serving config.
    "ceilnomadv":("CEILING R=192 2pass, NOMADV",
                  TP + SHAPE + " LLAMA_TEMPORAL_NOMADV=1", 6, NEXP),
    # enforced swap WITHOUT the two-pass graph split: separates policy from graph.
    "enforce":   ("CEILING R=192 enforce, 1-pass",
                  "LLAMA_TEMPORAL_ENFORCE=1 " + SHAPE, 6, NEXP),
    "temporal":  ("TEMPORAL Pixel cfg (t4)",        TP + SHAPE, 4, K),
    # The two-pass split exists to overlap the fetch with resident compute. On this SoC
    # the fetch is already fully hidden (streaming costs nothing), so the split may be
    # pure overhead -- streaming with the SINGLE-pass enforced swap instead.
    "enforce18": ("TEMPORAL R=18 single-pass",  "LLAMA_TEMPORAL_ENFORCE=1 " + SHAPE, 6, K),
    "enforce18t4":("TEMPORAL R=18 single-pass t4","LLAMA_TEMPORAL_ENFORCE=1 " + SHAPE, 4, K),
    "t6":        ("temporal t6",                    TP + SHAPE, 6, K),
    "t8":        ("temporal t8",                    TP + SHAPE, 8, K),
    "split1":    ("temporal split=1",  TP + "LLAMA_TEMPORAL_SPLIT=1 LLAMA_TEMPORAL_FETCH_THREADS=6", 6, K),
    "split3":    ("temporal split=3",  TP + "LLAMA_TEMPORAL_SPLIT=3 LLAMA_TEMPORAL_FETCH_THREADS=6", 6, K),
    "fw4":       ("temporal fetch_thr=4", TP + "LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=4", 6, K),
    "fw8":       ("temporal fetch_thr=8", TP + "LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=8", 6, K),
    # --- eviction-cost levers (S3-38). The madvise is ~40 us per call x 135 calls/token
    # here; these are the knobs that already exist, so they cost nothing to test. Each
    # was measured on the fetch-bound Pixel where eviction barely mattered, so none of
    # those verdicts transfer (pitfall #19).
    "ev_defer":  ("temporal EVICT_DEFER",   TP + SHAPE + " LLAMA_TEMPORAL_EVICT_DEFER=1", 6, K),
    "ev_nolock": ("temporal JANITOR_NOLOCK",TP + SHAPE + " LLAMA_TEMPORAL_JANITOR_NOLOCK=1", 6, K),
    "ev_both":   ("temporal DEFER+NOLOCK",  TP + SHAPE +
                  " LLAMA_TEMPORAL_EVICT_DEFER=1 LLAMA_TEMPORAL_JANITOR_NOLOCK=1", 6, K),
    # MADV_FREE is in BASE; this arm drops it, i.e. falls back to MADV_DONTNEED.
    "ev_dontneed":("temporal MADV_DONTNEED",
                   (TP + SHAPE).replace("", ""), 6, K),
}
# ev_dontneed must OMIT LLAMA_TEMPORAL_MADV_FREE rather than set it to 0 (presence-parsed)
NO_MADV_FREE = {"ev_dontneed"}


def sh(cmd, timeout=3600):
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.replace("\r", "") + r.stderr.replace("\r", "")


def clocks():
    out = sh("for c in 0 6; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq; done")
    v = [int(x) for x in out.split() if x.isdigit()]
    return dict(zip([0, 6], v)) if len(v) == 2 else {}


def wait_cool(timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        c = clocks()
        if c and all(c[k] >= RATED[k] * 0.98 for k in c):
            return True
        time.sleep(20)
    return False


def _tis(out, marker):
    """Parse one `time_in_state` snapshot (khz -> jiffies) following its marker."""
    try:
        body = out.split(marker, 1)[1]
    except IndexError:
        return {}
    d = {}
    for line in body.splitlines()[1:]:
        f = line.split()
        if len(f) != 2 or not f[0].isdigit() or not f[1].isdigit():
            break          # snapshot ends at the next non-"<khz> <time>" line
        d[int(f[0])] = int(f[1])
    return d


def run_arm(key, ntok, reps, tag=""):
    label, extra, threads, R = ARMS[key]
    # Doze caps scaling_max_freq to ~55% of rated on this device and the cool-gate cannot
    # tell "throttled" from "capped" -- it just blocks for its full timeout. Hold it awake.
    sh("svc power stayon true", timeout=60)
    # a resident arm needs ~5.7 GB and background apps creep back between arms; on a
    # device with 12.5 GB of zram, not fitting is silent (see PEAKSWAP below).
    if R >= NEXP:
        sh("am kill-all", timeout=120)
        # 3 s was not enough immediately after a reboot: background apps were still
        # coming up and a plain-ceiling arm swapped 2635 MB and reported a void 51.6.
        time.sleep(15)
    cool = wait_cool()
    # This device cannot be DVFS-pinned (no root), and pinning exists precisely to remove
    # the asymmetry between a continuously-busy arm and one that idles on storage (S3-23:
    # worth 1.25x on the Pixel). Since it cannot be removed here, MEASURE it: cpufreq
    # time_in_state deltas give the exact residency histogram over the arm at zero
    # sampling cost. (A 5 Hz shell poll was tried first and was itself ~20 forks/sec of
    # load on the device under test -- do not reintroduce it.)
    s = ("#!/system/bin/sh\n" + f"cd {DEV}\n" + "echo 1000 > /proc/self/oom_score_adj\n"
         + f"echo TIS0_BEFORE; cat {P0}/stats/time_in_state\n"
         + f"echo TIS6_BEFORE; cat {P6}/stats/time_in_state\n")
    base = BASE
    if key in NO_MADV_FREE:
        # presence-parsed: MADV_FREE=0 would still select MADV_FREE. Drop the variable.
        base = " ".join(w for w in BASE.split() if not w.startswith("LLAMA_TEMPORAL_MADV_FREE"))
    for kv in (base + " " + extra + f" LLAMA_TEMPORAL_R={R}").split():
        s += f"export {kv}\n"
    s += (f"./llama-bench-temporal -m {MODEL} -t {threads} -p 0 -n {ntok} -r {reps} "
          f"-mmp 0 -ot _exps=CPU -o csv &\n"
          "P=$!\n"
          # zram is 12.5 GB here: a resident arm that does not fit will not fail, it
          # will swap and still print a plausible number. Sample the peak.
          "MAXSW=0\n"
          "while kill -0 $P 2>/dev/null; do\n"
          "  SW=$(grep VmSwap /proc/$P/status 2>/dev/null | tr -dc 0-9)\n"
          "  [ -n \"$SW\" ] && [ \"$SW\" -gt \"$MAXSW\" ] && MAXSW=$SW\n"
          "  sleep 1\n"
          "done\n"
          "wait $P\n"
          "echo PEAKSWAP_KB=$MAXSW\n"
          f"echo TIS0_AFTER; cat {P0}/stats/time_in_state\n"
          f"echo TIS6_AFTER; cat {P6}/stats/time_in_state\n")
    with open("/tmp/_sam_arm.sh", "w") as f:
        f.write(s)
    subprocess.run(["adb", "push", "/tmp/_sam_arm.sh", f"{DEV}/_sam_arm.sh"], capture_output=True)
    t0 = time.time()
    out = sh(f"sh {DEV}/_sam_arm.sh", timeout=5400)
    wall = time.time() - t0
    dec = sd = None
    for line in out.splitlines():
        f = line.split(",")
        if len(f) > 40 and f[0].strip('"') == "0badc06a":
            try: dec, sd = float(f[-2].strip('"')), float(f[-1].strip('"'))
            except ValueError: pass
    prof = re.search(r"temporal-fetchprof: .*", out)
    pool = re.search(r"temporal-pool: fetches=.*", out)
    swap = re.search(r"PEAKSWAP_KB=(\d+)", out)
    sw = int(swap.group(1)) // 1024 if swap else -1
    ck = ""
    parts = []
    for pol in ("0", "6"):
        b = _tis(out, f"TIS{pol}_BEFORE")
        a = _tis(out, f"TIS{pol}_AFTER")
        if not b or not a:
            continue
        # residency-weighted mean frequency over exactly this arm
        num = den = 0
        for f_khz, t in a.items():
            d = t - b.get(f_khz, 0)
            if d > 0:
                num += f_khz * d
                den += d
        if den:
            parts.append(f"cpu{pol}={num/den/1e6:.2f}GHz")
    if parts:
        ck = "  mean_clk " + " ".join(parts)
    flag = "" if cool else "  [COOL TIMEOUT]"
    print(f"{label+tag:32s} t={threads} R={R:<3} decode={dec} sd={sd} "
          f"wall={wall:.0f}s peak_swap={sw}MB{ck}{flag}", flush=True)
    if prof: print(f"   {prof.group(0)[:145]}", flush=True)
    if pool: print(f"   {pool.group(0)[:145]}", flush=True)
    if dec is None:
        print("   !! no decode row -- tail:\n   " + "\n   ".join(out.splitlines()[-12:]), flush=True)
    if sw > 200:
        print(f"   !! {sw} MB swapped -- this arm did NOT fit in RAM, treat the number as void",
              flush=True)
    return dec


if __name__ == "__main__":
    argv = sys.argv[1:]
    ntok, reps = 48, 3
    if "--n" in argv:
        i = argv.index("--n"); ntok = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    if "--reps" in argv:
        i = argv.index("--reps"); reps = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    seq = argv or ["ceiling", "temporal"]
    res = {}
    for i, a in enumerate(seq):
        res.setdefault(a, []).append(run_arm(a, ntok, reps, tag=f" #{i+1}"))
    print("\n=== summary (STOCK GOVERNOR -- not comparable to Pixel pinned numbers) ===")
    for a, v in res.items():
        v = [x for x in v if x]
        if v:
            print(f"{ARMS[a][0]:32s} {[round(x,2) for x in v]}  mean={sum(v)/len(v):.2f}")
