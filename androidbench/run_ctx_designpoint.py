"""Pixel 10a: design-point turnover across context depth (produces
results/ctx_designpoint_2026-08-24.csv; see CTX_DESIGNPOINT_2026-08-24.md).

Measures temporal residency at R=k against a directly-measured, same-shape,
SAME-DEPTH ceiling, for the fine (E=192,k=18) and k24 (E=256,k=24) shapes at
ctx depth 0/1024/2048/4096.

CONFIG INVARIANTS -- each one is a bug this harness has already been burned by:

  * TWOPASS is REQUIRED on streamed arms. With LLAMA_TEMPORAL_REPACK=1 and no
    TWOPASS, the residency logic (FIFO trim in ggml_tm_ensure, ggml-cpu.c:2930)
    sits inside mul_mat_id, which CPU_REPACK bypasses -- so R silently means
    "lazy-load, never evict" and the arm measures a nearly-resident model.
    Pitfalls #8/#25. TWOPASS survives repack because it drives residency from
    ggml_temporal_window_fill, a separate graph op.
  * TWOPASS is FORBIDDEN on ceiling arms: the denominator is `ceilplain`
    (fully resident, streaming machinery off) per run_samsung.py.
  * R is inert under TWOPASS for R >= top_k (window is sized n_expert_used, not
    R -- llama-graph.cpp:1930), so this script yields the R=k operating point,
    NOT an R-curve. For a real R-curve use run_rcurve_norepack.py. Pitfall #26.
  * Ceilings must be measured at EVERY depth reported (a resident model also
    slows with context: 32.4 -> 14.6 tok/s over 0 -> 4096). Pitfall #27.
  * Ceiling models are sized to fit at the DEEPEST context tested, because the
    KV cache competes for the same RAM (e112 fit at ctx=0 and swapped 510 MB at
    ctx=1024). Pitfall #28.

VALIDITY GUARDS (an arm that trips any of these is written with void=True):
  * evictions == 0 on a streamed arm -> no_turnover, VOID, and NOT retried
    (config fault, not thermal noise).
  * residency-weighted mean clock < 97% of rated across the arm's actual
    runtime, from cpufreq time_in_state deltas -- not just a pre-run sample.
    Pitfall #29.
  * peak VmSwap > 100 MB -> the arm did not fit in RAM.
  * sustained cool-gate (3 consecutive samples >= 98% of rated) before each arm,
    with the governor pin re-asserted periodically so a lost pin self-heals.

Depth is passed as `-d`; llama-bench defaults it to 0, and the 2026-07-24 curve
ran at depth 0 while bench.py-era scripts used 1024. Always state the depth.
Pitfall #30.

Caller must ensure >=80% battery on AC; this script does not gate on it.
"""
import csv, json, re, subprocess, sys, time

DEV = "/data/local/tmp/tmoe"
POLICIES = ["/sys/devices/system/cpu/cpufreq/policy0",
            "/sys/devices/system/cpu/cpufreq/policy4",
            "/sys/devices/system/cpu/cpufreq/policy7"]
RATED = {0: 1950000, 4: 2600000, 7: 3105000}

MODELS = {
    "fine": dict(gguf="qwen3moe-rand-fine-Q4pure.gguf",
                 side="qwen3moe-rand-fine-Q4pure-repacked.bin", K=18, E=192),
    "k24":  dict(gguf="qwen3moe-rand-k24-Q4pure.gguf",
                 side="k24-repacked.bin", K=24, E=256),
    "e141n_ceiling": dict(gguf="qwen3moe-rand-e141n-Q4pure.gguf",
                           side=None, K=141, E=141),   # fully resident, no side-file
    "e112_ceiling":  dict(gguf="qwen3moe-rand-e112-Q4pure.gguf",
                           side="e112-repacked.bin", K=112, E=112),
    # Conservative same-shape ceilings for the context sweep: e112/e141n were sized for
    # ctx=0 headroom and don't reliably fit once a longer context's KV-cache shares the
    # same RAM budget (confirmed: e112 swapped 510MB at ctx=1024 this morning, despite
    # fitting fine overnight -- background app memory creep between sessions). Decode is
    # empirically E-independent when fully resident, so shrinking E buys headroom without
    # changing what's being measured.
    "e80_ceiling":   dict(gguf="qwen3moe-rand-e80-Q4pure.gguf",
                           side=None, K=80, E=80),
    "e100n_ceiling": dict(gguf="qwen3moe-rand-e100n-Q4pure.gguf",
                           side=None, K=100, E=100),
}

OUT_CSV = "run_TWOPASS_verified.csv"
FIELDS = ["ts", "phase", "model", "R", "ctx", "rep", "decode_tps", "stddev_tps",
          "prefill_tps", "fetches", "evictions", "fetched_mib", "avg_fetch_us",
          "peak_swap_mb", "clk0", "clk4", "clk7", "mean_clk0", "mean_clk4", "mean_clk7",
          "throttled", "void", "no_turnover", "swaps", "cmd"]

REST_S = 60   # minimum rest between arms regardless of cool-gate outcome; heat that
              # doesn't yet show as a clock drop can still be building (BENCHMARK_GATES
              # M4/M5: back-to-back hot batches read 20-30% low even when each individual
              # arm's pre-check looked clean).


def sh(cmd, timeout=3600):
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.replace("\r", "") + r.stderr.replace("\r", "")


def clocks():
    out = sh("for p in 0 4 7; do cat /sys/devices/system/cpu/cpufreq/policy$p/scaling_cur_freq; done")
    v = [int(x) for x in out.split() if x.isdigit()]
    return dict(zip([0, 4, 7], v)) if len(v) == 3 else {}


def tis_snapshot():
    """time_in_state per policy: {policy: {khz: jiffies}}. Same technique run_samsung.py
    uses to measure residency-weighted mean clock over an arm's ACTUAL runtime, not just
    a point-sample before it starts -- a pre-run-only check missed mid-run throttling
    once already this session (e112 ceiling read 20.0 vs a historical 27.5-36 tok/s,
    root-caused to policy4/7 sitting at ~86% of rated mid-run while Thermal Status still
    read 0 -- the exact false-negative the ledger's pitfall L1 describes)."""
    out = sh("for p in 0 4 7; do echo P$p; "
             "cat /sys/devices/system/cpu/cpufreq/policy$p/stats/time_in_state; done")
    snap, cur = {}, None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("P") and line[1:].isdigit():
            cur = int(line[1:]); snap[cur] = {}
        elif cur is not None:
            parts = line.split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                snap[cur][int(parts[0])] = int(parts[1])
    return snap


def mean_clock_khz(before, after):
    """Residency-weighted mean clock per policy over the (before,after) tis window."""
    out = {}
    for p in before:
        num = den = 0
        for khz, t_after in after.get(p, {}).items():
            d = t_after - before[p].get(khz, 0)
            if d > 0:
                num += khz * d
                den += d
        out[p] = (num / den) if den else None
    return out


def repin_governor():
    sh("su -c 'for p in " + " ".join(POLICIES) + "; do echo performance > $p/scaling_governor; done'")


def wait_cool(timeout_s=1800, sustain_checks=3, sustain_gap_s=5, repin_every_s=120):
    """Block until clocks read >=98% of rated for `sustain_checks` consecutive samples
    (not just once -- a single clean sample right before launch is what missed the e112
    throttle: clocks can look fine an instant before a run and still be mid-drift).
    Periodically re-asserts the governor pin so a lost pin (reboot, doze, a watchdog
    restart -- all have happened to this device before per LEDGER.md) self-heals instead
    of silently voiding every arm for the rest of the night."""
    t0 = time.time()
    streak = 0
    last = {}
    last_repin = 0
    while time.time() - t0 < timeout_s:
        if time.time() - last_repin > repin_every_s:
            repin_governor()
            last_repin = time.time()
        c = clocks()
        last = c
        if c and all(c[k] >= RATED[k] * 0.98 for k in c):
            streak += 1
            if streak >= sustain_checks:
                return c
        else:
            streak = 0
        time.sleep(sustain_gap_s)
    return None if not last or not all(last.get(k, 0) >= RATED[k] * 0.98 for k in RATED) else last


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_arm(phase, model_key, R, ctx, ngen=128, reps=3):
    m = MODELS[model_key]
    env = (f"LLAMA_TEMPORAL_REPACK=1 "
           f"LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_MADV_FREE=1 "
           f"LLAMA_TEMPORAL_R={R} LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6 "
           f"LLAMA_TEMPORAL_SPIN_US=5000 LLAMA_TEMPORAL_FETCHPROF=1")
    if m["side"]:
        env += f" LLAMA_TEMPORAL_REPACK_FILE={m['side']}"
    # TWOPASS is REQUIRED here, and removing it earlier was an error that invalidated a
    # whole night of data. Mechanism (verified in engine source + empirically):
    #   * The R-curve/turnover logic (FIFO trim to R, swap_prob eviction) lives in
    #     ggml_tm_ensure, inside mul_mat_id -- which CPU_REPACK bypasses entirely
    #     (documented pitfall #8, "bit three separate times"). With REPACK=1 and no
    #     TWOPASS, LLAMA_TEMPORAL_R degenerates to "lazy-load on first touch, NEVER
    #     evict": measured evictions=0, hook_calls=0, byte-identical fetches across
    #     R=18/36/48, and the pool sitting 6.3x over its nominal R=18 cap. That measures
    #     an effectively-fully-resident model, not temporal residency.
    #   * TWOPASS drives residency from ggml_temporal_window_fill, a SEPARATE graph op,
    #     so it survives repack -- evictions are real (measured 51840 vs 0).
    # Scope limit to remember: window_fill sizes the window as n_expert_used (top_k), NOT
    # R (llama-graph.cpp:1930), so under TWOPASS every R >= top_k behaves identically.
    # This config yields the R=top_k operating point only; a genuine R=36/48 curve needs
    # NO_REPACK + LLAMA_TEMPORAL_SWAP_PROB (the FIFO-trim path).
    # Ceiling arms (R == E, fully resident) must NOT get TWOPASS: the headline denominator
    # is run_samsung.py's "ceilplain" (fully resident, streaming machinery not running).
    # TWOPASS there would enforce a swap/layer even at full residency -- not a ceiling.
    streamed = R < m["E"]
    if streamed:
        env += " LLAMA_TEMPORAL_TWOPASS=1"

    if R >= m["E"] * 0.8:   # a large/fully-resident arm -- background apps creep back
        sh("am kill-all 2>/dev/null")   # between runs (run_samsung.py precedent) and can
        time.sleep(15)                  # push a marginal ceiling model into swap
    c = wait_cool()
    if c is None:
        log(f"{phase} {model_key} R={R} ctx={ctx}: !! COOL-GATE TIMED OUT (never sustained "
            f"98% of rated for {3} consecutive checks) -- SKIPPING this arm, not recording "
            f"throttled data. Device likely needs a real rest; check thermal state.")
        return None
    clk_str = ",".join(f"{k}={v}" for k, v in c.items())
    log(f"{phase} {model_key} R={R} ctx={ctx}: cool-gate PASSED (sustained) clocks {clk_str}")

    tis_before = tis_snapshot()
    s = ("#!/system/bin/sh\n" + f"cd {DEV}\n" + "echo 1000 > /proc/self/oom_score_adj\n")
    for kv in env.split():
        s += f"export {kv}\n"
    depth_flag = f"-d {ctx} " if ctx else ""   # ctx=0 omits -d entirely, matching the
                                                # historical ad-hoc scripts that produced
                                                # the actual headline curve (llama-bench's
                                                # own default is -d 0 anyway, confirmed via
                                                # --help, so this is belt-and-suspenders)
    s += (f"./llama-bench-temporal -m {m['gguf']} -t 4 -p 0 -n {ngen} {depth_flag}"
          f"-r {reps} -mmp 0 -ot \"_exps=CPU\" -o csv &\n"
          "P=$!\n"
          "MAXSW=0\n"
          "while kill -0 $P 2>/dev/null; do\n"
          "  SW=$(grep VmSwap /proc/$P/status 2>/dev/null | tr -dc 0-9)\n"
          "  [ -n \"$SW\" ] && [ \"$SW\" -gt \"$MAXSW\" ] && MAXSW=$SW\n"
          "  sleep 1\n"
          "done\nwait $P\necho PEAKSWAP_KB=$MAXSW\n")
    with open("/tmp/_run_k24ctx.sh", "w") as f:
        f.write(s)
    subprocess.run(["adb", "push", "/tmp/_run_k24ctx.sh", f"{DEV}/_run_k24ctx.sh"],
                    capture_output=True)
    out = sh(f"su -c 'sh {DEV}/_run_k24ctx.sh'", timeout=1800)
    tis_after = tis_snapshot()
    mean_clk = mean_clock_khz(tis_before, tis_after)
    throttled = any(mean_clk.get(p) is not None and mean_clk[p] < RATED[p] * 0.97
                     for p in RATED)
    mean_clk_str = ",".join(f"{p}={mean_clk.get(p):.0f}" if mean_clk.get(p) else f"{p}=?"
                             for p in (0, 4, 7))
    log(f"  residency-weighted mean clock over the run: {mean_clk_str}"
        + ("  !! THROTTLED MID-RUN" if throttled else ""))

    dec = sd = pp = None
    for line in out.splitlines():
        f_ = line.split(",")
        if len(f_) > 30 and f_[0].strip('"') == "0badc06a":
            try:
                is_gen = f_[-4].strip('"') != "0"
                if is_gen:
                    dec, sd = float(f_[-2].strip('"')), float(f_[-1].strip('"'))
                else:
                    pp = float(f_[-2].strip('"'))
            except (ValueError, IndexError):
                pass
    pool = re.search(r"temporal-pool: fetches=(\d+) fetched_mib=([\d.]+) evictions=(\d+) "
                      r".*avg_fetch_us=([\d.]+)", out)
    swap = re.search(r"PEAKSWAP_KB=(\d+)", out)
    sw_mb = int(swap.group(1)) // 1024 if swap else -1

    # RESIDENCY GUARD -- the check whose absence invalidated an entire night of data.
    # A streamed arm that never evicted did not stream: the pool lazily loaded experts
    # and kept them, so the number describes an effectively-resident model. Such an arm
    # is VOID by construction, no matter how clean its clocks and swap look.
    ev = int(pool.group(3)) if pool else -1
    swaps_m = re.search(r"ENFORCE on, swaps=(\d+)", out)
    n_swaps = int(swaps_m.group(1)) if swaps_m else -1
    no_turnover = streamed and ev <= 0
    if no_turnover:
        log(f"  !! NO EVICTIONS on a streamed arm (evictions={ev}, swaps={n_swaps}) -- "
            f"residency was NOT enforced; this measures a lazily-loaded resident model. VOID.")

    void = sw_mb > 100 or throttled or dec is None or no_turnover
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), phase=phase, model=model_key, R=R,
               ctx=ctx, rep=reps, decode_tps=dec, stddev_tps=sd, prefill_tps=pp,
               fetches=pool.group(1) if pool else "", evictions=pool.group(3) if pool else "",
               fetched_mib=pool.group(2) if pool else "", avg_fetch_us=pool.group(4) if pool else "",
               peak_swap_mb=sw_mb, clk0=c.get(0), clk4=c.get(4), clk7=c.get(7),
               mean_clk0=mean_clk.get(0), mean_clk4=mean_clk.get(4), mean_clk7=mean_clk.get(7),
               throttled=throttled, void=void, no_turnover=no_turnover, swaps=n_swaps, cmd=f"{model_key} R={R} d={ctx}")
    write_row(row)
    log(f"  -> decode={dec} +/-{sd} prefill={pp} swap={sw_mb}MB pool={pool.group(0) if pool else 'NONE'}")
    if sw_mb > 100:
        log(f"  !! {sw_mb}MB swapped -- VOID this arm, did not fit in RAM")
    if throttled:
        log(f"  !! VOID this arm -- mean clock dropped below 97% of rated during the run")
    if dec is None:
        log("  !! NO DECODE ROW -- tail:\n   " + "\n   ".join(out.splitlines()[-15:]))
    log(f"  resting {REST_S}s before next arm")
    time.sleep(REST_S)
    return row


def run_arm_retrying(phase, model_key, R, ctx, ngen=128, reps=3, max_extra_rest=3):
    """run_arm, but if an attempt comes back void (swap or mid-run throttle), rest longer
    and retry rather than silently losing that data point. Every attempt is still written
    to the CSV (void=True included) for a full audit trail; the loop just keeps trying
    for a clean one so morning's dataset is as complete as possible."""
    extra_rest = 0
    for attempt in range(1, max_extra_rest + 2):
        row = run_arm(phase, model_key, R, ctx, ngen=ngen, reps=reps)
        if row is None:
            return None   # cool-gate never sustained -- see run_arm's own log for why
        if not row["void"]:
            return row
        if row.get("no_turnover"):
            log("  no-turnover is a CONFIG fault, not thermal noise -- retrying cannot fix "
                "it. Stopping this arm immediately.")
            return row
        extra_rest += 120
        log(f"  attempt {attempt} was void -- resting an extra {extra_rest}s before retry "
            f"({max_extra_rest + 1 - attempt} retries left)")
        time.sleep(extra_rest)
    log(f"  !! {phase} {model_key} R={R} ctx={ctx}: gave up after "
        f"{max_extra_rest + 1} void attempts -- leaving void in the record")
    return row


def write_row(row):
    import os
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    log("=== PHASE 1: VALIDATE vs HISTORY (fine R=18=K, ctx=0). Historical: 5.65 tok/s, "
        "34% of ceiling. TWOPASS on, evictions MUST be > 0 or the arm is void. ===")
    run_arm_retrying("validate", "fine", 18, 0)

    log("=== PHASE 1b: CONTROL -- R=36 under TWOPASS should be IDENTICAL to R=18, because "
        "window_fill sizes the window as top_k, not R (llama-graph.cpp:1930). If these "
        "differ, my reading of the engine is wrong and everything below is suspect. ===")
    run_arm_retrying("control_R_inert", "fine", 36, 0)

    log("=== PHASE 2: fine R=18 across context depths (real turnover this time) ===")
    for ctx in (1024, 2048, 4096):
        run_arm_retrying("ctx_sweep", "fine", 18, ctx)

    log("=== PHASE 3: k24 R=24=K across context depths ===")
    for ctx in (0, 1024, 2048, 4096):
        run_arm_retrying("ctx_sweep", "k24", 24, ctx)

    log("DONE. Results in " + OUT_CSV)
    log("Ceiling denominators reuse the already-measured e80/e100n ceilplain arms "
        "(fully resident, no TWOPASS) from run_k24_ctx_results_n128.csv -- those are "
        "unaffected by pitfall #8 since a fully-resident model is not supposed to evict.")


if __name__ == "__main__":
    main()
