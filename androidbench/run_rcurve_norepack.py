#!/usr/bin/env python3
"""The R-curve the TWOPASS regime structurally cannot produce, plus a true replication
attempt of the historical Pixel numbers.

WHY THIS CONFIG:
  * The FIFO trim that actually enforces "hold at most R experts" lives in ggml_tm_ensure
    (ggml-cpu.c:2930, `while (t->n_resident > g_tm_R) ... evict`), inside mul_mat_id.
    CPU_REPACK bypasses mul_mat_id entirely (pitfall #8), which is why every repacked
    arm reported evictions=0 and identical fetches across R. So an honest R-curve
    REQUIRES LLAMA_NO_REPACK=1.
  * TWOPASS is deliberately OFF: its window is sized n_expert_used (top_k), not R
    (llama-graph.cpp:1930), so under TWOPASS every R >= top_k is identical -- verified
    empirically (R=18: 20.70 tok/s / R=36: 20.51, byte-identical fetch+evict counts).
  * SWAP_PROB is left at 0, i.e. NATURAL router jitter. ANDROID_OPTIM_PROGRESS.md:79
    measures that jitter at 0.85 experts/layer/token. Hypothesis under test: at R=K
    (zero slack) each jittered-in expert forces an evict+refetch and thrashes, while
    R=36/48/64 give slack for recently-used experts to survive -- which would explain
    the historical 5.65 / 10.49 / 10.85 / 11.76 shape at R=18/36/48/64.

PITFALL #6 (never mix kernel families): the denominator here MUST also be non-repacked,
so the ceiling arm is e80 at R=E under LLAMA_NO_REPACK=1 -- NOT the repacked e80 ceiling
measured earlier. Non-repacked compute is ~1.33x slower, so absolute tok/s in this table
is NOT comparable to the TWOPASS table; only the %-of-ceiling column is.
"""
import csv, os, re, subprocess, sys, time
sys.path.insert(0, ".")
from run_ctx_designpoint import (DEV, MODELS, RATED, sh, clocks, tis_snapshot, mean_clock_khz,
                          wait_cool, log)

OUT_CSV = "run_norepack_rcurve.csv"
FIELDS = ["ts", "model", "R", "ctx", "decode_tps", "stddev_tps", "ceiling_tps",
          "fetches", "evictions", "fetched_mib", "avg_fetch_us", "experts_per_layer_tok",
          "peak_swap_mb", "mean_clk0", "mean_clk4", "mean_clk7", "throttled", "void", "note"]
REST_S = 60


def write_row(row):
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_arm(model_key, R, ctx=0, ngen=128, reps=3, note=""):
    m = MODELS[model_key]
    streamed = R < m["E"]
    env = ("LLAMA_NO_REPACK=1 LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_MADV_FREE=1 "
           "LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6 "
           f"LLAMA_TEMPORAL_SPIN_US=5000 LLAMA_TEMPORAL_R={R}")
    # no TWOPASS, no SWAP_PROB -> natural jitter + FIFO trim to R

    if not streamed:
        sh("am kill-all 2>/dev/null"); time.sleep(15)
    c = wait_cool()
    if c is None:
        log(f"{model_key} R={R} ctx={ctx}: COOL-GATE TIMEOUT -- skipping")
        return None
    log(f"{model_key} R={R} ctx={ctx}: cool-gate PASSED {c}")

    tis_b = tis_snapshot()
    s = f"#!/system/bin/sh\ncd {DEV}\necho 1000 > /proc/self/oom_score_adj\n"
    for kv in env.split():
        s += f"export {kv}\n"
    depth = f"-d {ctx} " if ctx else ""
    s += (f"./llama-bench-temporal -m {m['gguf']} -t 4 -p 0 -n {ngen} {depth}"
          f"-r {reps} -mmp 0 -ot \"_exps=CPU\" -o csv &\nP=$!\nMAXSW=0\n"
          "while kill -0 $P 2>/dev/null; do\n"
          "  SW=$(grep VmSwap /proc/$P/status 2>/dev/null | tr -dc 0-9)\n"
          "  [ -n \"$SW\" ] && [ \"$SW\" -gt \"$MAXSW\" ] && MAXSW=$SW\n  sleep 1\ndone\n"
          "wait $P\necho PEAKSWAP_KB=$MAXSW\n")
    open("/tmp/_nr.sh", "w").write(s)
    subprocess.run(["adb", "push", "/tmp/_nr.sh", f"{DEV}/_nr.sh"], capture_output=True)
    out = sh(f"su -c 'sh {DEV}/_nr.sh'", timeout=2400)
    tis_a = tis_snapshot()
    mc = mean_clock_khz(tis_b, tis_a)
    throttled = any(mc.get(p) is not None and mc[p] < RATED[p] * 0.97 for p in RATED)

    dec = sd = None
    for line in out.splitlines():
        f_ = line.split(",")
        if len(f_) > 30 and f_[0].strip('"') == "0badc06a":
            try:
                dec, sd = float(f_[-2].strip('"')), float(f_[-1].strip('"'))
            except ValueError:
                pass
    pool = re.search(r"fetches=(\d+) fetched_mib=([\d.]+) evictions=(\d+).*?avg_fetch_us=([\d.]+)", out)
    sw = re.search(r"PEAKSWAP_KB=(\d+)", out)
    sw_mb = int(sw.group(1)) // 1024 if sw else -1
    ev = int(pool.group(3)) if pool else -1
    fetches = int(pool.group(1)) if pool else -1
    # slices/token / 45 layers / 3 slices-per-expert = experts swapped per layer per token
    epl = fetches / (ngen * reps) / 45 / 3 if fetches > 0 else 0.0
    no_turnover = streamed and ev <= 0
    void = sw_mb > 100 or throttled or dec is None or no_turnover
    row = dict(ts=time.strftime("%H:%M:%S"), model=model_key, R=R, ctx=ctx,
               decode_tps=dec, stddev_tps=sd, ceiling_tps="",
               fetches=fetches, evictions=ev,
               fetched_mib=pool.group(2) if pool else "",
               avg_fetch_us=pool.group(4) if pool else "",
               experts_per_layer_tok=round(epl, 3), peak_swap_mb=sw_mb,
               mean_clk0=mc.get(0), mean_clk4=mc.get(4), mean_clk7=mc.get(7),
               throttled=throttled, void=void, note=note)
    write_row(row)
    log(f"  -> decode={dec} +/-{sd} evict={ev} fetches={fetches} "
        f"churn={epl:.2f} experts/layer/tok swap={sw_mb}MB"
        + ("  !! THROTTLED" if throttled else "")
        + ("  !! NO TURNOVER" if no_turnover else ""))
    time.sleep(REST_S)
    return row


def retry(model_key, R, ctx=0, note="", tries=3):
    for i in range(tries):
        r = run_arm(model_key, R, ctx, note=note)
        if r is None or not r["void"]:
            return r
        if r.get("note") and r["evictions"] <= 0 and R < MODELS[model_key]["E"]:
            log("  no-turnover is a config fault; not retrying")
            return r
        log(f"  void (attempt {i+1}) -- extra rest before retry")
        time.sleep(120 * (i + 1))
    return r


def main():
    log("=== NO_REPACK ceiling (pitfall #6: denominator must be same kernel family) ===")
    retry("e80_ceiling", 80, 0, note="norepack ceilplain")

    log("=== R-CURVE, natural jitter, NO_REPACK. Historical targets at ctx=0: "
        "R=18->5.65, R=36->10.49, R=48->10.85, R=64->11.76 tok/s ===")
    for R in (18, 36, 48, 64):
        retry("fine", R, 0, note="replication target")

    log("DONE -> " + OUT_CSV)


if __name__ == "__main__":
    main()
