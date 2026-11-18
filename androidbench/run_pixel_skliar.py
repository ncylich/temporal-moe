#!/usr/bin/env python3
"""Pixel 10a: Skliar's cache-conditional routing on the plain 1e19 MoE (PIXEL_SKLIAR_LEDGER.md).

Same phone protocol as the trained model (run_ctx_designpoint.py --model trained_fine --noroot): llama-bench,
-t 4, -p 0, -n 128, -d 1024, -r 3, -mmp 0, -ot "_exps=CPU", the production fetch flags, 60 s rest and the same
void rules. Differences are listed in the ledger: real text (TEMPORAL_BENCH_CTX/GEN) and a stricter cool gate.
The arms other than skliar_stream and trained_deploy were used for gates and exploration only.

    sh results/pixel_skliar/run_timing.sh                                  # the timed rows (tag t2)
    python3 run_pixel_skliar.py gate --texts 1 2 3 4 5 --reps 3 --env TEMPORAL_BENCH_REPREFILL=0
    python3 run_pixel_skliar.py dump --texts 1 --arms skliar_stream        # routing dump for the check scripts
    sh results/pixel_skliar/run_gates_shexp.sh; sh results/pixel_skliar/run_timing_shexp.sh   # S1-7, shared expert first
"""
import argparse, json, os, re, subprocess, time

import run_ctx_designpoint as H

DEV = H.DEV
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "pixel_skliar")
BENCH = "llama-bench-skliar"
GGUF = "moe_fine_1e19-full-q4_0.gguf"
SIDE = "moe_fine_1e19-full-q4_0-repacked.bin"
K, E, LAM = 18, 192, os.environ.get("LAM", "0.7207")      # LAM=<lambda> overrides Skliar's bias (ledger S1-7)
CTX = int(os.environ.get("CTX", "1024"))
SK = f"LLAMA_TEMPORAL_ROUTE=skliar TEMPORAL_CB_C={K} TEMPORAL_CB_LAMBDA={LAM} TEMPORAL_CB_J=1"

# R experts per layer in RAM and the routing flags. Streamed arms need TWOPASS (run_ctx_designpoint.py).
ARMS = {
    "stock_resident":  (E, ""),                                                # all experts in RAM, stock top-k
    "skliar_stream":   (K, "LLAMA_TEMPORAL_TWOPASS=1 " + SK),                  # Skliar, pool on flash
    "ours_stream":     (K, "LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ROUTE=decode"),   # our rule, pool on flash
    "skliar_resident": (E, SK),                                                # the rules with all experts in RAM
    "ours_resident":   (E, "LLAMA_TEMPORAL_ROUTE=decode"),
    # gate only: the streamed arms' own graph (two-pass) with every expert kept in RAM
    "skliar_ram2p":    (E, "LLAMA_TEMPORAL_TWOPASS=1 " + SK),
    "ours_ram2p":      (E, "LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ROUTE=decode"),
    # the trained model's deploy arm exactly as in TRAINED_MODEL_LEDGER.md step 6 (trained GGUF, llama-bench's
    # random tokens), on this build: shows the trained rule's path is unchanged by patch 0003
    "trained_deploy":  (K, "LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ROUTE=trained"),
}
TRAINED = ("fine_1e19-q4_0.gguf", "fine_1e19-q4_0-repacked.bin")
# S1-7: the trained model on the same five real texts as the Skliar arm, pool on flash and (gate) all in RAM
ARMS["trainedtext_stream"] = (K, "LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ROUTE=trained")
ARMS["trainedtext_ram2p"] = (E, "LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ROUTE=trained")
# The 11 B random-weight model of CTX_DESIGNPOINT_2026-08-24.md (45 MoE layers, 192 experts, 18 active). Its
# router is not meaningful, so loads are imposed (LLAMA_TEMPORAL_ROUTE=replay): one per layer per token, or
# drawn from Skliar's measured histogram on the 1.26 B model (the timed rows of ledger S1-5).
BIG = ("qwen3moe-rand-fine-Q4pure.gguf", "qwen3moe-rand-fine-Q4pure-repacked.bin")
SKLIAR_HIST = "5118,15783,2825,850,272,76,27,8,1"
ARMS["big_one_load"] = (K, "LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ROUTE=replay TEMPORAL_REPLAY_HIST=0,1")
ARMS["big_skliar_loads"] = (K, "LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ROUTE=replay TEMPORAL_REPLAY_HIST=" + SKLIAR_HIST)


def thermal():
    """(Thermal Status, VIRTUAL-SKIN deg C). soc_therm, which the trained-model run gated on, read a constant
    27.854 C through this whole session, so the gate here adds the framework's status and logs the skin value."""
    out = H.sh("dumpsys thermalservice 2>/dev/null")
    st = re.search(r"Thermal Status: (\d+)", out)
    sk = re.findall(r"mValue=([0-9.]+), mType=3, mName=VIRTUAL-SKIN,", out)     # cached value first, current value last
    return (int(st.group(1)) if st else None, float(sk[-1]) if sk else None)


SKIN_MAX = float(os.environ.get("SKIN_MAX", "34.0"))


def wait_cool(timeout_s=3600):
    """run_ctx_designpoint.wait_cool_noroot (soc_therm <= 33 C and every scaling_max_freq at rating, three
    samples 10 s apart) and, in addition, Thermal Status 0 and VIRTUAL-SKIN <= SKIN_MAX. Same gate for every arm."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st, sk = thermal()
        if st == 0 and sk is not None and sk <= SKIN_MAX:
            c = H.wait_cool_noroot(timeout_s=300)
            if c is not None:
                c.update(thermal_status=st, skin_c=sk)
                return c
        time.sleep(15)
    return None


def run(arm, v, reps, ngen, tag, extra="", dump=False, cool=True):
    R, route = ARMS[arm]
    name = f"{arm}_v{v}_{tag}"
    trained = arm == "trained_deploy" or arm.startswith("big")        # llama-bench's random tokens
    gguf, side = BIG if arm.startswith("big") else TRAINED if arm.startswith("trained") else (GGUF, SIDE)
    text = "" if trained else (f"TEMPORAL_BENCH_CTX=tok/ctx1024_v{v}.ctx TEMPORAL_BENCH_GEN=tok/ctx1024_v{v}.gen "
                               f"TEMPORAL_BENCH_REPREFILL=1 TEMPORAL_ROUTE_HIST=_hist.json ")
    env = (f"LLAMA_TEMPORAL_REPACK=1 LLAMA_TEMPORAL_REPACK_FILE={side} "
           f"LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_MADV_FREE=1 "
           f"LLAMA_TEMPORAL_R={R} LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6 "
           f"LLAMA_TEMPORAL_SPIN_US=5000 LLAMA_TEMPORAL_FETCHPROF=1 {route} "
           f"{text}TEMPORAL_BENCH_TOKLOG=_tok.csv {extra}").split()
    if dump:
        env.append("LLAMA_TEMPORAL_ROUTE_DUMP2=_route.bin")
    streamed = R < E
    if not streamed:
        H.sh("am kill-all 2>/dev/null"); time.sleep(15)
    c = {}
    if cool:
        c = wait_cool()
        if c is None:
            H.log(f"{name}: cool gate timed out, skipped"); return None
        H.log(f"{name} R={R}: cool gate passed {c}")
    s = "#!/system/bin/sh\n" + f"cd {DEV}\n" + "echo 1000 > /proc/self/oom_score_adj\nrm -f _tok.csv _hist.json _route.bin _bench.json\n"
    for kv in env:
        s += f"export {kv}\n"
    s += (f"./{BENCH} -m {gguf} -t 4 -p 0 -n {ngen} -d {CTX} -r {reps} -mmp 0 -ot \"_exps=CPU\" -o json > _bench.json &\n"
          "P=$!\nMAXSW=0\nHWM=0\nMF7=99999999\n"
          "while kill -0 $P 2>/dev/null; do\n"
          "  SW=$(grep VmSwap /proc/$P/status 2>/dev/null | tr -dc 0-9)\n"
          "  [ -n \"$SW\" ] && [ \"$SW\" -gt \"$MAXSW\" ] && MAXSW=$SW\n"
          "  H=$(grep VmHWM /proc/$P/status 2>/dev/null | tr -dc 0-9)\n"
          "  [ -n \"$H\" ] && HWM=$H\n"
          "  F=$(cat /sys/devices/system/cpu/cpufreq/policy7/scaling_max_freq)\n"
          "  [ \"$F\" -lt \"$MF7\" ] && MF7=$F\n"
          "  sleep 0.5\n"
          "done\nwait $P\necho PEAKSWAP_KB=$MAXSW\necho PEAKHWM_KB=$HWM\necho MINMAXF7=$MF7\n")
    with open("/tmp/_run_skliar.sh", "w") as f:
        f.write(s)
    subprocess.run(["adb", "push", "/tmp/_run_skliar.sh", f"{DEV}/_run_skliar.sh"], capture_output=True)
    out = H.sh(f"sh {DEV}/_run_skliar.sh 2>&1", timeout=3600)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True); os.makedirs(os.path.join(OUT, "toklog"), exist_ok=True)
    with open(os.path.join(OUT, "logs", name + ".log"), "w") as f:
        f.write(" ".join(env) + "\n" + out)
    g = lambda pat: (int(re.search(pat, out).group(1)) if re.search(pat, out) else None)
    mf7 = g(r"MINMAXF7=(\d+)")
    throttled = mf7 is None or mf7 < H.RATED[7]
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), arm=arm, text=v, tag=tag, R=R, ctx=CTX, n_gen=ngen, reps=reps,
               env=" ".join(env), peak_swap_mb=(g(r"PEAKSWAP_KB=(\d+)") or 0) // 1024,
               peak_rss_mb=round((g(r"PEAKHWM_KB=(\d+)") or 0) / 1024, 1), min_max_freq7=mf7,
               soc_temp_c=c.get("soc_temp"), skin_c_before=c.get("skin_c"), rooted=False)
    row["thermal_status_after"], row["skin_c_after"] = thermal()
    try:
        j = json.loads(H.sh(f"cat {DEV}/_bench.json"))[0]
        row.update(samples_ts=j["samples_ts"], decode_tps=j["avg_ts"], stddev_tps=j["stddev_ts"],
                   ms_per_tok=1000.0 / j["avg_ts"], build_commit=j.get("build_commit"))
    except Exception as e:
        row["error"] = repr(e)
    pool = re.search(r"temporal-pool: fetches=(\d+) fetched_mib=([\d.]+) evictions=(\d+) .*avg_fetch_us=([\d.]+) "
                     r"qwait_hi_us=([\d.]+) wait_ms=([\d.]+)", out)
    if pool:
        row.update(fetches=int(pool.group(1)), fetched_mib=float(pool.group(2)), evictions=int(pool.group(3)),
                   avg_fetch_us=float(pool.group(4)), wait_ms=float(pool.group(6)))
    rt = re.search(r"temporal-route2: .*decode_layer_tokens=(\d+) loads=(\d+) loads_per_layer_token=([\d.]+) .*hist=\[([\d,]*)\]", out)
    if rt:
        row.update(decode_layer_tokens=int(rt.group(1)), loads=int(rt.group(2)), loads_per_layer_token=float(rt.group(3)),
                   hist=[int(x) for x in rt.group(4).split(",") if x])
    tr = re.search(r"temporal-route: .*layer_tokens=(\d+) cold_fills=(\d+) swaps=(\d+) swaps_per_layer_token=([\d.]+)", out)
    if arm.startswith("trained") and tr:
        row.update(swaps_per_layer_token=float(tr.group(4)), route_layer_tokens=int(tr.group(1)))
        rt = tr
    row["route_lines"] = [l for l in out.splitlines() if l.startswith("temporal-")]
    no_turnover = streamed and row.get("evictions", -1) <= 0
    row["void"] = bool(row["peak_swap_mb"] > 100 or throttled or "decode_tps" not in row or no_turnover
                       or (arm != "stock_resident" and rt is None))
    row["throttled"], row["no_turnover"] = throttled, no_turnover
    tl = os.path.join(OUT, "toklog", name + ".csv")
    subprocess.run(["adb", "pull", f"{DEV}/_tok.csv", tl], capture_output=True)
    try:                                                   # tokens 2..n of each repetition: slowest one, and how many took over 200 ms
        import csv
        with open(tl) as f:
            ms = [int(x["ns"]) / 1e6 for x in csv.DictReader(f) if int(x["rep"]) >= 0 and int(x["i"]) > 0]
        row["max_token_ms"], row["tokens_over_200ms"] = round(max(ms), 1), sum(m > 200 for m in ms)
    except Exception:
        pass
    if dump:
        subprocess.run(["adb", "pull", f"{DEV}/_route.bin", os.path.join(OUT, f"route_{name}.bin")], capture_output=True)
        H.sh(f"rm -f {DEV}/_route.bin")
    with open(os.path.join(OUT, "rows.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    H.log(f"  -> {row.get('decode_tps')} tok/s sd {row.get('stddev_tps')} trials {row.get('samples_ts')} rss {row['peak_rss_mb']} MB "
          f"loads {row.get('loads_per_layer_token')} hist {row.get('hist')} evictions {row.get('evictions')} wait_ms {row.get('wait_ms')}"
          + ("  !! VOID" if row["void"] else ""))
    if "decode_tps" not in row:
        H.log("  no decode row; tail:\n   " + "\n   ".join(out.splitlines()[-12:]))
    return row


def hashes(arm, v, tag):
    import csv
    with open(os.path.join(OUT, "toklog", f"{arm}_v{v}_{tag}.csv")) as f:
        return [(x["rep"], x["i"], x["logit_hash"]) for x in csv.DictReader(f)]


def gate(v, extra, tag, reps=1, ngen=128, rules=("skliar",)):
    """Gates (a) and (b) on text v, on the timing binary and the timing protocol.
    (a) the streamed arm's selected experts equal the Python reference on the engine's own router logits;
    (b) every decoded token's logits are bit-identical (64-bit hash of the logit vector) between the pool on
        flash (R = 18, evictions > 0) and every expert in RAM (R = 192, evictions = 0), and between two
        streamed runs."""
    import sys
    res = {"text": v, "tag": tag}
    repo = os.path.dirname(HERE)
    for rule, mode in (("skliar", "skliar"), ("ours", "decode")):
        if rule not in rules:
            continue
        ex = (extra + " TEMPORAL_BENCH_LOGIT_HASH=1").strip()
        a = run(f"{rule}_stream", v, reps, ngen, tag + "-gateA", ex, dump=True, cool=False)
        b = run(f"{rule}_ram2p", v, reps, ngen, tag + "-gateB", ex, cool=False)
        c = run(f"{rule}_stream", v, reps, ngen, tag + "-gateC", ex, cool=False)
        ha, hb, hc = hashes(f"{rule}_stream", v, tag + "-gateA"), hashes(f"{rule}_ram2p", v, tag + "-gateB"), hashes(f"{rule}_stream", v, tag + "-gateC")
        dump = os.path.join(OUT, f"route_{rule}_stream_v{v}_{tag}-gateA-dump.bin")
        os.rename(os.path.join(OUT, f"route_{rule}_stream_v{v}_{tag}-gateA.bin"), dump)
        chk = json.loads(subprocess.run([sys.executable, os.path.join(repo, "gpubench", "released", "route_check.py"), "--dump", dump,
                                         "--mode", mode, "--K", str(K), "--E", str(E), "--C", str(K), "--lam", LAM],
                                        capture_output=True, text=True).stdout.strip().splitlines()[-1])
        ties = json.loads(subprocess.run([sys.executable, os.path.join(HERE, "pixel_skliar_ties.py"), dump],
                                         capture_output=True, text=True).stdout.strip().splitlines()[-1])
        os.remove(dump)                                     # 25 MB each; the check results are kept
        r = dict(route_check=chk, ties=ties, tokens_hashed=len(ha),
                 flash_equals_ram=(ha == hb and len(ha) > 0), flash_repeat_equal=(ha == hc),
                 evictions_flash=a.get("evictions"), evictions_ram=b.get("evictions"), fetches_ram=b.get("fetches"),
                 loads_flash=a.get("loads"), loads_ram=b.get("loads"), hist=a.get("hist"))
        r["pass"] = bool(chk["decode_equal"] == chk["decode_tokens"] and (mode == "decode" or chk["loads_equal"] == chk["decode_tokens"])
                         and ties["engine_selection_not_a_valid_topk"] == 0 and r["flash_equals_ram"] and r["flash_repeat_equal"]
                         and (r["evictions_flash"] or 0) > 0 and r["evictions_ram"] == 0)
        res[rule] = r
        H.log(f"gate text {v} {rule}: {'PASS' if r['pass'] else 'FAIL'} decode {chk['decode_equal']}/{chk['decode_tokens']} "
              f"loads_equal {chk['loads_equal']} prompt {chk['prompt_equal']}/{chk['prompt_tokens']} ties_at_cut {ties['tie_at_cut']} "
              f"invalid_topk {ties['engine_selection_not_a_valid_topk']} flash==ram {r['flash_equals_ram']} repeat {r['flash_repeat_equal']} "
              f"evictions {r['evictions_flash']}/{r['evictions_ram']}")
    os.makedirs(os.path.join(OUT, "gates"), exist_ok=True)
    with open(os.path.join(OUT, "gates", f"gate_{tag}.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def run_retrying(arm, v, reps, ngen, tag, extra):
    """As run_arm_retrying: a void attempt stays in rows.jsonl, then a longer rest and another try."""
    rest = 0
    for attempt in range(4):
        row = run(arm, v, reps, ngen, tag, extra)
        H.log(f"  resting {H.REST_S}s"); time.sleep(H.REST_S)
        if row is None or not row["void"] or row.get("no_turnover"):
            return row
        rest += 120
        H.log(f"  void attempt {attempt + 1}; resting {rest}s more"); time.sleep(rest)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["time", "dump", "quick", "gate"])
    ap.add_argument("--arms", nargs="+", default=["skliar_stream"])
    ap.add_argument("--texts", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--ngen", type=int, default=128)
    ap.add_argument("--rules", nargs="+", default=["skliar"])
    ap.add_argument("--tag", default="t1"); ap.add_argument("--env", nargs="*", default=[])
    a = ap.parse_args()
    H.NOROOT = True
    for v in a.texts:
        if a.what == "gate":
            gate(v, " ".join(a.env), a.tag, a.reps, a.ngen, a.rules); continue
        for arm in a.arms:
            if a.what == "quick":                      # exploration only: no cool gate, not for the results table
                run(arm, v, a.reps, a.ngen, a.tag + "-quick", " ".join(a.env), cool=False); time.sleep(20)
            elif a.what == "dump":
                run(arm, v, a.reps, a.ngen, a.tag + "-dump", " ".join(a.env), dump=True, cool=False)
            else:
                run_retrying(arm, v, a.reps, a.ngen, a.tag, " ".join(a.env))
    H.log("DONE")
