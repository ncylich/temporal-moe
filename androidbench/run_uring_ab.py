#!/usr/bin/env python3
"""A/B/A/B the io_uring fetch path against the blocking-pread worker pool, in-engine.

Both arms run as ROOT: io_uring is EPERM in the `shell` SELinux domain on this
device, so the domain would otherwise be confounded with the fetch path (pitfall
#6: both arms must differ in exactly one thing). A `preadroot` arm is included so
the root-vs-shell effect is measured separately rather than assumed to be zero.

Per-arm output: decode tok/s (+/- sd from llama-bench's own reps), the engine's
FETCHPROF line, and -- when --ufs is passed -- the UFS driver's own view of the
same run, which is what actually answers "does the device overlap the burst?":

    overlap = read_req_latency_sum / read_total_busy
      ~1  => the device services the burst one request at a time
      ~6  => six requests are genuinely in flight together

Usage:  ./run_uring_ab.py [--ufs] [--reps N] arm [arm ...]
        arms: pread uring uringsq
"""
import subprocess, sys, time, re

DEV   = "/data/local/tmp/tmoe"
MODEL = "qwen3moe-rand-fine-Q4pure.gguf"
SIDE  = "qwen3moe-rand-fine-Q4pure-repacked.bin"
K     = 18
UFS   = "/sys/devices/platform/13200000.ufs/monitor"
CHUNK = 110592          # 108 KiB: one part of a split=2 expert fetch

RATED = {0: 1950000, 4: 2600000, 7: 2147000}

PROD = (f"LLAMA_TEMPORAL_REPACK=1 LLAMA_TEMPORAL_REPACK_FILE={SIDE} "
        "LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_MADV_FREE=1 "
        f"LLAMA_TEMPORAL_R={K} LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6 "
        "LLAMA_TEMPORAL_SPIN_US=5000 LLAMA_TEMPORAL_FETCHPROF=1")

# label, extra env, run-as-root, device-side pre-command (block-layer knobs)
Q = "/sys/block/sda/queue"
ARMS = {
    "pread":   ("A pread pool (root)",    "",                        True,  ""),
    "uring":   ("B io_uring 1-submitter", "LLAMA_TEMPORAL_URING=1",  True,  ""),
    "uringsq": ("C io_uring SQPOLL",      "LLAMA_TEMPORAL_URING=1 LLAMA_TEMPORAL_URING_SQPOLL=1", True, ""),
    "preadsh": ("Z pread pool (shell)",   "",                        False, ""),
    # sda runs mq-deadline with rq_affinity=2; dm-63 (what the ledger recorded) is
    # scheduler=none, so the BACKING device's scheduler was never actually checked.
    "noop":    ("D pread, sda sched=none", "",                       True,  f"echo none > {Q}/scheduler"),
    "rqaff0":  ("E pread, rq_affinity=0", "",                        True,  f"echo 0 > {Q}/rq_affinity"),
    "uringnoop": ("F uring, sda sched=none", "LLAMA_TEMPORAL_URING=1", True, f"echo none > {Q}/scheduler"),
}
STOCK_SDA = f"echo mq-deadline > {Q}/scheduler; echo 2 > {Q}/rq_affinity"


def sh(cmd, timeout=3600):
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.replace("\r", "") + r.stderr.replace("\r", "")


def clocks():
    out = sh("for c in 0 4 7; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq; done")
    v = [int(x) for x in out.split() if x.isdigit()]
    return dict(zip([0, 4, 7], v)) if len(v) == 3 else {}


def wait_cool(timeout_s=600):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        c = clocks()
        if c and all(c[k] >= RATED[k] * 0.98 for k in c):
            return True
        time.sleep(20)
    return False


def push_script(env, ntok, reps, ufs, pre=""):
    s = "#!/system/bin/sh\n"
    # every arm restores stock first, then applies its own knob, so an arm never
    # inherits the previous arm's block-layer state
    s += STOCK_SDA + "\n"
    if pre:
        s += pre + "\n"
    s += (f"echo SDAQ sched=$(cat {Q}/scheduler) rq_affinity=$(cat {Q}/rq_affinity) "
          f"max_sectors_kb=$(cat {Q}/max_sectors_kb)\n")
    if ufs:
        s += f"echo 0 > {UFS}/monitor_enable\necho {CHUNK} > {UFS}/monitor_chunk_size\n"
    s += f"cd {DEV}\necho 1000 > /proc/self/oom_score_adj\n"
    for kv in (PROD + " " + env).split():
        s += f"export {kv}\n"
    if ufs:
        s += f"echo 1 > {UFS}/monitor_enable\n"
    s += (f"./llama-bench-temporal -m {MODEL} -t 4 -p 0 -n {ntok} -r {reps} -mmp 0 "
          f"-ot _exps=CPU -o csv\n")
    if ufs:
        # counters MUST be read while still enabled: writing monitor_enable=0 zeroes
        # the driver's whole monitor struct and every field then reads back 0.
        s += (f"echo UFSMON nr=$(cat {UFS}/read_nr_requests) "
              f"lat_sum=$(cat {UFS}/read_req_latency_sum) "
              f"lat_avg=$(cat {UFS}/read_req_latency_avg) "
              f"lat_min=$(cat {UFS}/read_req_latency_min) "
              f"lat_max=$(cat {UFS}/read_req_latency_max) "
              f"busy=$(cat {UFS}/read_total_busy) "
              f"sectors=$(cat {UFS}/read_total_sectors)\n"
              f"echo 0 > {UFS}/monitor_enable\n")
    with open("/tmp/_ab_arm.sh", "w") as f:
        f.write(s)
    subprocess.run(["adb", "push", "/tmp/_ab_arm.sh", f"{DEV}/_ab_arm.sh"], capture_output=True)


def run_arm(key, ntok, reps, ufs, tag=""):
    label, env, as_root, pre = ARMS[key]
    if not wait_cool():
        print(f"  {label}: COOL TIMEOUT (flagged)", flush=True)
    push_script(env, ntok, reps, ufs and as_root, pre if as_root else "")
    cmd = f"su -c 'sh {DEV}/_ab_arm.sh'" if as_root else f"sh {DEV}/_ab_arm.sh"
    t0 = time.time()
    out = sh(cmd, timeout=3600)
    wall = time.time() - t0
    dec = sd = None
    for line in out.splitlines():
        f = line.split(",")
        if len(f) > 40 and f[0].strip('"') == "0badc06a":
            try:
                dec, sd = float(f[-2].strip('"')), float(f[-1].strip('"'))
            except ValueError:
                pass
    prof = ""
    m = re.search(r"temporal-fetchprof: .*", out)
    if m:
        prof = m.group(0)
    pool = ""
    m = re.search(r"temporal-pool: fetches=.*", out)
    if m:
        pool = m.group(0)
    mon = ""
    m = re.search(r"UFSMON .*", out)
    if m:
        mon = m.group(0)
    sdaq = ""
    m = re.search(r"SDAQ .*", out)
    if m:
        sdaq = m.group(0)
    print(f"{label+tag:30s} decode={dec} sd={sd} wall={wall:.0f}s  {sdaq}", flush=True)
    if prof: print(f"   {prof[:150]}", flush=True)
    if pool: print(f"   {pool[:150]}", flush=True)
    if mon:
        d = dict(kv.split("=") for kv in mon.split()[1:])
        nr, lsum, busy = int(d["nr"]), int(d["lat_sum"]), int(d["busy"])
        if nr and busy:
            # every driver latency field is in MICROseconds (verified: QD1 probe,
            # 40 requests, sum 11741 us, busy 11712 us, overlap 1.00)
            print(f"   UFS nr={nr} dev_lat_avg={lsum/nr:.0f}us "
                  f"min={int(d['lat_min'])}us max={int(d['lat_max'])}us "
                  f"busy={busy/1e3:.0f}ms OVERLAP={lsum/busy:.2f} "
                  f"MiB={int(d['sectors'])//2048}", flush=True)
    if dec is None:
        print("   !! no decode row -- raw tail:", flush=True)
        print("   " + "\n   ".join(out.splitlines()[-12:]), flush=True)
    return dec


if __name__ == "__main__":
    argv = sys.argv[1:]
    ufs = "--ufs" in argv
    argv = [a for a in argv if a != "--ufs"]
    reps = 3
    if "--reps" in argv:
        i = argv.index("--reps"); reps = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    ntok = 48
    if "--n" in argv:
        i = argv.index("--n"); ntok = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    seq = argv or ["pread", "uring", "pread", "uring"]
    res = {}
    for i, a in enumerate(seq):
        d = run_arm(a, ntok, reps, ufs, tag=f" #{i+1}")
        res.setdefault(a, []).append(d)
    print("\n=== summary (interleaved) ===")
    for a, v in res.items():
        v = [x for x in v if x]
        if v:
            print(f"{ARMS[a][0]:30s} {[round(x,2) for x in v]}  mean={sum(v)/len(v):.2f}")
