#!/usr/bin/env python3
"""What request sizes does the block layer ACTUALLY issue for our expert fetches?

The UFS driver's monitor counts only requests whose size == monitor_chunk_size, so
one run per candidate size. This is driver-level ground truth (the same instrument
that proved the fused 648 KiB read becomes 512+136 in S3-33), not inference.

It also reports, per size:
    OVERLAP = read_req_latency_sum / read_total_busy
      ~1  the device services these requests one at a time
      ~N  N of them are genuinely in flight together
Units of every latency field are MICROseconds (verified against a QD1 probe:
40 requests, sum 11741 us, busy 11712 us, overlap 1.00).

    ./ufs_chunk_sweep.py [--uring] [size ...]
"""
import subprocess, sys, re

DEV   = "/data/local/tmp/tmoe"
MODEL = "qwen3moe-rand-fine-Q4pure.gguf"
SIDE  = "qwen3moe-rand-fine-Q4pure-repacked.bin"
UFS   = "/sys/devices/platform/13200000.ufs/monitor"

PROD = (f"LLAMA_TEMPORAL_REPACK=1 LLAMA_TEMPORAL_REPACK_FILE={SIDE} "
        "LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_MADV_FREE=1 "
        "LLAMA_TEMPORAL_R=18 LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6 "
        "LLAMA_TEMPORAL_SPIN_US=5000 LLAMA_TEMPORAL_FETCHPROF=1")

SIZES = [110592, 221184, 663552, 524288, 4096, 131072]


def sh(cmd, timeout=1800):
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.replace("\r", "") + r.stderr.replace("\r", "")


def run(chunk, env, ntok=24):
    s = ("#!/system/bin/sh\n"
         f"echo 0 > {UFS}/monitor_enable\necho {chunk} > {UFS}/monitor_chunk_size\n"
         f"cd {DEV}\necho 1000 > /proc/self/oom_score_adj\n")
    for kv in (PROD + " " + env).split():
        s += f"export {kv}\n"
    s += (f"echo 1 > {UFS}/monitor_enable\n"
          f"./llama-bench-temporal -m {MODEL} -t 4 -p 0 -n {ntok} -r 1 -mmp 0 "
          f"-ot _exps=CPU -o csv\n"
          # counters MUST be read while still enabled: writing monitor_enable=0
          # zeroes the whole monitor struct in the driver, and every field reads 0.
          f"echo UFSMON nr=$(cat {UFS}/read_nr_requests) "
          f"sum=$(cat {UFS}/read_req_latency_sum) avg=$(cat {UFS}/read_req_latency_avg) "
          f"min=$(cat {UFS}/read_req_latency_min) max=$(cat {UFS}/read_req_latency_max) "
          f"busy=$(cat {UFS}/read_total_busy) sectors=$(cat {UFS}/read_total_sectors)\n"
          f"echo 0 > {UFS}/monitor_enable\n")
    with open("/tmp/_chunk.sh", "w") as f:
        f.write(s)
    subprocess.run(["adb", "push", "/tmp/_chunk.sh", f"{DEV}/_chunk.sh"], capture_output=True)
    out = sh(f"su -c 'sh {DEV}/_chunk.sh'", timeout=1800)
    dec = None
    for line in out.splitlines():
        f = line.split(",")
        if len(f) > 40 and f[0].strip('"') == "0badc06a":
            try: dec = float(f[-2].strip('"'))
            except ValueError: pass
    m = re.search(r"UFSMON (.*)", out)
    if not m:
        print(f"{chunk:>8} : no UFSMON line"); return
    d = dict(kv.split("=") for kv in m.group(1).split())
    nr, sm, busy = int(d["nr"]), int(d["sum"]), int(d["busy"])
    prof = re.search(r"wall/fetch=(\d+)us", out)
    w = prof.group(1) if prof else "?"
    if nr == 0:
        print(f"{chunk:>8} ({chunk//1024:>4} KiB) : nr=0            decode={dec} wall/fetch={w}us")
        return
    print(f"{chunk:>8} ({chunk//1024:>4} KiB) n={ntok:<3}: nr={nr:<7} dev_avg={sm/nr:>6.0f}us "
          f"min={int(d['min']):>5}us max={int(d['max']):>6}us sum={sm/1000:>7.0f}ms "
          f"busy={busy/1000:>7.0f}ms OVERLAP={sm/busy:>4.2f} MiB={int(d['sectors'])//2048:<6} "
          f"decode={dec} wall/fetch={w}us")


if __name__ == "__main__":
    argv = sys.argv[1:]
    env = "LLAMA_TEMPORAL_URING=1" if "--uring" in argv else ""
    ntoks = [24]
    if "--n" in argv:
        i = argv.index("--n")
        ntoks = [int(x) for x in argv[i + 1].split(",")]
        argv = argv[:i] + argv[i + 2:]
    sizes = [int(a) for a in argv if a.isdigit()] or SIZES
    print(f"fetch path: {'io_uring' if env else 'pread pool'}")
    for c in sizes:
        for n in ntoks:
            run(c, env, ntok=n)
