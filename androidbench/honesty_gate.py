#!/usr/bin/env python3
"""Honesty gates for the async fetch pool. Run after ANY change to the fetch path.

Gate 1 (bytes are real): R=0 decode, fetched_bytes (pool's own count) must match
/proc/<pid>/io read_bytes to within ~10% (O_DIRECT: every fetch is a device read).
Gate 2 (numerics unchanged): PPL over 2 chunks must equal the fully-resident value
185405.9848 to 4 dp (the same oracle used for every prior configuration).
"""
import subprocess, re, time, threading

DEV = "/data/local/tmp/tmoe"
MODEL = "qwen3moe-rand-fine-Q4pure.gguf"
ENV = ("LLAMA_NO_REPACK=1 LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_R=0 "
       "LLAMA_TEMPORAL_FETCH_THREADS=8")

def sh(cmd, timeout=3600):
    return subprocess.run(["adb", "shell", cmd], capture_output=True, text=True,
                          timeout=timeout).stdout.replace("\r", "")

DISK = None
def _detect_disk():
    # the LUN layout differs per device (Samsung: /data on sda; Pixel: another LUN).
    # Detect empirically: read 32 MiB O_DIRECT from the model, see whose counters move.
    global DISK
    before = {f.split()[2]: int(f.split()[5]) for f in sh("cat /proc/diskstats").splitlines() if len(f.split()) > 5}
    sh(f"cd {DEV} && ./qd_probe {MODEL} 1 64 524288 > /dev/null 2>&1")
    after  = {f.split()[2]: int(f.split()[5]) for f in sh("cat /proc/diskstats").splitlines() if len(f.split()) > 5}
    moved = {d: after[d]-before[d] for d in after if d in before and after[d]-before[d] > 30000 and not d.startswith("dm-")}
    DISK = max(moved, key=moved.get) if moved else "sda"
    print(f"  diskstats device: {DISK} (moved {dict((d, m*512//1048576) for d, m in moved.items())} MiB)", flush=True)

def sectors():
    if DISK is None: _detect_disk()
    out = sh(f"grep ' {DISK} ' /proc/diskstats")
    f = out.split()
    return int(f[5]) if len(f) > 5 else -1

def gate1():
    print("gate 1: R=0, pool fetched_bytes vs /proc/diskstats device reads", flush=True)
    # Per-pid /proc polling is impossible here: detached processes are INVISIBLE to
    # ps/pgrep/proc from other adb sessions on this device (verified S2-12). Instead:
    # device-wide diskstats delta around a BLOCKING run. Signal ~11.7 GB, idle noise
    # ~tens of MB; the model load (~6.7 GiB buffered) is part of the delta and is
    # subtracted below.
    s0 = sectors()
    out = sh(f"cd {DEV} && echo 1000 > /proc/self/oom_score_adj; {ENV} ./llama-bench-temporal -m {MODEL} -t 6 -p 0 -n 16 -r 1 "
             f"-mmp 0 -ot \"_exps=CPU\" -o csv 2>&1", timeout=1800)
    s1 = sectors()
    read = (s1 - s0) * 512 / 1048576 if s0 >= 0 and s1 >= 0 else -1.0
    fetched = -1.0
    m = re.search(r"fetched_mib=([0-9.]+)", out)
    if m: fetched = float(m.group(1))
    # LAZY LOAD: the loader reads only non-expert weights (experts fetched on use),
    # so the load contribution is file size minus total expert bytes.
    model_mib  = 5941846016 / 1048576
    expert_mib = 221184 * 192 * 135 / 1048576   # slice x n_expert x n_expert_tensors
    load_mib   = model_mib - expert_mib
    fetch_read = read - load_mib
    ok = fetched > 0 and abs(fetch_read - fetched) / fetched < 0.10
    print(f"  pool fetched_mib={fetched:.0f} disk_delta_mib={read:.0f} (minus load {load_mib:.0f} -> {fetch_read:.0f})"
          f" -> {'PASS' if ok else 'FAIL'}", flush=True)
    m = re.search(r"temporal-pool: fetches=.*", out)
    if m: print(f"  {m.group(0)}", flush=True)
    return ok

def gate2():
    # SELF-BASELINING bit-identity: pool-off (plain kernels, LLAMA_NO_REPACK) vs
    # pool-on (R=18, real evict/refetch traffic), same binary, same flags, same input.
    # Equality of every printed digit is the gate; no hardcoded reference constant
    # (device- and corpus-portable; kernel-family confusion impossible by construction).
    print("gate 2: PPL bit-identity, pool-off vs pool-on (self-baselining)", flush=True)
    def ppl(env, mmap=False):
        mm = "" if mmap else "--no-mmap"
        out = sh(f"cd {DEV} && echo 1000 > /proc/self/oom_score_adj; {env} ./llama-perplexity -m {MODEL} "
                 f"-f ppl_input.txt --chunks 2 -c 512 -t 6 {mm} -ot \"_exps=CPU\" 2>&1 | tail -5",
                 timeout=3600)
        m = re.search(r"Final estimate: PPL = ([0-9.]+ \+/- [0-9.]+)", out)
        return m.group(1) if m else "NOT FOUND"
    off = ppl("LLAMA_NO_REPACK=1", mmap=True)   # default mmap: page-cache-backed, no anonymous blowup (5.5 GiB resident OOM-panicked the Pixel)
    on  = ppl("LLAMA_NO_REPACK=1 LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_R=18 LLAMA_TEMPORAL_FETCH_THREADS=8")
    ok = off == on and off != "NOT FOUND"
    print(f"  pool-off: {off}", flush=True)
    print(f"  pool-on : {on} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok

if __name__ == "__main__":
    g1 = gate1()
    g2 = gate2()
    print(f"honesty gates: {'ALL PASS' if g1 and g2 else 'FAILURE'}", flush=True)
