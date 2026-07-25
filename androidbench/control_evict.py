#!/usr/bin/env python3
"""Positive control for the two-step (madvise+fadvise) residency controller.

Question: does LLAMA_TEMPORAL_R actually evict expert pages, or is it the §4 silent
no-op again?  Router-independent design: warm the whole file, start the engine with
R=0 (streamed regime), and watch from OUTSIDE the process:

  - file page-cache residency via pagecache_tool mincore (fadvise eviction is visible
    system-wide; if the controller works, residency collapses from ~75% toward the
    ~18.5% non-expert fraction)
  - /proc/<pid>/io read_bytes (only real block-device reads count; page-cache hits
    don't increment it)
  - decode tok/s vs the 3.75 tok/s streaming roofline (513 MiB/token / 1.92 GB/s)

Arms: (A) TEMPORAL_MMAP=2, no R -> controller off, expect residency stays high,
read_bytes ~ 0, fast decode.  (B) TEMPORAL_MMAP=2, R=0 -> controller on, expect
residency collapse + read_bytes growth + decode <= 3.75 tok/s.
The pair discriminates; a single arm does not.
"""
import subprocess, sys, time, json

DEV = "/data/local/tmp/tmoe"
MODEL = "qwen3moe-rand-fine-Q4_K_M.gguf"
BIN = "llama-bench-temporal"

def sh(cmd, timeout=3600):
    return subprocess.run(["adb", "shell", cmd], capture_output=True, text=True,
                          timeout=timeout).stdout.replace("\r", "")

def residency():
    out = sh(f"cd {DEV} && ./pagecache_tool resident {MODEL}")
    for tok in out.split():
        if tok.startswith("pct="):
            return float(tok.split("=")[1])
    return -1.0

def run_arm(name, env, ngen=16, prompt=32, reps=1):
    print(f"\n=== arm {name}: env='{env}' ===", flush=True)
    r0 = residency()
    print(f"residency before launch: {r0:.1f}%", flush=True)
    log = f"{DEV}/control_{name}.log"
    # The subshell writes its own pid then execs the benchmark, so bench.pid IS the
    # benchmark pid -- pidof fails on the >15-char comm and pgrep -f can self-match.
    cmd = (f"cd {DEV} && rm -f control_{name}.log bench.pid && "
           f"( echo $$ > bench.pid; exec env {env} ./{BIN} -m {MODEL} -t 6 "
           f"-p {prompt} -n {ngen} -r {reps} -v -o csv ) > control_{name}.log 2>&1 &")
    sh(cmd)
    # poll pid, io, residency until the process exits
    t0 = time.time()
    traj = []
    last_io = -1
    pid = None
    while True:
        p = sh(f"P=$(cat {DEV}/bench.pid 2>/dev/null); [ -n \"$P\" ] && [ -d /proc/$P ] && echo $P").strip()
        if p:
            pid = p.split()[0]
            io = sh(f"cat /proc/{pid}/io 2>/dev/null")
            rb = -1
            for line in io.splitlines():
                if line.startswith("read_bytes:"):
                    rb = int(line.split()[1])
            last_io = rb if rb >= 0 else last_io
            res = residency()
            traj.append((round(time.time()-t0, 1), rb, res))
            print(f"  t={traj[-1][0]:7.1f}s  read_bytes={rb/1e6:9.1f} MB  file_resident={res:5.1f}%", flush=True)
            time.sleep(3)
        else:
            if pid is not None or time.time()-t0 > 30:
                break
            time.sleep(1)
    out = sh(f"cat {log}")
    print(f"--- engine output ({name}) ---")
    print(out)
    print(f"final read_bytes (last alive sample): {last_io/1e6:.1f} MB")
    print(f"residency after exit: {residency():.1f}%")
    return last_io, out, traj

if __name__ == "__main__":
    arms = sys.argv[1:] or ["A", "B"]
    # warm the file once at the start; arm A shouldn't disturb it much
    if "A" in arms or "B" in arms:
        print("warming file...", flush=True)
        sh(f"cd {DEV} && ./pagecache_tool read {MODEL}", timeout=600)
        print(f"residency after warm: {residency():.1f}%", flush=True)
    # LLAMA_NO_REPACK=1 everywhere: with repack on, decode reads 5.4 GiB of anonymous
    # repacked copies and file-page eviction measures nothing (found 2026-07-23).
    if "A" in arms:
        run_arm("A", "LLAMA_NO_REPACK=1 LLAMA_TEMPORAL_MMAP=2")
    if "B" in arms:
        run_arm("B", "LLAMA_NO_REPACK=1 LLAMA_TEMPORAL_MMAP=2 LLAMA_TEMPORAL_R=0")
