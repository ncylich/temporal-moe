#!/usr/bin/env python3
"""One slot-pool run with faithful io accounting.

Launches llama-bench-temporal detached with a pidfile, polls /proc/<pid>/io from an
on-device loop (0.2 s) into a trajectory file, and reports: CSV rows, temporal-pool
stats, and final read_bytes (real block-device traffic) for cross-checking against
fetched_mib. Usage: pool_probe.py "<env>" [extra llama-bench args]
"""
import subprocess, sys, time

DEV = "/data/local/tmp/tmoe"
MODEL = "qwen3moe-rand-fine-Q4_K_M.gguf"

def sh(cmd, timeout=1800):
    return subprocess.run(["adb", "shell", cmd], capture_output=True, text=True,
                          timeout=timeout).stdout.replace("\r", "")

def main():
    env = sys.argv[1]
    extra = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "-t 6 -p 32 -n 16 -r 1"
    # launcher: pidfile + exec; poller: on-device loop appending timestamped read_bytes
    sh(f"cd {DEV} && rm -f pool.pid pool.log pool.io.traj; "
       f"nohup sh -c 'echo $$ > pool.pid; exec env {env} ./llama-bench-temporal "
       f"-m {MODEL} -mmp 0 -ot \"_exps=CPU\" {extra} -o csv' > pool.log 2>&1 &")
    time.sleep(1)
    sh(f"cd {DEV} && nohup sh -c 'P=$(cat pool.pid); "
       f"while [ -d /proc/$P ]; do "
       f"echo \"$(cat /proc/uptime | cut -d\" \" -f1) $(grep read_bytes /proc/$P/io | cut -d\" \" -f2)\" >> pool.io.traj; "
       f"sleep 0.2; done' >/dev/null 2>&1 &")
    # wait for completion
    while True:
        alive = sh(f"P=$(cat {DEV}/pool.pid 2>/dev/null); [ -n \"$P\" ] && [ -d /proc/$P ] && echo yes").strip()
        if alive != "yes":
            break
        time.sleep(3)
    time.sleep(1)
    log = sh(f"cat {DEV}/pool.log")
    for line in log.splitlines():
        if line.startswith('"0badc06a') or "temporal-pool" in line:
            print(line)
    traj = sh(f"cat {DEV}/pool.io.traj").splitlines()
    if traj:
        t0, b0 = traj[0].split()
        tN, bN = traj[-1].split()
        print(f"io: first_sample={int(b0)/1e6:.0f} MB, final={int(bN)/1e6:.0f} MB, "
              f"span={float(tN)-float(t0):.1f}s, n={len(traj)}")
        # crude per-phase rates
        prev_t, prev_b = float(t0), int(b0)
        for line in traj[:: max(1, len(traj)//12)][1:]:
            t, b = line.split()
            dt, db = float(t) - prev_t, int(b) - prev_b
            print(f"  t+{float(t)-float(t0):6.1f}s  {int(b)/1e6:8.0f} MB  ({db/dt/1e6:6.0f} MB/s)")
            prev_t, prev_b = float(t), int(b)

if __name__ == "__main__":
    main()
