#!/usr/bin/env python3
"""Measurement driver for the Samsung llama.cpp benchmark.

All gates from BENCHMARK_GATES.md are enforced here; run_android_bench.sh
delegates to this so there is exactly one implementation of the guardrails.

Two things this adds beyond those gates, both from failures observed on THIS device:

1. CLOCK GATE, not thermal gate. Thermal Status read 0 on this device while
   scaling_max_freq sat at 2227200 against a 4742400 rating -- a 2.1x clock clamp that
   the thermal counter did not report. Every run records the observed max clock, and a
   run whose clock sags below --min-clock-pct of rated is marked degraded.

2. PAGE-CACHE STATE, recorded per run. On Android the "resident vs streamed" axis is
   page-cache-resident vs faulted-from-UFS, and mmap'd weights live in the same RAM we
   claim not to be using. A swap that re-reads a cached page is free and proves nothing.
   Residency is measured with mincore() via pagecache_tool, and cold state is produced
   with posix_fadvise(DONTNEED) and verified, not assumed.
"""
from __future__ import annotations
import argparse, csv, io, json, os, subprocess, sys, threading, time
from dataclasses import dataclass, field, asdict

DEV_DIR = "/data/local/tmp/tmoe"
# Device this campaign was run on. Override for any other handset with
# ANDROID_SERIAL (adb's own variable) or EXPECT_SERIAL, matching run_android_bench.sh.
SERIAL = os.environ.get("ANDROID_SERIAL") or os.environ.get("EXPECT_SERIAL") or "RFGL42B1VLW"
HW_MAX = {0: 3628800, 6: 4742400, 7: 4742400}   # cpuinfo_max_freq, verified on device
TOOL = f"{DEV_DIR}/pagecache_tool"


def sh(cmd: str, timeout: int = 3600) -> str:
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.replace("\r", "")


def host(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.replace("\r", "")


# ---------------------------------------------------------------- device state
def clocks() -> dict[int, int]:
    out = sh("for c in 0 6 7; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq; done")
    vals = [int(x) for x in out.split() if x.isdigit()]
    return dict(zip([0, 6, 7], vals)) if len(vals) == 3 else {}


def thermal() -> int:
    out = sh("dumpsys thermalservice | grep -i 'Thermal Status'")
    try:
        return int(out.strip().split(":")[-1])
    except Exception:
        return -1


def meminfo() -> dict[str, int]:
    out = sh("grep -E '^(MemAvailable|Cached|MemFree|SwapFree):' /proc/meminfo")
    d = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2:
            d[p[0].rstrip(":")] = int(p[1])          # kB
    return d


def battery() -> tuple[bool, int]:
    out = sh("dumpsys battery")
    powered = any(f"{s} powered: true" in out for s in ("AC", "USB", "Wireless"))
    lvl = 0
    for line in out.splitlines():
        if line.strip().startswith("level:"):
            lvl = int(line.split(":")[1]); break
    return powered, lvl


def read_sectors() -> int:
    """Device-wide sectors read from sda (512 B each). Cross-check for per-process io."""
    out = sh("grep ' sda ' /proc/diskstats")
    f = out.split()
    return int(f[5]) if len(f) > 5 else 0


def parse_io_read_bytes() -> int:
    """read_bytes of the last benchmarked process, captured while it was still alive."""
    out = sh(f"cat {DEV_DIR}/last_io.txt 2>/dev/null")
    for line in out.splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return -1


def residency(path: str) -> tuple[float, float]:
    """(resident_mib, pct) of a file currently in the page cache, via mincore()."""
    out = sh(f"{TOOL} resident {path}")
    mib = pct = 0.0
    for tok in out.split():
        if tok.startswith("resident_mib="): mib = float(tok.split("=")[1])
        if tok.startswith("pct="):          pct = float(tok.split("=")[1])
    return mib, pct


def evict(path: str) -> float:
    """fadvise(DONTNEED) then return the VERIFIED post-eviction residency pct."""
    out = sh(f"{TOOL} evict {path}")
    for line in out.splitlines():
        if line.startswith("after_evict"):
            for tok in line.split():
                if tok.startswith("pct="):
                    return float(tok.split("=")[1])
    return -1.0


def warm(path: str) -> float:
    sh(f"{TOOL} read {path}")
    return residency(path)[1]


def wait_until_cool(min_pct: float = 98.0, timeout_s: int = 1200, quiet: bool = False) -> bool:
    """Block until every core's scaling_max_freq is back at its rating AND thermal==0.

    This is the M4/M5 'bench from rest' rule made mechanical. Decay was observed within
    four consecutive invocations on this device, so it is enforced, not advised.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        c, th = clocks(), thermal()
        if c and th == 0 and all(c[k] >= HW_MAX[k] * min_pct / 100 for k in c):
            return True
        if not quiet:
            print(f"    cooling... clocks={c} thermal={th} ({int(time.time()-t0)}s)", flush=True)
        time.sleep(20)
    return False


# ---------------------------------------------------------------- measurement
@dataclass
class Run:
    label: str
    backend: str                  # cpu | opencl
    cache: str                    # warm | cold
    prefill_tps: float | None = None
    decode_tps: float | None = None
    prefill_sd: float = 0.0
    decode_sd: float = 0.0
    clock_before: dict = field(default_factory=dict)
    clock_min: dict = field(default_factory=dict)
    thermal_before: int = -1
    thermal_after: int = -1
    resident_pct_before: float = -1.0
    resident_pct_after: float = -1.0
    mem_avail_before_kb: int = 0
    mem_avail_after_kb: int = 0
    status: str = "ok"
    cmd: str = ""
    read_bytes: int = -1          # bytes actually fetched from the block device
    disk_sectors_read: int = 0    # device-wide cross-check (512 B sectors)


class ClockSampler(threading.Thread):
    """Samples scaling_max_freq during the run; the MINIMUM is what we report.

    A run that starts and ends at full clock can still have spent its middle at half
    clock -- before/after sampling alone would miss exactly the effect that produced the
    428-vs-285 tok/s confusion.
    """
    def __init__(self, period=3.0):
        super().__init__(daemon=True)
        self.period, self.stop_flag, self.min_seen = period, False, {}

    def run(self):
        while not self.stop_flag:
            c = clocks()
            for k, v in c.items():
                if k not in self.min_seen or v < self.min_seen[k]:
                    self.min_seen[k] = v
            time.sleep(self.period)


def measure(model: str, label: str, backend: str, cache: str, threads: int, ubatch: int,
            batch: int, ngl: int, ctx: int, ngen: int, reps: int, pin: str | None,
            prompt: int = 512, extra: str = "", min_clock_pct: float = 90.0,
            cool_first: bool = True, evict_hz: int = 0, temporal_r: int = -1,
            env_extra: str = "") -> Run:
    path = f"{DEV_DIR}/{model}"
    r = Run(label=label, backend=backend, cache=cache)

    if cool_first:
        if not wait_until_cool():
            r.status = "cool_timeout"

    # --- cache state: produce it, then VERIFY it (never assume) -------------------
    if cache == "cold":
        pct = evict(path)
        if pct > 5.0:
            r.status = "evict_failed"       # audit-1: do not record a false cold claim
        r.resident_pct_before = pct
    else:
        r.resident_pct_before = warm(path)

    r.clock_before, r.thermal_before = clocks(), thermal()
    r.mem_avail_before_kb = meminfo().get("MemAvailable", 0)

    # Both CPU arms use the SAME binary (llama-bench-temporal, which carries the
    # LLAMA_TEMPORAL_MMAP patch); only the env var differs. Running an A/B across two
    # different builds would confound the mmap policy with any other build difference.
    if backend == "opencl":
        binary, ldpath, env = "./llama-bench-cl", "LD_LIBRARY_PATH=. ", ""
    else:
        binary, ldpath = "./llama-bench-temporal", ""
        env = ""
        if backend == "cpu-temporal":
            env = "LLAMA_TEMPORAL_MMAP=1 "      # whole-file MADV_RANDOM
        elif backend == "cpu-temporal2":
            env = "LLAMA_TEMPORAL_MMAP=2 "      # experts-only MADV_RANDOM, rest WILLNEED
        if evict_hz:
            env += f"LLAMA_TEMPORAL_EVICT_HZ={evict_hz} "   # forced expert turnover
        if temporal_r >= 0:
            env += f"LLAMA_TEMPORAL_R={temporal_r} "        # deterministic residency regime
        if env_extra:
            env += env_extra.rstrip() + " "                 # slot-pool: ODIRECT/SWAP_PROB
    ldpath = env + ldpath
    pinpfx = f"taskset -p {pin} $$ >/dev/null && " if pin else ""
    cmd = (f"cd {DEV_DIR} && {pinpfx}{ldpath}{binary} -m {model} -ngl {ngl} -t {threads} "
           f"-ub {ubatch} -b {batch} -p {prompt} -n {ngen} -d {ctx} -r {reps} {extra} -o csv")
    r.cmd = cmd

    # Bytes ACTUALLY read from the block device, not bytes we intended to read.
    # /proc/<pid>/io read_bytes is per-process and counts only real block-device traffic
    # (page-cache hits do not increment it), so it distinguishes "streamed from UFS" from
    # "re-read a page we already had". The process is polled because the counter
    # disappears when it exits. diskstats is captured device-wide as a cross-check.
    sectors_before = read_sectors()
    # Resolve the benchmark's own pid by name: `cmd &` backgrounds a SUBSHELL (because
    # cmd starts with `cd ... &&`), and the subshell's io counters are all zero, which
    # silently reads as "no storage traffic" -- a believable wrong number.
    bin_name = binary.lstrip("./")
    wrapped = (f"rm -f {DEV_DIR}/last_io.txt; ({cmd}) & "
               f"sleep 1; BPID=$(pidof {bin_name} | tr ' ' '\\n' | head -1); "
               f"while [ -n \"$BPID\" ] && kill -0 $BPID 2>/dev/null; do "
               f"cat /proc/$BPID/io > {DEV_DIR}/last_io.txt 2>/dev/null; sleep 0.5; done; "
               f"wait")

    sampler = ClockSampler(); sampler.start()
    out = sh(wrapped)
    sampler.stop_flag = True; sampler.join(timeout=5)
    r.clock_min = sampler.min_seen
    r.read_bytes = parse_io_read_bytes()
    r.disk_sectors_read = max(0, read_sectors() - sectors_before)

    r.thermal_after = thermal()
    r.mem_avail_after_kb = meminfo().get("MemAvailable", 0)
    r.resident_pct_after = residency(path)[1]

    rows = [x for x in csv.DictReader(io.StringIO(out)) if x.get("avg_ts")]
    if not rows:
        r.status = "no_output"
        return r
    for row in rows:
        if int(row["n_gen"]) == 0:
            r.prefill_tps, r.prefill_sd = float(row["avg_ts"]), float(row["stddev_ts"])
        else:
            r.decode_tps, r.decode_sd = float(row["avg_ts"]), float(row["stddev_ts"])

    # --- clock gate: the signal Thermal Status failed to give us ------------------
    if r.clock_min:
        sag = min(r.clock_min[k] / HW_MAX[k] for k in r.clock_min)
        if sag < min_clock_pct / 100:
            r.status = f"degraded_clock_{sag*100:.0f}pct"
    if r.thermal_before or r.thermal_after:
        r.status = "throttled" if r.status == "ok" else r.status
    # audit-1: a zero/absent metric is never 'ok'
    if (r.decode_tps or 0) <= 0:
        r.status = "error_zero_decode"
    return r


def to_row(r: Run, model: str) -> list:
    """FLAME-MoE serving-benchmark schema (see emit_row.py for column semantics)."""
    note = (f"[{r.status}] backend={r.backend};cache={r.cache};"
            f"clock_before={r.clock_before};clock_min={r.clock_min};"
            f"thermal={r.thermal_before}->{r.thermal_after};"
            f"resident_pct={r.resident_pct_before:.1f}->{r.resident_pct_after:.1f};"
            f"mem_avail_kb={r.mem_avail_before_kb}->{r.mem_avail_after_kb};"
            f"read_bytes={r.read_bytes};disk_mib={r.disk_sectors_read*512/1048576:.0f};"
            f"serial={SERIAL};soc=SM8850;cmd={r.cmd}")
    return ["decode", model, f"android-{r.backend}", r.label, "", "",
            f"{r.prefill_tps:.3f}" if r.prefill_tps else "",
            f"{r.decode_tps:.4f}" if r.decode_tps else "",
            "", note, f"{r.decode_sd:.4f}", 0]


def preflight(model: str, require_battery: int = 80):
    if f"{SERIAL}\tdevice" not in host(["adb", "devices"]).replace("  ", "\t"):
        if SERIAL not in host(["adb", "devices"]):
            sys.exit(f"FATAL: expected serial {SERIAL} not attached (M10)")
    powered, lvl = battery()
    if not powered:
        sys.exit("FATAL: device not on external power (M1)")
    if lvl < require_battery:
        sys.exit(f"FATAL: battery {lvl}% < {require_battery}% (M1) - charge and wait")
    if not sh(f"test -f {DEV_DIR}/{model} && echo yes").strip():
        sys.exit(f"FATAL: {model} not on device")
    print(f"preflight ok: battery={lvl}% powered={powered}", flush=True)
    return lvl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="adhoc")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "opencl"])
    ap.add_argument("--cache", default="warm", choices=["warm", "cold"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--ubatch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--ngen", type=int, default=128)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pin", default=None)
    ap.add_argument("--min-battery", type=int, default=80)
    ap.add_argument("--no-cool", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    preflight(a.model, a.min_battery)
    r = measure(a.model, a.label, a.backend, a.cache, a.threads, a.ubatch, a.batch,
                a.ngl, a.ctx, a.ngen, a.reps, a.pin, a.prompt, cool_first=not a.no_cool)
    print(json.dumps(asdict(r), indent=2))
    if a.out:
        with open(a.out, "a", newline="") as f:
            csv.writer(f).writerow(to_row(r, a.model))
