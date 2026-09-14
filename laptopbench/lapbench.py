#!/usr/bin/env python3
"""lapbench.py -- one driver for the laptop temporal-MoE decode benchmark (PLAN.md section 8).

Runs under Windows Python. Native-Windows stages run directly; WSL2 stages run through
`wsl -d Ubuntu-24.04 -e bash -lc "bash <script>"`, one generated script per command so the
exact text that ran is on disk next to its log. Memory caps are applied by rewriting
%UserProfile%\\.wslconfig and `wsl --shutdown`, then waiting for the distro to come back and
reading MemTotal inside it.

Ported from androidbench/bench.py (measure() discipline: verify state, never assume it;
poll the process for its io and status counters while it is alive; a zero metric is never
"ok") and androidbench/emit_row.py (row schema). The adb transport and the Android probes
are replaced; nothing else is loosened.

Invariants enforced in code (PLAN.md section 8):
  1. `arms` and `sweep` refuse a bench binary whose sha256 is not stamped as gated in
     gates.json (and the stamp must say every gate passed).
  2. A run whose pool counters disagree with the arm requested is not recorded as a row;
     it goes to refused.jsonl with the counters and the reason (pitfall #18).
  3. `--set KEY=0` is refused: several engine flags are presence-parsed (pitfall #17).
  4. Every row carries env, cap (requested and in effect), flags, binary hash, clock probe,
     peak RSS, pool counters and --tag.

Stages: check probe compute build gates arms sweep session pack report
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# ----------------------------------------------------------------------------- constants
DISTRO = "Ubuntu-24.04"
WIN_ROOT = Path(r"C:\tmoe")
WSL_ROOT = "~/tmoe"                                    # expanded inside the distro
WSL_UNC = Path(r"\\wsl.localhost") / DISTRO            # read WSL files from Windows
MODEL = "qwen3moe-rand-fine-Q4pure.gguf"
MODEL_SHA256 = "d8a3bdf4a9c4a1563ad694718a44e155b7588e5ca47d3ee569bf9a8b8c3a2229"
SIDEFILE = "qwen3moe-rand-fine-Q4pure-repacked.bin"
FORK_COMMIT = "61f6d1b4"
OFFICIAL_TAG = "b9959"                                 # upstream base of the fork, for native T1

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
LOGS = RESULTS / "logs"
PPL_INPUT = REPO / "androidbench" / "ppl_input.txt"

# model geometry (PLAN.md 1.1): 45 layers, 192 experts, 18 active, one expert = 3 x 216 KiB
N_LAYER, N_EXPERT, K_ACTIVE = 45, 192, 18
SLICE_BYTES = 221184
EXPERT_BYTES = 3 * SLICE_BYTES                         # 663552 = 648 KiB
EXPERT_MIB = EXPERT_BYTES / 1048576                    # 0.6328
NONEXPERT_BYTES_HINT = None                            # filled from the model file size
BYTES_PER_TOKEN = N_LAYER * EXPERT_BYTES               # 28.5 MiB, one swap per layer per token

# Pixel production flags (androidbench/ENGINE_FLAGS.md). TWOPASS is added per arm.
PROD_FLAGS = {
    "LLAMA_TEMPORAL_REPACK": "1",
    "LLAMA_TEMPORAL_REPACK_FILE": None,                # filled with the side-file path
    "LLAMA_TEMPORAL_ODIRECT": "1",
    "LLAMA_TEMPORAL_MADV_FREE": "1",
    "LLAMA_TEMPORAL_SPLIT": "2",
    "LLAMA_TEMPORAL_FETCH_THREADS": "6",
    "LLAMA_TEMPORAL_SPIN_US": "5000",
}
# flags read as getenv(...) != NULL: setting them to 0 turns them ON (pitfall #17)
PRESENCE_FLAGS = {"LLAMA_TEMPORAL_TWOPASS", "LLAMA_TEMPORAL_FUSED", "LLAMA_TEMPORAL_FETCHPROF",
                  "LLAMA_TEMPORAL_URING", "LLAMA_TEMPORAL_URING_SQPOLL",
                  "LLAMA_TEMPORAL_URING_IOPOLL", "LLAMA_TEMPORAL_TRACE", "LLAMA_NO_REPACK",
                  "LLAMA_TEMPORAL_REPACK"}

ENGINE_ARGS = "-t 4 -p 0 -n 128 -r 8 -mmp 0 -ot _exps=CPU"
WARMUP_ARGS = "-t 4 -p 0 -n 32 -r 1 -mmp 0 -ot _exps=CPU"
PROBE_ARGS = "-t 4 -p 0 -n 256 -r 4 -mmp 0"          # ~1024 resident tokens, ~20 s at 50 tok/s
CLOCK_TOL = 0.03                                        # PLAN 5.4: >3% under first probe = degraded

ARMS = {
    # key: (tier, label, R, twopass, cap)
    "a": ("ceiling", "ceiling", 192, False, "12G"),
    "b": ("resident_control", "resident_control", 192, True, "12G"),
    "c": ("deploy", "deploy_R18", 18, True, "4G"),
    "e": ("floor", "floor_R18", 18, False, "4G"),
}
CAP_MB = {"12G": 12288, "4G": 4096, "2.5G": 2560}

DRY = False
FORCE = False
_CMD_SEQ = 0
_TAG = "untagged"
PROVISIONAL = ""                                       # --provisional NOTE: stamped in every row/json
NATIVE = False                                         # --env linux: run shell stages locally, no wsl -e
CURRENT_CAP: str | None = None                         # linux: cap applied per run via systemd-run
IS_LINUX = sys.platform.startswith("linux")


# ----------------------------------------------------------------------------- utilities
def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def say(msg: str) -> None:
    print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg: str, code: int = 2) -> None:
    print(f"REFUSED: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def sha256_file(path: Path, cache: bool = True) -> str:
    """sha256 of a file, cached by (size, mtime) so a 6 GB model is not rehashed per batch."""
    cache_file = RESULTS / "hash_cache.json"
    st = path.stat()
    key = str(path)
    if cache and cache_file.exists():
        c = json.loads(cache_file.read_text())
        e = c.get(key)
        if e and e["size"] == st.st_size and abs(e["mtime"] - st.st_mtime) < 1e-6:
            return e["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    d = h.hexdigest()
    if cache:
        c = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        c[key] = {"size": st.st_size, "mtime": st.st_mtime, "sha256": d}
        RESULTS.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(c, indent=1))
    return d


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))


def jload(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def decode_out(b: bytes) -> str:
    """wsl.exe's own messages are UTF-16; everything the distro prints is UTF-8."""
    if b[:64].count(b"\x00") > 8:
        try:
            return b.decode("utf-16-le", errors="replace").replace("\r", "")
        except Exception:
            pass
    return b.decode("utf-8", errors="replace").replace("\r", "")


def log_dir() -> Path:
    d = LOGS / _TAG
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- transports
class Result:
    def __init__(self, rc: int, out: str, err: str, wall: float, cmd: str, script: str = ""):
        self.rc, self.out, self.err, self.wall, self.cmd, self.script = rc, out, err, wall, cmd, script

    @property
    def text(self) -> str:
        return self.out + "\n" + self.err


def run_win(args: list[str], timeout: int = 7200, label: str = "") -> Result:
    """Run a native Windows command; the exact argv is printed and logged."""
    global _CMD_SEQ
    _CMD_SEQ += 1
    cmdline = subprocess.list2cmdline(args)
    say(f"[win {_CMD_SEQ:04d}{' ' + label if label else ''}] {cmdline}")
    if DRY:
        return Result(0, "", "", 0.0, cmdline)
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, timeout=timeout)
    r = Result(p.returncode, decode_out(p.stdout), decode_out(p.stderr), time.time() - t0, cmdline)
    (log_dir() / f"win_{_CMD_SEQ:04d}{'_' + label if label else ''}.log").write_text(
        f"$ {cmdline}\nrc={r.rc} wall={r.wall:.1f}s\n--- stdout ---\n{r.out}\n--- stderr ---\n{r.err}\n",
        encoding="utf-8")
    return r


def run_ps(script: str, timeout: int = 600, label: str = "") -> Result:
    return run_win(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout, label)


def run_wsl(script: str, timeout: int = 7200, label: str = "", login: bool = True) -> Result:
    """Run a bash script in the Linux environment under test.

    WSL2: the script is written to C:\\tmoe\\logs\\<tag>\\ (LF endings) and executed as
    `wsl -d Ubuntu-24.04 -e bash -lc "bash /mnt/c/...sh"`.
    Linux native (--env linux): the script is written to ~/tmoe/logs/<tag>/ and executed as
    `bash <script>` directly; no wsl.exe anywhere.
    Either way the exact text that ran is on disk and is what --dry-run prints."""
    global _CMD_SEQ
    _CMD_SEQ += 1
    name = f"cmd_{_CMD_SEQ:04d}{'_' + label if label else ''}.sh"
    body = "#!/bin/bash\nset -o pipefail\n" + script.strip("\n") + "\n"
    if NATIVE:
        sdir = Path(wsl_home()) / "tmoe" / "logs" / _TAG
        spath = sdir / name
        args = ["bash", str(spath)]
        kind = "linux"
    else:
        sdir = WIN_ROOT / "logs" / _TAG
        spath = sdir / name
        mnt = f"/mnt/c/tmoe/logs/{_TAG}/{name}"
        args = ["wsl", "-d", DISTRO, "-e", "bash", "-lc" if login else "-c", f"bash {mnt}"]
        kind = "wsl"
    cmdline = subprocess.list2cmdline(args)
    say(f"[{kind} {_CMD_SEQ:04d}{' ' + label if label else ''}] {cmdline}\n" +
        "\n".join("      | " + l for l in body.splitlines()))
    if DRY:
        return Result(0, "", "", 0.0, cmdline, body)
    sdir.mkdir(parents=True, exist_ok=True)
    spath.write_bytes(body.encode("utf-8"))
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, timeout=timeout)
    r = Result(p.returncode, decode_out(p.stdout), decode_out(p.stderr), time.time() - t0, cmdline, body)
    (log_dir() / f"{kind}_{_CMD_SEQ:04d}{'_' + label if label else ''}.log").write_text(
        f"$ {cmdline}\n--- script ---\n{body}\nrc={r.rc} wall={r.wall:.1f}s\n--- stdout ---\n{r.out}\n--- stderr ---\n{r.err}\n",
        encoding="utf-8")
    return r


_HOME: str | None = None


def wsl_home() -> str:
    """$HOME of the Linux environment under test (the distro's, or the live USB user's)."""
    global _HOME
    if _HOME:
        return _HOME
    if NATIVE:
        _HOME = os.path.expanduser("~") if IS_LINUX else "/home/" + (os.environ.get("USERNAME") or "user")
        return _HOME
    if DRY:
        return "/home/mohsen"
    r = run_wsl("echo $HOME", label="home")
    h = r.out.strip().splitlines()[-1] if r.out.strip() else ""
    if not h.startswith("/"):
        die(f"cannot resolve $HOME in {DISTRO}: {r.text[-300:]}")
    _HOME = h
    return h


def wsl_path(rel: str) -> str:
    """~/tmoe/<rel> spelled absolutely (bash does not expand ~ inside quotes)."""
    return f"{wsl_home()}/tmoe/{rel}".rstrip("/")


def wsl_read(rel: str) -> str:
    """Read a file under ~/tmoe: locally on Linux native, through \\\\wsl.localhost from Windows."""
    if NATIVE:
        p = Path(wsl_home()) / "tmoe" / rel
    else:
        p = WSL_UNC / wsl_home().lstrip("/").replace("/", "\\") / "tmoe" / rel.replace("/", "\\")
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


# ----------------------------------------------------------------------------- WSL memory cap
WSLCONFIG = Path(os.environ["USERPROFILE"]) / ".wslconfig"


def wslconfig_text(cap: str) -> str:
    return (f"# written by laptopbench/lapbench.py {now()} (cap {cap})\n"
            f"[wsl2]\nmemory={CAP_MB[cap]}MB\nswap=0\n")


def current_cap_mb() -> int | None:
    if not WSLCONFIG.exists():
        return None
    m = re.search(r"^memory\s*=\s*(\d+)\s*(MB|GB)?", WSLCONFIG.read_text(), re.M | re.I)
    if not m:
        return None
    n = int(m.group(1))
    return n * 1024 if (m.group(2) or "").upper() == "GB" else n


def wsl_memtotal_mb() -> int:
    r = run_wsl("grep -E '^(MemTotal|SwapTotal):' /proc/meminfo; swapon --show --noheadings | wc -l",
                label="memtotal", login=False)
    m = re.search(r"MemTotal:\s+(\d+) kB", r.out)
    return int(m.group(1)) // 1024 if m else -1


def wsl_restart() -> None:
    run_win(["wsl", "--shutdown"], label="shutdown")
    if DRY:
        return
    t0 = time.time()
    while time.time() - t0 < 180:
        p = subprocess.run(["wsl", "-d", DISTRO, "-e", "true"], capture_output=True)
        if p.returncode == 0:
            time.sleep(3)
            return
        time.sleep(3)
    die(f"{DISTRO} did not come back within 180 s of wsl --shutdown")


def set_cap(cap: str) -> dict:
    """Apply a memory cap and return what is in effect.

    WSL2: rewrite .wslconfig, restart the VM, read MemTotal inside it (PLAN 5.3).
    Linux native: the cap is a cgroup limit applied per engine run through
    `systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0`; wrap.sh reads back the
    scope's memory.max and the row refuses if it is not the cap requested."""
    global CURRENT_CAP
    if cap not in CAP_MB:
        die(f"unknown cap {cap}; use one of {list(CAP_MB)}")
    want = CAP_MB[cap]
    if NATIVE:
        CURRENT_CAP = cap
        say(f"cap: {cap} will be applied per run as systemd-run MemoryMax={want}M MemorySwapMax=0")
        return {"cap": cap, "cap_mb": want, "memtotal_mb": None, "mechanism": "systemd-run --scope MemoryMax"}
    if current_cap_mb() != want or "swap=0" not in (WSLCONFIG.read_text() if WSLCONFIG.exists() else ""):
        say(f"cap: writing {WSLCONFIG} memory={want}MB swap=0 and restarting {DISTRO}")
        if not DRY:
            if WSLCONFIG.exists() and not (WIN_ROOT / "wslconfig.orig").exists():
                shutil.copy(WSLCONFIG, WIN_ROOT / "wslconfig.orig")
            WSLCONFIG.write_text(wslconfig_text(cap))
        wsl_restart()
    else:
        # config already says so; verify the running VM agrees, else restart it
        mt = wsl_memtotal_mb()
        if not DRY and not (0.80 * want <= mt <= 1.02 * want):
            say(f"cap: VM MemTotal {mt} MB disagrees with configured {want} MB; restarting")
            wsl_restart()
    mt = wsl_memtotal_mb()
    if not DRY and not (0.80 * want <= mt <= 1.02 * want):
        die(f"cap {cap}: MemTotal inside {DISTRO} is {mt} MB, expected ~{want} MB")
    return {"cap": cap, "cap_mb": want, "memtotal_mb": mt}


# ----------------------------------------------------------------------------- host state
def win_state() -> dict:
    """Laptop state per PLAN 5.4 (M1/M9), all read programmatically, never from Task Manager."""
    ps = r"""
$o = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes' -ErrorAction SilentlyContinue
$b = Get-CimInstance BatteryStatus -Namespace root\wmi -ErrorAction SilentlyContinue | Select-Object -First 1
$bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
$sc = (powercfg /getactivescheme) -join ' '
$st = (powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String 'Current AC Power Setting Index').Line.Trim()
$hi = (powercfg /query SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE | Select-String 'Current AC Power Setting Index').Line.Trim()
$cpu = (Get-CimInstance Win32_Processor).LoadPercentage
$free = [math]::Round((Get-PSDrive C).Free/1GB,1)
$ws = (Get-Service WSearch -ErrorAction SilentlyContinue).Status
$mp = Get-MpComputerStatus -ErrorAction SilentlyContinue
$ex = try { (Get-MpPreference -ErrorAction Stop).ExclusionPath -join ';' } catch { 'unreadable' }
[pscustomobject]@{
  overlay_ac = $o.ActiveOverlayAcPowerScheme; scheme = $sc; standby_ac = $st; hibernate_ac = $hi
  power_online = $b.PowerOnline; battery_pct = $bat.EstimatedChargeRemaining
  cpu_load_pct = $cpu; c_free_gb = $free; wsearch = "$ws"
  defender_rtp = $mp.RealTimeProtectionEnabled; defender_exclusions = "$ex"
} | ConvertTo-Json -Compress
"""
    r = run_ps(ps, label="winstate")
    if DRY:
        return {"dry": True}
    try:
        d = json.loads(r.out.strip().splitlines()[-1])
    except Exception:
        die(f"cannot read host state: {r.text[-500:]}")
    d["best_performance"] = (d.get("overlay_ac") or "").lower() == "ded574b5-45a0-4f42-8737-46345c09c238"
    d["standby_ac_zero"] = d.get("standby_ac", "").endswith("0x00000000")
    d["hibernate_ac_zero"] = d.get("hibernate_ac", "").endswith("0x00000000")
    d["ts"] = now()
    return d


def linux_state() -> dict:
    """Laptop state on the live USB: AC from /sys/class/power_supply, load, free disk, governor."""
    r = run_wsl("""
for p in /sys/class/power_supply/*; do n=$(basename $p); [ -f $p/online ] && echo "online_$n=$(cat $p/online)"; [ -f $p/status ] && echo "status_$n=$(cat $p/status)"; [ -f $p/capacity ] && echo "capacity_$n=$(cat $p/capacity)"; done
echo load="$(cut -d' ' -f1 /proc/loadavg)"
echo governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)
echo max_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null)
echo free_gb=$(df -BG """ + WSL_ROOT + """ | tail -1 | awk '{print $4}' | tr -d G)
echo swap_lines=$(swapon --show --noheadings | wc -l)
echo systemd_run=$(command -v systemd-run || echo missing)
""", label="linuxstate", login=False)
    if DRY:
        return {"dry": True, "linux": True}
    d = dict(l.split("=", 1) for l in r.out.splitlines() if "=" in l)
    ac = any(k.startswith("online_") and v.strip() == "1" for k, v in d.items())
    st = {"linux": True, "power_online": ac, "cpu_load_pct": int(float(d.get("load", "0")) * 100 / 8),
          "c_free_gb": float(d.get("free_gb", "0")), "governor": d.get("governor"), "max_khz": d.get("max_khz"),
          "swap_lines": int(d.get("swap_lines", "0")), "systemd_run": d.get("systemd_run"),
          "best_performance": True, "standby_ac_zero": True, "hibernate_ac_zero": True,
          "overlay_ac": "n/a (linux native)", "scheme": f"governor={d.get('governor')}",
          "defender_exclusions": "n/a (linux native)", "raw": d, "ts": now()}
    return st


def defender_exclusions_ok(state: dict) -> tuple[bool, str]:
    if NATIVE:
        return True, "n/a (linux native, no Defender in the path)"
    """Exclusion lists need elevation to read. Accept (1) a readable list covering C:\\tmoe and
    the distro's VHDX directory, or (2) C:\\tmoe\\DEFENDER_EXCLUSIONS.txt written by hand listing
    the paths that were excluded (recorded in every row as 'attested')."""
    vhdx = distro_basepath()
    need = [str(WIN_ROOT).lower(), vhdx.lower()]
    ex = (state.get("defender_exclusions") or "").lower()
    # unelevated, Get-MpPreference returns the literal "N/A: Must be an administrator to view exclusions"
    if ex and ex != "unreadable" and "administrator" not in ex and not ex.startswith("n/a"):
        ok = all(any(p.startswith(n) for p in ex.split(";")) for n in need)
        return ok, f"read:{ex}"
    att = WIN_ROOT / "DEFENDER_EXCLUSIONS.txt"
    if att.exists():
        t = att.read_text().lower()
        ok = all(n in t for n in need)
        return ok, f"attested:{att.read_text().strip().replace(chr(10), ';')}"
    return False, "unreadable (needs elevation) and no C:\\tmoe\\DEFENDER_EXCLUSIONS.txt attestation"


def distro_basepath() -> str:
    r = run_ps(r"(Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss\*' | "
               r"Where-Object DistributionName -eq '" + DISTRO + "').BasePath", label="basepath")
    if DRY:
        return r"C:\Users\mohsen\AppData\Local\wsl\{distro}"
    return r.out.strip().splitlines()[-1].replace("\\\\?\\", "") if r.out.strip() else ""


# ----------------------------------------------------------------------------- check
def check(env: str, quick: bool = False) -> dict:
    """Refuses on any failure unless --force (then the row says forced)."""
    st = linux_state() if NATIVE else win_state()
    fails = []
    if DRY:
        say("check: dry run, state not evaluated")
        return st
    if NATIVE and st.get("systemd_run") == "missing":
        fails.append("systemd-run missing: the MemoryMax cap cannot be applied")
    if NATIVE and st.get("swap_lines"):
        fails.append("swap is active on the live USB; `sudo swapoff -a` first (pitfall #20)")
    if not st.get("power_online"):
        fails.append("not on AC (M1)")
    if not st["best_performance"]:
        fails.append(f"power mode overlay is not Best performance: {st.get('overlay_ac')}")
    if not (st["standby_ac_zero"] and st["hibernate_ac_zero"]):
        fails.append(f"AC standby/hibernate timeouts not 0: {st.get('standby_ac')} / {st.get('hibernate_ac')}")
    if (st.get("cpu_load_pct") or 0) > 25:
        fails.append(f"host busy: CPU load {st['cpu_load_pct']}% (M9)")
    if (st.get("c_free_gb") or 0) < 20:
        fails.append(f"C: free {st['c_free_gb']} GB < 20")
    ok, how = defender_exclusions_ok(st)
    st["defender_exclusions_check"] = how
    if not ok:
        fails.append(f"Defender exclusions do not cover C:\\tmoe and the VHDX dir: {how}")
    if not quick:
        if not NATIVE:
            mp = WIN_ROOT / "models" / MODEL
            if not mp.exists():
                fails.append(f"model missing: {mp}")
            else:
                h = sha256_file(mp)
                st["model_sha256_win"] = h
                if h != MODEL_SHA256:
                    fails.append(f"model sha256 mismatch on Windows: {h}")
        if env in ("wsl", "linux", "all"):
            r = run_wsl(f"""
set -e
cd {wsl_path('llama.cpp')} && echo fork=$(git rev-parse --short=8 HEAD) dirty=$(git status --porcelain | wc -l)
df -BG {wsl_path('')} | tail -1 | awk '{{print "wsl_free_gb="$4}}'
grep -E '^(MemTotal|SwapTotal):' /proc/meminfo
echo nproc=$(nproc) load="$(cut -d' ' -f1-3 /proc/loadavg)"
M={wsl_path('models/' + MODEL)}
if [ -f "$M" ]; then echo model_present=1; else echo model_present=0; fi
""", label="wslcheck", login=False)
            m = re.search(r"fork=(\w+) dirty=(\d+)", r.out)
            if not m or not m.group(1).startswith(FORK_COMMIT):
                fails.append(f"WSL fork is not at {FORK_COMMIT}: {r.out.strip()[:200]}")
            elif int(m.group(2)) != 0:
                fails.append("WSL fork checkout is dirty")
            if "model_present=1" not in r.out:
                if NATIVE:
                    fails.append(f"model missing at {wsl_path('models/' + MODEL)} (PLAN 5.4: read it from the NTFS volume through ntfs3, or copy it)")
                else:
                    say("check: model absent in WSL; copying from C:\\tmoe\\models")
                    run_wsl(f"mkdir -p {wsl_path('models')} && cp /mnt/c/tmoe/models/{MODEL} {wsl_path('models/' + MODEL)}",
                            label="copymodel", timeout=3600, login=False)
            hr = run_wsl(f"sha256sum {wsl_path('models/' + MODEL)} | cut -d' ' -f1", label="wslsha", timeout=1800, login=False)
            hw = hr.out.strip().splitlines()[-1] if hr.out.strip() else ""
            st["model_sha256_wsl"] = hw
            if hw != MODEL_SHA256:
                fails.append(f"model sha256 mismatch in WSL: {hw}")
            st["wsl_check_raw"] = r.out.strip()
    st["check_fails"] = fails
    st["forced"] = bool(fails) and FORCE
    st["provisional"] = PROVISIONAL
    if fails and not FORCE:
        die("check failed:\n  - " + "\n  - ".join(fails))
    if fails:
        say("check: FAILURES OVERRIDDEN by --force: " + "; ".join(fails))
    else:
        say("check ok: AC, Best performance, standby 0, exclusions " + how.split(":")[0] +
            f", load {st.get('cpu_load_pct')}%")
    return st


# ----------------------------------------------------------------------------- probe (T0)
FIO_WIN = WIN_ROOT / "tools" / "fio" / "fio" / "fio.exe"
BS_LIST = ["4k", "32k", "108k", "216k", "648k", "1024k"]
QD_LIST = [4, 8, 12]


def fio_common(bs: str, qd: int, engine: str, fname: str, runtime: int = 20) -> str:
    return (f"--name=qd{qd}_{bs} --filename={fname} --size=8G --rw=randread --direct=1 "
            f"--iodepth={qd} --bs={bs} --ioengine={engine} --runtime={runtime} --time_based "
            f"--norandommap --randrepeat=0 --group_reporting --output-format=json")


def fio_parse(text: str) -> dict:
    j = json.loads(text[text.index("{"):text.rindex("}") + 1])
    job = j["jobs"][0]
    rd = job["read"]
    pct = rd.get("clat_ns", {}).get("percentile", {})
    return {"lat_mean_us": rd["lat_ns"]["mean"] / 1e3,
            "lat_p50_us": pct.get("50.000000", 0) / 1e3,
            "lat_p99_us": pct.get("99.000000", 0) / 1e3,
            "bw_MBps": rd["bw_bytes"] / 1e6, "iops": rd["iops"],
            "io_bytes": rd["io_bytes"], "total_ios": rd["total_ios"],
            "runtime_ms": rd["runtime"], "error": job.get("error", 0), "fio_version": j.get("fio version")}


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot) if ss_tot else 1.0


def probe_env(env: str) -> dict:
    out = {"env": env, "ts": now(), "sizes": {}, "qd": {}, "cmds": []}
    if env == "windows":
        pdir = WIN_ROOT / "probe"
        pdir.mkdir(exist_ok=True)
        fname = "C\\:\\tmoe\\probe\\probe.bin"                  # fio's Windows drive-colon escape
        engine = "windowsaio"
        def fio(args: str, label: str) -> Result:
            return run_win([str(FIO_WIN), "--thread"] + args.split(), timeout=600, label=label)
        # pre-write, non-sparse, random data, then verify the size on disk
        prep = (f"--name=prep --filename={fname} --size=8G --rw=write --bs=1M --direct=1 "
                f"--ioengine={engine} --fallocate=none --refill_buffers --end_fsync=1 --output-format=json")
        if not (pdir / "probe.bin").exists() or (pdir / "probe.bin").stat().st_size < 8 * 2 ** 30:
            r = fio(prep, "prep"); out["cmds"].append(r.cmd)
        if not DRY:
            out["file_bytes"] = (pdir / "probe.bin").stat().st_size
            r = run_ps(f"fsutil sparse queryflag {pdir / 'probe.bin'}", label="sparse")
            out["sparse_query"] = r.text.strip()[-200:]
    else:
        fname = wsl_path("probe/probe.bin")
        engine = "libaio"
        def fio(args: str, label: str) -> Result:
            return run_wsl(f"mkdir -p {wsl_path('probe')}\nfio {args}", timeout=600, label=label, login=False)
        prep = (f"--name=prep --filename={fname} --size=8G --rw=write --bs=1M --direct=1 "
                f"--ioengine={engine} --fallocate=none --refill_buffers --end_fsync=1 --output-format=json")
        r = run_wsl(f"test -f {fname} && stat -c '%s %b' {fname} || echo missing", label="probestat", login=False)
        need = True
        if not DRY and "missing" not in r.out:
            sz, blk = r.out.split()[:2]
            need = int(sz) < 8 * 2 ** 30 or int(blk) * 512 < 0.99 * int(sz)
        if need:
            r = fio(prep, "prep"); out["cmds"].append(r.cmd)
        if not DRY:
            r = run_wsl(f"stat -c 'size=%s blocks512=%b' {fname}; filefrag {fname} 2>/dev/null | tail -1",
                        label="probestat2", login=False)
            out["file_stat"] = r.out.strip()
            m = re.search(r"size=(\d+) blocks512=(\d+)", r.out)
            if m and int(m.group(2)) * 512 < 0.99 * int(m.group(1)):
                die(f"probe file is sparse in {env}: {r.out.strip()} (S3-32 hole trap)")
    for bs in BS_LIST:
        r = fio(fio_common(bs, 1, engine, fname), f"fio_{bs}_qd1")
        out["cmds"].append(r.cmd)
        if not DRY:
            out["sizes"][bs] = fio_parse(r.out)
            s = out["sizes"][bs]
            say(f"  {env} {bs:>6} QD1: mean {s['lat_mean_us']:.1f} us  p50 {s['lat_p50_us']:.1f}  bw {s['bw_MBps']:.0f} MB/s  ios {s['total_ios']}")
    for qd in QD_LIST:
        r = fio(fio_common("216k", qd, engine, fname), f"fio_216k_qd{qd}")
        out["cmds"].append(r.cmd)
        if not DRY:
            out["qd"][str(qd)] = fio_parse(r.out)
            s = out["qd"][str(qd)]
            say(f"  {env} 216k QD{qd}: mean {s['lat_mean_us']:.1f} us  bw {s['bw_MBps']:.0f} MB/s")
    # DRAM number
    if env == "windows":
        ps = ("$n=268435456; $a=New-Object byte[] $n; $b=New-Object byte[] $n; "
              "[Buffer]::BlockCopy($a,0,$b,0,$n); $sw=[Diagnostics.Stopwatch]::StartNew(); "
              "for($i=0;$i -lt 16;$i++){[Buffer]::BlockCopy($a,0,$b,0,$n)}; $sw.Stop(); "
              "'dram_copy_MBps=' + [math]::Round(16*256/$sw.Elapsed.TotalSeconds)")
        r = run_ps(ps, label="dram")
        out["cmds"].append(r.cmd)
        if not DRY:
            m = re.search(r"dram_copy_MBps=(\d+)", r.out)
            out["dram"] = {"method": "single-thread .NET BlockCopy 256 MiB x16 (read+write counted once)",
                           "MBps": int(m.group(1)) if m else -1}
    else:
        r = run_wsl("sysbench memory --memory-block-size=1M --memory-total-size=16G --memory-oper=read --threads=1 run",
                    label="dram", login=False)
        out["cmds"].append(r.cmd)
        if not DRY:
            m = re.search(r"\(([\d.]+) MiB/sec\)", r.out)
            out["dram"] = {"method": "sysbench memory read 1M blocks 16G single thread",
                           "MBps": float(m.group(1)) * 1.048576 if m else -1}
    if DRY:
        return out
    xs = [float(b[:-1]) for b in BS_LIST]
    ys = [out["sizes"][b]["lat_mean_us"] for b in BS_LIST]
    a, b, r2 = linfit(xs, ys)
    ys50 = [out["sizes"][bb]["lat_p50_us"] for bb in BS_LIST]
    a50, b50, r250 = linfit(xs, ys50)
    out["fit"] = {"fixed_us": a, "per_kib_us": b, "r2": r2,
                  "fixed_us_p50": a50, "per_kib_us_p50": b50, "r2_p50": r250,
                  "bw_MBps_by_qd": {"1": out["sizes"]["216k"]["bw_MBps"],
                                    **{q: out["qd"][q]["bw_MBps"] for q in out["qd"]}}}
    out["cached"] = out["sizes"]["4k"]["lat_mean_us"] < 30.0
    out["fio_version"] = out["sizes"]["4k"]["fio_version"]
    say(f"  {env} fit: latency = {a:.1f} us + {b:.3f} us/KiB (r2 {r2:.3f}); 4k QD1 {ys[0]:.1f} us"
        f"{'  ** CACHED, environment fails T0 **' if out['cached'] else ''}")
    return out


def probe(env: str) -> None:
    envs = ["windows", "wsl"] if env == "all" else [env]
    res = jload(RESULTS / "probe.json", {})
    for e in envs:
        if e == "wsl":
            set_cap("12G")
        res[e] = probe_env(e)
        res[e]["tag"] = _TAG
        res[e]["provisional"] = PROVISIONAL
        if not DRY:
            jdump(RESULTS / "probe.json", res)
    if not DRY:
        say(f"probe.json written: {RESULTS / 'probe.json'}")


# ----------------------------------------------------------------------------- compute (T1)
def parse_bench_csv(text: str) -> dict:
    rows = [x for x in csv.DictReader(io.StringIO(text)) if x.get("avg_ts")]
    d = {}
    for row in rows:
        if int(row["n_gen"]) == 0:
            d["prefill_tps"], d["prefill_sd"] = float(row["avg_ts"]), float(row["stddev_ts"])
        else:
            d["decode_tps"], d["decode_sd"] = float(row["avg_ts"]), float(row["stddev_ts"])
            d["n_gen"], d["n_ubatch"], d["n_depth"] = int(row["n_gen"]), row.get("n_ubatch", ""), row.get("n_depth", "")
            d["n_threads"] = row.get("n_threads", "")
    return d


def official_bench_exe() -> Path:
    d = WIN_ROOT / "tools" / f"llamacpp-{OFFICIAL_TAG}"
    exe = d / "llama-bench.exe"
    if not exe.exists():
        die(f"official {OFFICIAL_TAG} Windows CPU build not found at {exe} (see ledger for the download step)")
    return exe


def compute(env: str) -> None:
    envs = ["windows", "wsl"] if env == "all" else [env]
    res = jload(RESULTS / "compute.json", {})
    for e in envs:
        entry = {"env": e, "ts": now(), "tag": _TAG, "runs": {}, "provisional": PROVISIONAL}
        if e == "windows":
            exe = official_bench_exe()
            entry["binary"] = str(exe)
            entry["binary_sha256"] = None if DRY else sha256_file(exe)
            entry["binary_note"] = f"official ggml-org release {OFFICIAL_TAG} win-cpu-x64 (no native compiler on this machine)"
            for t in (4, 8):
                r = run_win([str(exe), "-m", str(WIN_ROOT / "models" / MODEL), "-t", str(t), "-p", "0", "-n", "128",
                             "-r", "8", "-mmp", "0", "-o", "csv"], timeout=3600, label=f"t1_t{t}")
                if not DRY:
                    entry["runs"][f"t{t}"] = {**parse_bench_csv(r.out), "wall_s": r.wall, "rc": r.rc, "cmd": r.cmd,
                                              "stderr_tail": r.err[-400:]}
                    say(f"  windows -t {t}: {entry['runs'][f't{t}'].get('decode_tps')} tok/s")
        else:
            set_cap("12G")
            b = wsl_path("bin/llama-bench-temporal")
            entry["binary"] = b
            entry["binary_note"] = f"fork tree built in {e} with GGML_NATIVE=ON, no LLAMA_TEMPORAL_* env (stock behaviour)"
            hr = run_wsl(f"test -f {b} && sha256sum {b} | cut -d' ' -f1 || echo MISSING", label="benchsha", login=False)
            if not DRY and (not hr.out.strip() or "MISSING" in hr.out):
                die(f"compute: {b} is missing; run `build --env {e}` first (it copies the fork's llama-bench there)")
            entry["binary_sha256"] = None if DRY else hr.out.strip().splitlines()[-1]
            for t in (4, 8):
                r = run_wsl(f"cd {wsl_path('')} && env -u LLAMA_TEMPORAL_R {b} -m {wsl_path('models/' + MODEL)} "
                            f"-t {t} -p 0 -n 128 -r 8 -mmp 0 -o csv", timeout=3600, label=f"t1_t{t}", login=False)
                if not DRY:
                    entry["runs"][f"t{t}"] = {**parse_bench_csv(r.out), "wall_s": r.wall, "rc": r.rc, "cmd": r.cmd,
                                              "stderr_tail": r.err[-400:]}
                    say(f"  wsl -t {t}: {entry['runs'][f't{t}'].get('decode_tps')} tok/s")
        res[e] = entry
        if not DRY:
            jdump(RESULTS / "compute.json", res)
    if not DRY:
        decide()


def decide() -> None:
    """PLAN 5.2: WSL2 within 15% of native on QD1 latency and within 5% on compute -> WSL2."""
    p, c = jload(RESULTS / "probe.json", {}), jload(RESULTS / "compute.json", {})
    if not ("windows" in p and "wsl" in p and "windows" in c and "wsl" in c):
        say("decide: need probe and compute for both environments first")
        return
    d = {"ts": now()}
    lw, lwsl = p["windows"]["sizes"]["4k"]["lat_mean_us"], p["wsl"]["sizes"]["4k"]["lat_mean_us"]
    d["t0_4k_qd1_us"] = {"windows": lw, "wsl": lwsl, "wsl_over_native": lwsl / lw}
    d["t0_cached"] = {"windows": p["windows"]["cached"], "wsl": p["wsl"]["cached"]}
    d["t1"] = {}
    for t in ("t4", "t8"):
        a = c["windows"]["runs"].get(t, {}).get("decode_tps")
        b = c["wsl"]["runs"].get(t, {}).get("decode_tps")
        d["t1"][t] = {"windows": a, "wsl": b, "wsl_over_native": (b / a) if a and b else None}
    t0_ok = (not d["t0_cached"]["wsl"]) and (not d["t0_cached"]["windows"]) and lwsl <= 1.15 * lw and 60 <= lwsl <= 200
    t1_ok = all(v["wsl_over_native"] is not None and v["wsl_over_native"] >= 0.95 for v in d["t1"].values())
    d["t0_pass"], d["t1_pass"] = t0_ok, t1_ok
    d["decision"] = "wsl" if (t0_ok and t1_ok) else "linux-native (rule 2): needs Mohsen; if unavailable, rule 3 (Windows port), not this session"
    jdump(RESULTS / "decision.json", d)
    say(f"decision: T0 wsl/native {lwsl / lw:.3f} (wsl {lwsl:.1f} us, native {lw:.1f} us) pass={t0_ok}; "
        f"T1 " + ", ".join(f"{k} {v['wsl_over_native']:.3f}" for k, v in d['t1'].items() if v['wsl_over_native']) +
        f" pass={t1_ok} -> {d['decision']}")


# ----------------------------------------------------------------------------- build
WRAP_SH = r'''#!/bin/bash
# wrap.sh <outprefix> <cmd...>: run cmd, poll /proc/<pid>/{status,io} while alive, keep peaks.
out="$1"; shift
"$@" > "$out.stdout" 2> "$out.stderr" &
pid=$!
hwm=0; swp=0; rss=0
while kill -0 "$pid" 2>/dev/null; do
  s=$(cat /proc/$pid/status 2>/dev/null)
  h=$(printf '%s\n' "$s" | awk '/^VmHWM/{print $2}')
  w=$(printf '%s\n' "$s" | awk '/^VmSwap/{print $2}')
  r=$(printf '%s\n' "$s" | awk '/^VmRSS/{print $2}')
  [ -n "$h" ] && [ "$h" -gt "$hwm" ] && hwm=$h
  [ -n "$w" ] && [ "$w" -gt "$swp" ] && swp=$w
  [ -n "$r" ] && [ "$r" -gt "$rss" ] && rss=$r
  cat /proc/$pid/io > "$out.io" 2>/dev/null
  sleep 0.25
done
wait "$pid"; rc=$?
cg=$(awk -F: '{print $3}' /proc/self/cgroup | head -1)
mm=$(cat /sys/fs/cgroup$cg/memory.max 2>/dev/null || echo unknown)
echo "rc=$rc vmhwm_kb=$hwm vmswap_peak_kb=$swp vmrss_peak_kb=$rss pid=$pid cgroup=$cg memory_max=$mm" > "$out.meta"
exit $rc
'''


def build(env: str) -> None:
    if env not in ("wsl", "linux"):
        die("build: --env wsl or linux only (no native Windows compiler; the Windows port is out of scope)")
    llama = wsl_path("llama.cpp")
    bind = wsl_path("bin")
    r = run_wsl(f"""
set -e
cd {llama}
git rev-parse --short=8 HEAD; git status --porcelain | wc -l
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF > {wsl_path('logs')}/build_configure.log 2>&1
cmake --build build --config Release -j 8 --target llama-bench llama-perplexity > {wsl_path('logs')}/build.log 2>&1
mkdir -p {bind}
cp build/bin/llama-bench {bind}/llama-bench-temporal
cp build/bin/llama-perplexity {bind}/llama-perplexity
cat > {bind}/wrap.sh <<'WRAPEOF'
{WRAP_SH}WRAPEOF
chmod +x {bind}/wrap.sh
sha256sum {bind}/llama-bench-temporal {bind}/llama-perplexity {bind}/wrap.sh
grep -m1 -E 'CMAKE_C_COMPILER:|CMAKE_CXX_COMPILER:' build/CMakeCache.txt || true
gcc --version | head -1
grep -c . {wsl_path('logs')}/build.log
""", timeout=3600, label="build")
    if DRY:
        return
    if r.rc != 0:
        die(f"build failed: {r.text[-800:]}")
    hashes = dict(re.findall(r"^([0-9a-f]{64})\s+\S+/(\S+)$", r.out, re.M))
    inv = {v: k for k, v in hashes.items()}
    info = {"ts": now(), "tag": _TAG, "env": env, "provisional": PROVISIONAL, "fork_commit": r.out.strip().splitlines()[0],
            "bench_sha256": inv.get("llama-bench-temporal"), "ppl_sha256": inv.get("llama-perplexity"),
            "wrap_sha256": inv.get("wrap.sh"), "compiler": [l for l in r.out.splitlines() if "gcc" in l.lower()][:2],
            "cmake": "cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF; "
                     "cmake --build build --config Release -j 8 --target llama-bench llama-perplexity"}
    # side-file: built here, never copied (PLAN section 3, G3)
    side = wsl_path("models/" + SIDEFILE)
    model = wsl_path("models/" + MODEL)
    r2 = run_wsl(f"""
set -e
cd {wsl_path('')}
test -f {model}
if [ ! -f {side} ]; then
  LLAMA_TEMPORAL_REPACK_DUMP={side} {bind}/llama-bench-temporal -m {model} 2>&1 | tail -5
fi
stat -c 'size=%s blocks512=%b' {side}
sha256sum {side} | cut -d' ' -f1
""", timeout=3600, label="sidefile")
    if r2.rc != 0:
        info["sidefile_error"] = r2.text[-800:]
        jdump(RESULTS / "build.json", info)
        die(f"side-file dump failed: {r2.text[-800:]}")
    m = re.search(r"size=(\d+) blocks512=(\d+)", r2.out)
    info["sidefile"] = {"path": side, "size": int(m.group(1)), "blocks512": int(m.group(2)),
                        "on_disk_fraction": int(m.group(2)) * 512 / int(m.group(1)),
                        "sha256": r2.out.strip().splitlines()[-1], "dump_tail": r2.out[-600:]}
    jdump(RESULTS / "build.json", info)
    say(f"build ok: bench {info['bench_sha256'][:12]} ppl {info['ppl_sha256'][:12]} side-file "
        f"{info['sidefile']['size'] / 2 ** 30:.2f} GiB ({info['sidefile']['on_disk_fraction']:.3f} on disk)")


# ----------------------------------------------------------------------------- engine runs (WSL)
POOL_RE = re.compile(r"temporal-pool: fetches=(\d+) fetched_mib=([\d.]+) evictions=(\d+) tensors=(\d+) "
                     r"hook_calls=(\d+) hook_miss=(\d+) avg_fetch_us=([\d.]+) qwait_hi_us=([\d.]+) "
                     r"wait_ms=([\d.]+) odirect=(\d) workers=(\d+) sibling=(\d) split=(\d+)")


def parse_pool(text: str) -> dict | None:
    m = POOL_RE.search(text)
    if not m:
        return None
    keys = ["fetches", "fetched_mib", "evictions", "tensors", "hook_calls", "hook_miss", "avg_fetch_us",
            "qwait_hi_us", "wait_ms", "odirect", "workers", "sibling", "split"]
    d = {k: (float(v) if "." in v else int(v)) for k, v in zip(keys, m.groups())}
    s = re.search(r"temporal-pool: ENFORCE on, swaps=(\d+)", text)
    d["swaps"] = int(s.group(1)) if s else 0
    a = re.search(r"temporal-pool: active, R=(\d+)", text)
    d["R_active"] = int(a.group(1)) if a else None
    fp = re.search(r"temporal-fetchprof: .*", text)
    d["fetchprof"] = fp.group(0) if fp else None
    return d


def env_string(flags: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in flags.items() if v is not None)


def arm_flags(R: int, twopass: bool, overrides: dict | None = None, no_repack: bool = False) -> dict:
    f = dict(PROD_FLAGS)
    f["LLAMA_TEMPORAL_REPACK_FILE"] = wsl_path("models/" + SIDEFILE)
    if no_repack:
        f.pop("LLAMA_TEMPORAL_REPACK"); f.pop("LLAMA_TEMPORAL_REPACK_FILE")
        f["LLAMA_NO_REPACK"] = "1"
    f["LLAMA_TEMPORAL_R"] = str(R)
    if twopass:
        f["LLAMA_TEMPORAL_TWOPASS"] = "1"
    for k, v in (overrides or {}).items():
        if v is None:
            f.pop(k, None)
        else:
            f[k] = v
    return f


def parse_overrides(sets: list[str]) -> dict:
    """--set KEY=VAL, repeatable. KEY=0 is refused for every key (pitfall #17); KEY= (empty)
    removes the flag, which is the only way to turn a presence-parsed flag off."""
    o = {}
    for s in sets or []:
        if "=" not in s:
            die(f"--set needs KEY=VAL: {s}")
        k, v = s.split("=", 1)
        if v.strip() == "0":
            die(f"--set {k}=0 refused: {k} would be ENABLED by that if it is presence-parsed "
                f"(pitfall #17); use --set {k}= to remove it")
        o[k] = None if v == "" else v
    return o


def engine_run(label: str, flags: dict | None, args: str, timeout: int = 7200, drop_caches: bool = False,
               binary: str = "bin/llama-bench-temporal") -> dict:
    """One engine invocation inside WSL under wrap.sh; returns everything the row needs."""
    binp = wsl_path(binary)
    model = wsl_path("models/" + MODEL)
    outp = wsl_path(f"logs/{_TAG}/{label}")
    envs = ("env " + env_string(flags) + " ") if flags else ""
    pre = "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'\n" if drop_caches else ""
    # Linux native: the cap is a transient cgroup scope around this one run (PLAN 5.3)
    capw = ""
    if NATIVE and CURRENT_CAP:
        capw = f"systemd-run --user --scope --quiet -p MemoryMax={CAP_MB[CURRENT_CAP]}M -p MemorySwapMax=0 "
    # block device backing the model's filesystem (sdd in WSL2, nvme0n1pN on the live USB)
    script = f"""
mkdir -p {wsl_path('logs/' + _TAG)}
cd {wsl_path('')}
DEV=$(df --output=source {model} | tail -1 | sed 's#^/dev/##')
{pre}S0=$(awk -v d="$DEV" '$3==d{{print $6}}' /proc/diskstats); grep -E '^(MemAvailable|MemFree|Cached):' /proc/meminfo
{capw}{wsl_path('bin/wrap.sh')} {outp} {envs}{binp} -m {model} {args} -o csv
RC=$?
S1=$(awk -v d="$DEV" '$3==d{{print $6}}' /proc/diskstats)
echo "disk_dev=$DEV disk_sectors_delta=$((S1-S0))"
grep -E '^(MemAvailable|MemFree|Cached):' /proc/meminfo
cat {outp}.meta; cat {outp}.io
echo '--- stderr tail ---'; tail -n 25 {outp}.stderr
echo '--- stdout ---'; cat {outp}.stdout
exit $RC
"""
    r = run_wsl(script, timeout=timeout, label=label, login=False)
    d = {"label": label, "flags": flags or {}, "args": args, "cmd": r.cmd, "script": r.script,
         "rc": r.rc, "wall_s": r.wall, "ts": now(), "binary": binp}
    if DRY:
        return d
    stderr = wsl_read(f"logs/{_TAG}/{label}.stderr")
    stdout = wsl_read(f"logs/{_TAG}/{label}.stdout")
    (log_dir() / f"{label}.stderr").write_text(stderr, encoding="utf-8")
    (log_dir() / f"{label}.stdout").write_text(stdout, encoding="utf-8")
    d.update(parse_bench_csv(stdout))
    d["pool"] = parse_pool(stderr)
    m = re.search(r"rc=(-?\d+) vmhwm_kb=(\d+) vmswap_peak_kb=(\d+) vmrss_peak_kb=(\d+)", r.out)
    if m:
        d["rc"] = int(m.group(1)); d["vmhwm_mib"] = int(m.group(2)) / 1024
        d["vmswap_peak_mib"] = int(m.group(3)) / 1024; d["vmrss_peak_mib"] = int(m.group(4)) / 1024
    m = re.search(r"memory_max=(\S+)", r.out)
    d["cgroup_memory_max"] = m.group(1) if m else None
    if NATIVE and CURRENT_CAP:
        want = CAP_MB[CURRENT_CAP] * 1048576
        d["cap_in_effect"] = d["cgroup_memory_max"] == str(want)
        if not d["cap_in_effect"]:
            say(f"WARNING: cgroup memory.max {d['cgroup_memory_max']} is not the requested {want}; the row will be refused")
    m = re.search(r"disk_dev=(\S+)", r.out)
    d["disk_dev"] = m.group(1) if m else None
    m = re.search(r"^read_bytes: (\d+)", r.out, re.M)
    d["read_bytes"] = int(m.group(1)) if m else -1
    m = re.search(r"disk_sectors_delta=(-?\d+)", r.out)
    d["disk_read_mib"] = int(m.group(1)) * 512 / 1048576 if m else None
    mem = re.findall(r"MemAvailable:\s+(\d+) kB", r.out)
    d["mem_avail_mib"] = [int(x) // 1024 for x in mem]
    d["stderr_tail"] = stderr[-1500:]
    d["oom"] = bool(re.search(r"Killed|out of memory|Cannot allocate|std::bad_alloc|failed to allocate", stderr + r.out, re.I))
    return d


# ----------------------------------------------------------------------------- gates
def gates(env: str) -> None:
    if env not in ("wsl", "linux"):
        die("gates: --env wsl or linux only")
    b = jload(RESULTS / "build.json")
    if not b and not DRY:
        die("gates: no build.json; run build first")
    set_cap("12G")                                        # G2 runs R=192, which needs the full model
    bench_hash = None
    if not DRY:
        hr = run_wsl(f"sha256sum {wsl_path('bin/llama-bench-temporal')} {wsl_path('bin/llama-perplexity')} | cut -d' ' -f1",
                     label="gatesha", login=False)
        bench_hash, ppl_hash = hr.out.split()[:2] if len(hr.out.split()) >= 2 else (None, None)
        if bench_hash != b["bench_sha256"]:
            die(f"gates: binary hash {bench_hash} differs from build.json {b['bench_sha256']}; rebuild first")
    g = {"ts": now(), "tag": _TAG, "env": env, "provisional": PROVISIONAL, "bench_sha256": bench_hash, "results": {}}
    model_bytes = 0
    if not DRY:
        sr = run_wsl(f"stat -c %s {wsl_path('models/' + MODEL)}", label="modelsize", login=False)
        model_bytes = int(sr.out.split()[-1]) if sr.out.strip() else 0
    expert_total = SLICE_BYTES * N_EXPERT * N_LAYER * 3
    load_mib = (model_bytes - expert_total) / 1048576   # lazy load reads only non-expert weights

    # ---- G1: bytes are real -------------------------------------------------------------
    # Pool fetched_mib against /proc/<pid>/io read_bytes (block reads only; page-cache hits do
    # not count). Caches are dropped first so the loader's non-expert read is a known block
    # read of (file - experts) bytes, subtracted as in honesty_gate.py. R=18 with no policy
    # (fetch on miss) moves ~7 GB in 16 tokens, so the 10% tolerance is dominated by fetches.
    r1 = engine_run("g1_bytes", arm_flags(18, False), "-t 4 -p 0 -n 16 -r 1 -mmp 0 -ot _exps=CPU",
                    drop_caches=True, timeout=3600)
    if not DRY:
        pool = r1.get("pool") or {}
        fetched = pool.get("fetched_mib", 0.0)
        read_mib = r1["read_bytes"] / 1048576 if r1["read_bytes"] > 0 else -1
        fetch_read = read_mib - load_mib
        rel = abs(fetch_read - fetched) / fetched if fetched else 9.9
        ok = fetched > 0 and rel < 0.10
        g["results"]["G1"] = {"pass": ok, "fetched_mib": fetched, "read_mib": read_mib, "load_mib_subtracted": load_mib,
                              "fetch_read_mib": fetch_read, "rel_err": rel, "disk_read_mib": r1["disk_read_mib"],
                              "pool": pool, "rc": r1["rc"], "decode_tps": r1.get("decode_tps"), "label": "g1_bytes"}
        say(f"G1 {'PASS' if ok else 'FAIL'}: pool fetched {fetched:.0f} MiB, proc read {read_mib:.0f} MiB "
            f"(-{load_mib:.0f} load = {fetch_read:.0f}), rel err {rel * 100:.1f}%, diskstats {r1['disk_read_mib']:.0f} MiB")

    # ---- G2: numerics exact, R=18 vs R=192, same binary, same flags ----------------------
    def ppl(label: str, flags: dict) -> tuple[str, dict]:
        r = engine_run(label, flags, f"-f {wsl_path('temporal-moe/androidbench/ppl_input.txt')} --chunks 2 -c 512 -t 4 --no-mmap -ot _exps=CPU",
                       timeout=7200, binary="bin/llama-perplexity")
        if DRY:
            return "", r
        m = re.search(r"Final estimate: PPL = ([0-9.]+ \+/- [0-9.]+)", r["stderr_tail"] + wsl_read(f"logs/{_TAG}/{label}.stderr"))
        return (m.group(1) if m else "NOT FOUND"), r
    a, ra = ppl("g2_ppl_R192", arm_flags(192, True))
    bb, rb = ppl("g2_ppl_R18", arm_flags(18, True))
    if not DRY:
        ok = a == bb and a != "NOT FOUND"
        g["results"]["G2"] = {"pass": ok, "ppl_R192": a, "ppl_R18": bb, "pool_R192": ra.get("pool"), "pool_R18": rb.get("pool"),
                              "rc": [ra["rc"], rb["rc"]]}
        say(f"G2 {'PASS' if ok else 'FAIL'}: R=192 PPL {a} | R=18 PPL {bb}")

    # ---- G3: repack real: LLAMA_NO_REPACK vs repacked, identical PPL, different tok/s -----
    c, rc_ = ppl("g3_ppl_norepack_R18", arm_flags(18, True, no_repack=True))
    s1 = engine_run("g3_tps_repack", arm_flags(18, True), "-t 4 -p 0 -n 64 -r 3 -mmp 0 -ot _exps=CPU", timeout=3600)
    s2 = engine_run("g3_tps_norepack", arm_flags(18, True, no_repack=True), "-t 4 -p 0 -n 64 -r 3 -mmp 0 -ot _exps=CPU", timeout=3600)
    if not DRY:
        t1, t2 = s1.get("decode_tps") or 0, s2.get("decode_tps") or 0
        sd = max(s1.get("decode_sd") or 0, s2.get("decode_sd") or 0)
        differ = t1 > 0 and t2 > 0 and abs(t1 - t2) > max(0.03 * max(t1, t2), 2 * sd)
        ok = (c == bb and c != "NOT FOUND") and differ
        g["results"]["G3"] = {"pass": ok, "ppl_norepack": c, "ppl_repack": bb, "tps_repack": t1, "tps_norepack": t2,
                              "sd": sd, "differ": differ, "pool_repack": s1.get("pool"), "pool_norepack": s2.get("pool")}
        say(f"G3 {'PASS' if ok else 'FAIL'}: PPL norepack {c} vs repack {bb}; tok/s repack {t1:.2f} vs norepack {t2:.2f}")
    if DRY:
        return
    g["all_pass"] = all(v["pass"] for v in g["results"].values()) and len(g["results"]) == 3
    g["gated_hashes"] = [bench_hash] if g["all_pass"] else []
    allg = load_gates()
    allg[env] = g
    jdump(RESULTS / "gates.json", allg)
    say(f"gates.json[{env}] written; all_pass={g['all_pass']}")
    if not g["all_pass"]:
        sys.exit(1)


def load_gates() -> dict:
    """gates.json is keyed by environment; a legacy flat file is treated as the wsl entry."""
    allg = jload(RESULTS / "gates.json", {})
    if "results" in allg:
        allg = {"wsl": allg}
    return allg


def require_gated(env: str = "wsl") -> str:
    """Invariant 1: the bench binary's current hash must be stamped as gated for this environment."""
    g = load_gates().get(env)
    if DRY:
        return "dry"
    hr = run_wsl(f"sha256sum {wsl_path('bin/llama-bench-temporal')} | cut -d' ' -f1", label="armsha", login=False)
    h = hr.out.strip().splitlines()[-1] if hr.out.strip() else ""
    if not g or not g.get("all_pass") or h not in g.get("gated_hashes", []):
        die(f"binary {h[:12]} is not gated for {env} (gates.json all_pass={g.get('all_pass') if g else None}, "
            f"gated={[x[:12] for x in (g or {}).get('gated_hashes', [])]}); run gates first")
    return h


# ----------------------------------------------------------------------------- arms
def expected_counters(arm: str, R: int, twopass: bool, tokens: int) -> dict:
    """What the pool line must show for the arm requested (PLAN 1.2, pitfall #18)."""
    if R >= N_EXPERT and not twopass:
        return {"fetches": (0, 0), "evictions": (0, 0), "swaps": (0, 0)}
    if R >= N_EXPERT and twopass:
        return {"fetches": (0, 0), "evictions": (1, 10 ** 12), "swaps": (1, 10 ** 12)}
    fill = N_LAYER * R
    if twopass:                                            # one swap per layer per token
        return {"fetches": (int(0.90 * N_LAYER * tokens), int(1.10 * N_LAYER * (tokens + 2)) + fill),
                "evictions": (1, 10 ** 12), "swaps": (1, 10 ** 12), "mib_per_fetch": (0.60, 0.67)}
    # vanilla floor: free top-k with fetch on miss, ~16 of 18 experts missed per layer
    return {"fetches": (N_LAYER * 8 * tokens, N_LAYER * K_ACTIVE * (tokens + 2) + fill),
            "evictions": (0, 10 ** 12), "swaps": (0, 0), "mib_per_fetch": (0.60, 0.67)}


def verify_counters(arm: str, R: int, twopass: bool, tokens: int, pool: dict | None) -> tuple[bool, str]:
    if pool is None:
        return False, "no temporal-pool line in stderr"
    exp = expected_counters(arm, R, twopass, tokens)
    for k, (lo, hi) in exp.items():
        if k == "mib_per_fetch":
            v = pool["fetched_mib"] / pool["fetches"] if pool["fetches"] else 0
        else:
            v = pool.get(k, -1)
        if not (lo <= v <= hi):
            return False, f"{k}={v} outside [{lo}, {hi}] for arm {arm} R={R} twopass={twopass} tokens={tokens}"
    return True, "ok"


def clock_probe(session: dict) -> dict:
    """PLAN 5.4: a resident stock llama-bench before each batch, compared with the session's first."""
    set_cap("12G")
    r = engine_run(f"clock_{int(time.time())}", None, PROBE_ARGS, timeout=1800)
    if DRY:
        return {"tok_s": None}
    tps = r.get("decode_tps") or 0.0
    if session.get("clock_ref_tok_s") is None and tps > 0:
        session["clock_ref_tok_s"] = tps
        session["clock_ref_ts"] = now()
    ref = session.get("clock_ref_tok_s") or tps
    d = {"tok_s": tps, "ref_tok_s": ref, "ratio": (tps / ref) if ref else 0, "degraded": tps < (1 - CLOCK_TOL) * ref,
         "label": r["label"], "wall_s": r["wall_s"]}
    say(f"clock probe: {tps:.2f} tok/s vs session ref {ref:.2f} ({d['ratio'] * 100:.1f}%)"
        f"{'  ** DEGRADED **' if d['degraded'] else ''}")
    return d


def one_batch(arm: str, tier: str, label: str, R: int, twopass: bool, cap: str, rnd: int, session: dict,
              overrides: dict, rest: int, bench_hash: str, extra_note: str = "", cap_override: str | None = None) -> dict | None:
    st = check("wsl", quick=True)
    probes = []
    p = clock_probe(session)
    probes.append(p)
    tries = 0
    while not DRY and p["degraded"] and tries < 2:
        tries += 1
        say(f"batch {label} r{rnd}: clock degraded, resting {rest} s and re-probing ({tries}/2)")
        time.sleep(rest)
        p = clock_probe(session); probes.append(p)
    cap = cap_override or cap
    capinfo = set_cap(cap)
    flags = arm_flags(R, twopass, overrides)
    tag = f"{label}_r{rnd}_{cap.replace('.', 'p')}"
    engine_run(f"{tag}_warmup", flags, WARMUP_ARGS, timeout=3600)
    m = engine_run(tag, flags, ENGINE_ARGS, timeout=7200)
    if DRY:
        return None
    tokens = 128 * 8
    ok, why = verify_counters(arm, R, twopass, tokens, m.get("pool"))
    row = {"tag": _TAG, "env": "linux" if NATIVE else "wsl", "provisional": PROVISIONAL, "arm": arm, "tier": tier,
           "label": label, "R": R, "twopass": twopass,
           "round": rnd, "cap": cap, "cap_mb": capinfo["cap_mb"], "memtotal_mb": capinfo["memtotal_mb"],
           "cgroup_memory_max": m.get("cgroup_memory_max"),
           "flags": flags, "overrides": overrides, "engine_args": ENGINE_ARGS, "cmd": m["cmd"],
           "binary_sha256": bench_hash, "gated": True,
           "clock_probe": probes[-1], "clock_probes_all": probes, "degraded": probes[-1]["degraded"],
           "decode_tok_s": m.get("decode_tps"), "decode_sd": m.get("decode_sd"), "tokens": tokens,
           "wall_s": m["wall_s"], "rc": m["rc"], "pool": m.get("pool"), "vmhwm_mib": m.get("vmhwm_mib"),
           "vmswap_peak_mib": m.get("vmswap_peak_mib"), "read_bytes": m["read_bytes"], "disk_read_mib": m["disk_read_mib"],
           "mem_avail_mib": m["mem_avail_mib"], "host": {k: st.get(k) for k in ("overlay_ac", "scheme", "power_online",
                                                                             "battery_pct", "cpu_load_pct", "forced")},
           "counters_ok": ok, "counters_note": why, "ts": now(), "note": extra_note,
           "status": "ok"}
    if (row["decode_tok_s"] or 0) <= 0 or m["rc"] != 0:
        row["status"] = f"error_rc{m['rc']}" + ("_oom" if m.get("oom") else "")
    elif NATIVE and not m.get("cap_in_effect", True):
        ok, why = False, f"cgroup memory.max={m.get('cgroup_memory_max')} is not the requested cap {cap}"
    elif (row["vmswap_peak_mib"] or 0) > 0:
        row["status"] = "swapped"                            # pitfall #20
    elif probes[-1]["degraded"]:
        row["status"] = "degraded_clock"
    if not ok:
        row["status"] = "refused_counters"
        append_jsonl(RESULTS / "refused.jsonl", row)
        say(f"REFUSED row {label} r{rnd}: {why}")
        return row
    append_jsonl(RESULTS / "runs.jsonl", row)
    say(f"ROW {label} r{rnd} cap {cap}: {row['decode_tok_s']:.2f} tok/s sd {row['decode_sd']:.2f}  "
        f"fetches={row['pool']['fetches']} fetched_mib={row['pool']['fetched_mib']:.0f} evictions={row['pool']['evictions']} "
        f"swaps={row['pool']['swaps']}  VmHWM {row['vmhwm_mib']:.0f} MiB  status={row['status']}")
    return row


def session_state() -> dict:
    return jload(RESULTS / f"session_{_TAG}.json", {"tag": _TAG, "started": now(), "clock_ref_tok_s": None})


def save_session(s: dict) -> None:
    if not DRY:
        jdump(RESULTS / f"session_{_TAG}.json", s)


def already_done(label: str, rnd: int, cap: str) -> bool:
    for r in read_jsonl(RESULTS / "runs.jsonl"):
        if r["tag"] == _TAG and r["label"] == label and r["round"] == rnd and r["cap"] == cap and r["status"] == "ok":
            return True
    return False


def arms(env: str, which: str, n: int, Rs: list[int], rest: int, sets: list[str], resume: bool, demo: bool) -> None:
    if env not in ("wsl", "linux"):
        die("arms: --env wsl or linux only")
    overrides = parse_overrides(sets)
    bench_hash = require_gated(env)
    check(env)
    sess = session_state()
    order = [a for a in "abce" if a in which]              # interleaved a, b, c, e per round
    schedule: list[tuple] = []
    for rnd in range(1, n + 1):
        for a in order:
            tier, label, R, tp, cap = ARMS[a]
            schedule.append((a, tier, label, R, tp, cap, rnd))
    if "a" in which:
        schedule.append(("a", *ARMS["a"], n + 1))            # closing ceiling shows drift (PLAN 1.3)
    if "d" in which:
        for rnd in range(1, n + 1):
            for R in Rs:
                schedule.append(("d", "deploy", f"deploy_R{R}", R, True, "4G", rnd))
    say(f"arms schedule: {len(schedule)} batches, rest {rest} s: " +
        " ".join(f"{s[2]}:r{s[6]}" for s in schedule))
    for i, (a, tier, label, R, tp, cap, rnd) in enumerate(schedule):
        if resume and already_done(label, rnd, cap):
            say(f"skip {label} r{rnd} (done)")
            continue
        row = one_batch(a, tier, label, R, tp, cap, rnd, sess, overrides, rest, bench_hash)
        save_session(sess)
        if not DRY and i < len(schedule) - 1:
            say(f"rest {rest} s")
            time.sleep(rest)
    if demo:
        memdemo(sess, overrides, rest, bench_hash)
    if not DRY:
        report()


def memdemo(sess: dict, overrides: dict, rest: int, bench_hash: str) -> None:
    """PLAN 1.4: the ceiling must FAIL to start under 4 GB; deploy must run at 4 GB and 2.5 GB."""
    say("memory demonstration")
    d = jload(RESULTS / f"memdemo_{_TAG}.json", {"tag": _TAG, "ts": now()})
    # ceiling at 4 GB: expected to fail (OOM / allocation failure). No clock probe needed.
    capinfo = set_cap("4G")
    m = engine_run("memdemo_ceiling_4G", arm_flags(192, False, overrides), ENGINE_ARGS, timeout=3600)
    if not DRY:
        dm = run_wsl("(sudo -n dmesg 2>/dev/null || dmesg 2>/dev/null || journalctl -k --no-pager 2>/dev/null) | grep -i -E 'out of memory|oom-kill|killed process' | tail -3",
                     label="dmesg", login=False)
        d["ceiling_4G"] = {"rc": m["rc"], "decode_tok_s": m.get("decode_tps"), "vmhwm_mib": m.get("vmhwm_mib"),
                           "oom": m.get("oom"), "dmesg": dm.out.strip()[-500:], "stderr_tail": m["stderr_tail"][-600:],
                           "cap": capinfo, "failed_to_start": (m["rc"] != 0 or not m.get("decode_tps"))}
        say(f"ceiling @4G: rc={m['rc']} decode={m.get('decode_tps')} -> "
            f"{'FAILED TO START (expected)' if d['ceiling_4G']['failed_to_start'] else 'RAN (unexpected: cap not binding)'}")
        time.sleep(min(rest, 120))
    # deploy at 2.5 GB, full protocol with clock probe (deploy at 4 GB already has n=3 rows)
    tier, label, R, tp, cap = ARMS["c"]
    row = one_batch("c", tier, label + "_cap2p5", R, tp, cap, 1, sess, overrides, rest, bench_hash,
                    extra_note="memory demonstration", cap_override="2.5G")
    if not DRY:
        d["deploy_2.5G"] = {k: row.get(k) for k in ("decode_tok_s", "decode_sd", "vmhwm_mib", "vmswap_peak_mib", "rc", "status", "pool")} if row else None
        rows4 = [r for r in read_jsonl(RESULTS / "runs.jsonl") if r["tag"] == _TAG and r["label"] == "deploy_R18" and r["status"] == "ok"]
        d["deploy_4G"] = {"n": len(rows4), "vmhwm_mib": [r["vmhwm_mib"] for r in rows4],
                          "decode_tok_s": [r["decode_tok_s"] for r in rows4]}
        rowsa = [r for r in read_jsonl(RESULTS / "runs.jsonl") if r["tag"] == _TAG and r["arm"] == "a" and r["status"] == "ok"]
        d["ceiling_12G"] = {"n": len(rowsa), "vmhwm_mib": [r["vmhwm_mib"] for r in rowsa]}
        jdump(RESULTS / f"memdemo_{_TAG}.json", d)


# ----------------------------------------------------------------------------- sweep
def sweep(env: str, knob: str, n: int, rest: int, sets: list[str], base_arm: str) -> None:
    if env not in ("wsl", "linux"):
        die("sweep: --env wsl or linux only")
    if not knob or "=" not in knob:
        die("--knob NAME=v1,v2 required")
    name, vals = knob.split("=", 1)
    vals = [v for v in vals.split(",")]
    if any(v.strip() == "0" for v in vals):
        die(f"--knob {name}=0 refused (pitfall #17); use an empty value to remove the flag")
    overrides = parse_overrides(sets)
    bench_hash = require_gated(env)
    check(env)
    sess = session_state()
    tier, label, R, tp, cap = ARMS[base_arm]
    for rnd in range(1, n + 1):
        for v in vals:                                       # A/B/A/B interleaved
            o = dict(overrides); o[name] = None if v == "" else v
            one_batch(base_arm, tier, f"{label}_{name.replace('LLAMA_TEMPORAL_', '')}={v or 'unset'}", R, tp, cap, rnd,
                      sess, o, rest, bench_hash, extra_note=f"sweep {name}")
            save_session(sess)
            if not DRY:
                time.sleep(rest)
    if not DRY:
        report()


# ----------------------------------------------------------------------------- report / pack
def bound_tok_s(ceiling: float, fit: dict, n_inflight: int = 12) -> dict:
    """PLAN section 4."""
    fixed, per_kib = fit["fixed_us"], fit["per_kib_us"]
    fetch_qd1_s = N_LAYER * (fixed + 648 * per_kib) / 1e6
    bw = fit["bw_MBps_by_qd"].get(str(n_inflight)) or fit["bw_MBps_by_qd"].get("12")
    fetch_qdn_s = BYTES_PER_TOKEN / (bw * 1e6)
    b = 1.0 / max(1.0 / ceiling, fetch_qdn_s)
    return {"fetch_QD1_s": fetch_qd1_s, "fetch_QDn_s": fetch_qdn_s, "bw_MBps_at_QDn": bw, "n_inflight": n_inflight,
            "bound_tok_s": b, "bound_QD1_tok_s": 1.0 / max(1.0 / ceiling, fetch_qd1_s)}


def report() -> None:
    rows = [r for r in read_jsonl(RESULTS / "runs.jsonl") if r["tag"] == _TAG]
    if not rows:
        say("report: no rows for this tag"); return
    ok = [r for r in rows if r["status"] == "ok"]
    ceil = [r["decode_tok_s"] for r in ok if r["arm"] == "a"]
    ceiling = sum(ceil) / len(ceil) if ceil else None
    print("\n" + "=" * 110)
    print(f"LAPTOP i7-11370H / PM981a / WSL2 -- tag {_TAG} -- {len(rows)} rows ({len(ok)} ok)")
    print("=" * 110)
    print(f"{'label':<26}{'cap':>5}{'n':>3}{'tok/s':>9}{'sd':>7}{'ratio':>8}{'VmHWM MiB':>11}{'fetch/tok':>10}{'MiB/tok':>9}{'evict':>8}{'swaps':>7}  status")
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault((r["label"], r["cap"]), []).append(r)
    for (label, cap), rs in groups.items():
        good = [r for r in rs if r["status"] == "ok"]
        if good:
            t = [r["decode_tok_s"] for r in good]
            mean = sum(t) / len(t)
            sd = (sum((x - mean) ** 2 for x in t) / len(t)) ** 0.5
            hwm = max(r["vmhwm_mib"] or 0 for r in good)
            p = good[-1]["pool"] or {}
            tok = good[-1]["tokens"]
            print(f"{label:<26}{cap:>5}{len(good):>3}{mean:>9.2f}{sd:>7.2f}"
                  f"{(mean / ceiling * 100 if ceiling else 0):>7.1f}%{hwm:>11.0f}{p.get('fetches', 0) / tok:>10.1f}"
                  f"{p.get('fetched_mib', 0) / tok:>9.2f}{p.get('evictions', 0):>8}{p.get('swaps', 0):>7}  "
                  + ",".join(sorted({r['status'] for r in rs})))
        else:
            print(f"{label:<26}{cap:>5}{0:>3}{'-':>9}{'-':>7}{'-':>8}{'-':>11}  " + ",".join(sorted({r['status'] for r in rs})))
    pr = jload(RESULTS / "probe.json", {}).get("wsl", {}).get("fit")
    dep = [r["decode_tok_s"] for r in ok if r["label"] == "deploy_R18" and r["cap"] == "4G"]
    if ceiling and pr and dep:
        b = bound_tok_s(ceiling, pr)
        dm = sum(dep) / len(dep)
        print(f"\nbound (PLAN 4): fetch_QD1 {b['fetch_QD1_s'] * 1e3:.1f} ms/token, fetch_QD{b['n_inflight']} {b['fetch_QDn_s'] * 1e3:.1f} ms/token "
              f"at {b['bw_MBps_at_QDn']:.0f} MB/s; bound {b['bound_tok_s']:.2f} tok/s (QD1 bound {b['bound_QD1_tok_s']:.2f})")
        print(f"deploy {dm:.2f} tok/s = {dm / ceiling * 100:.1f}% of ceiling {ceiling:.2f}, {dm / b['bound_tok_s'] * 100:.1f}% of bound")
    ref = [r for r in read_jsonl(RESULTS / "refused.jsonl") if r["tag"] == _TAG]
    if ref:
        print(f"\nrefused rows (counters disagreed): {len(ref)}")
        for r in ref:
            print(f"  {r['label']} r{r['round']}: {r['counters_note']}")
    print()


CSV_HEADER = ["phase", "model", "tier", "setup", "ubatch", "context", "prefill_ms", "decode_tok_s", "peak_vram_mib",
              "note", "decode_tok_s_std", "copied_bytes_per_token"]


def to_csv_row(r: dict) -> list:
    """emit_row.py schema, one row per measured batch. peak_vram_mib carries VmHWM (peak RSS, the
    laptop's analogue); copied_bytes_per_token carries the bytes fetched from the NVMe per token."""
    p = r.get("pool") or {}
    note = (f"[{r['status']}] R={r['R']};twopass={int(r['twopass'])};cap={r['cap']};memtotal_mb={r['memtotal_mb']};"
            f"round={r['round']};fetches={p.get('fetches')};fetched_mib={p.get('fetched_mib')};evictions={p.get('evictions')};"
            f"swaps={p.get('swaps')};clock_probe={r['clock_probe'].get('tok_s')};clock_ref={r['clock_probe'].get('ref_tok_s')};"
            f"vmhwm_mib=VmHWM;read_bytes={r['read_bytes']};binary={r['binary_sha256'][:12]};tag={r['tag']};"
            f"flags={env_string(r['flags'])};args={r['engine_args']}" + (f";{r['note']}" if r.get("note") else ""))
    setup = "laptop-linux-cpu" if r.get("env") == "linux" else "laptop-wsl2-cpu"
    if r.get("provisional"):
        note = f"[provisional: {r['provisional']}] " + note
    return ["decode", MODEL, r["tier"], setup, "", "", "",
            f"{r['decode_tok_s']:.4f}", f"{r['vmhwm_mib']:.0f}" if r.get("vmhwm_mib") else "", note,
            f"{r['decode_sd']:.4f}", int(p.get("fetched_mib", 0) * 1048576 / r["tokens"]) if r.get("tokens") else 0]


def pack() -> None:
    rows = [r for r in read_jsonl(RESULTS / "runs.jsonl") if r["tag"] == _TAG and r["status"] in ("ok",)]
    csvp = REPO / "results" / "ablations" / "serving_benchmarks_laptop.csv"
    if DRY:
        say(f"pack (dry): would append {len(rows)} rows to {csvp}, copy results json/jsonl to "
            f"{REPO / 'comms' / 'laptop' / _TAG}, tar logs from {LOGS / _TAG} and {WIN_ROOT / 'logs' / _TAG}")
        return
    csvp.parent.mkdir(parents=True, exist_ok=True)
    new = not csvp.exists()
    existing = csvp.read_text(encoding="utf-8") if csvp.exists() else ""
    with open(csvp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_HEADER)
        for r in rows:
            if f"tag={r['tag']}" in existing and f"round={r['round']}" in existing and r["label"] in existing:
                continue
            w.writerow(to_csv_row(r))
    out = REPO / "comms" / "laptop" / _TAG
    out.mkdir(parents=True, exist_ok=True)
    for name in ("probe.json", "compute.json", "decision.json", "build.json", "gates.json", "runs.jsonl", "refused.jsonl",
                 f"session_{_TAG}.json", f"memdemo_{_TAG}.json"):
        if (RESULTS / name).exists():
            shutil.copy(RESULTS / name, out / name)
    tarp = out / f"logs_{_TAG}.tar.gz"
    with tarfile.open(tarp, "w:gz") as t:
        if (LOGS / _TAG).exists():
            t.add(LOGS / _TAG, arcname=f"logs/{_TAG}")
        sdir = WIN_ROOT / "logs" / _TAG
        if sdir.exists():
            t.add(sdir, arcname=f"scripts/{_TAG}")
    say(f"pack: {len(rows)} rows -> {csvp}; artifacts -> {out}; logs -> {tarp}")
    envs = sorted({r.get("env", "wsl") for r in rows}) or ["wsl"]
    msg = (f"laptopbench: {len(rows)} decode rows, tag {_TAG}, {'/'.join(envs)} on i7-11370H + PM981a"
           + (f" (provisional: {PROVISIONAL})" if PROVISIONAL else "")
           + f"\n\nRows in results/ablations/serving_benchmarks_laptop.csv; probe, compute, gates,\n"
             f"runs.jsonl and the log tarball under comms/laptop/{_TAG}/.")
    r = run_win(["git", "-C", str(REPO), "add", "results/ablations/serving_benchmarks_laptop.csv", f"comms/laptop/{_TAG}",
                 "laptopbench/LEDGER.md", "laptopbench/lapbench.py", "laptopbench/results"], label="gitadd")
    r = run_win(["git", "-C", str(REPO), "-c", "core.hooksPath=.githooks", "commit", "-q", "-m", msg], label="gitcommit")
    say(f"pack: commit rc={r.rc} {r.text.strip()[-300:]}")


# ----------------------------------------------------------------------------- main
def main() -> None:
    global DRY, FORCE, _TAG, PROVISIONAL, NATIVE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "probe", "compute", "build", "gates", "arms", "sweep", "session", "pack", "report", "decide"])
    ap.add_argument("--env", default="wsl", choices=["wsl", "linux", "windows", "all"])
    ap.add_argument("--arms", default="a,b,c,d,e", help="subset, default all")
    ap.add_argument("--R", default="24,36,48", help="sweep points for (d)")
    ap.add_argument("--n", type=int, default=3, help="batches per arm")
    ap.add_argument("--rest", type=int, default=300, help="seconds between batches")
    ap.add_argument("--cap", default=None, choices=list(CAP_MB), help="apply a cap and exit (with check)")
    ap.add_argument("--set", action="append", default=[], help="engine env override KEY=VAL, repeatable; KEY=0 refused")
    ap.add_argument("--knob", default=None, help="NAME=v1,v2 for sweep")
    ap.add_argument("--base-arm", default="c", help="arm the sweep varies (default c)")
    ap.add_argument("--tag", default=None, help="session label stamped in every row")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="override a failed check; recorded in the row")
    ap.add_argument("--no-demo", action="store_true", help="arms: skip the memory demonstration")
    ap.add_argument("--provisional", default="", metavar="NOTE",
                    help="stamp every row and json with provisional=NOTE (e.g. Defender exclusions attested, not read)")
    a = ap.parse_args()
    DRY, FORCE, PROVISIONAL = a.dry_run, a.force, a.provisional
    NATIVE = a.env == "linux"
    if NATIVE and not IS_LINUX and not DRY:
        die("--env linux runs on the live USB itself (Linux Python); on Windows only --dry-run is allowed")
    if a.env == "all" and IS_LINUX:
        die("--env all means windows+wsl; on the live USB use --env linux")
    if a.stage in ("arms", "sweep", "gates", "pack", "session", "report") and not a.tag:
        die("--tag is required for this stage")
    _TAG = a.tag or "untagged"
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    which = a.arms.replace(",", "")
    Rs = [int(x) for x in a.R.split(",") if x]
    if a.stage == "check":
        if a.cap:
            set_cap(a.cap)
        st = check(a.env)
        print(json.dumps(st, indent=1))
    elif a.stage == "probe":
        check(a.env, quick=True)
        probe(a.env)
    elif a.stage == "compute":
        check(a.env)
        compute(a.env)
    elif a.stage == "decide":
        decide()
    elif a.stage == "build":
        build(a.env)
    elif a.stage == "gates":
        check(a.env)
        gates(a.env)
    elif a.stage == "arms":
        arms(a.env, which, a.n, Rs, a.rest, a.set, a.resume, not a.no_demo)
    elif a.stage == "sweep":
        sweep(a.env, a.knob, a.n, a.rest, a.set, a.base_arm)
    elif a.stage == "session":
        check(a.env)
        if not (RESULTS / "probe.json").exists() or a.env not in jload(RESULTS / "probe.json", {}):
            probe(a.env)
        if not (RESULTS / "compute.json").exists() or a.env not in jload(RESULTS / "compute.json", {}):
            compute(a.env)
        if not a.resume or not (RESULTS / "build.json").exists():
            build(a.env)
        if not a.resume or not (jload(RESULTS / "gates.json") or {}).get("all_pass"):
            gates(a.env)
        arms(a.env, which, a.n, Rs, a.rest, a.set, a.resume, not a.no_demo)
    elif a.stage == "report":
        report()
    elif a.stage == "pack":
        pack()


if __name__ == "__main__":
    main()
