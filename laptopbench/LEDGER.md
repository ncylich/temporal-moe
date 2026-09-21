# Laptop temporal-MoE ledger

Machine: Mohsen's laptop. Entries L1-N in the style of androidbench/LEDGER.md: what was
run, the exact command, the numbers, what was retracted. Every run, including failed and
degraded ones, appears here.

---

## VERDICT (read this first)

**Nothing has been timed.** Phase 0 stopped on missing inputs and a WSL that cannot start
until Windows is restarted. No number below is a result.

---

## 0. Phase 0 -- checkout and preflight

### L1-0 -- Phase 0 stopped: four inputs missing, WSL2 blocked pending a restart

Session ran from native Windows PowerShell 5.1 (not WSL), 2026-09-13, ~21:55-22:10 local.

**Inputs, as received vs. as required**

| input | status |
|---|---|
| Repo `https://github.com/ncylich/temporal-moe.git` | reachable |
| Branch `laptop` | **absent on origin** (`git ls-remote --heads origin laptop` returns nothing) |
| `laptopbench/PLAN.md` | **absent on every branch** (searched all 9 remote branches; `androidbench/` is present on all of them) |
| Fork URL | **not supplied** (placeholder `<FORK_URL>`); the repo does not name it either (`git grep` over androidbench docs and README: no llama.cpp URL) |
| Model URL | **not supplied** (placeholder `<MODEL_URL>`) |
| Model sha256 | **not supplied** (placeholder `<SHA256>`) |
| WSL distro `Ubuntu-24.04` | **not installed**; WSL2 refuses to start (see below) |

Per the operating rules ("if any input is missing, the branch lacks laptopbench/PLAN.md,
or the sha256 does not match, stop after Phase 0 and report"): stopped after Phase 0.
The model was not generated locally (generator needs 21 GB; machine has 16).

**What was done**

```
git clone C:\Users\mohsen\Desktop\temporalmoe-all\temporal-moe C:\tmoe\temporal-moe   # local clone: origin pack is 738 MiB, network clone stalled
git -C C:\tmoe\temporal-moe remote set-url origin https://github.com/ncylich/temporal-moe.git
git -C C:\tmoe\temporal-moe fetch origin
git -C C:\tmoe\temporal-moe checkout -B main origin/main      # 68645b22 Merge pull request #2 from ncylich/fix-hook-exec-bit
git -C C:\tmoe\temporal-moe checkout -b laptop                # LOCAL branch, created from main because origin has none; not pushed
python -m venv C:\tmoe\venv                                   # Python 3.11.9
```

Not done, because it depends on the missing inputs or on WSL: fork clone (both sides),
model download and hash, WSL-side workspace, apt packages, WSL venv, Windows fio install.
No Defender, Search, power-plan, .wslconfig or firmware setting was changed.

**Machine (recorded from CIM/powercfg, not Task Manager)**

| item | value |
|---|---|
| OS | Windows 11 Home 25H2, build 10.0.26200 (wsl reports 26200.9445) |
| CPU | 11th Gen Intel Core i7-11370H @ 3.30 GHz, 4C/8T |
| RAM | 8 x 2 GiB Micron 53E1G32D4NQ-046 LPDDR4x, 4267 MT/s configured; 15.79 GiB total |
| Disk | SAMSUNG MZVLB1T0HBLR-000L2 (PM981a), NVMe, 1024 GB, Healthy |
| C: free | 167.3 GB free of 450 GB |
| WSL | app 2.7.14.0, kernel 6.18.33.2-2 (from `wsl --version`); **no distributions installed** |
| Python (Windows) | 3.11.9; no `py` launcher |
| Python (WSL) | n/a, no distro |
| Power plan | **Balanced** (381b4222-...), not Best performance |
| AC | PowerOnline=True, battery 98%, not charging/discharging |
| Sleep | Modern Standby (S0 Low Power Idle) is the only standby the firmware offers; S3 disabled; Hibernate and Fast Startup available |
| Defender | real-time protection ON; exclusion lists not readable without elevation |
| Windows Search | WSearch running/Automatic; no crawl-scope rule mentions tmoe or a VHDX |
| fio (Windows) | not on PATH, not installed |
| Uptime | 5 days 4 h |

**Why WSL2 cannot start, exactly**

`wsl --status`:
> WSL2 is unable to start since virtualization is not enabled on this machine. Please ensure
> the "Virtual Machine Platform" optional component is enabled and virtualization is turned
> on in your computer's firmware settings.

`Win32_OptionalFeature`: VirtualMachinePlatform = Enabled, Microsoft-Windows-Subsystem-Linux
= Enabled, HypervisorPlatform = Disabled. Setup event log, 2026-09-13:

```
21:29:45  Id 7   Initiating changes to turn on update VirtualMachinePlatform ...
21:29:50  Id 13  A reboot is necessary before the selectable update VirtualMachinePlatform ... can be turned on
21:33:03  Id 7   Initiating changes to turn on update Microsoft-Windows-Subsystem-Linux ...
21:33:08  Id 13  A reboot is necessary before the selectable update Microsoft-Windows-Subsystem-Linux ... can be turned on
```

CBS RebootPending = True, PendingFileRenameOperations present. Virtualization-based
security reports Running and `HypervisorPresent` = True, so VT-x is on in firmware; the
`VirtualizationFirmwareEnabled=False` from Win32_Processor is the usual artifact of
querying under a running hypervisor. Conclusion: a Windows restart is required, not a
firmware change. After the restart, `Ubuntu-24.04` still has to be installed
(`wsl --install -d Ubuntu-24.04`, interactive first-user setup) before the WSL half of
Phase 0 can run.

**Blocked on Mohsen**

1. Restart Windows (WSL/VMP enablement pending).
2. Install Ubuntu-24.04 in WSL after the restart.
3. Push branch `laptop` with `laptopbench/PLAN.md`, or say where the plan lives.
4. Supply fork URL, model URL, model sha256.
5. Decide: power plan to Best performance, Modern Standby off, Defender/Search exclusions
   for C:\tmoe and the distro VHDX. These are system/security settings; not changed here.

**Retracted:** nothing (nothing was measured).

### L1-0b -- after the restart: WSL2 starts, Ubuntu-24.04 installed and provisioned; inputs still missing

Mohsen restarted Windows (uptime 00:01:36 at resume; CBS RebootPending cleared). `wsl --status`
now returns only `Default Version: 2`, no virtualization error. Confirms L1-0: the blocker was
the pending feature enablement, not firmware.

**Commands (Windows PowerShell)**

```
wsl --install -d Ubuntu-24.04 --no-launch            # "Distribution successfully installed"
wsl -d Ubuntu-24.04 -u root -e bash -lc "useradd -m -s /bin/bash -G sudo mohsen; \
   echo 'mohsen ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/mohsen; \
   printf '[user]\ndefault=mohsen\n[boot]\nsystemd=true\n' > /etc/wsl.conf"
wsl -d Ubuntu-24.04 -u root -e bash -lc "apt-get update && apt-get install -y build-essential cmake fio sysbench python3-venv git"
wsl --terminate Ubuntu-24.04
wsl -d Ubuntu-24.04 -e bash -lc "mkdir -p ~/tmoe/models ~/tmoe/logs; git clone /mnt/c/tmoe/temporal-moe ~/tmoe/temporal-moe; \
   cd ~/tmoe/temporal-moe; git remote set-url origin https://github.com/ncylich/temporal-moe.git; git fetch origin; \
   git checkout -B main origin/main; python3 -m venv ~/tmoe/venv"
```

The WSL clone was taken from the local Windows clone (source only; the checkout itself is on
the ext4 root, not /mnt/c) and re-pointed at GitHub. It sits on `main` at 68645b22, same as
the Windows clone; it has no `laptop` branch yet because origin has none.

**WSL-side facts**

| item | value |
|---|---|
| Distro | Ubuntu 24.04.4 LTS, WSL2, default user mohsen (uid 1000, sudo NOPASSWD), systemd on |
| Kernel | 6.18.33.2-microsoft-standard-WSL2 |
| CPUs visible | 8 |
| RAM visible | 7 GiB total (WSL default, no .wslconfig yet; the 12 GB / 4 GB / 2.5 GB caps come later from the driver) |
| Root fs | /dev/sdd ext4, 1007G, 2.8G used, 953G free |
| VHDX | C:\Users\mohsen\AppData\Local\wsl\{8193fb13-76e8-49c9-965b-4daa7b3aaff0}\ext4.vhdx, 3.01 GB |
| Toolchain | gcc 13.3.0, cmake 3.28.3, fio 3.36, sysbench 1.0.20 |
| Python | 3.12.3 system; venv at ~/tmoe/venv |

**Still not done, and why**

| item | reason |
|---|---|
| Fork clone, both sides | fork URL not supplied |
| Model download, sha256, copy into ~/tmoe/models | model URL and sha256 not supplied |
| `laptopbench/PLAN.md` | branch `laptop` still absent on origin (re-checked at resume) |
| Windows fio under C:\tmoe\tools | third-party executable download; left for Mohsen to place, or to approve explicitly. Not needed until Phase 2, which is blocked anyway |
| Power plan (Balanced -> Best performance), Modern Standby, Defender/Search exclusions for C:\tmoe and the VHDX path above | system/security settings, Mohsen's call |

**Retracted:** nothing (still nothing measured).

### L1-1 -- Phase 0 completed after the inputs arrived (all but the model), Phase 1 driver written and dry-run

Inputs received 2026-09-13 ~23:00: branch `laptop` at ffbdd5a4 (PLAN.md, SESSION_PROMPT.md), fork
`https://github.com/ncylich/llama.cpp.git` branch `temporal-moe` commit 61f6d1b4, model URL and sha256.

**Phase 0 items closed**

| item | result |
|---|---|
| repo | `C:\tmoe\temporal-moe` and `~/tmoe/temporal-moe` on `laptop`; L1-0/L1-0b rebased onto ffbdd5a4 |
| fork, both sides | `C:\tmoe\llama.cpp`, `~/tmoe/llama.cpp` at 61f6d1b4e "temporal-MoE: expert slot pool with streamed residency on CPU". Tip diff touches ggml/include/ggml-cpu.h, ggml/src/ggml-cpu/ggml-cpu.c, ggml/src/ggml-cpu/repack.cpp, src/llama-context.cpp, src/llama-graph.cpp, src/llama-mmap.cpp, src/llama-model-loader.cpp, tools/llama-bench/llama-bench.cpp. Upstream base 0badc06a = tag b9959 (2026-07-10) |
| **model** | **NOT AVAILABLE.** `https://huggingface.co/ncylich/temporal-moe-extras/resolve/main/serving/qwen3moe-rand-fine-Q4pure.gguf` returns HTTP 404 `X-Error-Code: EntryNotFound` (15-byte body). The repo API lists 413 files on its only branch `main`, last modified 2026-09-08T06:56Z; no `serving/` path and no `.gguf` anywhere. Datasets namespace: 401. PLAN 10.2 dates the upload 2026-09-14, after this session started. Re-checked 23:35 and 23:50: unchanged |
| fio, Windows | axboe/fio release fio-3.42 (2026-04-07), `fio-3.42-x64.msi`, 2,834,432 bytes, sha256 `d6bc1c0eb7a4b3bd2810e6c0ce605917a4671cc126c9dae5be7eb4891464a5c6`; extracted with `msiexec /a ... TARGETDIR=C:\tmoe\tools\fio` (administrative extract, nothing registered). `C:\tmoe\tools\fio\fio\fio.exe` = fio-3.42, sha256 `fe1bc26d83cf050fcaeef8b498bae7709f07f47d86902125236a31ccae3a3121`, lists `windowsaio` |
| stock llama-bench for native T1 | no C/C++ compiler on the Windows side (cl, clang, gcc, cmake, VS absent) and a toolchain is outside the allowed installs. Used the option the prompt allows: official ggml-org release b9959 (the fork's exact upstream base), `llama-b9959-bin-win-cpu-x64.zip`, 18,210,062 bytes, sha256 `e7b44f74a8413b96fc79551cebae517d1f5371ca4aec28d40d0a5589db0783b0`, extracted to `C:\tmoe\tools\llamacpp-b9959\`; `llama-bench.exe` sha256 `9e9a998886ec233ef5f3a07a142fd1de26444de3413b6c4cfb4efd63b33199a8`, loads `ggml-cpu-icelake.dll` (AVX-512 variant) on this CPU. Confound for T1: MSVC icelake variant against gcc 13 `-march=native`; a T1 gap of a few percent cannot be separated from the compiler |
| WSL build | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF`, `--target llama-bench llama-perplexity`, gcc 13.3.0, "Adding CPU backend variant ggml-cpu: -march=native", BUILD_EXIT=0 (`~/tmoe/logs/build*.log`). The driver's `build` stage re-runs this incrementally, copies to `~/tmoe/bin/llama-bench-temporal`, records hashes; the side-file dump waits on the model |
| power | Mohsen set the power mode: registry `ActiveOverlayAcPowerScheme = ded574b5-45a0-4f42-8737-46345c09c238` (Best performance overlay). `powercfg /getactivescheme` still prints the base plan "Balanced (381b4222...)" because on Windows 11 the mode is an overlay on the plan; both are recorded per row. Set per instruction: `powercfg /change standby-timeout-ac 0; powercfg /change hibernate-timeout-ac 0` -> STANDBYIDLE AC index 0x00000000, HIBERNATEIDLE AC 0x00000000. Modern Standby remains the firmware's only standby mode; with the AC timeouts at 0 and the lid open it is never entered. Processor AC: PROCTHROTTLEMAX 100%, PROCTHROTTLEMIN 5% |
| Defender | real-time protection ON. Exclusion lists need elevation: `Get-MpPreference` returns "N/A: Must be an administrator" for every exclusion field and the registry key is access-denied. Defender operational log (event 5007) 22:01-22:25 shows a platform update and service restart, **no `Exclusions\Paths` entries**, so as of 23:50 nothing evidences exclusions for `C:\tmoe` or `C:\Users\mohsen\AppData\Local\wsl\{8193fb13-...}`. The driver accepts either a readable list or `C:\tmoe\DEFENDER_EXCLUSIONS.txt` written by Mohsen naming the excluded paths (recorded as "attested" in every row). Until one exists `check` refuses, and any stage run needs `--force`, which is stamped in the row |
| Windows Search | WSearch running; crawl-scope WorkingSetRules contain nothing under C:\tmoe or the VHDX directory, and neither path is inside the default indexed scope (AppData\Local is default-excluded). Recorded, not changed |
| WSL memory | default VM: MemTotal 7831 MB, **2 GB swap on /dev/sdc** (pitfall #20). The driver's `.wslconfig` sets `swap=0` with every cap and records the VmSwap peak per run; a swapped run is status `swapped`, never `ok` |
| venvs | Windows `C:\tmoe\venv` (3.11.9), WSL `~/tmoe/venv` (3.12.3); the driver needs only the stdlib |

**Phase 1: `laptopbench/lapbench.py`** (Windows Python; stages check, probe, compute, build, gates,
arms, sweep, session, pack, plus report and decide). Design points, each traceable to a plan rule:

- Transport: every WSL command is a generated bash script under `C:\tmoe\logs\<tag>\cmd_NNNN_<label>.sh`
  run as `wsl -d Ubuntu-24.04 -e bash -c "bash /mnt/c/tmoe/logs/<tag>/cmd_NNNN.sh"` (`-lc` for the build),
  so the exact text that ran is on disk and is what `--dry-run` prints (pitfall #16 by construction).
- Cap: rewrites `%UserProfile%\.wslconfig` (`memory=<MB>`, `swap=0`), `wsl --shutdown`, waits for the
  distro, reads `MemTotal` inside and refuses if it is not within 80-102% of the cap. Per row:
  `cap`, `cap_mb`, `memtotal_mb`.
- `measure()` port: `~/tmoe/bin/wrap.sh` launches the engine and polls `/proc/<pid>/status`
  (VmHWM, VmSwap, VmRSS peaks) and `/proc/<pid>/io` every 0.25 s while it lives; diskstats delta on
  `sdd` as the device-wide cross-check; MemAvailable before and after; a zero decode or nonzero exit
  is never `ok`.
- Invariant 1: `arms` and `sweep` recompute the bench binary's sha256 and refuse unless `gates.json`
  has `all_pass` and lists that hash. Invariant 2: expected pool counters per arm (a: fetches=0,
  evictions=0; b: fetches=0, evictions>0, swaps>0; c and d: 45 fetches per token within 10% plus the
  45xR fill, 0.60-0.67 MiB per fetch; e: 8-18 fetches per layer per token); a disagreeing run goes
  to `refused.jsonl` with the counters, not to `runs.jsonl`. Invariant 3: `--set KEY=0` and
  `--knob NAME=0` exit with the pitfall-17 message; `--set KEY=` (empty) removes a flag.
  Invariant 4: the row carries tag, env, arm, tier, R, twopass, cap, flags, engine args, binary
  hash, clock probe (value, session reference, ratio, degraded), VmHWM, VmSwap peak, pool counters,
  read_bytes, host state (overlay, scheme, AC, battery, load, forced).
- Clock gate (PLAN 5.4): before each batch, cap 12G, stock resident `llama-bench -t 4 -p 0 -n 256 -r 4
  -mmp 0` (1024 tokens, about 20 s at the expected 50 tok/s); the session's first reading is the
  reference, more than 3% under it is degraded, then rest and re-probe twice, and the batch carries
  `degraded_clock` if still low. Then the arm's own cap is applied (a second restart for 4 GB arms),
  one warmup (`-n 32 -r 1`), then the measured run with the plan's engine line.
- Order: rounds of a, b, c, e for `--n`, a closing a, then d rounds over `--R`; `--rest` (300 s)
  between batches; `--resume` skips a (label, round, cap) already `ok`.
- Memory demonstration inside `arms`: ceiling at cap 4G recorded with rc, OOM detection and dmesg;
  deploy at 2.5G with the full protocol; VmHWM for all of them in `memdemo_<tag>.json`.
- Gates: G1 = `R=18`, no policy, `-n 16`, caches dropped first, pool `fetched_mib` against
  `/proc/<pid>/io read_bytes` minus the lazy loader's non-expert read (file minus expert bytes), 10%;
  G2 = `llama-perplexity` PROD+TWOPASS at R=192 vs R=18, `--chunks 2 -c 512 -t 4 --no-mmap`, every
  digit equal; G3 = `LLAMA_NO_REPACK=1` vs repacked at R=18 TWOPASS: PPL equal and tok/s apart by
  more than 3% and 2 sd.
- Rows: `pack` writes `results/ablations/serving_benchmarks_laptop.csv` in `emit_row.py`'s 12 columns.
  **Disagreement noted:** `emit_row.py` cannot be invoked verbatim; it hardcodes `android-cpu` and
  tier `ceiling` and reads one raw llama-bench CSV. The driver writes the same schema itself:
  `setup=laptop-wsl2-cpu`, `tier` in {ceiling, resident_control, deploy, floor}, `peak_vram_mib`
  carries VmHWM (peak RSS) and says so in `note`, `copied_bytes_per_token` carries bytes fetched from
  the NVMe per token (the laptop analogue of the CUDA swap copy; Android wrote 0).
- Compute (T1) on Windows uses the official b9959 binary; in WSL the fork's own build with no
  `LLAMA_TEMPORAL_*` variable, which is stock behaviour.
- DRAM number: WSL `sysbench memory --memory-oper=read --threads=1`; Windows has no sysbench, so a
  single-thread .NET `Buffer.BlockCopy` of 256 MiB x16 is recorded, labelled as a different method.

**Dry runs** (`--dry-run` for probe, compute, build, gates, arms, sweep, pack; 1608 lines) are saved
in `laptopbench/results/dryrun_commands.txt`; the arms schedule for `--n 3 --arms a,b,c,e` is 13
batches: (a b c e) x3, closing a, then the 2.5 GB deploy and the 4 GB ceiling for the demonstration.
Representative commands:

```
C:\tmoe\tools\fio\fio\fio.exe --thread --name=qd1_4k --filename=C\:\tmoe\probe\probe.bin --size=8G --rw=randread --direct=1 --iodepth=1 --bs=4k --ioengine=windowsaio --runtime=20 --time_based --norandommap --randrepeat=0 --group_reporting --output-format=json
fio --name=qd1_4k --filename=/home/mohsen/tmoe/probe/probe.bin --size=8G --rw=randread --direct=1 --iodepth=1 --bs=4k --ioengine=libaio --runtime=20 --time_based --norandommap --randrepeat=0 --group_reporting --output-format=json
C:\tmoe\tools\llamacpp-b9959\llama-bench.exe -m C:\tmoe\models\qwen3moe-rand-fine-Q4pure.gguf -t 4 -p 0 -n 128 -r 8 -mmp 0 -o csv
env -u LLAMA_TEMPORAL_R /home/mohsen/tmoe/bin/llama-bench-temporal -m /home/mohsen/tmoe/models/qwen3moe-rand-fine-Q4pure.gguf -t 4 -p 0 -n 128 -r 8 -mmp 0 -o csv
LLAMA_TEMPORAL_REPACK_DUMP=/home/mohsen/tmoe/models/qwen3moe-rand-fine-Q4pure-repacked.bin /home/mohsen/tmoe/bin/llama-bench-temporal -m /home/mohsen/tmoe/models/qwen3moe-rand-fine-Q4pure.gguf
/home/mohsen/tmoe/bin/wrap.sh <out> env LLAMA_TEMPORAL_REPACK=1 LLAMA_TEMPORAL_REPACK_FILE=<side> LLAMA_TEMPORAL_ODIRECT=1 LLAMA_TEMPORAL_MADV_FREE=1 LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6 LLAMA_TEMPORAL_SPIN_US=5000 LLAMA_TEMPORAL_R=18 LLAMA_TEMPORAL_TWOPASS=1 /home/mohsen/tmoe/bin/llama-bench-temporal -m <model> -t 4 -p 0 -n 128 -r 8 -mmp 0 -ot _exps=CPU -o csv
```

Real `check --env windows` at 23:49 refused with exactly the two open items ("Defender exclusions do
not cover ... read:n/a: must be an administrator", "model missing"), which is the behaviour wanted.
One driver bug found by the dry run itself: `pack --dry-run` wrote a header-only CSV and an empty
`comms/laptop/s2/`; both deleted, `pack` now returns before writing under `--dry-run`.

**Blocked:** model (HF 404); Defender exclusions unverifiable. **Next:** a preliminary T0 probe under
`--force` (tag `s0pre`) to validate the fio path and answer the WSL cached-read question early; the
`s1` probe is re-run once exclusions read correctly; T1 and everything after wait for the model.

**Retracted:** nothing.

### L1-1b -- Preliminary T0 (tag s0pre, `--force`): WSL2 QD1 latency is 1.70x native; provisional until the Defender re-run

`python lapbench.py probe --env all --tag s0pre --force` (forced only because the Defender exclusion
state is unverifiable; nothing else failed check). Both probe files pre-written, 8 GiB, random data,
non-sparse (Windows: `fsutil sparse queryflag` "NOT set as sparse"; WSL: `blocks512=16777224` for
8589934592 bytes, 6 extents). fio 3.42 `windowsaio --thread` native, fio 3.36 `libaio` in WSL2,
`--direct=1 --norandommap --randrepeat=0`, 20 s per point. One run each; n=1, so a ledger note.

| QD1 size | native mean / p50 (us) | WSL2 mean / p50 (us) | WSL2 / native (mean) |
|---|---|---|---|
| 4k | 86.3 / 68.1 | 146.9 / 134.1 | **1.70** |
| 32k | 122.3 / 102.9 | 181.8 / 171.0 | 1.49 |
| 108k | 158.9 / 138.2 | 237.9 / 218.1 | 1.50 |
| 216k | 202.2 / 179.2 | 288.7 / 268.3 | 1.43 |
| 648k | 376.5 / 325.6 | 508.8 / 473.1 | 1.35 |
| 1024k | 510.4 / 448.5 | 624.3 / 577.5 | 1.22 |

| 216k bandwidth | native MB/s | WSL2 MB/s | WSL2 / native |
|---|---|---|---|
| QD1 | 1089 | 763 | 0.70 |
| QD4 | 2882 | 2510 | 0.87 |
| QD8 | 2980 | 2837 | 0.95 |
| QD12 | 2984 | 2775 | 0.93 |

Fits (mean latency): native **106.3 us + 0.403 us/KiB** (r2 0.994); WSL2 **174.1 us + 0.465 us/KiB**
(r2 0.983). p50 fits: native 89.5 + 0.357, WSL2 161.2 + 0.431. So the VM adds about 68 us of fixed
cost per request and 15% per byte; at depth 8-12 the device saturates near 3.0 GB/s either way.
Neither environment is cached (4k QD1 far above 30 us). DRAM, not comparable across methods:
sysbench single-thread read in WSL2 38.97 GB/s; .NET single-thread BlockCopy native 14.2 GB/s
(a copy, counted once). Compare with the Pixel's UFS: 163 us + 0.63 us/KiB (S3-32); this NVMe is
faster on both terms even through the VM.

**What it means for PLAN 5.2, provisionally.** T0's rule is "WSL2 within 15% of native on QD1
latency". At 1.70x on 4k (1.22-1.50x at the expert-sized reads) WSL2 fails T0 by a wide margin. If
the clean re-run agrees, rule 2 applies: Linux native on a live USB, which needs Mohsen. Two caveats
before that call is made: (1) this run was forced past the Defender check, and the host's real-time
filter sits on the VHDX file that every WSL2 read passes through, so an exclusion could recover some
of the fixed cost; (2) the decision also needs T1, which needs the model. **Not decided yet.**

For the section-4 bound (once a ceiling exists): per token 45 x (fixed + 648 x per_KiB) = 45 x 367 us
= 16.5 ms native, 45 x 475 us = 21.4 ms WSL2 at QD1; at QD12, 28.5 MiB / 2.98 GB/s = 10.0 ms native,
/ 2.78 GB/s = 10.8 ms WSL2. The storage bound therefore sits near 60 tok/s at QD1 native and about
47 tok/s at QD1 WSL2, above the expected 50-60 tok/s ceiling, i.e. the laptop should be compute-bound
if the fetch overlaps well; that is what arm (c) against (b) will test.

Artifacts: `laptopbench/results/probe.json` (both envs, tag s0pre), logs in
`laptopbench/results/logs/s0pre/`, scripts in `C:\tmoe\logs\s0pre\`.

**Retracted:** nothing.

### L1-1c -- Orchestrator decision (rule 3, Windows native), model verified, clean T0 at n=2, port drafted; push blocked

**Decisions received 2026-09-14 ~00:15** (PLAN.md updated at 268fc1b6): Windows native is the reported
environment; WSL2's one remaining job is the correctness oracle (G1-G3 on the x86 build, G2 PPL
to every digit); the Windows port of PLAN section 6 is in scope on fork branch `temporal-moe-win`;
no live USB. Local branch `laptop` rebased onto 268fc1b6.

**Model.** The resolve URL returned 200 at 00:18:52 (`x-linked-size` 5,941,846,016). Downloaded at
00:26-00:29 into `C:\tmoe\models\qwen3moe-rand-fine-Q4pure.gguf`, 5,941,846,016 bytes, sha256
`d8a3bdf4a9c4a1563ad694718a44e155b7588e5ca47d3ee569bf9a8b8c3a2229` = expected. The pipeline's
`compute` stage copies it into `~/tmoe/models/` and re-hashes there (recorded with T1 in L1-2).

**Defender.** Mohsen is adding the two exclusions from an elevated shell; per instruction the session
proceeds under `C:\tmoe\DEFENDER_EXCLUSIONS.txt` (an attestation naming `C:\tmoe` and the distro's
VHDX directory) and every row and json from here carries
`provisional = "Defender exclusions attested in C:\tmoe\DEFENDER_EXCLUSIONS.txt, Get-MpPreference output pending; WSL2 rows pending Linux-native re-measurement"`.
Driver bug found on the first attempt: unelevated `Get-MpPreference` returns the literal text
"N/A: Must be an administrator to view exclusions" in the exclusion field, which the check treated
as a readable (empty) list instead of falling back to the attestation; fixed (c81a6344).

**Clean T0, tag `s1-provisional`** (attestation in place, before the download started; same fio
commands as L1-1b; one run per point, so with L1-1b native T0 is at n=2):

| QD1 size | native mean / p50 (us) | WSL2 mean / p50 (us) | ratio (mean) |
|---|---|---|---|
| 4k | 100.7 / 84.5 | 143.8 / 134.1 | 1.43 |
| 32k | 140.2 / 127.5 | 188.7 / 177.2 | 1.35 |
| 108k | 195.1 / 175.1 | 253.4 / 234.5 | 1.30 |
| 216k | 229.8 / 205.8 | 305.4 / 280.6 | 1.33 |
| 648k | 405.2 / 374.8 | 485.9 / 448.5 | 1.20 |
| 1024k | 527.9 / 481.3 | 652.7 / 602.1 | 1.24 |
| 216k bw QD1/4/8/12 MB/s | 958 / 2714 / 2873 / 2867 | 721 / 2649 / 2905 / 2840 | |

Fits (mean): native **130.5 us + 0.402 us/KiB** (r2 0.985; p50 115.3 + 0.372); WSL2 **178.7 us +
0.471 us/KiB** (r2 0.986; p50 166.2 + 0.433). Against L1-1b (106.3 + 0.403 native, 174.1 + 0.465
WSL2): the per-KiB terms repeat to 1%, the native fixed term moved 106 -> 131 us (the 4k point
86 -> 101 us), the WSL2 fixed term 174 -> 179. Run-to-run spread of the native fixed cost is
therefore about 20% at n=2 and the bound in section 4 will be quoted with both fits (the
orchestrator named 106 + 0.40; the clean run says 131 + 0.40). WSL2's excess over native is
1.2-1.4x at every size, so the rule-3 decision does not move. Neither environment cached.
DRAM: native .NET copy 15.5 GB/s, WSL2 sysbench read 36.9 GB/s (different methods, both single-thread).

**Port, branch `temporal-moe-win` at 0f2bdc0d6 (local, from 61f6d1b4).** The pool was entirely
inside `#if defined(__linux__)` with no-op stubs elsewhere, so a Windows binary of `temporal-moe`
silently has no pool. The port adds `ggml/src/ggml-cpu/temporal-port.h`: on Linux every name is a
macro expanding to the exact pre-port call (pthread, clock_gettime, C11 atomics, pread, madvise,
posix_memalign), so the Linux preprocessed source is unchanged; on Windows the same names are
overlapped `ReadFile` on a `FILE_FLAG_NO_BUFFERING` handle (offset in the OVERLAPPED, per-thread
event, 4 KiB-aligned reads as before), `DiscardVirtualMemory` for both MADV flavours (resolved at run
time; the SDK hides it below `_WIN32_WINNT` 0x0603), SRW lock + condition variable in exclusive mode,
`CreateThread`, `QueryPerformanceCounter` into `struct timespec`, `Interlocked*` and
`ReadAcquire8/WriteRelease8` under MSVC cl (clang-cl keeps `<stdatomic.h>`),
`GetFinalPathNameByHandle` in place of readlink, `SetThreadAffinityMask`, and no-op fadvise.
Linux-only under their own guards: ioprio_set, RWF_HIPRI preadv2, the preadv scatter of FUSED (on
Windows one unbuffered read of the fused region into a bounce buffer plus three memcpys), io_uring
(requesting `LLAMA_TEMPORAL_URING` on Windows aborts with a message). The rewrite of ggml-cpu.c was
scripted (`C:\tmoe\tools\port_ggml_cpu.py`, kept out of the repo) with every structural
substitution asserted to match exactly once and a final scan proving no POSIX call remains outside a
`__linux__` block. Loader: `_open(_O_RDONLY|_O_BINARY)` / `_dup` for the side-file descriptor. Dump
tool: `_fseeki64/_ftelli64/_chsize_s`, because MSVC `off_t` is 32-bit and the side-file is 6 GB.
Diff: 3 files, +311/-221 plus the 230-line header. **Not compiled anywhere yet**: no Windows
compiler is installed (Build Tools pending), and the Linux compile check waits for the WSL oracle
run to finish so the build does not perturb it.

**Push blocked.** `git push` to `ncylich/temporal-moe` and `ncylich/llama.cpp` is refused with
"Permission ... denied to mohsenfayyaz": the only stored credential on this machine is that GitHub
account (Windows credential manager), no `gh`, no fork. Both branches exist locally only until
Mohsen grants push rights or supplies a credential. The orchestrator's "push now" cannot be honoured.

**Retracted:** nothing.

### L1-2 -- WSL2 oracle: G1 exact, G2 = 185548.9246 +/- 4953.87147, G3 fails on kernel-family numerics; T1 reference; two engine findings

Environment decision: rule 3 (Windows native), taken by the orchestrator at 268fc1b6 on the L1-1b/L1-1c
T0 numbers (WSL2 4k QD1 1.43-1.70x native). The driver's own `decide` on the s1 data says the same
thing: T0 wsl/native 1.429 (143.8 vs 100.7 us), fail; T1 wsl/native 1.231 at -t 4 and 0.741 at -t 8,
fail. WSL2's only remaining job is the oracle below.

**Binary under test (WSL2, x86, gcc 13.3 `-march=native`, fork 61f6d1b4):**
`~/tmoe/bin/llama-bench-temporal` sha256 `b5498ebdde0d5da29d71f6796c349c7ab4013e0157c270ae9d8fafbfbdd624c3`,
`~/tmoe/bin/llama-perplexity` sha256 `7d9d65d8dd8e9cea58e31b592f560f910cfd8d0c9508ce015445e8d6dbde9e0c`.
Side-file built here: `~/tmoe/models/qwen3moe-rand-fine-Q4pure-repacked.bin`, 11,674,939,392 bytes,
98.2% allocated on disk (the rest are the 4 KiB alignment gaps the dump tool leaves), sha256
`cc5b8c60a699eff74e2fa70b44f04b4731f79d7ca53a202f765c6ab4a9a887f9`. Model in WSL re-hashed =
d8a3bdf4...2229. Full flag set for every gate run (the Pixel production set):
`LLAMA_TEMPORAL_REPACK=1 LLAMA_TEMPORAL_REPACK_FILE=<side-file> LLAMA_TEMPORAL_ODIRECT=1
LLAMA_TEMPORAL_MADV_FREE=1 LLAMA_TEMPORAL_SPLIT=2 LLAMA_TEMPORAL_FETCH_THREADS=6
LLAMA_TEMPORAL_SPIN_US=5000` plus `LLAMA_TEMPORAL_R=<R>` and, for the PPL runs,
`LLAMA_TEMPORAL_TWOPASS=1`. Cap 12G (.wslconfig memory=12288MB swap=0; MemTotal 11.9 GB). All rows
provisional (Defender attestation, L1-1c).

**T1 reference (stock behaviour, no LLAMA_TEMPORAL_ variable), `-p 0 -n 128 -r 8 -mmp 0`, one batch each:**

| | -t 4 tok/s (sd) | -t 8 tok/s (sd) | binary |
|---|---|---|---|
| Windows native | 24.05 (6.45) | 36.16 (4.71) | official b9959 win-cpu-x64, `ggml-cpu-icelake.dll`, sha256 9e9a9988... |
| WSL2 | 29.60 (3.97) | 26.79 (7.65) | fork build above |

The two binaries differ in compiler and kernel dispatch, so the cross-environment ratio is not a VM
measurement; it is reference only. Standard deviations of 15-30% on single batches say the
clock-probe and n=3 discipline of the arms stage is needed before any of these is quoted. The plan's
1.6 band (50-60 tok/s ceiling) was optimistic for LPDDR4x-4267: about 1 GB of weights move per
token here (0.5 GB non-expert + 45 x 18 x 648 KiB), and 24-36 tok/s is 24-36 GB/s of effective
bandwidth against a ~34 GB/s dual-channel peak, i.e. the ceiling is memory-bound where it should be.

**G1, bytes are real: PASS, exact.** Two identical runs of `R=18`, no policy, `-t 4 -p 0 -n 16 -r 1
-mmp 0 -ot _exps=CPU`; the first warms the loader's buffered read, the second is measured:

| pool `fetched_mib` | `/proc/<pid>/io read_bytes` | rel. error | diskstats (sdd) | rchar |
|---|---|---|---|---|
| 1466.9 MiB (6954 fetches) | 1466.86 MiB | **0.003%** | 1466.9 MiB | 1667.0 MiB = pool + 199 MiB loader |

*Finding 1 (method).* The first attempt (00:42) did it honesty_gate.py's way: drop caches, one run,
subtract the loader's non-expert bytes (file minus experts = 199.1 MiB). It read 2206 MiB at the
block level for 1467 MiB of pool traffic, a 37% miss, while `rchar` matched pool + loader to 0.1%.
The extra 540 MiB is kernel readahead on the loader's buffered reads spilling into the skipped expert
regions (135 gaps, a few MiB each on the WSL2 virtual disk). Warming the buffered part first removes
it; the unbuffered fetch path cannot be served from cache by construction. G1 is now stricter than
before, not looser: the tolerance is unchanged and the comparison is direct.

**G2, numerics exact: PASS.** `llama-perplexity -f androidbench/ppl_input.txt --chunks 2 -c 512 -t 4
--no-mmap -ot _exps=CPU`, PROD + TWOPASS:

| arm | Final estimate | pool line |
|---|---|---|
| R=192 (load-time repack, resident) | **PPL = 185548.9246 +/- 4953.87147** | fetches=0 evictions=0 swaps=0 |
| R=18 (streamed from the side-file) | **PPL = 185548.9246 +/- 4953.87147** | fetches=13095 fetched_mib=2762.2 evictions=0 |

Identical in every printed digit, and identical again to the 00:43 first attempt (both runs both
times). **This is the oracle the Windows binary must reproduce.** (The Pixel's ARM value for the
same input was 185405.9848; the x86 and ARM kernel families differ in the 4th significant digit,
which is expected and irrelevant to the gate.) TWOPASS shows `swaps=0` here because the two-pass
split only engages at n_tokens == 1 (decode); perplexity is prefill, so the run exercises the
single-pass residency path, which is the one G1 measures too.

**G3, repack real: FAIL as written.** Restated at R=192 (see finding 2), plain Q4_0 kernels
(`LLAMA_NO_REPACK=1`, no side-file, nothing streams) against the repacked path:

| arm | PPL | tok/s (-n 64 -r 3) |
|---|---|---|
| NO_REPACK, R=192 | 185544.4891 +/- 4953.07021 | 25.92 |
| repacked, R=192 (side-file bytes at R=18 gave the same digits) | 185548.9246 +/- 4953.87147 | 28.95 (sd 1.30) |

tok/s differ by 12% (> 3% and > 2 sd): the repacked kernel is in use. PPL differs at the 5th
significant digit. Because G2 already shows side-file bytes == load-time-repacked bytes to every
digit, the side-file is byte-correct (pitfall #9 is excluded); what differs is the AVX-512 repacked
GEMM's accumulation order against the plain `vec_dot_q4_0_q8_0`. On the Pixel the two ARM kernels
agreed, which is why the plan wrote "identical PPL"; on x86 they do not. **Not redefined here.** The
driver keeps `all_pass=false` for this binary, so nothing is timed on it; the orchestrator is asked
whether G3 on x86 should read "streamed-repacked == resident-repacked to every digit (G2) and
tok/s differs between kernel families", which the data above satisfies, or something else.

*Finding 2 (engine).* The first G3 attempt ran `LLAMA_NO_REPACK=1` with `TWOPASS=1` at R=18 through
llama-perplexity and hung for the full 7200 s timeout: stderr stops after "temporal-pool: active",
`read_bytes: 0`, rchar 206 MB (the non-expert weights). Mechanism, from the source: in two-pass mode
`ggml_tm_ensure` only orders the compute and submits nothing (window_fill does the submitting in
decode), and the plain `mul_mat_id` waits for experts that are never submitted; the repacked kernel's
`ggml_tm_wait_src_expert` submits on demand, which is why the repacked R=18 run works. No arm uses
NO_REPACK with TWOPASS at R < 192. Gate timeouts are now 1800 s.

**Not timed:** nothing on this binary (all_pass=false). WSL2 is not used for arms in any case.

**Artifacts:** `laptopbench/results/gates.json` (keyed by env), `compute.json`, `decision.json`,
`build.json`, logs in `laptopbench/results/logs/s2-provisional/` and `~/tmoe/logs/s2-provisional/`.

**Retracted:** the 00:43 "G1 FAIL 990%" and "G2 NOT FOUND" lines in `pipeline_gates_s2.log` were a
driver defect (the Windows-side copy of the WSL logs was empty because pathlib collapsed the
`\\wsl.localhost` prefix), not engine results; the WSL-side logs of that attempt carry the same
pool numbers and the same PPL digits as the rerun.

### L1-2b -- Port branch verified on Linux: builds, reproduces the oracle to every digit, semantic diff is cosmetic

`temporal-moe-win` (0f2bdc0d6, on 61f6d1b4) built in a WSL2 worktree (`~/tmoe/llama.cpp-win`, same
cmake line as the oracle build, gcc 13.3 `-march=native`): BUILD_EXIT=0; the only warnings are the
loader's five pre-existing `-Wmissing-declarations` (`llama_temporal_register_hot` and friends,
unchanged from 61f6d1b4).

**Functional check on the port binary's Linux build** (same model, side-file and flags as G2, R=18,
TWOPASS, `llama-perplexity --chunks 2 -c 512 -t 4 --no-mmap`):
`Final estimate: PPL = 185548.9246 +/- 4953.87147`, pool `fetches=13095 fetched_mib=2762.2` -- the
oracle's digits and the oracle run's fetch counts exactly.

**"Linux build byte-identical" -- not literally, and here is the whole difference.** Binary hashes of
the two trees differ, but that comparison is confounded twice (absolute worktree path in `__FILE__`
strings; build number 9961 vs 9960 baked into the library SONAMEs). The honest test is the
preprocessed `ggml-cpu.c` with the build's own flags (`-O3 -DNDEBUG -std=gnu11 -march=native ...`),
line markers stripped, paths normalised: 833 differing lines, of which everything but 117 is
whitespace and the typedef aliases (`tm_mutex_t` for `pthread_mutex_t`, `tm_cond_t`, `tm_au64_t`,
`tm_ai32_t`, `tm_file_t`, `tm_ssize_t`, `tm_off_t`). After normalising those, the residue is:
the seven typedef lines themselves; macro-parenthesised arguments (`(g_tm_odirect) ?`, `(use_free) ?`);
`if (!((fd) >= 0))` for `if (fd < 0)`; the three `g_tm_uring*` flag declarations moved ahead of the
io_uring block; `pthread_create/pthread_detach` wrapped in `do { pthread_t th; ... } while (0)`; and
shifted `__LINE__` numbers inside `GGML_ASSERT`/`ggml_abort` message strings (the pool region grew by
69 lines). No call, type, initializer or control flow differs on Linux. Verification artifacts:
`~/tmoe/logs/pp/{llama.cpp,llama.cpp-win}.i`, `a.n`, `b.n`.

**Windows compile: not yet.** Build Tools 2022 are still absent at 02:58 (no cl.exe, no clang-cl.exe,
no installer process). The driver's `build --env windows` checks for them before every attempt and
refuses otherwise; `dryrun_windows.txt` holds the exact cmake lines it will run.

**Retracted:** nothing.

### L1-3 -- Windows build of the port, smoke tests, and the four deviations the overnight run carries

Build Tools 2022 arrived 2026-09-15 ~13:20 (MSVC 14.44.35207, clang-cl, CMake). Mohsen added the
Defender exclusions from an elevated shell; Defender's own log (event 5007) records both paths
(`...Exclusions\Paths\C:\tmoe = 0x0`, `...Paths\C:\Users\mohsen\AppData\Local\wsl\{8193fb13-...} = 0x0`),
which the driver now accepts as evidence; the `Get-MpPreference` paste is still pending, so rows stay
provisional. Push to both GitHub repos is still refused (`mohsenfayyaz` has no write access).

**Windows build** (`build --env windows`): `cmake -S C:\tmoe\llama.cpp -B build-win -G "Visual Studio 17 2022"
-A x64 -T ClangCL -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF ...`, targets llama-bench and
llama-perplexity, copied to `C:\tmoe\bin\llama-bench-temporal.exe` / `llama-perplexity.exe` with their DLLs.
CPU backend variant `/arch:AVX512`. First compile stopped on one error: the four pool call sites inside
`mul_mat_id` (lines 3376-3438) were still under `#if defined(__linux__)`; widened (3d8cc14a). The side-file
dumped on NTFS by the ported dump tool has sha256 `cc5b8c60...887f9`, **identical to the WSL side-file**: the
Windows repack and the 64-bit seek/truncate path produce the same bytes. Latest binaries: see build.json
(`windows`), rebuilt after each port change below; the overnight gates hash the final one.

**Smoke tests on the ported engine (deploy config, `-n 4`, host not quiet, informational only):**

| run | rc | pool line | peak wset / peak commit |
|---|---|---|---|
| deploy, no cap, first port | 0 | fetches=2970 (2430 fill + 4x135) fetched 626.5 MiB evictions=540 swaps=180 | 823 / **5963 MiB** |
| deploy, job cap 4G, first port | 1 "failed to load model" | none | commit refused at allocation |
| deploy, no cap, reserve-mode port | 0 | same counters | 799 / **1000 MiB** |
| deploy, cap 4G and 2.5G, reserve-mode | 0 | same counters, 12.4 / 11.8 tok/s (fill-dominated) | 798 / 1001 MiB, job limit read back |
| deploy, cap 768M | abort: "FATAL VirtualAlloc(MEM_COMMIT, 110592 bytes) failed (1455)" | none | fails to start, as a cap should |
| (e) literal, repacked, R=18 no policy | 0 | fetches=5862 **evictions=0 swaps=0** | residency grows, not a floor |
| (e) repacked + SWAP_PROB=1.0 | 0 | identical to the literal run | SWAP_PROB ignored by the repacked kernel |
| (e) NO_REPACK + SWAP_PROB=1.0 | 0 | fetches=14112 (2920/token) **evictions=9252** (2313/token) hook_calls=675 | the vanilla floor |
| deploy + TRACE + FETCHPROF | 0 | fetchprof: 1.00 syscalls/fetch, wall 1153 us, sys 1146 us, **outside_sys 7 us**, 0 short reads; trace 5.9 MB, 62,789 events | instrumentation works on Windows |
| `llama-perplexity` R=18, oracle input | 0 | fetches=13101 fetched 2763.5 MiB | **PPL = 185468.5410 +/- 4951.23870** |
| same, repeated; and R=64 | 0 | 13101 both | 185468.5410 both: deterministic, residency-invariant |

Zero-copy note: the reserved buffer comes from `VirtualAlloc` (64 KiB aligned), so the pool's
"O_DIRECT zero-copy fetch ENABLED" path is taken on Windows; in WSL2 the heap buffer (`dst%4096=64`)
forced the bounce+memcpy path. The plain-kernel floor is in the plain CPU buffer and bounces.

**Port change 3, reserve-mode weight buffers (9cff55d1c).** Finding: ggml allocates the 5.3 GB expert
buffer with `_aligned_malloc`, and Windows charges commit for every committed page, touched or not, so
a `JOB_OBJECT_LIMIT_PROCESS_MEMORY` of 4 GB refused the *deploy* configuration at allocation, before the
lazy loader could skip anything (5963 MiB peak commit against 823 MiB peak working set). PLAN 5.3's
premise ("-mmp 0 commits everything, so the ceiling fails to start") was true of the ceiling and, on
Windows, of every configuration. Fix, Windows only, no-ops elsewhere: while the loader allocates weight
buffers with the pool configured (`LLAMA_TEMPORAL_R` set), CPU buffers of 1 GiB or more are
`VirtualAlloc(MEM_RESERVE)`d; `set_tensor`, `memset_tensor`, `cpy_tensor`, `clear`, the repack
`set_tensor`, the loader's direct `read_raw` and the pool's fetch commit the pages they write; eviction
decommits (`VirtualFree MEM_DECOMMIT`, both MADV flavours); free releases. Commit charge = resident
experts + non-expert weights + KV/compute, i.e. the memory claim, and the ceiling (which commits all
5.3 GB through `set_tensor`) fails at 4 GB with a clear "commit refused". Consequences: a refetch pays
commit + zero-fill (the MADV_DONTNEED cost model; MADV_FREE's overwrite-in-place does not exist here),
and computing on an evicted expert faults instead of reading zeros.

**Deviation 1: arm (e).** The plan's literal (e) (R=18, no TWOPASS, no ENFORCE, production flags)
never evicts on this fork: eviction, trim and SWAP_PROB live in `ggml_tm_ensure`, which only the plain
`mul_mat_id` calls; the repacked kernel has `ggml_tm_wait_src_expert` only (pitfalls #8 and #18, and
G1 in L1-2 showed `evictions=0`). Its fetches decay toward zero as residency grows, so it would report
a second ceiling. The overnight run records it once as `floor_R18_literal` with status `documentary`,
and reports as the vanilla floor `floor_R18_swap1_norepack`: `LLAMA_NO_REPACK=1`, R=18,
`LLAMA_TEMPORAL_SWAP_PROB=1.0`, no TWOPASS (the CUDA bench's prescribed turnover: every needed expert
refetched each op, 45 x 18 x 3 = 2430 slices per token expected, 2920 measured with sibling prefetch).
The kernel family differs from the other arms (plain vs repacked, 12% on tok/s at R=192); the floor is
storage-bound by two orders of magnitude, so that confound is noted, not corrected.

**Deviation 2: counters are per slice.** PLAN 1.2's "fetches ~45, fetched_mib ~28.5 per token" reads
as per expert; the pool counts per slice: 135 fetches of 216 KiB per token (28.5 MiB), 135 evictions,
45 swaps, plus a fill of 45 x R x 3. Verified exactly on the 4-token smoke run; the driver's
invariant-2 bands are set to that.

**Deviation 3: G2 against the oracle.** Windows R=18 gives 185468.5410, the WSL oracle 185548.9246;
6 more slice fetches (13101 vs 13095) say the router chose differently at some point. Windows is
deterministic (repeat identical) and residency-invariant (R=64 identical, same fetch count), which
argues for compiler rounding (clang-cl vs gcc on the same AVX-512 kernels, the same phenomenon as G3's
kernel-family difference) rather than a corrupted slice, but the decisive test is Windows R=192 vs
Windows R=18 on the same binary (6.6 GB; overnight, first thing). The overnight gates run with
`--gates-x86`: G2 passes on same-binary identity and records the oracle comparison; G3 passes on tok/s
differing between kernel families and records the PPL comparison. Both forms, and the as-written
verdicts, are stored in gates.json and stamped in every row's `provisional` note. If the orchestrator
rules otherwise, the rows are void; nothing is hidden.

**Deviation 4: ceiling-vs-cap semantics.** With reserve-mode, "the ceiling must fail to start at 4 GB"
holds through the commit refusal; deploy at 4 GB and 2.5 GB runs (smoke: 1001 MiB commit). Peak RSS is
reported as the peak working set; peak commit is recorded alongside.

**Overnight plan** (`C:\tmoe\tools\overnight.ps1`, detached; waits for load <= 20% and >= 9 GB free for
five consecutive minutes before starting; every stage retried on a check refusal and resumed):
gates --gates-x86 -> arms n=3 (a b c e interleaved x3, closing a, e-literal once, d at R=24/36/48 x3,
memory demonstration) -> attribute (a, b, c traced + fetch-profiled, decomposition (b)/(a) and (c)/(b))
-> sweeps in PLAN 7 order, n=3, interleaved, rest 180 s: FETCH_THREADS {4,6,8,12}, SPLIT {1,2,3},
SPIN_US {300,1000,2000,5000}, FUSED {off,on}, THREADS {4,8}, NOMADV {off,on} (diagnostic), SPINNERS {2,6},
EVICT_DEFER {off,on}, JANITOR_NOLOCK {off,on} -> pack (CSV rows, comms/laptop/w1, local commit) -> report.
Not in the sweep: URING (Linux-only), WORKER_AFFINITY (no effect on the Pixel; SetThreadAffinityMask
is wired if wanted), MADV_FREE vs DONTNEED (one call on Windows).

**Retracted:** nothing.

### L1-4 -- Windows gates: G1 0.35%, G2 same-binary identical, oracle difference attributed to the compiler; overnight run started

Runner: Scheduled Task `tmoe-overnight` (restart on failure, allowed on battery, wake to run; the first
detached attempt died when the laptop was unplugged and entered Modern Standby at 15:26). Battery-side
standby/hibernate timeouts set to 0 like the AC side. Started 23:15:58 with load 14%, 8.4 GB free, AC on.
Binary under test: `C:\tmoe\bin\llama-bench-temporal.exe` sha256 `36e143792cb6...` (fork
`temporal-moe-win` f9c46b374, clang-cl, /arch:AVX512), `llama-perplexity.exe` `3c614e174f89...`, Windows
side-file sha256 cc5b8c60... (= WSL's). Flags: the Pixel production set as in L1-2. Rows provisional
(Defender paste pending; gates in x86 form).

**G1, bytes are real on NTFS: PASS.** R=18, no policy, `-n 16 -r 1`, warm run first, then measured:
pool `fetched_mib` **1504.2**, process `ReadTransferCount` 1698.0 MiB minus the loader's 199.1 MiB =
**1498.9 MiB, rel. error 0.35%**; physical-disk read delta 1510.4 MiB (device_ok); 16,078 read calls.
The per-process counter on Windows counts requested bytes (buffered and unbuffered alike), so the
loader term is subtracted deterministically; the disk counter proves the device delivered them.

**G2, numerics exact: PASS on the same binary; the oracle is not matched, and the reason is now known.**

| arm (Windows binary) | Final estimate |
|---|---|
| R=192, nothing fetched | **185468.5410 +/- 4951.23870** |
| R=18, streamed from the side-file | **185468.5410 +/- 4951.23870** |
| WSL2 oracle (gcc build, L1-2), both arms | 185548.9246 +/- 4953.87147 |

Identical to every digit between the resident and the streamed arm: the ported fetch path delivers
exactly the bytes the resident model has (this also closes L1-3's open question: no slice is
corrupted). The resident arm, which fetches nothing, already differs from the gcc oracle, so the
difference is in the compute, i.e. clang-cl and gcc round the same AVX-512 kernels differently
(fp contraction and vectorisation order). G3 below shows the same across the plain kernels. The
"every printed digit against the WSL oracle" form of G2 therefore cannot be met by any Windows build of
this tree, and the same-binary form is the correctness gate that means something here. Recorded as
`pass_same_binary=true, matches_oracle=false, x86_form=true`; the orchestrator's ruling stands
outstanding.

**G3, repack real: PASS in x86 form, FAIL as written.** Plain kernels (`LLAMA_NO_REPACK=1`, R=192):
PPL 185526.5380 +/- 4954.46146, 25.12 tok/s; repacked R=192: 34.46 tok/s (sd 1.94): **+37% from the
repacked kernel on this CPU** (ARM gained 33%). PPL differs between kernel families, as on WSL2 (where
the plain kernels gave 185544.4891, also different from Windows' plain kernels: the cross-compiler
effect is independent of the repack).

`gates.json[windows].all_pass = true` under `--gates-x86`; the driver timed nothing before this.
Arms started 23:18:03 (schedule of 23 batches, 300 s rests, then the memory demonstration, attribution,
sweeps, pack). Results follow in L1-5.

**Retracted:** nothing.

### L1-5 -- Windows-native results, tag w1: five arms n=3, memory demonstration, attribution, first sweeps; SPLIT=1 is the finding

Overnight run 2026-09-16 00:17 to 10:04 (Scheduled Task, restarted once at 00:17 after the
verify_counters bug in L1-4's first attempt; rows before the restart were kept, the refused one was
rerun). Binary 36e143792cb6 (fork temporal-moe-win f9c46b374, clang-cl), gates in x86 form (L1-4).
Protocol as PLAN 1.3: `-t 4 -p 0 -n 128 -r 8 -mmp 0 -ot _exps=CPU`, one warmup per batch, 300 s rests,
clock probe before every batch, job-object caps (12G / 4G / 2.5G) read back on every run. All rows
provisional (Defender paste pending; gates x86 form). 49 rows in `runs.jsonl`, 2 in `refused.jsonl`.

**The five arms** (mean over all completed batches +/- population sd; `ok` rows are the subset whose
clock probe was within 3% of the session's best, see the probe note):

| arm | flags | cap | n | tok/s | ratio to ceiling | peak working set | per token (pool line) |
|---|---|---|---|---|---|---|---|
| (a) ceiling | R=192 | 12G | 4 | **32.84 +/- 1.75** (31.10, 35.73, 31.94, 32.58) | 1.000 | 5707 MiB | fetches 0, evictions 0 |
| (b) resident control | R=192 + TWOPASS | 12G | 3 | **17.36 +/- 0.43** (17.71, 17.62, 16.76) | 0.529 | 5705 MiB | 112.1 fetches (23.7 MiB), 135 evictions, 45 swaps |
| (c) deploy | R=18 + TWOPASS | 4G | 3 | **15.17 +/- 0.13** (15.14, 15.35, 15.04) | **0.462** | **799 MiB** | 137.4 fetches (28.98 MiB), 135 evictions, 45 swaps |
| (d) R=24 / 36 / 48 | TWOPASS | 4G | 3 each | 15.28 / 15.23 / 15.54 | 0.465 / 0.464 / 0.473 | 799-800 MiB | as (c) |
| (e) floor | R=18, NO_REPACK, SWAP_PROB=1.0 | 4G | 3 | **0.82 +/- 0.01** | 0.025 | 1345 MiB | 2469.8 fetches (521 MiB), 2465 evictions |
| (e-literal) documentary | R=18, no policy, repacked | 4G | 1 | aborted at 27.5 s | -- | 3933 MiB at abort | residency grew until "VirtualAlloc(MEM_COMMIT) failed (1455)": the cap refused the next fetch. Pitfall #18 made visible by the commit cap |

Counters match the arm requested in every recorded row (invariant 2); (b)'s 112 fetches per token
are the steady-state refetch of evicted experts (L1-4 note). Peak commit (stamped from the restart on):
deploy 1078 MiB, control 6035 MiB.

**Memory demonstration (PLAN 1.4).** Ceiling under the 4 GB cap: **failed to start** --
`ggml-backend.cpp:2255: temporal: commit refused; the model does not fit the memory cap`, job peak
4274 MB at the limit, working set 4051 MiB at abort, no decode. Deploy under 4 GB: 15.17 tok/s (n=3),
799 MiB. Deploy under **2.5 GB: 15.49 tok/s, 799 MiB** (n=1). The reserve-mode port (L1-3) is what makes
this observation possible on Windows.

**Bound (PLAN 4, native fit from L1-1c: 130.5 us + 0.402 us/KiB; QD12 216k = 2867 MB/s).**
fetch_QD1 = 45 x (130.5 + 648 x 0.402) us = 17.6 ms/token; fetch_QD12 = 28.5 MiB / 2867 MB/s = 10.4 ms/token;
1/ceiling = 30.5 ms/token. The storage term is below the compute term at either depth, so the bound
is the ceiling itself, 32.84 tok/s, and **deploy sits at 46% of the bound** (with the L1-1b fit,
106 + 0.403: fetch_QD1 16.5 ms, same conclusion). Perfect overlap would hide the fetch entirely; the
engine does not, see the attribution.

**Attribution (PLAN 7 step 1; `attribute_w1.json`; traced runs are -n 32 -r 2, 64 tokens, tracing costs
~8%):**

| arm | tok/s traced | GEMV mean / median us | per token: wall, WAIT, EVICT | fetch parts / token | FETCHPROF |
|---|---|---|---|---|---|
| (a) | 39.66 | 5.3 / 5.3 | 18.3 ms, 0, 0 | 0 | -- |
| (b) | 25.66 | 7.2 / 6.8 | 32.0 ms, 3.6 ms, 3.4 ms | 38 (transient: 64 tokens) | 735 us/part, sys 719, outside 17 |
| (c) | 15.11 | 7.9 / 7.1 | 63.1 ms, **25.2 ms**, 3.9 ms | 264 (135 slices x 2 parts) | **1103 us/part** (108 KiB), sys 1090, outside 13, 1.00 syscalls/part, 0 short reads |

(b)/(a) = 0.647: the two-pass policy costs 35% before any byte moves, and it is a genuine per-GEMV
cost (mean and median both up, 5.3 -> 7.2 / 6.8), not a shootdown tail (S3-38's signature was mean
up, median flat). In steady state (b) also refetches (112/token) and lands at 17.4. (c)/(b) = 0.589:
streaming costs another 41% at the production flags, and the trace says where: 25 ms of every 63 ms
token is WAIT, because a 108 KiB part costs **1.1 ms inside the read call** with 12 in flight, against
195 us at QD1 in fio and 2867 MB/s at depth 12. Our code is not in it (13 us outside the syscall,
one call per part). The Windows overlapped unbuffered path pays a large per-request cost under
concurrency, which is exactly what the SPLIT sweep found:

**Sweeps done (n=3 each, all-row means):**

| knob | values -> tok/s | verdict |
|---|---|---|
| FETCH_THREADS | 4: 13.84, 6: 15.45, 8: 15.19, 12: 15.31 | flat above 6; 4 costs 10%. Pixel's 6 holds |
| **SPLIT** | **1: 22.82**, 2: 15.51, 3: 11.52 | **SPLIT=1 is +47% over the Pixel production value**: fewer, larger requests win on this path (the Pixel found the opposite, pitfall #19). Deploy at SPLIT=1 = **69% of the ceiling / of the bound**, 799 MiB |
| SPIN_US | 300: 16.02 (n=2), 1000: 15.50, 2000: 15.56, 5000: 15.55 (n=1 each) | incomplete; 300 looks 3% better, inside noise until n=3 |

Pending (blocked on host memory since 10:04, see below): SPIN_US rounds 2-3, FUSED (one 648 KiB read
per swap: the SPLIT trend says this is the next candidate), THREADS 4 vs 8, NOMADV diagnostic,
SPINNERS, EVICT_DEFER, JANITOR_NOLOCK, then pack.

**Clock-probe note (method).** The resident probe itself drifted 27.7 -> 42.3 tok/s over the night
(session best 42.3), so under the 3% rule 26 of 49 rows carry `degraded_clock`; yet the deploy rows
span 15.04-15.68 across probes of 31.5-41.5 (storage-bound, clock-insensitive) while the ceiling
tracks its probe (31-36). Both means are reported (all rows; `ok` only); the arms' own spread is 2-4x
smaller than the probe's, so the gate as written mostly measures the probe. Suggested to the
orchestrator: apply the 3% rule to the resident arms and record the probe for the streamed ones.

**Refused rows:** (b) round 1 under the old band (rerun, L1-4); (e-literal) as above.
**Blocked:** the remaining sweeps and `pack`: Chrome and VS Code opened at 10:00 on 09-16 and free
memory has been 4.5-5.2 GB since; the clock probe needs ~6 GB, the check needs 7.5 GB, so the runner
has correctly waited (heartbeat every 10 min in overnight.log). Push: `laptop` pushed at 37debb8e
(access granted); `ncylich/llama.cpp` still refuses `mohsenfayyaz`, so `temporal-moe-win`
(f9c46b374) is local only.

**Retracted:** nothing.

### L1-6 -- Rulings recorded, depth is a dimension, the prefill residency excursion and the window trim, SPLIT=1 production; queue A-G started

**Rulings (orchestrator, 2026-09-20), applied.**
1. Clock probe: recorded per row; the 3% `degraded_clock` rule applies to arm (a) only, the interleaved
   repeat of (a) is the drift check for streamed arms. 24 w1 rows relabelled `ok` with
   `status_orig=degraded_clock` and the reason; no reruns.
2. **Windows gate definition** (this is the record): G1 as written (pool `fetched_mib` vs per-process
   block reads minus the loader's, 10%, plus the physical-disk delta); G2 = same-binary identity of
   the R=192 and R=18 perplexities to every printed digit, with the WSL2 (gcc) oracle comparison
   recorded but not required (clang-cl and gcc round the same AVX-512 kernels differently, L1-4);
   G3 = tok/s differs between the repacked and plain kernel families by more than 3% and 2 sd, with
   the PPL comparison recorded (they differ on x86, L1-2). Driver flag `--gates-x86`; gates.json
   carries `x86_form`, `pass_same_binary`, `matches_oracle`, `pass_as_written`.
3. Pitfall #26: the R = 24/36/48 rows (`deploy_R24/36/48`, 9 rows) are relabelled tier
   `r_inert_control`: under TWOPASS the window is `n_expert_used`, R is inert, and identical 137.4
   fetches/token and 799 MiB at every R are that fact, not a memory-speed curve.
4. Pitfall #30: every w1 row so far is at depth 0 (`-d` default); `depth` is now stamped in every row
   and is the CSV `context` column. The paper's protocol is depth 1024.

**Branch `laptop` merges layer-lexicality at 3ebfa3d1** (androidbench current: CTX_DESIGNPOINT,
pitfalls 25-30 read). **Fork push landed**: `temporal-moe-win` is on `ncylich/llama.cpp`, tip
**43bec5f31** (port 0f2bdc0d6 + mul_mat_id guards 3d8cc14ae + commit-tracking residency 1c48d6871 +
reserve-mode buffers 9cff55d1c + f9c46b374 + window trim 751bdc4ce/43bec5f31).

**Production flag change: `LLAMA_TEMPORAL_SPLIT=1`** (L1-5: 22.8 vs 15.5 tok/s, +47%, n=3). Written into
the driver's production set; FUSED is promoted the same way only if queue B says so
(`production_flags.json`, applied before per-run overrides).

**Depth finding (smoke, deploy, cap 4G, -n 8):** decode streams correctly at every depth (evictions
135/token, swaps 45/token), but the depth prefill runs through the single-pass repacked path, which
fetches on miss and never evicts (pitfall #25), so every expert it touches stays resident when decode
starts: peak working set **2563 MiB at depth 1024, 3257 MiB at depth 4096, against 799 MiB at depth 0**
(commit 2988 / 3680 / 1001 MiB). Reported as-is, the memory claim at the paper's depth would be 3x
the depth-0 one.

**Engine change (43bec5f31): trim to the window.** `ggml_temporal_window_fill` now evicts every
resident expert outside the window whenever a layer holds more residents than the window
(`n_resident > K+1`), streamed configurations only (R < E; the resident control keeps its experts).
It cannot be tied to the window's first fill because llama-bench decodes a warmup token before the
depth prefill and re-prefills every rep (first attempt 751bdc4ce did nothing: identical counters).
`LLAMA_TEMPORAL_TRIM=0` disables it, value-parsed, for the A/B in queue D. Smoke at depth 1024:
evictions 8148 vs 1083 without the trim, same fetch counts; decode 12.6 vs 16.7 tok/s over 8 tokens
(the one-off eviction of ~7000 slices lands inside a short run; the sweep prices it at 1024 tokens).
The prefill excursion itself is inherent: the fork has no expert-major prefill path, so the peak
working set stays at the prefill's level; rows now carry a working-set time series with
`wset_last_mib` (decode phase) beside `vmhwm_mib` (peak). Smoke, deploy depth 1024, -n 32: climbs to
2473 MiB through the prefill, drops to **1067 MiB** at the first decode token and stays there. So
at depth 1024 the memory demonstration at 2.5 GB will be refused by the prefill's commit (~3.0 GB)
even though decode lives in 1.07 GB; that is what queue E will record.

**Gates re-run on the rebuilt binary** (bench `99225874a8f0`, perplexity `503ed717a15a`, fork
43bec5f31, clang-cl), x86 form, 17:01-17:03: G1 1504.2 MiB vs 1504.198 MiB block reads (rel. error
1e-6), device 1504.3 MiB; G2 R=192 = R=18 = 185468.5410 +/- 4951.23870 (oracle 185548.9246 recorded,
not matched); G3 repacked 32.45 vs plain 26.65 tok/s. `all_pass=true`; nothing was timed before it.

**Queue (Scheduled Task, `overnight2.ps1`), started 17:03:** A ceiling and deploy (production flags)
at depth 0/1024/2048/4096, n=3, interleaved, same-depth ceilings (28 batches) -> B FUSED off/on at
depth 1024 n=3 -> C NOMADV on the resident control at depth 1024 n=3 -> D SPIN_US, THREADS, NOMADV,
SPINNERS, EVICT_DEFER, JANITOR_NOLOCK, TRIM, FETCH_THREADS at depth 1024 -> E final session (tag w2):
a, b, c, e at depth 1024 n=3 plus the memory demonstration -> F prefill diagnostic (w2) -> G pack.
Driver: `--depth`, `prefill` stage, per-depth ceilings in `report`, `production_flags.json`.

**Retracted:** nothing. (The 24 relabelled rows change status, not numbers.)

### L1-7 -- Queue A, the context-depth arm (paper row): deploy against a same-depth ceiling at 0 / 1024 / 2048 / 4096

Windows native, binary 99225874a8f0 (fork 43bec5f31, gates L1-6), production flags with SPLIT=1,
`-t 4 -p 0 -n 128 -r 8 -mmp 0 -ot _exps=CPU -d <depth>`, interleaved per depth (ceiling, deploy),
three rounds plus a closing ceiling per depth, 300 s rests, job caps 12G (ceiling) / 4G (deploy).
2026-09-20 17:03 to 09-21 00:26. Rows provisional as before (Defender paste pending; x86 gates).

| depth | ceiling tok/s (n) | deploy tok/s (n=3) | deploy / same-depth ceiling | deploy peak wset / decode-phase | deploy per token |
|---|---|---|---|---|---|
| 0 | 32.84 +/- 1.75 (4, L1-5) | 22.82 +/- 0.05 (the SPLIT=1 sweep rows, identical flags) | **0.695** | 799 / 799 MiB | 137.4 fetches, 135 evictions, 45 swaps |
| 1024 | 25.89 +/- 0.32 (4) | 16.74 +/- 0.77 (15.90, 16.57, 17.76) | **0.647** | 2563 / 1052-1066 MiB | 144.3 fetches, 141.9 evictions, 45 swaps |
| 2048 | 20.42 +/- 0.43 (4) | 13.83 +/- 0.55 | **0.677** | 2821 / (field noisy, see below) | 144.6 / 142.3 / 45 |
| 4096 | 13.17 +/- 0.14 (4) | 9.93 +/- 0.22 | **0.754** | 3257 / (same) | 145.0 / 142.6 / 45 |

Same-depth ceilings, as pitfall #27 requires; the ceiling falls 2.5x from depth 0 to 4096 (the Pixel's
e80 fell 2.2x). Counters match the arm at every depth (invariant 2); the +9 fetches and +7 evictions
per token above depth 0 are the window trim's refetches after each rep's prefill (L1-6).

**Reading.** From depth 1024 the ratio rises with depth (0.647 -> 0.677 -> 0.754), the Pixel's
mechanism (fixed swap cost amortised against attention; Pixel fine-shape: 0.639 -> 0.700 -> 0.774 ->
0.834). The depth-0 point (0.695) sits above depth 1024, unlike the Pixel; two things differ at
depth 0 on this rig: no trim refetches (137 vs 144 fetches/token) and no prefill excursion, and the
depth-0 ceiling comes from the 09-16 sitting (probes 27.7-35.2) while today's probes sat at
32.8-34.8. The depth-0 deploy rows are the `deploy_R18_SPLIT=1` sweep rows (same flags as production
now); the resume logic skipped a fresh depth-0 batch because the label collided with the old SPLIT=2
rows, which is recorded rather than patched. Queue E re-measures a, b, c, e at depth 1024 in one
sitting.

**Memory at depth.** Peak working set grows with depth because the depth prefill fetches on miss
and never evicts (L1-6); decode-phase residency after the trim is ~1.05 GiB at depth 1024 (799 MiB
of experts and non-expert weights plus the KV cache). For depth 2048/4096 the `wset_last_mib` field
in these rows is the raw last sample and sometimes caught teardown (values from 185 to 2136 MiB);
from queue B on it is the median of the run's last fifth (72d17e50). The peaks are exact.

**Clock probes** in this sitting: 32.8-34.8 tok/s on every batch; the ceilings carry
`degraded_clock` only because the session's reference was the 09-16 best (42.3); the reference now
resets per sitting (6 h), so those flags are a bookkeeping artifact of the rule, and the per-depth
closing ceilings show no drift (25.53 -> 25.64 -> 26.08 -> 26.32 at 1024; sd 0.32).

**Retracted:** nothing.

### L1-8 -- Queue B: FUSED loses at depth 1024, SPLIT=1 stays production. Queue C: the policy cost splits into two-pass and eviction

Same binary and protocol as L1-7, depth 1024, deploy cap 4G, control cap 12G, 2026-09-21 00:26-01:58.

**B. FUSED (one 648 KiB read per swap) against the production three-slice reads, deploy, n=3 each,
interleaved:** FUSED off **18.04 +/- 0.26** (17.67, 18.17, 18.28); FUSED on **17.26 +/- 0.31** (16.83,
17.52, 17.43). Counters identical (144.3 fetches, 141.9 evictions per token), working set identical
(2563 / 2567 MiB peak, 1066 / 1070 MiB decode phase). Difference -4.3%, larger than 3% and than 2 sd,
against FUSED. On Windows the fused path is one unbuffered read plus three memcpys (L1-3); the Pixel
rejected it for a different reason (the 512 KiB block-layer split). **Not promoted;** production stays
at SPLIT=1, `production_flags.json` was not written (decision file
`production_flags_decision.json`).

Sitting note: the same deploy configuration measured 16.74 +/- 0.77 in queue A (17:03-00:26, three
rounds spread over seven hours) and 18.04 here (00:26-01:09); the closing-ceiling drift check per
depth stayed flat, so this is between-sitting variation of the streamed arm itself, not clocks. Queue
E measures everything in one sitting.

**C. NOMADV on the resident control (R=192 + TWOPASS), depth 1024, n=3 vs n=2:** eviction on
**18.18 +/- 1.36** (n=3); NOMADV=1 (state machine and fetches unchanged, no page release)
**20.09 +/- 0.13** (n=2; the third batch was refused by the host check, below). Fetches 112.1 per
token both, evictions 135.0 both (counted, not executed under NOMADV), decode-phase working set 1035
MiB vs 5974 MiB (NOMADV never releases, as intended). Against the depth-1024 ceiling 25.89 (L1-7):

| | tok/s | fraction of ceiling | cost |
|---|---|---|---|
| ceiling | 25.89 | 1.000 | |
| control, no page release (NOMADV) | 20.09 | 0.776 | two-pass split + wait: **22%** |
| control, production eviction | 18.18 | 0.702 | eviction (decommit + refault): **a further 7.4%** |
| deploy (streamed, SPLIT=1, from B) | 18.04 | 0.697 | streaming at R=18: **~0%** on top of the control |

So at depth 1024 the policy costs 30% of the ceiling, of which roughly three quarters is the
two-pass split itself and one quarter the eviction; streaming the experts from the NVMe on top of
that policy costs nothing measurable here, which is L1-5's "control below the streamed arm" resolved:
the control refetches 112 slices per token in steady state (L1-4) and pays the eviction on top, so
it is not a zero-traffic arm, and the streamed arm at SPLIT=1 is as fast as the control. Two caveats
until E: n=2 on the NOMADV arm, and the control's sd (1.36) is 5x deploy's.

**Refused batch.** The third NOMADV=1 batch was refused at 01:58 by the memory check: 6174 MB available
< 7500 MB. Chrome had been started again on the host (about 2 GB across its processes); the runner
is waiting for the host to be free. Queue D follows.

**Retracted:** nothing.

### L1-9 -- Queue D: the depth-1024 knob sweeps. Flat except threads (SMT hurts 46%) and a 2% eviction-deferral gain

Same binary (99225874a8f0), deploy at production flags (SPLIT=1), depth 1024, cap 4G, n=3 per value,
A/B/A/B interleaved, 180 s rests, 2026-09-21 02:00-08:31 (after Chrome was closed). Every row's
counters match the arm (144.3 fetches, 141.9 evictions, 45 swaps per token; peak 2563 MiB,
decode-phase 1066 MiB). The "unset" value of each knob is the production configuration measured
inside that sweep, so each comparison is same-sitting.

| knob | values -> tok/s (sd) | delta | verdict |
|---|---|---|---|
| FETCH_THREADS | 4: 18.44 (0.08), 6: 18.77 (0.02), 8: 18.60 (0.09), 12: 18.55 (0.02) | +/-2% | flat; 6 stays |
| SPIN_US | 300: 18.76 (0.68), 1000: 18.65, 2000: 18.70, 5000: 18.67 | <1% | flat; 5000 stays |
| SPINNERS | 2: 18.71 (0.05), 6: 18.76 (0.01) | +0.3% | flat |
| EVICT_DEFER | off: 18.75 (0.02), on: **19.13 (0.02)** | **+2.0%** | small, > 2 sd, < 3%: not promoted, noted (the Pixel rejected it at -4.1%, the Samsung was neutral; pitfall #19 again) |
| JANITOR_NOLOCK | off: 18.70 (0.10), on: 18.93 (0.02) | +1.2% | inside the 3% band; noted |
| THREADS | 4: 18.68 (0.03), 8: **10.08 (0.08)** | **-46%** | eight compute threads on four cores put the SMT siblings under the fetch workers and the spin waits; `-t 4` is the only sane setting (the Samsung saw t8 -20%) |
| FUSED (queue B) | off 18.04, on 17.26 | -4.3% | rejected (L1-8) |
| NOMADV on deploy | off: 18.55 (0.09); on: **aborted, all three batches** | | with no page release the streamed arm's residency grows without bound and the 4 GB commit cap refuses it ("commit refused" before the pool line, refused.jsonl); the cap turning an unbounded-residency arm into a hard failure instead of a quiet swap is pitfall #20 handled; the meaningful NOMADV A/B is the resident control's (L1-8) |
| TRIM | -- | | the sweep was refused 50 times by the driver's own pitfall-17 guard (`=0`); TRIM is value-parsed, the guard now exempts value-parsed flags, and the A/B runs after queue E/F |

Between sittings the same deploy configuration reads 16.74 (queue A, 17:00-00:30), 18.04 (queue B,
00:30-01:10) and 18.5-19.1 (queue D, 02:00-08:30); within a sweep the interleaved values agree to
0.02-0.10 sd. The streamed arm therefore drifts by up to 10% over hours while its own probes stay
within 3%; the closing-ceiling drift check per depth was flat. This is why queue E takes all four
arms in one sitting.

**Retracted:** nothing.
