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
