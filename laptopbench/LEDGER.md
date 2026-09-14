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
