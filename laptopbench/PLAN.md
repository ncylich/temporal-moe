# Temporal MoE on a personal laptop: measurement plan

Third serving measurement after the A6000 (`llamacpp-bench/`) and the phones (`androidbench/`).
The claim under test is the paper's: a temporal MoE serves in the memory of its active experts,
streaming the rest from storage, at a decode speed close to the fully resident model. On a
laptop the fast tier is DRAM, the slow tier is the NVMe, and the engine is the existing CPU
slot-pool fork. Nothing new is written until that fork has been measured on the new machine,
and every environment decision below is made by a number, not a preference.

First machine (Mohsen): i7-11370H (4 cores, 8 threads, AVX-512), 16 GB LPDDR4x-4267, Samsung
PM981a 1 TB NVMe, Windows 11. The Iris Xe iGPU is not used: batch-1 decode is
bandwidth-bound, the iGPU reads the same DRAM, and the streaming path never touches it.

## 1. The measurement

### 1.1 Model and what a swap costs

The fine 18-of-192 random-weight Qwen3-MoE of the Pixel campaign: 45 layers, hidden 1024,
expert width 384, uniform Q4_0. One expert is three 216 KiB slices, 648 KiB. All experts are
5.3 GiB; 18 resident per layer are 0.5 GiB; non-expert weights about 0.5 GB. The fully
resident model needs about 6 GB, the deploy configuration about 1.5 GB with KV cache.

Random weights make the router's choices meaningless, so swaps are prescribed, as the CUDA
bench does with `TEMPORAL_SWAP_PROB=1.0`. `LLAMA_TEMPORAL_TWOPASS=1` forces one random swap
per layer per token and splits the expert FFN into a resident sub-pass and a new-expert
sub-pass. One swap per layer per token is the trained rate, not a worst case:
`results/ablations/e1_swap_rate_by_layer.csv` gives `mean_swap_rate` 0.999 per layer. Per
token that is 45 swaps of 648 KiB, 28.5 MiB from the NVMe, and the pool's byte counter has
to agree.

### 1.2 The five arms

Mirrors setups a, b, c of `llamacpp-bench/README.md` and the decomposition in
`androidbench/LEDGER.md` S3-37c, where the two-pass policy alone cost 1.71x with every
expert resident and streaming added 10% on top. Without arm (b) a slow number cannot be
attributed.

| arm | flags | memory cap | pool line must show | isolates |
|---|---|---|---|---|
| (a) ceiling | `LLAMA_TEMPORAL_R=192`, production flags | 12 GB | `fetches=0` | fully resident. `BASELINE_POLICY.md`: the baseline is the largest E that fits, and 192 fits |
| (b) resident control | `R=192` + `TWOPASS=1` | 12 GB | `fetches=0`, `evictions>0` | the policy with zero bytes moved: two-pass split plus eviction |
| (c) deploy | `R=18` + `TWOPASS=1` | 4 GB | `fetches≈45`, `fetched_mib≈28.5` per token | the reported number |
| (d) R sweep | `R ∈ {24, 36, 48}` + `TWOPASS=1` | 4 GB | as (c) | memory against speed, as `ANDROID_OPTIM_PROGRESS.md` §3 |
| (e) vanilla floor | `R=18`, no `TWOPASS`, no `ENFORCE` | 4 GB | `fetches≈45×16` per token | free top-k with fetch on miss, the paper's vanilla-offload row |

Production flags start as the Pixel set in `androidbench/ENGINE_FLAGS.md` (`REPACK`,
`REPACK_FILE`, `ODIRECT`, `MADV_FREE`, `SPLIT=2`, `FETCH_THREADS=6`, `SPIN_US=5000`);
section 7 re-sweeps them.

### 1.3 Protocol

`llama-bench-temporal -m <model> -t 4 -p 0 -n 128 -r 8 -mmp 0 -ot "_exps=CPU"`. Decode only,
batch 1, no prefill, the Pixel protocol. Three batches per arm from rest with one warmup
(gate M4), five minutes between batches (M5), arms interleaved a, b, c, e, a so drift is
visible (pitfall #5). Every row records the pool counters, peak RSS, free memory, and the
clock probe of section 5.4.

### 1.4 Memory is demonstrated, not computed

The cap does two jobs. It keeps anything mmap'd from hiding misses in a warm page cache, and
it turns the memory claim into an observation: with `-mmp 0` the ceiling allocates 6 GB and
must fail to start under a 4 GB cap, deploy must run there, and should also run at 2.5 GB as
a tighter point. Peak RSS is reported for both. How the cap is applied depends on the
environment (section 5.3); what it must do does not.

### 1.5 What is reported

Rows go to `results/ablations/serving_benchmarks_laptop.csv` through `androidbench/emit_row.py`
with `tier` in {`ceiling`, `resident_control`, `deploy`, `floor`}, the sweep as `deploy` rows
with R in `note`. The paper gets a laptop column beside A6000 and Pixel in the style of
Table 2: ceiling tok/s, deploy tok/s and ratio, floor tok/s and ratio, measured peak RSS for
ceiling and deploy. Ratios are always against the ceiling measured in the same session on
the same binary. `laptopbench/LEDGER.md` carries every run including retractions, numbered
like `androidbench/LEDGER.md`.

### 1.6 Sanity bands

From section 4's physics on a PM981a with about 45 GB/s DRAM: (a) 50-60 tok/s, (c) 15-30,
(e) 2-3. Arm (b) has no prior and is the number the Samsung says to worry about. Deploy above
the ceiling or floor above deploy is a bug (pitfall #11). A deploy inside the band is not a
result until section 3's gates have passed on that binary.

## 2. Pass and fail, fixed before the first run

| step | passes when |
|---|---|
| T0 physics | fio 4K random read, QD1, unbuffered: WSL2 within 15% of native Windows, and absolute 60-200 us. Near 10 us means cached; that environment is out |
| T1 compute | stock `llama-bench`, fully resident: WSL2 within 5% of native Windows |
| G1 bytes are real | pool `fetched_bytes` matches the kernel's per-process read counter within 10% |
| G2 numerics exact | PPL identical in every printed digit between R=18 and R=192, same binary and flags |
| G3 repack real | `LLAMA_NO_REPACK` and repacked arms: identical PPL, different tok/s |
| (b) | recorded with `fetches=0`; its ratio to (a) reported whether or not it is small |
| (c) | tok/s, ratio to (a), and fraction of the section 4 bound. No target number |
| (e) | pool counters show about 16 fetches per layer before its tok/s is recorded |
| memory | ceiling fails to start under the 4 GB cap, deploy runs, peak RSS recorded for both |

## 3. Gates, before anything is timed

Both phone gates transfer. Only their transport changes.

- G1 (`androidbench/honesty_gate.py` gate 1): every fetch is an unbuffered device read, so
  the pool's byte count must match the OS per-process read counter. Catches a fetch path
  that quietly reads from cache and a side-file with a sparse hole (S3-32's first run).
- G2 (`androidbench/gate_ppl.py`): a missing residency barrier and a leaking eviction both
  looked like speedups on the Pixel. PPL is the oracle; greedy text on random weights is not.
- G3: confirms `ggml_temporal_repack_q4_0` emitted this CPU's interleaved layout rather than
  passing bytes through (pitfall #9). The side-file is built on the laptop, never copied.

Gates re-run after any change to the fetch path, the side-file, the flags, or the build.
The driver refuses to time a binary whose hash has not passed them.

## 4. Physics, so deploy has a bound to sit against

`LEDGER.md` S3-32 fitted one expert read as a fixed cost plus a per-KiB cost and derived
the deploy floor from it. The same fit on the PM981a is measured first, in every candidate
environment, on a pre-written non-sparse file:

```
fio --name=qd1 --filename=<probe.bin> --size=8G --rw=randread --direct=1 --iodepth=1 \
    --bs=$BS --runtime=20 --time_based --output-format=json
```

`BS` in 4k, 32k, 108k, 216k, 648k, 1024k, then `iodepth` 4, 8, 12 at 216k. `--ioengine` is
`libaio` on Linux and `windowsaio` on Windows. One DRAM number from `sysbench memory`.

With `fixed`, `per_KiB`, `bw(QD)` from the fit and the ceiling from arm (a):

```
bytes_per_token = 45 × 648 KiB ≈ 28.5 MiB
fetch_QD1       = 45 × (fixed + 648 × per_KiB)
fetch_QDn       = bytes_per_token / bw(n),   n = FETCH_THREADS × SPLIT
bound_tok_s     = 1 / max(1 / ceiling_tok_s, fetch_QDn)      perfect overlap
```

Deploy is reported as a fraction of `bound_tok_s`. T1 is the same for compute: stock
`llama-bench`, fully resident, `-t 4` and `-t 8`, in every candidate environment. It is the
session ceiling and the clock gate from then on.

## 5. Environment: three candidates, one rule

### 5.1 The candidates

| | WSL2 | Linux native (live USB) | Windows native |
|---|---|---|---|
| fork runs | unchanged | unchanged | after the port in section 6 |
| storage path | virtual SCSI over the NVMe | the NVMe | the NVMe |
| memory cap | `.wslconfig memory=` | `systemd-run -p MemoryMax=` | job object commit limit |
| G1 counter | `/proc/<pid>/io` | `/proc/<pid>/io` | `GetProcessIoCounters` |
| peak RSS | `VmHWM` | `VmHWM` | `PeakWorkingSetSize` |
| cost to start | an hour | an afternoon of Mohsen's time | about a week of engineering |
| touches his machine | no | boots from USB, writes nothing | no |

Native Linux and native Windows talk to the same drive with unbuffered reads. At the
physics level they are the same environment; T0 on both, when both exist, should agree
within noise. So "WSL is terrible and Windows is best" means the VM is the tax, and the
question is only how to get off the VM.

### 5.2 The rule

Run T0 and T1 on native Windows and WSL2 first. Both are cheap and neither needs the fork.

1. WSL2 passes T0 and T1: use WSL2. Proceed to section 3.
2. WSL2 fails, or later section 7 attributes a deploy gap to the virtual disk: move to Linux
   native on a live USB. Same fork, same scripts, same gates. Windows-native physics is
   reached without the port.
3. Linux native is unavailable (Mohsen declines to boot it, firmware blocks it, or the paper
   wants "runs on Windows" as a claim): do the Windows port. It is the only route that
   costs engineering, so it is chosen last and only by one of those three facts.

The choice is recorded in the ledger with the T0 and T1 numbers that made it.

### 5.3 Applying the cap

WSL2: `.wslconfig` `memory=12GB` for arms (a) and (b), `memory=4GB` and `2.5GB` for the
rest, `wsl --shutdown` between; the driver does this from the Windows side. Linux native:
`systemd-run --scope -p MemoryMax=4G`. Windows native: a job object with
`JOB_OBJECT_LIMIT_PROCESS_MEMORY`, which bounds commit, and `-mmp 0` commits everything, so
the ceiling fails to start exactly as on Linux.

### 5.4 Laptop state, whichever environment

- On AC, lid open, sleep and Modern Standby off, power plan Best performance;
  `powercfg /getactivescheme` recorded per run (gate M1).
- Nothing else running; Windows Search and Defender real-time scanning excluded for the
  benchmark tree, both compete for the NVMe (M9).
- Clock gate: a 20 second resident `llama-bench` before each batch replaces the Android
  `scaling_max_freq` probe. More than 3% under the session's first reading marks the batch
  degraded (M7), rest, retry. An 11370H holds turbo briefly and settles; the probe is the
  observable, not Task Manager.
- All files under one tree: `C:\tmoe` on Windows, `~/tmoe` on the ext4 root in WSL2 (never
  `/mnt/c`), `~/tmoe` on the USB's persistence volume with the model read from the NTFS
  volume through kernel `ntfs3`.

## 6. The Windows port, scoped

Inventory of the fork's diff (`434bd011`, 2437 lines, 8 files): 26 `pread`/`preadv` with
`O_DIRECT`, 28 `madvise`, 10 `mmap` (side-file), 7 `posix_fadvise`, about 40 direct
`pthread` mutex, condvar and create calls, 19 `clock_gettime`, one `sched_setaffinity`. The
io_uring path is already behind `#if defined(__linux__)` and stays Linux-only.

| POSIX | Windows | note |
|---|---|---|
| `pread` with `O_DIRECT` | `ReadFile` on a handle opened `FILE_FLAG_NO_BUFFERING` with an `OVERLAPPED` offset | buffers are already 4 KiB aligned for `O_DIRECT`; `preadv` for `FUSED` becomes one larger read |
| `madvise(MADV_DONTNEED / MADV_FREE)` | `DiscardVirtualMemory` | the eviction; S3-38's TLB cost question gets a second data point |
| `mmap` side-file, `MAP_POPULATE`, `MADV_WILLNEED` | `llama-mmap.cpp`'s existing Windows class plus `PrefetchVirtualMemory` | already in the tree |
| `pthread_*` | `ggml_mutex_*`, `ggml_cond_*`, `ggml_thread_create` in `ggml-cpu.c` | wrappers exist for both platforms; the fork bypassed them |
| `clock_gettime` | `ggml_time_us` | exists |
| `sched_setaffinity` | `SetThreadAffinityMask` | measured no effect on Pixel (S3-23); may drop |
| `/proc/<pid>/io`, `VmHWM` | `GetProcessIoCounters`, `GetProcessMemoryInfo` | in the driver, not the engine |

Roughly 400-600 lines behind an `#ifdef _WIN32`, MSVC or clang-cl build, then G1 through G3
from scratch on the new binary. One week including the gates. If chosen, the ledger's first
Windows entry is the G1 agreement number, because unbuffered I/O on NTFS is the part most
likely to be subtly wrong.

## 7. When deploy is short of the bound

Pitfall #12: instrument the engine before building a look-alike harness.

1. (b) far below (a): the policy. `LLAMA_TEMPORAL_TRACE=1` splits GEMV, WAIT and EVICT.
   `LLAMA_TEMPORAL_NOMADV=1`, diagnostic only, bounds the eviction share. S3-38 found the
   Samsung's eviction cost was TLB shootdown and that only a slot-pool redesign recovers it;
   on x86 a shootdown is an IPI across 8 hardware threads and may price differently.
2. (c) far below (b): storage. `LLAMA_TEMPORAL_FETCHPROF=1` reports time inside and outside
   syscalls and short reads. Outside-syscall time T0 did not predict is the virtual disk;
   that is rule 2 of section 5.2.
3. Both close: the laptop replicates, and the sweep below is headroom.

The Pixel flags were tuned to UFS and big.LITTLE. Pitfall #19: a rejection on one device
does not transfer. One knob per A/B, interleaved, n=3, gates re-run when the fetch path
changes.

| knob | Pixel | sweep here | why it may flip on NVMe |
|---|---|---|---|
| `FETCH_THREADS` × `SPLIT` | 6 × 2 | {4, 6, 8, 12} × {1, 2, 3} | NVMe scales past six in flight; T0's QD curve says how far |
| `URING`, `URING_IOPOLL` | rejected | on and off (Linux only) | NVMe has polled queues; UFS returned EOPNOTSUPP |
| `FUSED` | rejected, split at 512 KiB | on and off | read `max_sectors_kb` first |
| `SPIN_US` | 5000 | {300, 1000, 2000, 5000} | fixed latency about half |
| `MADV_FREE` | +7.6% Pixel, neutral Samsung | confirm once | already device-dependent |
| `-t` | 4 | 4 and 8 | SMT siblings, no little cores |
| `REPACK` | +1.33x ARM | on against `LLAMA_NO_REPACK` | x86 gain is its own number; same family both arms (S3-19) |

## 8. The command

One portable driver, `laptopbench/lapbench.py`, runs every stage in every environment. It is
the first deliverable, before any machine time. On Windows it runs under Windows Python and
drives WSL2 stages through `wsl -d Ubuntu-24.04 -e`; on a live USB it runs natively; after the
port it runs Windows-native. It replaces `androidbench/bench.py`'s `adb` transport and
Android probes and keeps its `measure()` discipline and `emit_row.py` schema.

```
lapbench.py <stage> --env {wsl,linux,windows,all} [options]

stages
  check     power plan, AC, cap in effect, exclusions, disk free, model sha256. Refuses on any failure.
  probe     T0: fio ladder and DRAM number; writes probe.json and the fitted fixed/per_KiB/bw(QD).
  compute   T1: stock llama-bench -t 4 and -t 8; writes compute.json; sets the session ceiling.
  build     cmake the fork for --env; dump the repacked side-file; record binary hash.
  gates     G1, G2, G3; writes gates.json; stamps the binary hash as gated.
  arms      the five arms, interleaved, rests, counters verified per run; appends runs.jsonl.
  sweep     one knob A/B/A/B from section 7; appends runs.jsonl.
  session   check, probe and compute if missing, build, gates, arms. Resumable.
  pack      emit_row.py to the CSV, tarball logs, commit results to comms/laptop/ on branch laptop.

options
  --arms a,b,c,d,e      subset, default all       --R 24,36,48       sweep points for (d)
  --n 3                 batches per arm            --rest 300         seconds between batches
  --cap 12G|4G|2.5G     applied per section 5.3    --set KEY=VAL      engine env overrides, repeatable
  --knob NAME=v1,v2     for sweep                  --tag NAME         session label in every row
  --resume              skip stages with fresh outputs                --dry-run  print the commands
  --force               override a failed check, recorded in the row as forced
```

Invariants the driver enforces rather than documents: no `arms` or `sweep` on a binary hash
without a passing `gates.json`; no run recorded whose pool counters disagree with the arm
requested (pitfall #18); every row carries env, cap, flags, binary hash, clock probe, peak
RSS, counters, and `--tag`; a `--set` of `KEY=0` is refused because a presence-parsed flag is
enabled by that (pitfall #17).

Two typical sessions:

```
python lapbench.py probe   --env all --tag s1          # T0 on Windows and WSL2
python lapbench.py compute --env all --tag s1          # T1 on both; decide per section 5.2
python lapbench.py session --env wsl --tag s2          # build, gates, five arms
python lapbench.py pack    --tag s2
```

## 9. Operation without inbound access to the laptop

The laptop is driven the way the pods are, through git, using the `relay` skill's
conventions on a branch named `laptop`. No SSH, no tunnel.

- The orchestrator commits an instruction, and any code, to `comms/orch/` on `laptop`.
- Mohsen pulls and runs one `lapbench.py` line from the instruction. A session is several
  hours unattended, which is why the driver's checks refuse rather than warn.
- He runs `pack`, which commits results under `comms/laptop/`, and pushes.
- The orchestrator's agent reads them, runs `androidbench/analyze.py` with `FILE_MIB` set to
  the new GGUF, writes the ledger entry and the next instruction.

## 10. Prerequisites that block everything

1. The fork: `https://github.com/ncylich/llama.cpp.git`, branch `temporal-moe`, commit `61f6d1b4` on upstream `0badc06a`.
   (Pushed 2026-09-14 from `~/Documents/llama.cpp-android`.)
2. The model: `https://huggingface.co/ncylich/temporal-moe-extras/resolve/main/serving/qwen3moe-rand-fine-Q4pure.gguf`, sha256 `d8a3bdf4a9c4a1563ad694718a44e155b7588e5ca47d3ee569bf9a8b8c3a2229`, 5.94 GB. This is the Pixel campaign's
   `qwen3moe-rand-fine-Q4pure.gguf`, uploaded 2026-09-14; `check` verifies the hash. The recipe
   to regenerate it is `androidbench/gen_model_fp16.py`, `convert_hf_to_gguf.py`, `llama-quantize`
   to pure Q4_0, and needs about 21 GB RAM.
3. `lapbench.py`, with the two gate scripts' transport removed. Written by the laptop session
   in its Phase 1.

## 11. Reuse map

| need | existing | change |
|---|---|---|
| engine, flags, production config | `~/Documents/llama.cpp-android`, `androidbench/ENGINE_FLAGS.md` | none; flags become the sweep's starting point |
| gates | `androidbench/honesty_gate.py`, `gate_ppl.py` | drop `adb shell` |
| driver discipline and schema | `androidbench/bench.py` `measure()`, `emit_row.py`, `analyze.py` | become `lapbench.py`; set `FILE_MIB` |
| latency model, floor derivation | `LEDGER.md` S3-32 | refit on the PM981a |
| decomposition, policy-cost warning | `LEDGER.md` S3-37c, S3-38 | reproduce as arm (b) |
| pitfalls and gates | `ANDROID_OPTIM_PROGRESS.md` line 213 on, `BENCHMARK_GATES.md` | M1, M4, M5, M7, M9 apply; M20-M23 do not |
| baseline definition | `BASELINE_POLICY.md` | E=192 fits, so the ceiling is the full model |
| model recipe | `gen_model_fp16.py`, `llamacpp-bench/gen_random_qwen3moe.py` | none |
| Windows mmap class | `src/llama-mmap.cpp` | reused by the port |
| remote loop | the `relay` and `poll-log` skills | branch `laptop`, results as commits |

## 12. Sequence

1. Mohsen does section 5.4; the laptop session writes `lapbench.py` in its Phase 1.
2. Session 1: `probe` and `compute` with `--env all`. Apply section 5.2. If rule 3, start
   section 6 and the schedule slips a week.
3. Session 2: `session` in the chosen environment. First laptop rows in the CSV, first ledger
   entries, the memory demonstration.
4. Section 7: attribute any gap, then sweep. Two or three sessions.
5. Final session: all five arms in one sitting, ledger verdict block modelled on the top of
   `androidbench/LEDGER.md`, paper column.

Out of scope: prefill, the iGPU, other laptops until this one has a number.
