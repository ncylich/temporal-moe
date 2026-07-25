# Android Temporal-MoE Optimization — Progress Log

**Historical record only.** All work below was performed on a **Samsung SM-S942U1**
(Snapdragon 8 Elite Gen 5 / SM8850, Android 16, 11.4 GB RAM, 256 GB UFS, no root),
serial RFGL42B1VLW, over two sessions ending 2026-07-23. That test setup no longer
exists; nothing in this file is a runnable procedure. Full experiment-by-experiment
detail, including every retraction: `androidbench/LEDGER.md` (§S2-1 … S2-20). Raw
logs: `androidbench/results/`. Engine changes: `~/Documents/llama.cpp-android`
(branch off commit `0badc06a`, uncommitted diff to `ggml-cpu.c`, `llama-model-loader.cpp`,
`llama-mmap.cpp`, `llama-bench.cpp`).

---

## 1. What was measured and how it was kept honest

Goal: measure **temporal expert residency** — keep R of E experts resident, stream the
rest from storage on use — on a phone, as the mobile analogue of the A6000 CUDA path
(fixed expert pool + `cudaMemcpyAsync` + prescribed turnover).

Mechanism (built in session 2, replacing the failed page-cache approach): an explicit
**slot pool**. Experts live in anonymous memory (`--mmap 0`, `-ot "_exps=CPU"`, repack
disabled); evict = `madvise(MADV_DONTNEED)`; fetch = `pread` O_DIRECT of the expert's
exact bytes before its rows are computed. Test model: synthetic Qwen3-MoE, 10.5B params,
45 layers, 192 experts top-18, random weights (seed 0), uniform Q4_0 (216 KiB/slice).

Every configuration had to pass two gates before any number was recorded:

1. **Bytes are real** — the pool's own fetched-byte counter must match device-wide
   `/proc/diskstats` deltas (O_DIRECT makes every fetch a device read by construction).
   Typical agreement: 0.9–5%.
2. **Numerics are exact** — perplexity bit-identical (all printed digits) to the
   same-binary, same-flags, pool-disabled baseline, through tens of thousands of
   evict/refetch cycles. Reference: 181920.0251 (uniform-Q4_0 model, plain kernels).

Method rules that mattered: ratios only against a same-session ceiling; gate on
`scaling_max_freq` (thermal status lies on this device); ≥80% battery on wall power;
n=3 for anything that matters; measured bytes, never intended bytes.

## 2. Result progression (decode tok/s, R=36 of 192 unless noted)

| step | R=36 decode | mechanism |
|---|---|---|
| synchronous fetch, QD1, no overlap | 9.59* | fetch on compute thread, before any GEMV |
| + async worker pool (QD8) | ~13 | all of an op's missing experts in flight at once |
| + same-layer sibling prefetch | ~15 | gate/up/down share one routing; prefetch siblings |
| + janitor eviction + per-expert overlap | 23.4 | madvise off critical path; resident experts computed while fetches land, fetched expert computed last |
| + uniform Q4_0 model | 27.27 ± 0.02 | all slices 216 KiB; ~8% fewer bytes/swap + regular burst timing |
| + spin-then-sleep expert wait | **28.27 ± 0.17** | bounded 300 µs spin beats futex wake for ~100–300 µs residual waits |

\* the 9.59 came from a mixed prefill+decode run; the decode-only equivalent was 7.87.
Cumulative: **~2.9–3.6× improvement**, every step gated, headline numbers n=3.

## 3. Final numbers (uniform-Q4_0 model, same session, full clocks)

| configuration | decode tok/s | % of ceiling | expert RAM |
|---|---|---|---|
| fully resident (ceiling) | 47.8 | 100% | 5.3 GiB (192/192) |
| temporal R=48 | 28.6 ± 0.5 (n=3) | 60% | 1.33 GiB (4× cut) |
| temporal R=36 | **28.3 ± 0.2 (n=3)** | **59%** | 1.0 GiB (5.3× cut) |
| temporal R=24 | 24.3 ± 0.2 (n=3) | 51% | 0.67 GiB (8× cut) |
| temporal R=18 (=top_k) | 14.8 (n=1) | 31% | 0.5 GiB (10.7× cut) |
| fully streamed R=0 | 2.2 | ~5% | ~0 |

Prescribed-turnover arms (the CUDA `TEMPORAL_SWAP_PROB` analogue) exist for p=0.1/0.3
at R=18 on the earlier mixed-quant model; see LEDGER S2-7.

## 4. The physics: why 59% and not ~100% like the A6000

Per-layer accounting at R=36 (all measured; components sum to observed totals):

| component | fully resident | temporal R=36 |
|---|---|---|
| attention + router | 72 µs | 72 µs |
| expert FFN (18 experts × 3 GEMVs) | 337 µs | 337 µs (bit-identical) |
| exposed storage wait | — | 289 µs |
| pool overhead | — | 62 µs |
| **layer total** | **409 µs** | **760 µs** |

- Router jitter (random weights) changes **0.85 experts/layer/token** — coincidentally
  right at the temporal-MoE design point of ~1 swap/layer — costing 3 slice-fetches
  (gate/up/down) ≈ 550 KiB cycled per layer per token, ~24 MiB/token total.
- Best random-read throughput of this UFS at 216 KiB: **2.4 GB/s at QD≥4** (beats its
  own 1.9 GB/s sequential; saturates by QD8). → **bandwidth floor ≈ 10 ms/token**
  against a 21 ms compute budget.
- Discovery is per-layer (same-token semantics: the routing decision IS the discovery),
  so each layer's fetch burst (~630 µs wall) can only hide behind that layer's own
  ~340 µs expert compute. The unhidden remainder ≈ 290 µs × 45 layers matches the
  measured 13 ms/token exactly.
- The A6000 comparison in one ratio: β = fetch-time / compute-time per token.
  Phone: 10/21 ≈ 0.5 (and >1 at R=18). A6000 (host RAM over PCIe4 at ~25 GB/s,
  10–20 µs/expert): β ≈ 0.05 — fully hideable. Same k-ratio, ~10× worse memory-tier
  ratio. The phone's true CUDA analogue is experts-on-NVMe, not experts-in-host-RAM.

## 5. Optimizations REJECTED on evidence (do not re-try without new facts)

| idea | verdict |
|---|---|
| split each 216 KiB read into 2×108 KiB parallel halves | lost every interleaved pair (probe's latency win < bandwidth cost in shallow bursts) |
| bundle gate+up+down into one contiguous side-file read | probed: 1×696 KiB (909 µs QD1) loses to 3×216 KiB parallel (~400 µs); UFS latency grows with request size; not IOPS-limited |
| LRU eviction instead of FIFO window | out of scope (cache-affinity ≠ temporal technique) and near-zero effect anyway; churn is real router turnover |
| coarse-granularity model (64×648 KiB experts) | more bytes cycled per swap; dropped |
| `--poll 0` (sleeping ggml threads) | slightly worse |
| `preadv2(RWF_HIPRI)` polled IO | kernel/driver ignores it |
| worker IO priority (RT / BE-0) | no competing IO to beat |
| hot-worker spin-peek | a worker is already awake when bursts arrive |
| 4 vs 8 fetch workers | per-fetch latency −34%, decode unchanged — latency×parallelism product is conserved (device floor signature) |
| **io_uring (SQPOLL / fixed buffers / IOPOLL)** | **`io_uring_setup` = EPERM: Android forbids io_uring to unrooted processes. The kernel-stack share of fetch latency (~135 µs of 257 µs idle) is unreachable from stock userspace.** |

## 6. Bugs the gates caught (why the gates exist)

- **Session-1 residency controller never evicted** (madvise-only silently drops nothing
  from page cache; verified two-step madvise+fadvise is required, byte-exact). A full
  night of regime numbers voided.
- **Dead-fd eviction**: controller held the loader's fd, which closes after load —
  every later fadvise failed EBADF silently. Fixed by `dup()`; found because R=0 decode
  broke the 3.75 tok/s streaming roofline.
- **PPL kernel-family confusion**: pool runs read 185387.1179 vs baseline 185405.9848;
  first blamed corruption, then (wrongly) retracted the blame, then proved: plain-CPU
  vs repacked kernels have different FP accumulation order, and `-ot "_exps=CPU"` is
  only honored literally when the pool env is set (loader line ~1427). Bit-identity
  gates must compare same-kernel-family baselines. (LEDGER S2-10/S2-11 — including the
  retraction of the retraction.)
- **preadv2 offset truncation**: 32-bit lo/hi offset split on a 64-bit ABI truncated
  offsets ≥4 GiB — fetches read the wrong file region. Caught as PPL=nan by the gate
  before any performance number existed.
- **Page-cache approach abandoned on evidence**: kernel evicts always-hot attention
  weights under fault pressure (residency fell below the non-expert floor), and
  CPU-repack meant decode never read the mmap'd pages at all.

## 7. Platform/tooling facts specific to this device (recorded for reuse)

- Detached (nohup'd) processes are **invisible to ps/pgrep//proc from other adb shell
  sessions**. Cross-session per-pid polling silently reads nothing; use device-wide
  `/proc/diskstats` deltas around blocking runs instead.
- `pgrep -f`/`pkill -f` with a pattern contained in your own wrapper's command line
  match/kill the wrapper itself (exit 137 chains). Match by comm (15-char truncated).
- Thermal Status reads 0 while clocks are clamped to ~50%; gate on `scaling_max_freq`,
  sampled during the run.
- `ffn_down_exps` rows are 384 wide — not divisible by 256 — so K-quants are
  geometrically impossible for them; quantizers silently intermix types (Q5_0
  fallback). Uniform 4-bit requires Q4_0 (`--pure q4_0`).
- UFS random 216 KiB reads at QD≥4 exceed the device's sequential rate; QD16 buys
  nothing over QD8; per-request latency grows with size beyond ~216 KiB.
- Android battery reporting: wall charger = "AC powered", Mac cable = "USB powered";
  arms are only comparable within one power state.

## 8. Where the remaining headroom is (in order of leverage)

1. **Trained temporally-coherent router** (the actual FLAME-MoE technique; random
   weights already sit at ~1 swap/layer, the anticipatory loss should go below it).
   Every swap avoided saves ~550 KiB and ~290 µs/layer exposure.
2. **Storage SKU**: larger-capacity UFS parts have more NAND dies → 1.3–2× the random
   throughput at the same queue depth. Free-space on a given device is irrelevant for
   reads.
3. **Rooted / userdebug device**: unlocks io_uring, raw block reads, and ftrace — the
   ~50% kernel-stack share of per-fetch latency becomes addressable, worth an estimated
   +3–5 tok/s at R=36 and would let the paper report a measured "Android platform tax".
4. Architectures with a shared expert (e.g. Gemma-4-26B-A4B: 128 routed + 1 shared)
   shrink the churn-exposed fraction by construction and are the natural next target —
   26B at Q4 only fits this class of phone *with* temporal residency.

---

# Addendum: Pixel 10a campaign (2026-07-24, overnight)

**Device**: Pixel 10a "stallion" (Tensor G4, 1×X4 + 3×A720 + 4×A520, 7.75 GB RAM,
128 GB SK hynix UFS 3.1, Android 16, Magisk root). Historical record; setup since torn down.

**The defining fact**: MemAvailable ~3.6 GB vs a 5.53 GiB model. The fully-resident
baseline is UNRUNNABLE — the one attempt OOM-kernel-panicked the device (pstore:
"System is deadlocked on memory"). On this hardware class, temporal residency is an
enabler, not an optimization.

**Engineering added for this class of device** (all honesty-gated): lazy expert load
(loader skips expert bytes when R < E; kills the OOM transient, faster loads),
oom_score_adj shielding, per-arm RSS/swap rails, diskstats device auto-detect,
self-baselining PPL gate (new deterministic corpus `ppl_input.txt`, checked in).

**Topology is decisive on 1+3+4**: ggml barriers every op mean each A520 little-core
thread divides decode ~3× (t4=13.8 → t5=4.9 → t6=1.1 tok/s at R=36). Winning config:
4 compute threads (prime+mids), 8 fetch workers.

**Final Pixel curve** (t4/w8, zero swap, gates green, n=3, ceiling 16.6–17.5 tok/s by
byte-model + stall-subtracted cross-checks; GEMV-proxy upper bound 23.6 is
shape-confounded, reported as range):

| R | RAM cut | decode tok/s | % of ceiling | Samsung same cut |
|---|---|---|---|---|
| 18 | 10.7× | 5.65 ± 1.0% | 32–34% | 31% |
| 36 | 5.3× | 10.49 ± 2.5% | 60–63% | 57% |
| 48 | 4× | 10.85 ± 1.2% | 62–65% | 60% |
| **64** | **3×** | **11.76 ± 4.2%** | **70.8% (vs byte-model ceiling)** | — |

**The Samsung relative-retention curve transferred** — slightly better on the Pixel,
because slower compute widens the per-layer fetch-hiding window.

**Levers measured on this device** (beyond the Samsung set): worker→little-core
affinity (no effect, falsified); split-2 reads (probe favorable, in-engine falsified —
same conserved latency×parallelism product as Samsung); non-expert-only repack (no
effect; pipeline is RAM-bound); io_uring under root (WORKS here, unlike stock Samsung:
SQPOLL+fixed-buffers 315 vs 334 µs — bounded at <2.5% of the 1.4 ms in-engine fetch
latency, integration not warranted); UFS clkgate/rpm tuning (~5%); 4 vs 8 workers
(8 wins; this storage needs the depth).

**At the goal's target configurations the 70% mark was NOT met** (R=18: 34%; R=36: 60–63% of the inferred ceiling). R=64 (a 3× cut) measured 11.76 tok/s = 70.8% of the byte-model ceiling — a valid curve point, not the goal, which specified R=18. The R=18 shortfall is arithmetically forced by random-weight router churn (2.09 experts/layer vs the ~1 swap/layer design point); a trained temporally-coherent router at design point projects R=18 to ~70–72% on this same hardware and engine. For the smaller windows: Exact requirements
for 70% (any one): per-fetch latency ≤1.1 ms at depth (UFS 4.x-class part),
swaps/layer ≤0.63 (trained-router temporal coherence vs the random router's 0.85),
R≥64 (needs ≥3.6 GB free — not available on 7.75 GB devices), or ≥2.1 GB/s storage.
Full detail: `androidbench/LEDGER.md` §S3-1–S3-7.

---

# MEASUREMENT PITFALLS — READ BEFORE BENCHMARKING ANYTHING ON THIS RIG

Every entry below is a mistake that was actually made in this campaign, produced a
confident wrong conclusion, and cost real time. They are listed as rules. Violating any one
of them has already, at least once, made a bad idea look good or a good idea look bad.

### 1. Never size an in-engine change from an idle-device probe
A storage change must be measured **with compute running**. Idle probes said a fused
1×648 KiB expert read beat 6×108 KiB by 18%; in-engine it lost by 12.3%. The ranking
*inverted*. (LEDGER S3-28b → S3-29 → S3-30.)

### 2. Never size a latency-critical burst from a saturated-throughput curve
Sustained GB/s requires many requests pipelined in flight. A single expert swap cannot
pipeline (same-token semantics forbid loading the next layer early), so only single-burst
**wall time** counts. Using the throughput curve predicted 435 µs where the real floor was
~680 µs. (S3-28 → S3-28b.)

### 3. Always check syscall return values in probe code
A probe that ignores a short `pread`/`preadv` return silently measures a smaller transfer
and reports it as fast. Verify the byte count, then re-verify the conclusion.

### 4. Measure the statistic the observer actually described
"For each fetch, the nearest preceding evict" was insensitive by construction (0.2% hits)
and produced a wrong "correlation, not causation" verdict. "For each evict end, is there a
fetch right after" — the claim actually made — was visible immediately in raw timestamps.
When someone reports a pattern, reproduce *their* statistic. (S3-25b → S3-26.)

### 5. Interleave A/B/A/B; a cross-build single run is not evidence
Thermal drift on this device is ±6% run-to-run, larger than most effects being chased. One
reading of 15.96 ± 4.30 nearly became a finding; interleaved repeats showed the change was
neutral. Same-binary env-gated A/B beats rebuilding. (S3-26.)

### 6. Both arms must be the same kernel family, same session, same governor
A repacked baseline vs a non-repacked temporal arm inflated the apparent gap ~1.33×.
Never mix repacked and non-repacked, and never compare across governor states. (S3-19.)

### 7. Pin the DVFS floor, or you are benchmarking the governor
The engine idles ~66% of a token waiting on storage, so `sched_pixel` drops cpu4 toward
357 MHz and the next compute burst runs while the clock ramps. A continuously-busy baseline
never does this. That artifact alone was 1.25× on per-GEMV. Pin `scaling_min_freq` (root) —
and report both pinned and stock, never silently one. (S3-23.)

### 8. Anything wired into the custom `mul_mat_id` silently no-ops under CPU_REPACK
`repack.cpp`'s `forward_mul_mat_id` is a separate code path with no pool logic. Residency
waits, tracing, and eviction awareness each had to be added there separately — and each
omission was silent. This bit three times. When touching the expert matmul, check
**both** kernels. (S3-19, S3-20, S3-24.)

### 9. Reading raw bytes into a CPU_REPACK buffer is silently wrong, not slow
`file->read_raw(cur->data, …)` bypasses the buffer's `set_tensor`, leaving plain Q4_0 in a
buffer whose kernel expects the interleaved layout. Fast, and numerically garbage. Every
"resident repacked" number taken before this fix was computing on wrong weights. (S3-24.)

### 10. E == K is a dense model, not an MoE
A baseline with all experts active every token is the friendliest possible memory case and
flatters the baseline for reasons unrelated to the technique. Use the largest E that fits
resident at the same K and per-expert width. See `androidbench/BASELINE_POLICY.md`. (S3-22.)

### 11. Fast-and-wrong looks like a win; gate before timing
A missing residency barrier gave 34.9 tok/s by computing on not-yet-fetched bytes; an
eviction leak gave 32 tok/s by silently becoming resident. Both were "speedups". Run the
correctness gate (resident vs streamed byte-identical output) **before** recording any
number. (S3-19, S3-24.)

### 12. Prefer instrumenting the engine over building a look-alike harness
Three of the errors above came from standalone harnesses that differed from the engine in a
way that mattered. When a probe and the engine disagree, add timestamps to the engine
rather than adding features to the probe.

### 13. The UFS driver's `monitor` counters are ZEROED when you disable the monitor
`echo 0 > monitor_enable` memsets the driver's whole monitor struct, so every counter
reads back 0. Reading the counters *after* disabling makes a working instrument report
`nr=0` for every request size — which looks exactly like "the device never saw our
requests", and invites a wrong mechanism (merging, splitting, wrong device). Read the
counters while the monitor is still enabled, then disable. (S3-36.)

### 14. A device-side average is meaningless until the startup fill is differenced out
The first token fills the window: 45 layers × R experts × 3 tensors × `split` parts, which
for R=18 is 4860 fetch parts before any steady-state swap happens. That burst has a totally
different queue occupancy from decode, and at `-n 24` it is 43% of all requests. Any
"average concurrency" or "average latency" over a whole run is a blend of two regimes. Run
two token counts and difference them (`Δsum / Δbusy`) to get the decode-phase value. Doing
this moved the measured overlap from 3.47 to 3.21 and, more importantly, revealed that
io_uring's headline overlap of 7.04 was entirely a fill-phase artifact. (S3-36.)

### 15. `llama-cli` is not usable on this fork; gate with `llama-perplexity`
With `--no-mmap` the cli path issues ~17M tiny read syscalls (2.9 GB of `rchar` in ~165-byte
reads) and did not finish a single model load in 35 minutes, while `llama-bench -mmp 0` and
`llama-perplexity --no-mmap` take the lazy-expert-load path and start in seconds. Two hours
went into "the gate is hanging" before this was the answer. The output gate is now
`androidbench/gate_ppl.py` (bit-identical PPL, self-baselining). (S3-36.)

### 16. Nesting quotes through `adb shell` silently produces a shell continuation, not an error
`adb shell "... -p \"a b c\""` reaches the device shell unbalanced; it prints `>` forever and
the binary never runs. The harness then compares two EMPTY outputs and reports them equal.
Write the command to a script file, `adb push` it, and run that. (S3-36.)

### 17. A presence-parsed env flag is ENABLED by setting it to 0
`LLAMA_TEMPORAL_TWOPASS` (and several siblings) are read as `getenv(...) != nullptr`, so
`LLAMA_TEMPORAL_TWOPASS=0` turns the two-pass path **on**. A "no two-pass" arm must OMIT
the variable. Setting it to 0 and reading the resulting tok/s as a no-two-pass ceiling
would have silently compared an arm against itself. (S3-37.)

### 18. Read the pool counters before believing an arm ran the configuration you asked for
An `ENFORCE`-without-`TWOPASS` arm at R=192 measured 64.66 tok/s and looked like proof
that the swap policy is free. Its pool line said `fetches=0 evictions=0` — it performed
no swaps at all and was a replicate of the plain arm. The same check invalidated a
single-pass STREAMED arm: `evictions=0` and 3110 MiB fetched against the two-pass arm's
4613 MiB, i.e. residency was never bounded, because all residency management lives in the
two-pass window-fill op and not in `mul_mat_id`. `fetches`, `evictions` and `fetched_mib`
are the proof that an arm is the configuration named in its label. (S3-37.)

### 19. A rejection measured on one device does not transfer to another
`EVICT_DEFER` was rejected at −4.1% on the fetch-bound Pixel. On the Samsung the fetch is
entirely hidden and the madvise is worth +26%, so that rejection says nothing there. Every
tuning decision in `ENGINE_FLAGS.md` is conditional on the compute:storage ratio of the
device it was measured on. (S3-37.)

### 20. On a device with large zram, "does not fit in RAM" is silent
The Samsung has 12.5 GB of zram. A resident arm that does not fit does not fail — it swaps
3450 MB, keeps running, and prints a plausible 7.70 tok/s. Sample peak `VmSwap` from
`/proc/<pid>/status` for the life of every arm and void any arm that swapped. (S3-37.)
