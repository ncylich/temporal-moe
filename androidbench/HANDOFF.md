# Handoff — temporal-MoE on Pixel 10a (2026-07-24, after S3-36)

## Read these first (permanent docs, not this file)

| doc | why |
|---|---|
| `../ANDROID_OPTIM_PROGRESS.md` § **MEASUREMENT PITFALLS** | 16 rules, each a mistake actually made here. Read before benchmarking anything. |
| `ENGINE_FLAGS.md` | every `LLAMA_TEMPORAL_*` flag, the production config, the measured physics constants. |
| `BASELINE_POLICY.md` | what a legitimate baseline is (largest E that fits resident at the same K and width). |
| `LEDGER.md` §S3-19…S3-36 | the full evidence trail, including every rejection and retraction. |
| `probes/README.md` | the storage probes, and the traps in using them. |

This file is the ephemeral part: current state and what to do next.

## State

Best measured: **~21 tok/s** (DVFS pinned) = **~69% of the E=112 resident ceiling** (~30.5),
at **9x less expert RAM**. Stock governor ~16 tok/s. Session start was 13.67 / 38%.

Where a layer goes: **453 us resident compute + ~420 us exposed stall + 28 us new-expert pass.**

Correctness gate: **`gate_ppl.py`** — bit-identical perplexity, self-baselining, currently
passing. (Do NOT use `llama-cli` for gates; pitfall #15.)

## Closed — do not re-litigate

- **Engine fetch-path overhead is 0 us.** `FETCHPROF` reports 1.00 syscalls/fetch and
  `outside_sys = 0` in every shape. No software layer left to strip. (S3-33)
- **The I/O "serialisation" does not exist.** ← NEW, S3-36. Measured at the UFS driver:
  during decode ~3.2 of our 6 requests are genuinely in flight, at ~350 us each, delivering
  **~0.97 GB/s** — and this is the same for the pread pool and for io_uring. Offering more
  concurrency raises per-request latency proportionally and leaves total device-busy time
  unchanged (Little's Law on a saturated resource). **The device is the limit.**
  The burst floor for a 648 KiB swap is therefore **~654 us**, not the 435 us the throughput
  curve suggests nor the 571 us the QD1 law suggests, and only ~126 us/layer of the current
  ~780 us is recoverable at all.
- **io_uring: built, gated, REJECTED -11.4%** (S3-36). Batched submission plug-merges the
  two adjacent parts of an expert into one 216 KiB request, undoing `SPLIT=2` (689 merged
  requests vs 140 for pread in the identical workload). IOPOLL is `EOPNOTSUPP` on
  SCSI/UFS; SQPOLL is worse; registered buffers are design-rejected (pinning defeats
  MADV_FREE eviction → unbounded residency → pitfall #11). Flag kept, off by default.
- **Block layer: nothing to win.** `/data` = dm-63 → sda34; sda runs mq-deadline with
  `rq_affinity=2`. Both `sched=none` and `rq_affinity=0` measured neutral in-engine,
  interleaved. blkio cgroup has no throttle set (cgroup-v2 exposes only `memory`).
  Queue depth is not the limit (`can_queue=31`, we offer 6).
- Scheduling: spinner cap, worker count, worker affinity, evict timing/deferral, spin
  budget — all measured neutral or negative.
- Request shaping: `SPLIT=2` (6 concurrent parts) optimal; 3-way and 1-way worse. Fused
  single-request **rejected, -12.3%**.
- UFS power management (clkgate, rpm_lvl, power/control, max_sectors_kb=1024): no effect.
- Bytes per swap: K=24/ff=288 cut bytes 22%, bought only +2.4%.

## The one thing left

### Sweep K upward — now the ONLY lever, and it is the right one
We are fetch-bound: compute 453 us vs a fetch whose **device floor is ~654 us**. Storage
time is bought and cannot be given back — so the move is to buy more compute with it.
Fetch cost is fixed per layer (one swap) regardless of K, so raising K raises compute
without raising the fetch. Where compute meets ~654-780 us the stall goes to ~0 and
temporal runs at ~100% of *its own* ceiling with a larger active model at roughly today's
absolute tok/s.

That reframes the headline from "69% of a resident model" to **"N× more active parameters
at the same speed and 9x less expert RAM."**

Sweep K = 24, 27, 31, 36; find where `compute(K) == fetch`. Variants `narrow` (E=288 K=27
ff=256) and `e176n` (its resident baseline) are defined in
`../llamacpp-bench/gen_random_qwen3moe.py` but **not generated**. Keep `ff` a multiple of 64
or `expert_bytes` stops being 4K-aligned and half the fetches fall off the zero-copy path.
Regenerate the resident baseline at the SAME shape (BASELINE_POLICY) — comparing a new K
against the old E=112 baseline is exactly the error §S3-22 exists to prevent.

## Hard floor
**~654 us per 648 KiB swap, measured at the device** (~0.97 GB/s for random 108 KiB
O_DIRECT reads). Even a perfect fetch caps at the resident ceiling (~30 tok/s), because
that is what the compute costs.

## Environment
- Device: Pixel 10a, rooted, serial `5C111JEA320125`. **Always run `probes/unpin.sh` when
  done** — leaving the DVFS floor pinned drains the battery.
- Reboot before any resident-model arm: this device sits at ~2.8 GB MemAvailable after a
  long session and a 3.5 GB resident load goes to zram and thrashes. After reboot: ~4.7 GB.
- Root is required for io_uring and for every sysfs knob; `su -c` works, and `$(...)` inside
  an `su -c "..."` string is expanded by the OUTER (shell-user) shell — escape it.
- Models in `/data/local/tmp/tmoe/`: `qwen3moe-rand-fine-Q4pure.gguf` (E192 K18 ff384),
  `qwen3moe-rand-k24-Q4pure.gguf` (E256 K24 ff288), `qwen3moe-rand-e112-Q4pure.gguf`
  (resident baseline), plus their `*-repacked.bin` side-files.
- Engine: `~/Documents/llama.cpp-android`, **uncommitted** diff to `ggml-cpu.c`,
  `ggml-cpu.h`, `repack.cpp`, `llama-graph.cpp`, `llama-model-loader.cpp`, `llama-bench.cpp`,
  `llama-context.cpp`, `llama-mmap.cpp`. Build: `cd build-android && ninja llama-bench
  llama-cli llama-perplexity` — and push **all three**; a stale `llama-perplexity` silently
  gates the wrong binary.
- Harnesses added this session: `gate_ppl.py` (correctness gate), `run_uring_ab.py`
  (interleaved arms + UFS overlap + block-layer knobs), `ufs_chunk_sweep.py` (what request
  sizes the device actually sees, and the overlap at each).
- Timeline artifact regenerates via `make_timeline.py` (never hand-write it):
  results/timeline_artifact.html
- The engine diff is uncommitted and represents two sessions. Consider committing it.
