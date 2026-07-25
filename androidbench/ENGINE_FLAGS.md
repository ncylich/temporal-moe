# Temporal-MoE engine: environment flags

Fork: `~/Documents/llama.cpp-android` (branch off `0badc06a`). All flags are read once in
`ggml_temporal_pool_register` (`ggml/src/ggml-cpu/ggml-cpu.c`) unless noted.

## Production configuration (best measured, Pixel 10a)

```
LLAMA_TEMPORAL_REPACK=1
LLAMA_TEMPORAL_REPACK_FILE=<model>-repacked.bin
LLAMA_TEMPORAL_TWOPASS=1
LLAMA_TEMPORAL_ODIRECT=1
LLAMA_TEMPORAL_MADV_FREE=1
LLAMA_TEMPORAL_R=<K>
LLAMA_TEMPORAL_SPLIT=2
LLAMA_TEMPORAL_FETCH_THREADS=6
LLAMA_TEMPORAL_SPIN_US=5000
  ./llama-bench-temporal -m <model>.gguf -t 4 -p 0 -n 48 -r 3 -mmp 0 -ot "_exps=CPU"
```

`-t 4` is required: 6 threads drags the little A520 cores into the ggml barriers and costs
~10x. Cool-gate on `scaling_max_freq` before every arm.

## Residency and policy

| flag | default | meaning |
|---|---|---|
| `LLAMA_TEMPORAL_R` | off | experts held resident. `< n_expert` enables the pool AND lazy expert load (loader skips reading expert data). `== n_expert` = fully resident baseline. |
| `LLAMA_TEMPORAL_TWOPASS` | off | two-pass expert FFN (resident sub-pass + new-expert sub-pass) and enforced 1 random swap/layer/token. Implies `ENFORCE`. |
| `LLAMA_TEMPORAL_ENFORCE` | off | enforced random swap without the two-pass split. |
| `LLAMA_TEMPORAL_SWAP_PROB` | 0 | prescribed turnover probability (the CUDA analogue). Not used with TWOPASS. |

## Fetch path

| flag | default | meaning |
|---|---|---|
| `LLAMA_TEMPORAL_ODIRECT` | off | O_DIRECT fetches. Makes every fetch a device read by construction (gate 1). |
| `LLAMA_TEMPORAL_SPLIT` | 1 | sub-reads per expert slice, 1..4. **2 is optimal** -- 6 concurrent parts per swap. |
| `LLAMA_TEMPORAL_FETCH_THREADS` | 4 | fetch worker threads. 6 optimal; more does not help (device saturates ~6-way). |
| `LLAMA_TEMPORAL_SPINNERS` | 2 | workers allowed to spin-poll the queue instead of sleeping. Raising it collapses median queue wait (151 -> 22 us) but does **not** change throughput -- the tail is device-bound (S3-28). |
| `LLAMA_TEMPORAL_WORKER_AFFINITY` | off | `"lo-hi"` CPU range for fetch workers. Measured: no effect (S3-23). |
| `LLAMA_TEMPORAL_FUSED` | off | one `preadv` of 3 iovecs per swap instead of 3(x split) reads. **Rejected: -12.3%** at 648 KiB because the block layer splits it 512+136 (`max_sectors_kb=512`). Requires the side-file's fused region. Might win for experts <= 512 KiB; untested there. |
| `LLAMA_TEMPORAL_URING` | off | io_uring fetch path: ONE submitter thread drains the queue and hands the whole burst to the kernel in a single `io_uring_enter`, replacing the 6-thread blocking-pread pool. **Rejected: -11.4%** (S3-36) -- batched submission lets the block layer plug-merge the two adjacent parts of an expert back into one 216 KiB request, silently undoing `SPLIT=2`. Kept as evidence; leave off. Requires root (EPERM in the `shell` SELinux domain). Mutually exclusive with `_FUSED`. |
| `LLAMA_TEMPORAL_URING_SQPOLL` | off | adds `IORING_SETUP_SQPOLL`. Worse even standalone at QD1 (485 vs 243 us). |
| `LLAMA_TEMPORAL_URING_IOPOLL` | off | adds `IORING_SETUP_IOPOLL`. **Unavailable on this device** -- CQE returns `-95 EOPNOTSUPP`; SCSI/UFS has no polled queues. |

## Eviction

| flag | default | meaning |
|---|---|---|
| `LLAMA_TEMPORAL_MADV_FREE` | off (DONTNEED) | use `MADV_FREE` instead of `MADV_DONTNEED`. **Worth 7.6% -- always set it** (S3-25). |
| `LLAMA_TEMPORAL_EVICT_DEFER` | off | hold evictions until in-flight fetches drain. **Rejected: -4.1%** -- it collides the madvise batch with the next layer's submit (S3-27). |
| `LLAMA_TEMPORAL_NOMADV` | off | skip the madvise entirely. **Diagnostic only** -- residency becomes unbounded. |
| `LLAMA_TEMPORAL_JANITOR_NOLOCK` | off | release the pool mutex during the madvise. Neutral; kept for hardware where the device is not the bottleneck (S3-26). |

## Repacked-layout streaming

| flag | default | meaning |
|---|---|---|
| `LLAMA_TEMPORAL_REPACK` | off | route experts to the `CPU_REPACK` buffer so `mul_mat_id` uses the fast interleaved ARM GEMM. Worth ~1.33x on compute. |
| `LLAMA_TEMPORAL_REPACK_FILE` | off | pre-repacked side-file to stream from. Slice offsets are `round_up_4096(gguf_offset)`, computed identically by the dump tool and the loader -- no index file. |
| `LLAMA_TEMPORAL_REPACK_DUMP` | off | one-shot: build the side-file and exit. `LLAMA_TEMPORAL_REPACK_DUMP=out.bin ./llama-bench-temporal -m model.gguf` |
| `LLAMA_NO_REPACK` | off | force literal (non-repacked) CPU placement. The pre-repack baseline. **Never mix repacked and non-repacked arms.** |

## Instrumentation

| flag | default | meaning |
|---|---|---|
| `LLAMA_TEMPORAL_TRACE` | off | chrome-trace JSON to `/data/local/tmp/tmoe/trace.json` at exit. Types: 0 GEMV, 1 WAIT, 2 FETCH, 3 EVICT, 4 ENSURE, 5 QWAIT(submit->dequeue; the dumper mislabels it "ROUTER"). Lanes: compute `ith`, workers `100+w`, janitor `200`. Costs ~8% throughput. |
| `LLAMA_TEMPORAL_FETCHPROF` | off | per-fetch phase accounting: syscalls/fetch, time inside vs **outside** syscalls, first-call time, slowest call, short-read count. **Use this instead of a standalone probe** when engine and probe disagree. |
| `LLAMA_TEMPORAL_SPIN_US` | 300 | expert-wait spin budget before sleeping. 5000 measured best; less is worse. |

## Physics reference (measured)

```
QD1 single-request latency = 163 us fixed + 0.63 us/KiB          (S3-32)
no device-side cache: re-reading the same offset is not faster   (S3-32)
engine fetch-path software overhead = 0 us                       (S3-33)
   (outside_sys = 0, 1.00 syscalls/fetch, 0 short reads, every shape)

DEVICE-SIDE, in-engine, decode phase, UFS driver monitor         (S3-36)
   concurrency actually in flight            ~3.2   (we offer 6)
   device latency per 108 KiB request        ~350 us
   delivered bandwidth                       ~0.97 GB/s
   => burst floor for a 648 KiB swap         ~654 us
   and this is INDEPENDENT of the submission mechanism: io_uring at
   concurrency 7.04 raised latency to 805 us and delivered 0.92 GB/s.

block layer: /data = dm-63 (scheduler=none) -> sda34; SDA runs mq-deadline,
   rq_affinity=2, nr_hw_queues=1, can_queue=31, nr_requests=128,
   max_sectors_kb=512 (stock; sdb/sdc/sdd confirm), max_segments=128.
   sched=none and rq_affinity=0 both measured NEUTRAL in-engine (S3-36).
```

**Do not derive a burst floor from the QD1 law (gives 571 us) or from the saturated
throughput curve (gives 435 us).** Both are wrong; the measured device floor is ~654 us.
See pitfall #2 and S3-36.
