# Handoff — temporal-MoE on Samsung SM-S942U1 (2026-07-24, S3-37)

Companion to `HANDOFF.md` (Pixel 10a). **Do not mix numbers between the two files.**
The Pixel is rooted and DVFS-pinned; this device is unrooted and stock-governor.

## Rig
- Samsung SM-S942U1, serial `RFGL42B1VLW`, SM8850, Android 16, **11.4 GB RAM**, 12.5 GB zram.
- **UNROOTED.** No DVFS pin, no UFS driver monitor, no block-layer knobs, no io_uring.
- CPU: 6x perf @3.63 GHz (policy0) + 2x prime @4.74 GHz (policy6). **No little cores.**
- Cool-gate reference: `{cpu0: 3628800, cpu6: 4742400}`.
- `/data/local/tmp/tmoe/`: `qwen3moe-rand-fine-Q4pure.gguf` (E192 K18 ff384) and its
  `-repacked.bin` side-file (11.67 GB, generated on-device), `llama-bench-temporal`,
  `llama-perplexity`, `ppl_input.txt`.
- Harness: `run_samsung.py` (arms + peak-VmSwap guard), `gate_ppl.py --resident 192 --noroot`.

## State (stock governor — not comparable to Pixel pinned numbers)

Matched threads (all `-t 6`), n=3 rounds, interleaved, all `peak_swap=0` (S3-37c):

| arm | tok/s |
|---|---|
| plain resident E=192, no swap machinery | **62.89** |
| two-pass + enforced swap, resident E=192 | 36.79 |
| **two-pass + enforced swap, streamed R=18** | **33.23** |
| same, madvise skipped (NOMADV, diagnostic only) | ~41 |

**Temporal = 53% of the plain resident ceiling, at ~10.6x less expert RAM.**
The policy costs **1.71x**; the streaming costs a further **9.7%**.

> **Settled in S3-37c.** This device cannot be DVFS-pinned (unrooted), so the concern was
> that the no-wait plain arm was flattered by the governor. Measured via
> `cpufreq/stats/time_in_state` deltas: the waiting arms run the perf cluster **higher**
> (3.27 GHz vs 2.58), not lower. The governor does not explain the gap; the policy cost is
> real. An earlier claim that "streaming is free" was **retracted** — those two arms had
> different thread counts.

## The one-line difference from the Pixel
**Streaming is free here.** R=18 streamed does 3.5x the I/O of the resident arm (4613 vs
1313 MiB) for zero throughput cost — the fetch is entirely hidden behind a ~2x faster
compute. The whole S3-36 storage problem does not exist on this SoC. The entire gap to
the ceiling is the **two-pass enforced-swap policy**, which costs 1.9x even fully resident.

## Closed
- E=192 fits fully resident (5.94 GB) **after `am kill-all`**. Without it, it swaps
  3450 MB into zram and reports a void 7.70 tok/s. The harness voids any arm that swaps.
- Correctness gate passes in its strongest form (resident vs streamed, PPL bit-identical).
  Note it does NOT exercise eviction (`evictions=0` in both gate arms).
- Thread count: t4 32.7, t6 31.5 (not separable), t8 26.3. Keep `-t 4`.
- Governor confound eliminated (S3-37c): waiting arms run the perf cluster higher, not lower.
- **Always diff the FULL arm config before comparing two arms.** `temporal` is `-t 4` and
  `ceiling` is `-t 6`; comparing them produced a wrong "streaming is free" conclusion that
  stood for two commits.
- **Single-pass streaming is unanswered, not rejected.** The `ENFORCE`-without-`TWOPASS`
  path does not evict at all (`evictions=0`, residency unbounded), so its 25.05 is not the
  technique. Wiring eviction into the single-pass path is a prerequisite to asking.

## Device hygiene learned the hard way: hold it awake (`svc power stayon true`), never poke
`thermalservice`, and if `scaling_max_freq` is stuck below rated, reboot rather than wait —
the cool-gate cannot tell "throttled" from "capped" and blocks silently.

## Next, in order of leverage
1. **The madvise.** Skipping it entirely is worth **+26%** here (32.6 -> 41.0). That is the
   largest identified lever on this device. `EVICT_DEFER` was rejected at -4.1% on the
   Pixel, but that was a fetch-bound machine and the rejection does not transfer (pitfall
   #19). Batching or deferring the reclaim into the compute window is worth measuring here.
2. **The two-pass graph split**, which costs ~1.5x beyond the madvise and exists solely to
   overlap a fetch this device no longer needs overlapped. Needs (a) eviction in the
   single-pass path, then (b) a fair single- vs two-pass comparison.
3. K sweep — but note the motivation is the OPPOSITE of the Pixel's. There, storage was
   bought and unused so more compute was free. Here compute is the scarce side.
