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

| arm | tok/s |
|---|---|
| plain resident E=192, no swap machinery | **62.5** |
| two-pass + enforced swap, resident E=192 | 32.6 |
| **two-pass + enforced swap, streamed R=18** | **32.5** |
| same, madvise skipped (NOMADV, diagnostic only) | 41.0 |

**Temporal = 52% of the plain resident ceiling, and 100% of its own same-policy ceiling.**

> **Read S3-37b before quoting the 52%.** This device cannot be DVFS-pinned (unrooted), and
> the plain ceiling is a no-wait arm while every policy arm idles on storage — the exact
> asymmetry pinning removes. The 100%-of-same-policy figure is solid (both arms wait alike);
> the 52% is **provisional** until the per-arm clock residency is measured. The measurement
> is wired into `run_samsung.py` (`time_in_state` deltas) and has NOT yet been run.

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
- **Single-pass streaming is unanswered, not rejected.** The `ENFORCE`-without-`TWOPASS`
  path does not evict at all (`evictions=0`, residency unbounded), so its 25.05 is not the
  technique. Wiring eviction into the single-pass path is a prerequisite to asking.

## Do this FIRST
Run `run_samsung.py ceilplain ceiling temporal ceilplain ceiling temporal` and read the
per-arm `mean_clk`. If the plain arm's residency-weighted clock is materially higher than
the policy arms', the 1.9x policy cost is partly governor and the ledger needs revising.
Everything in the next list is downstream of that answer.

Device hygiene learned the hard way: hold it awake (`svc power stayon true`), never poke
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
