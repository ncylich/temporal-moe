# Trained fine 1e19 model on the Pixel 10a: ledger

Model: `temporal_fine_g3_1e19/ckpt/iter_0004278` (HF `ncylich/temporal-moe-ckpts`), 14 layers
(layer 0 dense, 13 MoE), 192 routed experts per MoE layer, 18 active, 1.26 B parameters.
Training routing: rolling resident window, cold fill top-18, at most one swap per layer per
token, eviction `min_logit`. One entry per step, written before the next step starts.

## Step 1: converter and GGUF files (2026-11-10)

- `git cherry-pick 1e7e4cab` onto this branch: `scripts/serving_convert/` (README, RESULTS, scripts).
- Files were not rebuilt. They were copied from the converting session's scratch directory to a
  durable location outside the repo: `~/Documents/temporal-moe-serving/fine_1e19/`
  (`fine_1e19-f16.gguf`, `fine_1e19-q4_0.gguf`, `fine_1e19.safetensors`, `test_tokens.npz`,
  `test_tokens_2048.npz`, `ref_temporal_512.npy`, reference logs).
- Command: `shasum -a 256 *.gguf; ls -l`

| file | bytes | sha256 |
|---|---|---|
| fine_1e19-q4_0.gguf | 744,281,312 | c34d3445064a9b14cadacee1de9ec74d568995f078de6da0d8aef767b412c8ff |
| fine_1e19-f16.gguf | 2,631,311,648 | 47d771bb61157b517eed9c7f35e231cd6c18c2a4cdf17ac9431ac64a3e97738f |

Check: Q4_0 size equals the expected 744,281,312 bytes. Pass.

Reference (from the converting session's log, `ref_temporal_512.log`): routing temporal,
evict min_logit, k 18, 512 samples, 1,048,576 tokens, loss 3.19202 nats/token (SE 0.01587),
1.07187 bits per byte.

Device note found during step 1 (2026-11-10 23:24 device time): `su` is not available. The
Magisk app is installed and the bootloader is unlocked, but Magisk is not active this boot
(uptime 1 h 28 min, `/debug_ramdisk` empty). Shell-level access works: `/data/local/tmp/tmoe`
is writable, binaries run, VmHWM is readable. Not available: `drop_caches`, block-queue tuning,
frequency pinning.

## Step 2: trained routing in the engine (2026-11-10)

Engine checkout: `~/Documents/llama.cpp-trained-routing` (a `git worktree` of
`~/Documents/llama.cpp-android`), local branch `temporal-moe-trained-routing`, based on
`temporal-moe-win` @ 43bec5f3 (fetched from the `mine` remote). Nothing pushed to the fork.

What changed:

- New custom op `ggml_temporal_route` (ggml-cpu.c, compiled on every platform). Input: router
  logits [E, n_tokens] F32. Per layer and per token: cold fill = top-K of the logits; later
  tokens admit the best non-resident expert iff its logit is strictly above the worst
  resident's, evicting the lowest-logit resident. The newest admission sits in slot K-1.
  Works for any batch size, so a prefill advances the window token by token exactly like
  decode.
- `build_moe_ffn` uses it whenever trained routing is on. Gate weights are unchanged engine
  code: softmax over all experts, gathered to the window, renormalised, which equals the
  softmax over the resident logits.
- `ggml_temporal_window_fill` (two-pass, n_tokens == 1) no longer picks experts in trained
  mode. It takes the router's ids, evicts the expert the router just dropped, makes every
  window expert resident (a no-op for ones already resident or in flight), and keeps the
  existing trim.
- `llama_context::decode` resets every layer's window at position 0, and refuses a batch
  that mixes sequences while trained routing is on (the state is per layer, not per sequence).
- Flags: `LLAMA_TEMPORAL_ROUTE=trained` turns it on; it is also the default whenever
  `LLAMA_TEMPORAL_TWOPASS` is set. `LLAMA_TEMPORAL_ROUTE=random` restores the earlier random
  one-swap policy. With neither flag the engine runs stock top-k, as before.
  `LLAMA_TEMPORAL_ROUTE_DUMP=path` writes every token's window (int32 layer + K ids).
  At exit the engine prints `temporal-route: ... swaps_per_layer_token=...`.
- Host-build fixes, needed because `temporal-moe-win` does not build on macOS: guard two trace
  lines in `mul_mat_id`, add macOS stubs for `ggml_temporal_window_fill` and
  `ggml_tm_wait_src_expert`, and map `mincore`/`posix_fadvise` in `llama-model-loader.cpp`.
  None of these touch Linux or Windows code paths.

Check tool: `scripts/serving_convert/gguf_logits.cpp` gained an `nll` mode (one model load,
rows of seq+1 ids, KV cleared per row, writes per-token NLL). `check_engine_routing.py`
compares the engine's window dump with `ref_model.py` masks and per-token losses with
`ref_temporal_512.npy`.

Commands (host, Apple M4 Pro, CPU only, flash attention off, F32 KV):

```bash
LLAMA_TEMPORAL_ROUTE=trained LLAMA_TEMPORAL_ROUTE_DUMP=route_f32_4.bin \
  gguf_logits nll fine_1e19-f32.gguf rows4.i32 4 2048 nll_f32_4.f32
python scripts/serving_convert/check_engine_routing.py --weights fine_1e19.safetensors \
  --tokens test_tokens.npz --rows 4 --dump route_f32_4.bin --nll nll_f32_4.f32 \
  --ref-losses ref_temporal_512.npy
```

| GGUF | rows x tokens | resident sets equal to reference | engine loss | reference loss | diff |
|---|---|---|---|---|---|
| F32 | 4 x 2048 | 106,496 of 106,496 layer-tokens (13 layers) | 3.14413 | 3.14409 | +0.00004 |
| F16 | 4 x 2048 | 12,287 of 106,496 | 3.14407 | 3.14409 | -0.00002 |

- F32: the mask check passes exactly, token by token, on every layer.
- F16: the sets diverge from F16 rounding at near-ties, and the window is path-dependent, so
  a divergence persists. Mean overlap on row 0 is 15.73 of 18 experts (17.39 in layer 1, about
  15.4 in deeper layers). The loss is unaffected (-0.00002 nats, max per-token |d| 1.24,
  median 0.048). The resident-set equality check is therefore only meaningful on the F32
  file; F16 is checked by loss.
- Swap rate: both the reference and the engine swap on every non-cold token
  (swaps_per_layer_token = 1.0000 in all four rows). The trained router never settles on a
  window equal to its current top-18, so each layer admits exactly one expert per token.

Pending at the time of writing: the F16 score on the first 512 rows (1,048,576 tokens) against
the reference's 3.19202. Running on the host in the background; result is recorded below
under "Step 2, continued".

## Step 3: engine patch exported (2026-11-10)

- Engine commit `c1c21baa` on local branch `temporal-moe-trained-routing` (parent `43bec5f3`).
- `git format-patch 43bec5f3..HEAD -o scripts/serving_convert/engine_patch/` gives one patch,
  plus a README naming the base commit.
- Check: fresh `git clone` of the engine checkout, `git checkout 43bec5f3`, `git am` the patch:
  applies cleanly, resulting tree `4327c734` equals the exported branch's tree. Pass.
- temporal-moe commit `25f3efca` pushed as `trained-model-pixel`
  (`git push origin HEAD:refs/heads/trained-model-pixel`). Nothing pushed to `ncylich/llama.cpp`.

## Step 4: phone drivers generalized (2026-11-10)

- `run_ctx_designpoint.py`: model `trained_fine` (GGUF, side-file, K 18, E 192, 13 MoE
  layers, bytes, sha256, binary `llama-bench-trained`, build filter `c1c21baa`). The CSV row
  filter now uses the model's build commit instead of the hard-coded `0badc06a`. Named arms
  `deploy` (R=18, TWOPASS, trained routing), `resident_trained` (R=192, no TWOPASS, trained
  routing) and `resident_topk` (R=192, stock top-k); all share the production fetch and
  repacked-kernel flags. New columns: `arm`, `peak_rss_mb` (VmHWM, polled every 0.5 s),
  `swaps_per_layer_token` and `route_layer_tokens` (from the engine's `temporal-route:` line),
  `rooted`, `min_max_freq7`, `soc_temp_c`. `--noroot` runs without `su`: the cool gate becomes
  soc_therm <= 33 C with every policy's `scaling_max_freq` at its rating for three samples
  10 s apart, and an arm is void if policy 7's `scaling_max_freq` drops below rating during the
  run. Contexts above 2048 are labelled `ctx_sweep_speed_only_beyond_training_ctx`. Running
  without `--model` gives the legacy phases unchanged.
- `gate_ppl.py`: `--model trained_fine`, `--batch1` (-b 1 -ub 1 so every token takes the
  window path), `--plain` (repacked vs plain kernels, tolerance |ln PPL difference| < 0.01 nats,
  fixed before running).
- `honesty_gate.py`: `--model trained_fine`; model and expert byte counts read from the local
  GGUF (709.8 MiB file, 617.0 MiB routed experts) instead of constants.

Check: `python3 run_ctx_designpoint.py --model trained_fine --dry-run`

```
model trained_fine: fine_1e19-q4_0.gguf (device /data/local/tmp/tmoe), side-file fine_1e19-q4_0-repacked.bin, bench llama-bench-trained, build filter c1c21baa
local /Users/noahcylich/Documents/temporal-moe-serving/fine_1e19/fine_1e19-q4_0.gguf
  bytes 744,281,312 sha256 c34d3445064a9b14cadacee1de9ec74d568995f078de6da0d8aef767b412c8ff
  glm4moe: 14 layers, 1 dense, 13 MoE, 192 experts, 18 active
  arm deploy            R= 18 LLAMA_TEMPORAL_TWOPASS=1 LLAMA_TEMPORAL_ROUTE=trained
  arm resident_trained  R=192 LLAMA_TEMPORAL_ROUTE=trained
  arm resident_topk     R=192 (stock top-k)
DRY RUN OK
```

Pass.

## Step 5: phone setup and gates (2026-11-10, device time 23:4x to 00:3x)

Device: Pixel 10a `5C111JEA320125`, unrooted this boot (step 1 note), on AC power, 93% battery.
Everything runs as the `shell` user under `/data/local/tmp/tmoe`.

Setup:

- Pushed `llama-bench-trained`, `llama-perplexity-trained` (engine branch, Android build) and
  `gguf_logits-trained` (the check tool, statically linked against the same build).
- Pushed `fine_1e19-q4_0.gguf`; on-device sha256
  `c34d3445064a9b14cadacee1de9ec74d568995f078de6da0d8aef767b412c8ff` (matches step 1).
- Side-file `fine_1e19-q4_0-repacked.bin` built on the device:
  `LLAMA_TEMPORAL_REPACK_DUMP=fine_1e19-q4_0-repacked.bin ./llama-bench-trained -m fine_1e19-q4_0.gguf`.
  All 39 expert tensors repacked (q4_0_4x8). The tool then exits with
  `repack-dump: layer 0 missing slot 0`: its optional fused [gate|up|down] region assumes every
  layer has experts, and layer 0 here is dense. Only `LLAMA_TEMPORAL_FUSED` reads that region;
  it is off in every arm. The repacked-vs-plain gate below checks the mirror region.

Bug found and fixed during this step (engine commit `759d43c5`):
the first gate runs reported `layer_tokens=12813`, not a multiple of 13. A window dump showed
layer 13 saw 513 tokens where layers 1 to 12 saw 1025. glm4moe drops the rows that need no
logits before the last layer's FFN, so layer 13's window skipped those tokens. That breaks
the training rule in llama-perplexity (it scores half of each chunk) and after any prompt
prefill. The host check in step 2 was not affected because `gguf_logits nll` requests every
logit. Fix: under trained routing the rows are dropped after the layer loop. After the fix,
`layer_tokens=13325` = 13 x 1025.

Retracted: the first run of honesty gate 2 (PPL 301.2826, both arms) and the first
streamed-vs-resident gate (301.0689, both arms). Both passed, but on the binary with the
layer-13 bug. All gates below are reruns on `759d43c5`.

A second harness fault, caught by the engine's own check: llama-perplexity packs four
512-token sequences into one 2048-token batch, and trained routing refuses mixed-sequence
batches (`decode: trained temporal routing needs one sequence per batch`). Both PPL gates
now pass `-b 512`.

| gate | command | result | status |
|---|---|---|---|
| bytes are real | `python3 honesty_gate.py --model trained_fine` (gate 1: R=0, O_DIRECT, 16 decode tokens) | pool fetched 1071 MiB; diskstats delta 1122 MiB minus 93 MiB load = 1029 MiB; 3.9% apart (limit 10%) | pass |
| pool numerics | same, gate 2: LLAMA_NO_REPACK, pool off (mmap) vs pool on R=18, trained routing | 301.3581 +/- 22.06006 both | pass (bit-identical) |
| streamed vs resident, window path | `python3 gate_ppl.py --model trained_fine --resident 192 --batch1 --noroot` | R=192: 301.2954 +/- 21.91304; R=18: 301.2954 +/- 21.91304; 13,286 swaps each; R=18 arm 40,965 evictions | pass (bit-identical) |
| repacked vs plain kernels | `python3 gate_ppl.py --model trained_fine --plain --noroot` (R=192, no two-pass) | repacked 300.2781, plain 301.3581, \|ln ratio\| = 0.00359 nats (limit 0.01, set before running) | pass |

Notes:

- PPL is about 300 because `ppl_input.txt` is shuffled word salad. It is an identity oracle,
  not a quality number.
- `--batch1` (-b 1 -ub 1) makes every one of the 1025 tokens per chunk a one-token decode, so
  every token went through `ggml_temporal_window_fill` and the two-pass split.
- Under TWOPASS the R=192 arm still evicts the expert the router drops (39,858 evictions);
  TWOPASS ignores R >= K (pitfall #26). So this gate compares two streamed configurations
  whose difference is the memory cap. Streamed against truly resident is covered by honesty
  gate 2 (pool off) and, for timing, by arm `resident_trained`.
- Measured swap rate in every gate: swaps_per_layer_token = 1.0000 (0.9990 where warmup
  tokens are counted).
- The batch-1 two-pass PPL (301.2954) and the batched resident PPL (300.2781) differ by
  0.0034 nats in ln PPL: different accumulation order in the two-pass split, flipping
  near-ties in a path-dependent router. Same size as the kernel-family difference.

Step 3 redone after the fix: `git format-patch 43bec5f3..HEAD` now gives two patches; `git am`
on a fresh clone at 43bec5f3 reproduces tree `b69151b6` of engine commit `759d43c5`. Pass.

## Step 6: decode speed and peak memory on the Pixel 10a (2026-11-10, 01:55 to 02:39 host time)

Command: `python3 run_ctx_designpoint.py --model trained_fine --ctx 0 1024 2048 4096 --noroot
--out results/trained_fine_pixel_2026-11-10.csv` (log next to it). Binary `llama-bench-trained`
(engine 759d43c5), `-t 4 -p 0 -n 128 -r 3 -mmp 0 -ot "_exps=CPU"`, `-d <ctx>`; production fetch
flags (repacked kernel + side-file, O_DIRECT, MADV_FREE, SPLIT 2, 6 fetch threads, SPIN_US
5000). Arms: (a) `deploy` R=18, TWOPASS, trained routing; (b) `resident_trained` R=192, no
TWOPASS, trained routing; (c) `resident_topk` R=192, stock top-k (a different routing function
from the one the model was trained with).

Run conditions: unrooted, governor `sched_pixel` (not pinned), AC power. Cool gate: soc_therm
<= 33 C and every `scaling_max_freq` at rating, three samples 10 s apart; 60 s rest between
arms. Void if policy-7 `scaling_max_freq` dipped below 3.105 GHz during the arm, any swap
> 100 MB, no decode row, no evictions on deploy, or no router line on a trained arm. Host:
my own jobs stopped (the F16 scoring run was paused at row 240 for this); macOS daemons
(installd, BTLEServer) were busy and outside my control.

16 rows, 4 void. All four voids are at ctx 4096 (two deploy, two resident_trained), each for
policy-7 `scaling_max_freq` falling to 2.914 GHz mid-run; the driver rested longer and retried,
and the third attempt of each was clean. Void rows stay in the CSV.

| arm | ctx | decode tok/s (sd, 3 reps x 128) | / arm (b) | / arm (c) | VmHWM MB | swaps / layer / token | evictions |
|---|---|---|---|---|---|---|---|
| (a) deploy | 0 | 66.07 (3.35) | 0.437 | 0.443 | 509 | 1.0000 | 16,341 |
| (b) resident_trained | 0 | 151.33 (8.10) | 1 | 1.015 | 767 | 1.0000 | 0 |
| (c) resident_topk | 0 | 149.04 (8.52) | 0.985 | 1 | 767 | n/a | 0 |
| (a) deploy | 1024 | 32.91 (0.34) | 0.678 | 0.684 | 762 | 0.9999 | 19,845 |
| (b) resident_trained | 1024 | 48.56 (0.05) | 1 | 1.009 | 887 | 0.9999 | 0 |
| (c) resident_topk | 1024 | 48.11 (0.07) | 0.991 | 1 | 887 | n/a | 0 |
| (a) deploy | 2048 | 22.78 (0.23) | 0.759 | 0.777 | 892 | 1.0000 | 20,517 |
| (b) resident_trained | 2048 | 30.00 (0.26) | 1 | 1.024 | 976 | 1.0000 | 0 |
| (c) resident_topk | 2048 | 29.31 (0.28) | 0.977 | 1 | 976 | n/a | 0 |
| (a) deploy | 4096, speed only | 14.08 (0.24) | 0.842 | 0.850 | 1,096 | 1.0000 | 20,919 |
| (b) resident_trained | 4096, speed only | 16.72 (0.09) | 1 | 1.010 | 1,154 | 1.0000 | 0 |
| (c) resident_topk | 4096, speed only | 16.56 (0.20) | 0.990 | 1 | 1,153 | n/a | 0 |

ctx 4096 exceeds the 2,048-token training context: speed only, no quality claim.

What the peak-memory column does and does not show. VmHWM is the process peak over the whole
run, and for the deploy arm it is set by things other than the 18-expert window:

- MADV_FREE (production flag) leaves evicted pages counted in RSS until the kernel reclaims
  them. Separate memory-only runs (`_mem.sh`, deploy, 256 decode tokens, RSS sampled every
  0.25 s): ctx 0 with MADV_FREE VmHWM 452 MB, without it 214 MB (decode-phase RSS 203 MB,
  197 MB anonymous).
- At depth > 0 the `-d` prefill goes through the batched path, which fetches every expert a
  prompt token selects and does not evict until the next decode token trims. ctx 2048 without
  MADV_FREE: VmHWM 881 MB during the prefill, decode-phase RSS median 443 MB.
- So the decode working set of the deploy arm is about 203 MB at ctx 0 and 443 MB at ctx 2048
  (KV cache and compute buffers included), against 767 and 976 MB fully resident. The VmHWM
  column above is what a user would see as peak, and it includes the prefill.

Other observations:

- Swap rate is 1.0 per MoE layer per token in every trained arm: the trained router admits one
  expert per layer on every token. 13 x 3 slices = 39 expert-slice fetches per token.
- llama-bench prefills the depth once and restores the KV state for each rep; the router window
  is not part of that state, so reps 2 and 3 start from rep 1's final window. With one swap per
  token either way this does not change the fetch count; it does mean those reps' windows are
  not the ones the training rule would give for that KV. Speed only.
- Decode tokens in llama-bench are not natural text, so the window walk here is not the walk
  on real text; the swap rate measured on real test text (step 2, step 7) is also 1.0.
- Not comparable to the 2026-08-24 random-weight rows: different model (1.26 B vs 11 B
  geometry), unrooted, unpinned governor.

## Step 2, continued: host F16 loss on the first 512 test samples

Command (host, F16 GGUF, flash attention off, F32 KV, 8 threads, trained routing):
`LLAMA_TEMPORAL_ROUTE=trained gguf_logits nll fine_1e19-f16.gguf rows512.i32 512 2048 nll_host_f16_512.f32 threads=8`.
Paused at row 240 for step 6 and resumed on rows 240..511 (`rows_240_512.i32`). Rows 0..239
ran on the host build before engine fix `759d43c5`, rows 240..511 after; the fix only changes
which rows the last layer routes when some outputs are skipped, and `nll` requests every
output, so both builds route identically here.

| | samples | tokens | loss (nats/token) | bits per byte |
|---|---|---|---|---|
| PyTorch reference, temporal min_logit | 512 | 1,048,576 | 3.19202 | 1.07187 |
| engine, host F16, trained routing | 512 | 1,048,576 | 3.19221 | 1.07193 |

Difference +0.00018 nats, paired SE over samples 0.00013. RESULTS.md measured F16 rounding at
+0.00009 (plain attention) to +0.00039 (flash attention) under top-k. Check passes. Swap rate
over the run: 1.0000 per layer per token (7,237,907 swaps, part 2).

## Step 7: Q4_0 quality on the phone

Same 512 test samples (2048 tokens each, 1,048,576 tokens), trained routing, flash attention
on, F16 KV, Q4_0 GGUF (sha256 in step 1).

- Host: `LLAMA_TEMPORAL_ROUTE=trained gguf_logits nll fine_1e19-q4_0.gguf rows512.i32 512 2048 nll_host_q4_512.f32 fa kvf16 threads=8`
- Phone (unrooted, 4 threads): `gguf_logits-trained nll fine_1e19-q4_0.gguf rows512.i32 512 2048 nll_phone_q4_512.f32 fa kvf16 threads=4`.
  Run log `nll_phone_q4_512.log`; router report: `layers=13 layer_tokens=13631488 cold_fills=6656 swaps=13624366 swaps_per_layer_token=1.0000`.

Bits per byte = loss / 2.9780 (same divisor as step 2).

| | samples | tokens | loss (nats/token) | bits per byte | vs reference (paired SE) |
|---|---|---|---|---|---|
| PyTorch reference, temporal min_logit | 512 | 1,048,576 | 3.19202 | 1.07187 | |
| engine, host F16 | 512 | 1,048,576 | 3.19221 | 1.07193 | +0.00018 (0.00013) |
| engine, host Q4_0 | 512 | 1,048,576 | 3.21330 | 1.07902 | +0.02127 (0.00032) |
| engine, phone Q4_0 | 512 | 1,048,576 | 3.21334 | 1.07903 | +0.02131 (0.00032) |

Phone minus host Q4_0: +0.00004 nats, paired SE 0.00013, so the phone build matches the host
build. Per-token values differ (max 3.50 nats on one token) because the ARM and x86/Apple
kernels round differently and the router then takes different swaps on near ties, as in
step 2. Q4_0 costs +0.0211 nats over F16, the same cost RESULTS.md measured under top-k.
