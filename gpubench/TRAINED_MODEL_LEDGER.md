# Trained fine 1e19 model on the RTX A6000 (session G1)

Model: `temporal_fine_g3_1e19`, iteration 4278, 14 blocks (layer 0 dense, 13 MoE), 192 experts,
18 active, 1.26 B parameters, served as `fine_1e19-q4_0.gguf` (744,281,312 bytes, sha256
`c34d3445...b412c8ff`, architecture `glm4moe`). Goal: decode speed streamed (R=18 experts per
layer in VRAM, pool in CPU RAM) against the same model with all 192 experts per layer in VRAM,
both under the trained router.

Entries are in the order things happened. Context-0 runs are not run and not reported.

## G1-0 -- What the CUDA path needs (written 2026-11-11, before any pod was rented)

**Where the CUDA streaming code is.** It was never pushed to `ncylich/llama.cpp`. The fork has
only the CPU branches (`temporal-moe` @ `61f6d1b4`, `temporal-moe-win` @ `43bec5f3`), both on
upstream `0badc06a`. The CUDA path is `llamacpp-bench/systems_bench.patch` in this repo, a diff
against the same `0badc06` (5 files: `ggml-cuda.cu`, new `temporal.cu` / `temporal.cuh`,
`llama-model.cpp`, `models/qwen3moe.cpp`). The 11B numbers came from a pod-local commit chain of
that patch (`llamacpp-bench/README.md` names `2447b1a` and `883dfe9`); the repo patch contains
both of those changes (default-on one-swap decode, residency-masked prefill).

**Base for this work.** `43bec5f3` + `scripts/serving_convert/engine_patch/000{1,2}` (tree
`b69151b6`, as that README states) + `systems_bench.patch`. Checked locally: the CUDA patch
applies to that tree with `git apply --check`, no conflicts (the CPU branch and the CUDA patch
touch disjoint files).

**(a) Running `glm4moe`.** Read from the code, to be confirmed on the pod:

- Slot registration (`llama-model.cpp` hunk) is architecture-neutral: it walks every layer, skips
  layers without `ffn_{gate,up,down}_exps` (layer 0 here), and replaces the three pool tensors by
  R-slot VRAM tensors. The shared expert stays an ordinary VRAM tensor. `-ncmoe 14` puts the 39
  routed tensors in the CPU pool.
- The swap copy moves 16-byte vectors; one expert slice is 86,400 bytes = 5,400 vectors, so the
  alignment assumption holds for Q4_0 [800 x 192] and [192 x 800].
- Decode `mul_mat_id` on quantized weights with one token stays on the mmvq path, which is the
  path the CUDA graph check allows.
- `qwen3moe.cpp` hunk is the router-early experiment only; not used.

**(b) Swaps from the trained rule.** Three things are missing:

1. *The routing op itself.* The CPU patch inserts `ggml_custom_4d(..., ggml_temporal_route)`
   after the router logits. The CUDA backend does not run custom ops, so on a CUDA build the
   scheduler would hand that node to the CPU backend: logits to host, route, ids back, once per
   layer. That splits the decode graph into about 27 pieces, defeats CUDA graph capture, and would
   slow the all-in-VRAM baseline as well as the streamed arm, which would flatter the ratio. So the
   rule has to run as a CUDA kernel inside the one captured graph: per-layer window state on the
   device, logits [E, n_tokens] in, window ids [K, n_tokens] out, same tie-breaking and same slot
   order as `ggml_temporal_route` (cold fill top-K highest first; later tokens admit the best
   non-resident iff its logit is strictly above the worst resident's; newcomer in slot K-1).
   Gate weights need nothing new (softmax over all, gather, renormalise, already in the graph).
2. *Residency following the window.* `k_residency` already takes the selected ids and, in its
   one-swap path, loads the lowest-index selected expert that is not resident into the
   lowest-index slot whose expert is not selected. When the ids are the trained window, that is
   exactly the admitted expert and the evicted one, so steady state needs no change. It breaks
   when more than one selected expert is missing: the first token of a sequence (slots are seeded
   with experts 0..17, the cold fill is the top-18) and the first decode token after a prefill
   (the prefill streams through its own ring and leaves the decode slots stale). The one-swap
   path would then compute the other missing experts from slot 0. Needed: count the missing
   experts in the same parallel pass and fall back to the multi-swap scan when there is more than
   one.
3. *Prefill.* The expert-major prefill re-derives a resident set with its own index-based rule
   and masks the ids to it. Under trained routing the ids already are the window per token, so
   that mask has to be off; the prefill then computes exactly the experts the router chose.

Also needed for the gates: the window reset at sequence position 0 has to reach the device state
(`llama_context::decode` hook), counters for the `temporal-route:` line have to live on the
device and be read back at exit, `LLAMA_TEMPORAL_ROUTE_DUMP` has to write the device windows, and
a count of swap copies actually issued so the streamed arm's copy rate can be checked against the
router's swap rate.

**Arms this gives.** All in VRAM + trained (baseline); all in VRAM + stock top-k; R=18 slots with
`TEMPORAL_SWAP_PROB=0` (our kernel, no bytes moved, output wrong by construction, timing only, as
in the 11B row b); R=18 streamed.

**Risk noted up front.** At 11B the gap is 0.68 ms kernel overhead + 0.38 ms exposed transfer on a
5.0 ms token over 45 layers, about 15 us of kernel overhead per layer. With 13 MoE layers that
would be about 0.2 ms per token here, on a token that is likely near 1 ms.

## G1-1 -- Port, pod, build (2026-11-11 08:00-08:35 UTC)

**Port.** `scripts/serving_convert/engine_patch_cuda/0001-*.patch` (one commit, 4 files, +165/-11 on
top of the base of G1-0). Its README lists what it does. Written before the pod existed and
compiled on the first try; the items of G1-0 (b) are all in it.

**Pod.** The task asked for the data center of the "Noah 2TB" volume (EUR-IS-4) so an old build
tree could be reused. EUR-IS-4 had no A6000 (two create calls refused, `SUPPLY_CONSTRAINT`). The
user then asked for US-TX-1 with 2 to 4 cards; by the time of the call US-TX-1 reported no stock
and every multi-card request (EU-SE-1 x2 with and without a data-center pin) was refused. What
could be rented: **one RTX A6000 in EU-SE-1, pod `bcnqix1ewwwjr3`, $0.59/h, created 08:16 UTC**,
no network volume. Nothing from the old volume was needed: the CUDA patch and the bench scripts
are in this repo, and the model files still existed on the Mac.

| item | value |
|---|---|
| GPU | NVIDIA RTX A6000, 49,140 MiB, driver 595.91.07, PCIe x16 |
| host | Xeon Gold 6342 (96 threads visible, shared host), 503 GB RAM |
| image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, CUDA 12.4, GCC 11.4, system torch 2.4.1+cu124 (not reinstalled) |
| added | apt: cmake tmux git build-essential; pip: safetensors (no deps), huggingface_hub, tokenizers |
| work dir | `/workspace/tmoe` (network file system), F32 GGUF in `/root/m` |

**Model files.** `fine_1e19-q4_0.gguf` copied from the Mac, sha256 on the pod
`c34d3445...b412c8ff` (the published one). `fine_1e19.safetensors` copied (sha256 prefix
`5f5d2b8b622df033` on both ends). `fine_1e19-f32.gguf` was regenerated on the pod with
`to_gguf.py --ftype f32` because the 5.3 GB upload was slow; sha256
`990224067ecedf5632b15846f15e899effb49b43e0f379f54479bdc6829f424d`, the same as the laptop
ledger's (L3-2). `test_tokens.npz` is the Mac conversion session's file.

**Build.** `gpubench/build_engine.sh` on the pod, from the pushed branch: clone
`ncylich/llama.cpp` over https, `43bec5f3`, `git am` CPU patches (tree `b69151b6` asserted),
`git apply systems_bench.patch`, `git am` CUDA patch. Engine tree `aa969ec0`.
`cmake -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86`. sha256:
`llama-bench` `5f5f5209...`, `libggml-cuda.so` `acc5ed4b...`, `libllama.so` `cea2398d...`.

**Went wrong.** (1) The first upload used `rsync --info=progress2`, which the Mac's rsync does not
have; it exited at once and was replaced by `scp`. (2) `to_gguf.py` needs `huggingface_hub`, which
the image lacks; installed. (3) The local branch was created with a tool name in it and was renamed
to `a6000-trained-model` before the first push.

## G1-2 -- Flash attention does not run on CUDA for this model; all runs use `-fa 0`

The first smoke run used the 11B flags (`-fa 1`) and gave about 20 tok/s in every arm, stock
top-k included. Cause, from `GGML_SCHED_DEBUG=2`: 30 graph splits, every `FLASH_ATTN_EXT` node on
the CPU. The model has 50 heads of size 16 and the CUDA flash-attention kernels have no head size
16, so the backend declines the op. With 96 default threads on a shared host the CPU part took
about 50 ms per token.

With `-fa 0` attention is the ordinary matmul, softmax, matmul sequence, all on the GPU, and the
decode graph is a single CUDA split (checked with `GGML_SCHED_DEBUG=1`). **Every number below is
`-fa 0 -t 8`.** This is a departure from the 11B protocol (`-fa 1`). Its effect on the ratio: the
non-flash attention is likely slower than a flash kernel would be, and attention time is in both
arms, so a faster attention would lower speed kept. G1-5 bounds this by taking the
context-dependent time out altogether.

The 20 tok/s rows are in `gpubench/results/` nowhere; they were a smoke file on the pod
(`logs/smoke.jsonl`, tag `smoke`) and are void.

## G1-3 -- Gates

### Perplexity, streamed against all in VRAM (pass for the timed path)

`llama-perplexity -f input.txt -c 1024 --chunks 8 -ngl 99 -fa off -t 8`, input = four license
texts from `/usr/share/common-licenses` concatenated (79,146 bytes, sha256 `b8737097...`, an
identity oracle like the laptop's word salad, not a quality measurement). Logs in
`gpubench/results/logs/ppl/`. Engine tree `aa969ec0`.

| run | batch 1 (`-b 1 -ub 1`, the decode path that is timed) | batched (`-b 1024 -ub 1024`, the prefill path) |
|---|---|---|
| all in VRAM, trained routing | **16.8819 +/- 0.72000** | 16.8267 +/- 0.71693 |
| streamed R=18, trained routing | **16.8819 +/- 0.72000** | 16.7862 +/- 0.71507 |
| all in VRAM, stock top-k | 17.1077 +/- 0.72986 | 17.0846 +/- 0.72916 |

- Batch 1: identical in every printed digit, and the running estimate after each of the 8 chunks
  is identical too (`[1]18.9855 ... [8]16.8819` in both logs). Streamed route line:
  `layers=13 layer_tokens=106509 cold_fills=117 swaps=106392 swaps_per_layer_token=1.0000`, and
  108,238 slot swap-ins (the router's 106,392 swaps plus the cold fills that needed more than one).
- Batched: streamed differs from all-in-VRAM by 0.24% (16.7862 against 16.8267). **Not
  identical.** The batched streamed path is the expert-major prefill kernel, which groups rows by
  expert; the 11B ledger saw 0.02 to 0.04% from the same kernel and called it reordering. Here a
  small numeric change can flip a near-tie in the router and the window then stays different, so
  the effect is larger. Whether that is all of it is checked in the prefill entry (control with the
  same kernel and no streaming). It does not touch the decode measurement, which is batch 1.
- The two all-in-VRAM trained runs differ between batch sizes (16.8819 against 16.8267) for the
  same reason: batch 1 uses the mmvq kernel, batches use mmq.

### Routing against the PyTorch reference (fails as specified; cause is the stock CUDA forward, not the rule)

Same check as the laptop and the phone: F32 GGUF, plain attention, F32 KV, four 2,048-token test
rows, one decode call per row, `check_engine_routing.py`. `gguf_logits` got a `GGUF_LOGITS_GPU=1`
switch for this (it was CPU-only).

| run on the pod build | windows equal to the reference | route line | loss (reference 3.14409) |
|---|---|---|---|
| CPU only (control) | **106,496 of 106,496** | `swaps_per_layer_token=1.0000` | 3.14413 (+0.00004) |
| GPU, routing kernel `k_route` | **24,804 of 106,496** | `layers=13 layer_tokens=106496 cold_fills=52 swaps=106444 swaps_per_layer_token=1.0000` | 3.14432 (+0.00023) |
| GPU, routing op left on the CPU (`TEMPORAL_ROUTE_HOST=1`) | dump byte-identical to the row above (`cmp`) | same counts | 3.14432 |

What this shows:

- The kernel implements the rule exactly: fed the same GPU logits, `k_route` and the CPU op
  `ggml_temporal_route` (the code that matches the reference on three machines) write the same
  106,496 windows, slot order included.
- The GPU forward itself is not the CPU forward. Under stock top-k, with no temporal code in the
  path, the F32 model's output logits on GPU and CPU differ by a median of 3.5e-3 and up to 1e-2 at
  position 0 (256 test tokens, logit std 3.3). That is half-precision-sized, and it is in the stock
  CUDA backend.
- The windows leave the reference only at near-ties. `gpubench/route_divergence.py` takes every
  first divergence that no earlier layer can have caused (12 of them over the 4 rows) and prints the
  reference's margins there: the smallest relevant gap (two best non-residents, or two worst
  residents) is between 5e-5 and 1.2e-3 logits, median 1.8e-4, while the decision margins are 0.6 to
  4.6. After one flipped choice the window differs for a long time, because the rule is
  history-dependent, which is why 77% of layer-tokens differ after 12 flips.
- The loss moves by 0.0002 nats.

So the 106,496 gate cannot pass on this backend for a reason that is upstream of this work. The
served Q4 model is not affected in the sense that matters for timing: the swap rate is 1.0000 and
the streamed path equals the all-in-VRAM path. Unverified: where in the stock CUDA forward the
half-precision-sized error enters.

## G1-4 -- Decode, first pass (engine tree `aa969ec0`, tag `G1-dec1`)

```
python3 gpubench/run_decode.py --bin llama.cpp/build/bin --model models/fine_1e19-q4_0.gguf \
    --out results/decode.jsonl --tag G1-dec1 --ctx 1024 2048 4096 8192 16384
# per arm:  llama-bench -m ... -ngl 99 -fa 0 -t 8 -ub 1 -b 1 -p 0 -n 100 -d <ctx> -r 4 -o json [-ncmoe 14]
```

Four trials, first discarded, 100 generated tokens after the context is filled token by token.
tok/s is the mean of trials 2 to 4; all four trials are in `gpubench/results/decode.jsonl`.
Peak VRAM is the largest `nvidia-smi memory.used` during the process (it includes the CUDA
context).

| ctx | all in VRAM, trained | all in VRAM, stock top-k | R=18 slots, no bytes moved | streamed R=18 | speed kept | peak VRAM all / streamed |
|---|---|---|---|---|---|---|
| 1024 | 677.9 (1.475 ms) | 633.2 | 567.6 (1.762 ms) | 573.0 (1.745 ms) | 0.845 | 1064 / 526 MiB |
| 2048 | 589.7 (1.696 ms) | 559.5 | 510.0 (1.961 ms) | 502.3 (1.991 ms) | 0.852 | 1112 / 574 MiB |
| 4096 | 458.8 (2.179 ms) | 442.1 | 409.0 (2.445 ms) | 407.9 (2.452 ms) | 0.889 | 1204 / 666 MiB |
| 8192 | 315.6 (3.169 ms) | 305.4 | 288.8 (3.462 ms) | 285.1 (3.508 ms) | 0.903 | 1392 / 854 MiB |
| 16384 | 188.2 (5.315 ms) | 184.8 | 179.4 (5.574 ms) | 178.9 (5.589 ms) | 0.951 | 1772 / 1234 MiB |

- Route line in every trained arm: `swaps_per_layer_token=1.0000` (pass 2 at 1k:
  `layers=13 layer_tokens=18525 cold_fills=26 swaps=18498 swaps_per_layer_token=0.9999`; 18,525 =
  13 layers x (1 warmup token + 1,024 context tokens + 4 x 100 generated), so llama-bench fills the
  context once and generates from it four times).
- The stock top-k baseline is slower than the trained-routing baseline at every context (0.93 to
  0.98 of it), so the routing rule costs nothing; the stock arm pays for a full argsort.
- "No bytes moved" is the 11B row b: `TEMPORAL_SWAP_PROB=0`, slots and residency kernel active,
  copies skipped, output wrong by construction. It lands within noise of the streamed arm (at 1k it
  is slower than streamed by 1%), so at this slice size (84 KiB against 216 KiB) the exposed
  transfer is at or below the noise. Kernel overhead: 0.27 to 0.29 ms per token at every context,
  about 21 us per MoE layer.
- 4096 and above are past the 2,048-token training context: speed only.
- Peak VRAM at 16k for the stock top-k arm read 2033 MiB against 1772 for the trained arm; same
  weights and KV, not explained, not chased.

Diagnostics at 1k (`gpubench/results/decode_diag.jsonl`; run while a 5 GB upload was writing to the
pod, trials less steady than above, so indicative only): baseline with fusion off
(`TEMPORAL_NOFUSE=1`) 634 against 660 tok/s in the same session, so about 0.06 ms of the overhead
is the lost gate/up fusion; copy grid 8 or 32 blocks no better than the default 256; no-overlap
slower; CUDA graphs off 398 and 343 tok/s.

## G1-5 -- Decode, second pass and pooled result (engine tree `a7c18140`, tag `G1-dec2`)

Same command and protocol as G1-4 plus the `floor` arm. The engine was rebuilt between the passes
with CUDA patch 0002, which only adds an exit line (`temporal-cuda: ... slot swap-ins ...`); sha256
`llama-bench` `d67ca915...`, `libggml-cuda.so` `b7ec270a...`. Run 09:06-09:25 UTC with nothing else
on the pod.

tok/s, streamed R=18 / all 192 in VRAM, both under the trained router. "Pooled" is the mean of the
six kept trials per arm (trials 2 to 4 of each pass).

| ctx | pass 1 | pass 2 | pooled | pooled ms per token, all / no bytes moved / streamed | peak VRAM streamed / all |
|---|---|---|---|---|---|
| 1024 | 573.0 / 677.9 = 0.845 | 557.9 / 668.3 = 0.835 | 565.5 / 673.1 = **0.840** | 1.486 / 1.768 / 1.768 | 526 / 1064 MiB |
| 2048 | 502.3 / 589.7 = 0.852 | 502.7 / 581.2 = 0.865 | 502.5 / 585.5 = **0.858** | 1.708 / 1.959 / 1.990 | 574 / 1112 MiB |
| 4096 | 407.9 / 458.8 = 0.889 | 405.1 / 462.4 = 0.876 | 406.5 / 460.6 = **0.882** | 2.171 / 2.461 / 2.460 | 666 / 1204 MiB |
| 8192 | 285.1 / 315.6 = 0.903 | 288.5 / 315.9 = 0.914 | 286.8 / 315.7 = **0.908** | 3.168 / 3.451 / 3.487 | 854 / 1392 MiB |
| 16384 | 178.9 / 188.2 = 0.951 | 178.2 / 188.5 = 0.945 | 178.6 / 188.3 = **0.948** | 5.311 / 5.571 / 5.599 | 1234 / 1772 MiB |

- Spread between the passes: at most 1.3 points in the ratio and 2.7% in one arm's tok/s.
- Pass 2 other arms (1k / 2k / 4k / 8k / 16k): no bytes moved 563.6 / 511.1 / 403.7 / 290.8 /
  179.6; all in VRAM with stock top-k 623.5 / 550.4 / 436.9 / 306.1 / 185.1.
- **Superseded by G1-7**, which shows the "no bytes moved" arm still pays for launching the
  copy kernels, so the split below puts too much under kernel overhead and too little under
  transfer. Left as first written. Split of the gap, as the paper does for 11B: kernel overhead (no bytes moved minus all in VRAM)
  0.25 to 0.29 ms per token at every context; exposed transfer (streamed minus no bytes moved)
  between -0.00 and +0.04 ms, inside the pass-to-pass noise. The 11B figures were 0.68 ms and
  0.38 ms on a 5.0 ms token. Per MoE layer the kernel overhead is about 21 us here and about 15 us
  at 11B, so the fixed cost did not shrink with the model; the token did (1.49 ms against 5.0 ms),
  and the ratio stays near 0.84 only because the transfer term is gone.
- Streamed swap-ins against the router: at 1k the router made 18,498 swaps over 18,525
  layer-tokens and the slots took 18,814 swap-ins (the extra 316 are the two cold fills, 13 layers
  each, that need more than one).
- Memory: the saving is 538 MiB at every context (617 MiB of routed experts in VRAM against 58 MiB
  of slots); the nvidia-smi totals include the CUDA context and the KV cache, so the ratio of
  totals is 2.0x at 1k and falls with context. The streamed arm also holds a 617 MiB pinned copy of
  the expert pool in host RAM.

**Free top-k at the streamed memory (the 11B "vanilla offload" row, here with the real router).**
Arm `floor`: stock top-k router, R=18 slots, every selected expert that is not resident is
fetched (`TEMPORAL_UNIFIED=1 TEMPORAL_UNIFIED_NOFORCE1=1 -ncmoe 14`). No emulation is needed for
this model: the router's own misses drive it. 268.8 / 253.4 / 227.2 / 184.4 / 134.8 tok/s at 1k
to 16k, which is 0.40 / 0.44 / 0.49 / 0.58 / 0.72 of the trained all-in-VRAM baseline and 0.43 /
0.46 / 0.52 / 0.60 / 0.73 of the stock top-k all-in-VRAM one. At 1k it fetched 233,025 experts
over 18,525 layer-tokens, 12.6 per layer per token, against 1.0 under the trained rule. Its
perplexity has not been compared with the stock all-in-VRAM engine yet (open).

**Bound on the flash-attention caveat (G1-2).** The time that depends on context is the same in
both arms: from 1k to 2k both grow by 0.222 ms (pooled). Extrapolating both to no
context-dependent time at all gives 1.264 ms all in VRAM and 1.546 ms streamed, a ratio of 0.82.
So a faster attention kernel could move the 1k figure down from 0.84, but not below about 0.82.
This is arithmetic on the measured rows, not a measurement.

### What App. D can state, in plain terms

- Card and host: one RTX A6000 (49 GB) on a rented pod in the EU-SE-1 data center, Xeon Gold
  6342 host, CUDA 12.4. It is not the machine of the 11B runs.
- Model and engine: `fine_1e19-q4_0.gguf` (Q4_0), llama.cpp fork at `43bec5f3` plus the patch
  files in this repo, batch 1, trained router in every arm that is compared.
- Flash attention is off (`-fa 0`) because the CUDA flash-attention kernels do not support this
  model's head size of 16; with it on, attention runs on the CPU and every configuration, the
  stock one included, drops to about 20 tok/s. The 11B runs had it on.
- Protocol: context filled token by token, then 100 generated tokens timed, four trials, first
  discarded; two such passes, pooled.
- Perplexity at batch 1 is identical streamed and all in VRAM (16.8819 +/- 0.72000 on 8 chunks of
  1,024 tokens, every running estimate equal).
- Routing check against PyTorch on the GPU: 24,804 of 106,496 layer-token windows equal. The
  routing kernel writes exactly what the CPU routing code writes from the same logits, and the CPU
  build matches PyTorch on 106,496 of 106,496; the GPU forward pass itself differs from the CPU one
  by about 3e-3 in output logits under stock routing, and the windows leave the reference only at
  near-ties (gaps of 5e-5 to 1.2e-3 logits at the 12 independent first divergences). The loss on
  those rows moves by 0.0002 nats (3.14432 against 3.14409). One swap per layer per token
  (1.0000) in every run.

## G1-6 -- The other 11B tests, repeated on this model (engine tree `a7c18140`, 09:15-09:21 UTC)

All under the trained router unless marked, `-fa 0 -t 8`, four trials with the first discarded.
Rows: `gpubench/results/decode_swapn.jsonl`, `decode_ub2048.jsonl`, `prefill.jsonl`; console log
`gpubench/results/logs/runall.log`.

**What was not repeated and why.** The router-early row (11B row d) routes on the pre-attention
input; a trained model would have to be trained that way, so it has no honest counterpart here. The
coarse model has not been converted. A single 16,384-token micro-batch could not be created in any
arm (`failed to create context`): without flash attention the attention scores for one
micro-batch need 16384 x 16384 x 50 heads x 4 bytes = 53.7 GB, more than the card. Those three
rows are void.

**Pinned swaps per layer** (`TEMPORAL_SWAP_N=n`, the 11B floor curve; timing only, output wrong by
construction, sources cycled so every copy is cold), 1k context:

| swaps per layer | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| tok/s | 590.6 | 555.7 | 517.8 | 453.5 | 365.0 |
| ms per token | 1.693 | 1.799 | 1.931 | 2.205 | 2.740 |

From 0 to 8 swaps the token grows by 1.047 ms, 0.131 ms per swap per layer over 13 layers, which
is 10.1 us for the 259,200 bytes of one swap: an effective 25.7 GB/s (the 11B curve gave 29.4).

**Decode after a batched context fill** (`-ub 2048 -b 2048 -d <ctx>`, the 11B context sweep). The
first generated token then finds stale slots and reloads the window (the multi-swap fallback).

| ctx | streamed / all in VRAM tok/s | kept | peak VRAM streamed / all |
|---|---|---|---|
| 1024 | 574.9 / 676.3 | 0.850 | 946 / 1466 MiB |
| 2048 | 507.2 / 593.3 | 0.855 | 1946 / 2466 MiB |
| 4096 | 407.2 / 462.8 | 0.880 | 3248 / 3768 MiB |
| 8192 | 290.6 / 318.9 | 0.911 | 5856 / 6376 MiB |
| 16384 | 180.2 / 190.2 | 0.947 | 11070 / 11590 MiB |

Same ratios as the token-by-token fill. The VRAM peaks are dominated by the non-flash attention
buffers of the 2,048-token micro-batches; the 520 MiB difference is the experts.

**Prefill** (`llama-bench -p <ctx> -n 0 -ub <ub> -b <ub>`). A = all in VRAM, stock batched
`mul_mat_id`; D = all in VRAM through the expert-major kernel, nothing uploaded; C = streamed
expert-major (R=18 slots plus the 96-slot ring).

| test | ctx | micro-batch | A ms (tok/s) | D ms | C ms (tok/s) | C / A time | peak VRAM A / C |
|---|---|---|---|---|---|---|---|
| single micro-batch | 1024 | 1024 | 94 (10,875) | 190 | 192 (5,340) | 2.04 | 1372 / 850 MiB |
| single micro-batch | 4096 | 4096 | 690 (5,934) | 815 | 826 (4,960) | 1.20 | 6126 / 5604 MiB |
| single micro-batch | 16384 | 16384 | void | void | void | | |
| matched micro-batch | 16384 | 1024 | 4677 (3,503) | | 6146 (2,666) | 1.31 | 6588 / 6066 MiB |
| matched micro-batch | 16384 | 2048 | 4848 (3,380) | | 5805 (2,823) | 1.20 | 11472 / 10950 MiB |
| matched micro-batch | 16384 | 4096 | 5307 (3,087) | | 5967 (2,746) | 1.12 | 21242 / 20722 MiB |
| context sweep | 1024 | 2048 | 95 (10,724) | | 193 (5,318) | 2.02 | 1372 / 850 MiB |
| context sweep | 2048 | 2048 | 233 (8,792) | | 347 (5,896) | 1.49 | 2348 / 1826 MiB |
| context sweep | 4096 | 2048 | 554 (7,399) | | 790 (5,184) | 1.43 | 3652 / 3130 MiB |
| context sweep | 8192 | 2048 | 1538 (5,328) | | 2039 (4,019) | 1.33 | 6258 / 5736 MiB |

- C against D is 1.01 at both sizes where D was run, so streaming itself is free in prefill, as at
  11B. The cost is the expert-major kernel against the stock one (D / A = 2.0 at 1k, 1.18 at 4k):
  it issues one small matmul per active expert, and with 192-wide experts those are overhead-bound.
- The gap closes with context because attention, which is the same in both, takes over.
- Best micro-batch at 16k is 1024 for A and 2048 for C.

**Prefill perplexity** (`-b 1024 -ub 1024`, same input as G1-3, engine tree `a235a715`):

| router | A, stock kernel | D, expert-major, all in VRAM | C, streamed |
|---|---|---|---|
| trained | 16.8267 | 16.7862 | 16.7862 (twice) |
| stock top-k | 17.0846 | 17.0814 | 17.0814 (`TEMPORAL_PREFILL_FULLMOE=1`) |

C equals D in every digit under both routers, so the streamed prefill computes exactly what the
same kernel computes with everything in VRAM. The kernel itself differs from the stock one by
0.02% under top-k (the 11B figure) and 0.24% under trained routing, where a rounding-sized change
can flip a near-tie and the window then stays different. This closes the open point of G1-3.

**Free top-k floor perplexity** (batch 1): `TEMPORAL_UNIFIED_NOFORCE1=1` with the stock router
gives 17.1077 +/- 0.72986, identical to the stock all-in-VRAM engine, with 1,240,557 experts
fetched. This closes the open point of G1-5.

## G1-7 -- Where the time goes, and what was tried to recover it (09:24-09:43 UTC)

CUDA patches 0003 to 0005 add opt-in switches; engine trees `2ce47549` (0001 to 0004) and
`a235a715` (0001 to 0005). All rows at 1k context, `gpubench/results/decode_opt.jsonl`; baseline
in the same sessions 671 to 675 tok/s (1.482 to 1.490 ms). Every streamed variant below that moves
bytes was put through the batch-1 perplexity gate and gives 16.8819 +/- 0.72000
(`gpubench/results/logs/ppl_opt/`).

**Correction to G1-4 and G1-5.** Those entries read the "no bytes moved" arm
(`TEMPORAL_SWAP_PROB=0`) as kernel overhead and concluded that transfer was free. That arm still
launches the three copy kernels per layer (they return early) and adds a memset node per layer.
Taking the copy kernels out altogether gives a different split. Caught when two changes that
remove kernels (fusion kept, residency folded into the routing kernel) barely moved the streamed
number.

| configuration (1k) | tok/s | ms per token | over all-in-VRAM |
|---|---|---|---|
| all in VRAM | 673 | 1.486 | |
| slots, fused, routing kernel keeps the slots, no copy kernels (output wrong) | 654.4 | 1.528 | +0.04 |
| same, gate/up unfused | 633.1, 636.1 | 1.579, 1.572 | +0.09 |
| default machinery (residency kernel, unfused), no copy kernels (output wrong) | 619.9 | 1.613 | +0.13 |
| default, copy kernels launched but zero swaps (`TEMPORAL_SWAP_N=0`) | 590.6 | 1.693 | +0.21 |
| default streamed | 568 to 574 | 1.743 to 1.760 | +0.26 to +0.27 |

So, per token at 1k: about 0.13 ms is the slot machinery (0.05 lost fusion, 0.035 the residency
kernel, 0.04 the rest), about 0.08 ms is launching and joining three copy kernels on a second
stream, and about 0.06 ms is waiting for bytes. Transfer in the wide sense (everything the copies
cost) is about 0.14 ms, a little over half the gap. The physical part is small but not hideable:
one swap is 259,200 bytes and takes about 10 us at the measured 25.7 GB/s, and between the router
and the expert matmul of the same layer there are only a few microsecond-sized kernels. At 11B the
expert matmuls themselves were long enough to hide 71% of it.

**What was tried.**

| change | switch | streamed tok/s at 1k | effect |
|---|---|---|---|
| none | | 568.3, 573.8 | |
| keep gate/up fusion | `TEMPORAL_UNIFIED_FUSE=1` | 561.5 | none or worse: the fused kernel needs two of the three copies before it can start |
| one copy kernel, one event | `TEMPORAL_UNIFIED_LEAN=1` | 541.2 | worse (the 11B ledger saw the same) |
| one copy kernel on the compute stream | `TEMPORAL_UNIFIED_LEAN=2` | 545.7 | worse |
| routing kernel keeps the slots, no residency kernel | `TEMPORAL_ROUTE_SLOTS=1` | 572.4 | +0.5% |
| all three of the above | | 573.6 to 579.7 (grids 32 to 1024) | +1% |
| copies start at the router | `TEMPORAL_ROUTE_SLOTS=2` | 585.4, 585.4 | **+2%** |
| the same plus shared expert ordered first | `+ TEMPORAL_SHEXP_FIRST=1` | 585.7 | nothing more |
| the same plus fusion | | 581.3 to 583.5 | nothing more |
| copy grid 16 to 512 blocks | `TEMPORAL_COPY_BLOCKS` | 568 to 583 | 128 to 256 best |

`TEMPORAL_ROUTE_SLOTS=2` is the only change that helps beyond noise. It drops the residency kernel
and gives the copies the gate-weight operations to overlap with.

**Third pass, default and `TEMPORAL_ROUTE_SLOTS=2` side by side** (engine tree `a235a715`, tags
`G1-dec3`, `G1-dec3-slots2`, same protocol):

| ctx | all in VRAM | streamed, default | kept | streamed, `ROUTE_SLOTS=2` | kept |
|---|---|---|---|---|---|
| 1024 | 672.2 | 570.7 | 0.849 | 585.4 | 0.871 |
| 2048 | 588.6 | 505.3 | 0.858 | 521.0 | 0.885 |
| 4096 | 463.9 | 407.4 | 0.878 | 415.0 | 0.895 |
| 8192 | 316.5 | 289.6 | 0.915 | 294.2 | 0.930 |
| 16384 | 188.9 | 180.1 | 0.953 | 180.1 | 0.953 |

**Default configuration pooled over the three passes** (nine kept trials per arm):

| ctx | streamed / all in VRAM tok/s | kept | per pass |
|---|---|---|---|
| 1024 | 567.2 / 672.8 | **0.843** | 0.845, 0.835, 0.849 |
| 2048 | 503.4 / 586.5 | **0.858** | 0.852, 0.865, 0.859 |
| 4096 | 406.8 / 461.7 | **0.881** | 0.889, 0.876, 0.878 |
| 8192 | 287.7 / 316.0 | **0.911** | 0.903, 0.914, 0.915 |
| 16384 | 179.1 / 188.5 | **0.950** | 0.951, 0.945, 0.954 |

**Where the limit is.** With no copy kernels at all the streamed path would keep 0.97 at 1k
(1.528 against 1.486 ms). With the bytes fully exposed at the measured bandwidth and nothing else
(13 x 10 us) it would keep about 0.90. It keeps 0.84 by default and 0.87 with the one switch that
helps. The remaining 3 points are the cost of running copy kernels on a second stream inside the
captured graph; nothing tried here removed it. Not tried: predicting the next expert during
attention and loading it into a spare slot, which is the only way to give the copy a long window on
a model this small without changing the architecture.

**Not verified.** `ROUTE_SLOTS=2` has the batch-1 perplexity gate and one timing pass; it was not
run after a batched context fill, and the default configuration is the one with three passes.

## G1-8 -- Pod and cost

Pod `bcnqix1ewwwjr3` ran 08:16 to 09:44 UTC on 2026-11-11, 1.47 h at $0.59/h, about $0.87, and
was then stopped (not yet terminated; its 80 GB volume holds the build and the models). No second
pod could be rented.
