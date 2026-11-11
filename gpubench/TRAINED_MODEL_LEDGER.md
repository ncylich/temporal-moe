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

- Route line in every trained arm: `swaps_per_layer_token=1.0000` (for example at 1k:
  `layers=13 layer_tokens=17225 cold_fills=...`).
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
