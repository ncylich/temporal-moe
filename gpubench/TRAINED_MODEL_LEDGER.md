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
