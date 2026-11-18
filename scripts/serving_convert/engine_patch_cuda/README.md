# Engine patch: trained temporal routing on the CUDA streaming path

`0001-*.patch` runs the training-time routing rule as a CUDA kernel and lets the streamed
R-slot path follow it. `0002-*.patch` prints the number of slot swap-ins at exit for any router
(`temporal-cuda: ...`). `0003` to `0005` add switches that are off by default (last section).
They apply on top of three things, in this order:

1. `ncylich/llama.cpp` branch `temporal-moe-win` at commit `43bec5f3`
2. `../engine_patch/0001-*.patch` and `0002-*.patch` (`git am`; tree `b69151b6`)
3. `llamacpp-bench/systems_bench.patch` (`git apply`; the CUDA streaming path, never pushed to the fork)

```bash
git clone https://github.com/ncylich/llama.cpp && cd llama.cpp
git checkout -b cuda-trained 43bec5f3
git am  $REPO/scripts/serving_convert/engine_patch/000*.patch
git apply $REPO/llamacpp-bench/systems_bench.patch && git add -A && git commit -m "systems bench patch"
git am  $REPO/scripts/serving_convert/engine_patch_cuda/000*.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target llama-bench llama-perplexity -j
```

`gpubench/build_engine.sh` does exactly this. Nothing here was pushed to the fork.

## What it does

- `LLAMA_TEMPORAL_ROUTE=trained` on a CUDA build: the `ffn_moe_route` custom op that the CPU patch
  inserts is executed by the CUDA backend (`k_route` in `temporal.cu`) instead of falling back to
  the CPU. Window state is per layer on the device; the rule, slot order and tie-breaking are those
  of `ggml_temporal_route`. The decode graph stays one captured CUDA graph.
- Streamed path (`TEMPORAL_UNIFIED=1 -ncmoe <n_layer>`): `k_residency` receives the window as the
  selected ids. Its one-swap fast path already loads the admitted expert into the evicted one's
  slot. New: when more than one selected expert is missing (first token of a sequence, first
  decode token after a prefill) it falls back to the multi-swap scan.
- Expert-major prefill: the index-based residency mask is skipped, the ids already are each
  token's window.
- `llama_context` resets the device window at sequence position 0.
- At exit: `temporal-route-cuda: ... swaps_per_layer_token=<x>`, and for the streamed path
  `temporal-cuda: ... slot swap-ins ...=<n>`, the experts `k_residency` loaded.
- `LLAMA_TEMPORAL_ROUTE_DUMP=<file>` writes the device windows (same record format as the CPU).
  It synchronises after every route op, so single-token batches need `GGML_CUDA_DISABLE_GRAPHS=1`.
- `TEMPORAL_ROUTE_HOST=1` leaves the op to the CPU backend (cross-check only; it splits the graph).

Checks and numbers: `gpubench/TRAINED_MODEL_LEDGER.md`.

## Opt-in switches (patches 0003 to 0005)

Measured in ledger entry G1-7. All pass the batch-1 perplexity gate. Only the last one helps.

| switch | what it does | streamed decode at 1k |
|---|---|---|
| `TEMPORAL_UNIFIED_FUSE=1` | keeps the fused gate/up/GLU kernel on the streamed path | no gain |
| `TEMPORAL_UNIFIED_LEAN=1` or `2` | one copy kernel for the three projections (copy stream, or compute stream) | slower |
| `TEMPORAL_ROUTE_SLOTS=1` | the routing kernel keeps the VRAM slots, no residency kernel | +0.5% |
| `TEMPORAL_ROUTE_SLOTS=2` | the same, and the copies start at the router | +2% (0.85 to 0.87 speed kept) |
| `TEMPORAL_SHEXP_FIRST=1` | orders glm4moe's shared expert before the routed experts | no gain |

## Released checkpoints (patches 0006 and 0007)

For Qwen3.5-35B-A3B and Gemma4-26B-IT, unadapted. Checks and numbers: `gpubench/RELEASED_MODELS_LEDGER.md`;
drivers in `gpubench/released/`.

| switch | what it does |
|---|---|
| `LLAMA_TEMPORAL_ROUTE=decode` | our rule at decode time, as `analysis/residency/decode_state.py`: a prompt batch is routed with the stock top-k while the rule runs observe-only over it; each single-token batch is one rule step and routes inside the K residents |
| `LLAMA_TEMPORAL_ROUTE=skliar` + `TEMPORAL_CB_C`, `TEMPORAL_CB_LAMBDA` (`TEMPORAL_CB_J`, default 1) | Skliar et al.'s cache-conditional routing, as `analysis/residency/cache_bias.py`: LRU cache of C per layer, lambda times the running mean of max-min logit added to cached experts, top-1 guarantee; the prompt is routed freely and warms the cache |
| `TEMPORAL_UNIFIED=1 -ncmoe <n_layer>` | streamed: K slots per layer in VRAM, pool in pinned CPU memory. In both modes the routing kernel keeps the slots with one shared routine and the copies start at the router (the `TEMPORAL_ROUTE_SLOTS=2` design, forced on). With `skliar`, C must equal K (the cache is the slots) |
| `TEMPORAL_ROUTE_NOCOPY=1` / `=2` | timing controls, output wrong: copy kernels with an empty list / no copy kernels or events |
| `TEMPORAL_ROUTE_HIST=<json>` | loads-per-layer-token histogram at exit, format of the saved Python histograms |
| `LLAMA_TEMPORAL_ROUTE_DUMP2=<file>` | logits and ids per layer-token, plus a marker at each sequence reset, for `gpubench/released/route_check.py`; needs `GGML_CUDA_DISABLE_GRAPHS=1` |
| `TEMPORAL_BENCH_CTX`, `TEMPORAL_BENCH_GEN` | `llama-bench` fills the context and feeds the decode steps from token files (real text) instead of random tokens |
| `TEMPORAL_SWAP_HIST=w0,w1,...` (0012) | timing only, like `TEMPORAL_SWAP_N`: each layer-token copies n cold experts, n drawn on the device from the weights (w_n for n loads); `0,1` is exactly one load; realised counts are printed at exit as `temporal-swap-hist:` |
| `TEMPORAL_SHEXP_FIRST=1` (0007) | orders Qwen3.5's shared expert or Gemma4's dense MLP between the router and the routed experts, so the copies run under it; same ops, same result |
| `TEMPORAL_LOAD_FIRST=1` / `2` / `3` (0013, engine tree `bcf3fe54`) | streamed, routing op on the device, single-token steps; forces `TEMPORAL_ROUTE_SLOTS=2`. All values: the routing op (which starts the copies) is ordered right after the router logits, and `glm4moe`'s shared expert between it and the routed experts (`TEMPORAL_SHEXP_FIRST` does not do that for `glm4moe`). `1`: the routed experts run as two passes, experts already in their slots first with no wait, incoming experts after their copies, gate/up fused; `2`: the same unfused; `3`: order only (combine with `TEMPORAL_UNIFIED_FUSE=1`). Same values as the single pass. 0013 also makes `TEMPORAL_SWAP_N` / `TEMPORAL_SWAP_HIST` work on the routing-op path and `TEMPORAL_ROUTE_NOCOPY=1` apply to `k_route`. Results: `gpubench/LOAD_OVERLAP_LEDGER.md` |
| `TEMPORAL_FUSED=1` (0014, engine tree `71b82131`) | single-token steps, routing op on the device, `glm4moe` shapes with Q4_0 experts. One custom op per MoE layer replaces the shared expert, the routed experts, the gate-weight ops, the weighting and the sum; it runs as two launches and, streamed, moves the experts on the routing op's swap list from the pool into their slots itself (no copy kernels or second stream). Bit-identical to the unfused path. `TEMPORAL_FUSED_NOFOLD=1` leaves the gate weights to the graph's own ops; `TEMPORAL_FUSED_SKIP` is a profiling switch with wrong output. Results: `gpubench/LOAD_OVERLAP_LEDGER.md`, Part 3 |
| `TEMPORAL_QKV=1` (0015, engine tree `4610bc1d`) | single-token steps, any model whose Q, K and V projections are separate quantised matrices of one type and width. The graph places the three projections next to each other and the backend runs them as one quantisation of the input vector and one `mul_mat_vec_q` launch with three channels. Row arithmetic unchanged; perplexity equal in every digit. |
| `TEMPORAL_FUSED=3` (0015) | as `TEMPORAL_FUSED=1`, with the down tensor of each expert being loaded copied by blocks at the end of the gate/up launch instead of by the down launch. `TEMPORAL_FUSED_TAIL` sets the number of those blocks (default 14; 4 to 56 measure the same). |
| `TEMPORAL_QKV=2` (0016, engine tree `9238ff10`) | as `TEMPORAL_QKV=1`, and the RMS norm and weight that produce the projections' input are computed by the launch that quantises it. |
| `TEMPORAL_FUSED_ADD=1` (0016) | with `TEMPORAL_FUSED`: the residual add that follows the fused MoE op is done by the op's down launch. |
| `TEMPORAL_OPPROF=1` (0015, profiling) | with `GGML_CUDA_DISABLE_GRAPHS=1`: syncs around every node of single-token graphs and prints time per op kind at exit. A ranking, not a timing. |

Patch 0006 also carries an earlier `TEMPORAL_SHEXP_FIRST` for `qwen35moe` that puts the shared expert before
the router; 0007 replaces it. `gpubench/build_engine.sh` builds the server without its web UI
(`-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF`), whose asset step fails offline.
With `ENGINE_BASE_TGZ=<git archive of 43bec5f3>` the script builds without a route to GitHub (the base tree
hash is checked).
