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
| `TEMPORAL_SHEXP_FIRST=1` (0007) | orders Qwen3.5's shared expert or Gemma4's dense MLP between the router and the routed experts, so the copies run under it; same ops, same result |

Patch 0006 also carries an earlier `TEMPORAL_SHEXP_FIRST` for `qwen35moe` that puts the shared expert before
the router; 0007 replaces it. `gpubench/build_engine.sh` builds the server without its web UI
(`-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF`), whose asset step fails offline.
