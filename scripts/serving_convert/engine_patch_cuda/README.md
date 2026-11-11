# Engine patch: trained temporal routing on the CUDA streaming path

`0001-*.patch` runs the training-time routing rule as a CUDA kernel and lets the streamed
R-slot path follow it. It applies on top of three things, in this order:

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
- At exit: `temporal-route-cuda: ... swaps_per_layer_token=<x>`, and for the streamed path the
  number of slot swap-ins `k_residency` decided.
- `LLAMA_TEMPORAL_ROUTE_DUMP=<file>` writes the device windows (same record format as the CPU).
  It synchronises after every route op, so single-token batches need `GGML_CUDA_DISABLE_GRAPHS=1`.
- `TEMPORAL_ROUTE_HOST=1` leaves the op to the CPU backend (cross-check only; it splits the graph).

Checks and numbers: `gpubench/TRAINED_MODEL_LEDGER.md`.
