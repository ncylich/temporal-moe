# Temporal-MoE systems benchmark (llama.cpp)

Single source of truth for the real-engine, B=1 decode benchmark of temporal-MoE (rolling expert
residency: keep R of E experts VRAM-resident, stream ≤1 expert/layer/token on demand). Proves the
VRAM-reduction + decode-latency claim in a real inference engine, fairly. Code lives beside this file.

## Provenance
- **Engine:** llama.cpp CUDA fork @ base commit `0badc06`, sm_86 (RTX A6000), `Release`, `GGML_CUDA=ON`.
- **Model:** random-weight **Qwen3-MoE**, 192 experts / top-18, Q4_K_M (`qwen3moe-rand-Q4_K_M.gguf`,
  ~11.4B, the fine-grained 18-of-192 config upscaled ~10×). Weights are irrelevant to a latency/VRAM
  benchmark — a valid kernel on a realistic architecture is all that's needed.
- **Two tiers:** all-resident ceiling (all experts on GPU, **no** `-ncmoe`) vs temporal reduced
  (`-ncmoe 48` + `TEMPORAL_UNIFIED=1`: R=18 VRAM slots + 192-expert CPU pool, live swap + id-remap in the
  CUDA `mul_mat_id` hook). VRAM **1502 vs 7672 MiB = 5.1× total / 10.6× expert-tier**, graphs-on, decode
  bit-identical (verified via `llama-perplexity -ub 1` PPL to 4 dp — greedy text is a useless oracle on
  random weights).

## Results (decode @ 1024-token context, n=128, r=8; tok/s, higher = faster)

Four kernel setups isolate each cost (real swap rate p≈1.0): **a)** baseline — stock kernel, all
experts in VRAM (ceiling); **b)** our kernel, everything in VRAM / no swap (p=0) — our code overhead
with zero bytes moved; **c)** active-only in VRAM, router *after* attention — the **deploy** (5.1× VRAM
cut); **d)** router *before* attention, MLP after — router-early variant. Measured on two models of the
same total size (~11.3B) + active FLOPs, differing only in granularity: **fine** = 18-of-192 (moe_ff
384, ~840 KiB/swap); **coarse** = 6-of-64 (moe_ff 1152, ~2.5 MiB/swap).

| setup | fine 18-of-192 | vs a | coarse 6-of-64 | vs a |
|---|---|---|---|---|
| a) baseline (stock, all experts in VRAM) | 200 | 1.00× | 252 | 1.00× |
| b) our kernel, everything in VRAM / no swap (p=0) | 176 | 0.88× | 217 | 0.86× |
| c) active-only, router **after** attn — **deploy** | **165** | **0.83×** | **128** | **0.51×** |
| d) router **before** attn, MLP after — router-early | 173 *(grid 8)* | 0.87× | 143 *(grid 32)* | 0.57× |

**Fine-grained is far friendlier to expert streaming.** The swap-copy cost (gap b→c) is only ~11 tok/s
for fine (840 KiB swap, mostly hidden) but ~89 for coarse (2.5 MiB swap dominates decode) → coarse
deploy runs at 0.51× the ceiling vs fine's 0.83×. Our code overhead (a→b) is granularity-independent
(~0.86–0.88×). The router-early copy grid scales with swap size (fine 8 / coarse 32 — the bigger copy
needs more blocks to saturate PCIe); router-early nearly closes fine's small copy gap but recovers only
~1/6 of coarse's. (b splits further: overlap-machinery ~½ and residency bookkeeping ~½; the pure
graph/dispatch/layout change is ~0, i.e. baseline within noise.)

**Headline: temporal-MoE B=1 decode at the real routing swap rate = ~164 tok/s = 0.82× the all-resident
ceiling, at 5.1× VRAM reduction.** Three findings shaped it:

1. **Real swap rate is p≈1.0.** The trained models (probe `e1_swap_rate_by_layer.csv`) fire a swap
   almost every layer/token, and the mechanism hard-caps at ≤1 swap/layer — so the faithful deploy
   number is the ≤1-swap/layer (`force1`) timing, driven by the measured rate via `TEMPORAL_SWAP_PROB`.
2. **The swap copy is PCIe-physics-bound, not hideable at B=1 by default** — the B=1 compute window is
   ~3× smaller than the copy. It *can* be hidden by overlapping **attention** (router-early variant,
   below), but only if the copy kernel uses a small grid so it frees SMs for attention (`TEMPORAL_COPY_BLOCKS=8`).
3. **The residual gap is per-layer slot bookkeeping**, now ~95% eliminated: the residency kernel's
   assignment was parallelized (was a serial chain of dependent global-memory reads) and the remap was
   fused into it. What remains is the single-block residency kernel that must decide the ≤1 swap every
   layer on the critical path.

**Router-early is an architectural variant, not a free optimization:** routing on the pre-attention
input selects different experts, so a trained model would need to be trained that way. The systems
benchmark validates it's feasible and worth ~+6%; the ~164 headline is for the existing
post-attention-routed model.

Swap-rate → tok/s emulation curve: `results/phase0/figure_data/p2b_swaprate_emul.csv`.

## Build & run
```bash
git checkout 0badc06 && git apply systems_bench.patch     # or drop in temporal.{cu,cuh} + apply hunks
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target llama-bench llama-perplexity -j

# DEFAULT recipe: 1024-token prefill depth, 128 decode tokens timed (prefill speed excluded via -d).
# deploy (temporal, 5.1x VRAM cut, real swap rate p~1.0):
TEMPORAL_UNIFIED=1 TEMPORAL_UNIFIED_OVERLAP=1 TEMPORAL_SWAP_PROB=1.0 build/bin/llama-bench \
  -m qwen3moe-rand-Q4_K_M.gguf -ngl 99 -fa 1 -ncmoe 48 -ub 1 -b 1 -d 1024 -n 128 -r 8
# router-early variant (add):  TEMPORAL_ROUTER_EARLY=1 TEMPORAL_COPY_BLOCKS=8
# all-resident ceiling (NO -ncmoe):
build/bin/llama-bench -m qwen3moe-rand-Q4_K_M.gguf -ngl 99 -fa 1 -ub 1 -b 1 -d 1024 -n 128 -r 8
```

## Reproducing the GGUF models (was local-only; now scripted)

The two random-weight benchmark models are regenerated deterministically from the recipe by
`gen_random_qwen3moe.py` + `build_models.sh` (no local artifacts required):
```bash
bash build_models.sh <llama.cpp-dir> [out-root]     # -> qwen3moe-rand-{fine,coarse}-Q4_K_M.gguf
# out-root defaults to llamacpp-bench/models/, or set $MODELS_ROOT
```
Shared backbone (identical across granularities so total params + active FLOPs match): hidden 1024,
**45 layers**, 8 heads / 4 KV heads (head_dim 128), Qwen3 tokenizer vocab, tied embeddings, Q4_K_M.
Granularity: **fine** = 192 experts / top-18 / moe_ff 384; **coarse** = 64 / top-6 / moe_ff 1152
(E·moe_ff and top_k·moe_ff are invariant → same ~10 B total + active FLOPs). Depth (45) is pinned from
the original model's KV-cache slope in the context sweep (~89 KiB/token = n_layer·n_kv_heads·head_dim),
which reproduces all three fingerprints: all-resident ceiling **200.8 tok/s** (orig 200) / coarse 251
(orig 252), VRAM ≈ 7150 MiB (recorded 7672), KV ≈ 90 KiB/token.

## Vanilla-offload floor (Table 2)

The floor = what a **default** MoE does in our memory setup: only R=k expert slots resident, free top-k
every token, fetch-on-miss. It misses ~k·(1−k/E) experts **per layer** (fine 16.3, coarse 5.4) — many
swaps/layer, far more than the deploy's ≤1. Two ways to drive it, both moving real bytes:
- `TEMPORAL_UNIFIED_NOFORCE1=1` (budget=R): swaps every selected non-resident expert, evicting the
  lowest-index non-selected slot. Eviction is **not LRU** (forward-scan among non-selected slots). On the
  random-weight model the router has low diversity, so its *natural* miss rate is unrepresentative.
- `TEMPORAL_SWAP_N=n`: pin exactly n cold-miss swaps/layer (the faithful path — pins the miss rate the
  random model won't produce). Used for the Table-2 rows at n = round(k·(1−k/E)) and round(0.8k).

## Env knobs (read once, gated, default-off = stock behavior)
| env | effect |
|-----|--------|
| `TEMPORAL_UNIFIED=1` | on-device residency + graph-capturable swap (needs `-ncmoe`) |
| `TEMPORAL_UNIFIED_OVERLAP=1` | copy-pipelining overlap (gate/up/down copies issued at the gate op) |
| `TEMPORAL_SWAP_PROB=p` | drive decode at measured swap rate p∈[0,1] (1 swap on round(p·nlayers) layers) |
| `TEMPORAL_UNIFIED_FORCE1=1` | force the ≤1-swap/layer mechanism (== the p=1.0 deploy timing) |
| `TEMPORAL_ROUTER_EARLY=1` | route before attention so the swap-copy overlaps attention (arch variant) |
| `TEMPORAL_COPY_BLOCKS=n` | swap-copy grid width; **use 8 with router-early** to free SMs for attention |
| `TEMPORAL_SWAP_N=n` | **vanilla-offload floor**: force EXACTLY n cold-miss swap-copies per layer (n≤R), source experts cycled to defeat L2 caching. Emulates a default MoE with only R=k slots + fetch-on-miss at a pinned per-layer miss rate. |
| `TEMPORAL_UNIFIED_NOCOPY=1`, `TEMPORAL_NOFUSE=1`, `TEMPORAL_GRAPHDBG=1` | diagnostics |

## Problems solved (and how)
The hard engineering problems, so the next person doesn't re-derive them:

1. **Graph-capturable swap with device-computed indices.** The swap decides which expert → which slot
   *on-device during graph replay*, but a CUDA-graph memcpy node needs fixed host addresses (so the DMA
   engine is out). **Solved:** an SM copy kernel (`k_swapcopy`) reads a host-mapped CPU pool over PCIe by
   device-computed indices, alongside on-device residency + remap kernels — zero host sync, fully
   capturable and host-sync-free.
2. **Bit-identical overlap.** The first overlap recomputed the swapped expert and scattered it — its
   shape was data-dependent on the number of swaps, so it wasn't bit-identical. **Solved:**
   copy-pipelining — issue gate/up/down copies at the gate op, each GEMM waits *its own* event and reads
   a finished slot → bit-identical AND graph-capturable (verified via `llama-perplexity` PPL to 4 dp).
3. **The honest deploy number.** The residency kernel allowed up to R=18 swaps/layer, which the real
   mechanism (≤1 swap/layer) never produces — a pessimistic, non-faithful figure. **Solved:** emulate the
   mechanism at the **measured** routing swap rate (p≈1.0, from probe `e1_swap_rate_by_layer.csv`) via
   `TEMPORAL_SWAP_PROB` / `force1`.
4. **Hiding the PCIe copy — needs TWO things, not one.** At B=1 the copy (~840 KiB) is ~3× the
   expert-GEMV window, so it can't hide there. Routing before attention exposes a large overlap window,
   **but** the copy is a device SM-kernel, so at the full grid it monopolizes SMs and *blocks* attention.
   **Solved:** `TEMPORAL_ROUTER_EARLY` (issue the copy before attention) **plus** `TEMPORAL_COPY_BLOCKS=8`
   (PCIe-bound, so a few blocks saturate the link and free SMs) → attention overlaps the copy. Router-early
   *alone* does nothing; small grid *alone* does nothing.
5. **Per-layer bookkeeping latency (164→189).** The residency kernel's swap decision did dependent
   global-memory reads serially on one thread — a chain of ~400 ns stalls. **Solved:** parallelize the
   find with `atomicMin` reductions over the block (latency hidden across threads; same lowest-index
   nominee/victim → still bit-identical), and fuse the remap into the same kernel to drop a launch.

## Warnings — wrong things we did; don't repeat
- **Don't report a swap policy the mechanism never uses.** The `budget=R` path (up to 18 swaps/layer)
  gave a pessimistic ~117 tok/s; the mechanism swaps ≤1/layer (rest remapped to resident). Always drive
  it with the trained model's **measured** swap rate, not raw random-weight routing (worst-case churn).
- **Don't trust a roofline assuming a hardware path you can't use.** The Phase-2 roofline predicted ~196
  by assuming a DMA-engine copy + a compute window big enough to hide it — neither exists at B=1 under
  graph capture. Measure the real graphs-on path.
- **Don't assume a cost before measuring it.** We *assumed* disabling kernel fusion cost speed (it
  didn't — NOFUSE ≥ fused at B=1), *assumed* SM-contention on the copy could be fixed by grid size alone
  (flat sweep — it only matters once attention is the overlap target), and *assumed* fewer copy kernels
  would help (merging the 3 gate/up/down copies into 1 killed the pipelining and **hurt** the plain
  deploy — reverted). Diagnose first.
- **Don't confuse the two tiers' recipes.** Ceiling = all experts on GPU (**no** `-ncmoe`). Reduced =
  `-ncmoe` + `TEMPORAL_UNIFIED`. Passing `-ncmoe` to the ceiling offloads experts to CPU → ~7 tok/s, not
  the real ~199 ceiling.
- **Don't time on a shared GPU** (contention silently ~halved a number), **don't use greedy text as the
  correctness oracle** (random weights hit immediate EOS — use perplexity PPL), and **don't report
  short/empty-context runs** (`-p 0` empty context or n=96 short runs) — use `-d 1024 -n 128 -r 8`.

## Code manifest (this dir)
- `systems_bench.patch` — full diff vs `0badc06` (5 files): `temporal.cu` (residency/swap/remap/router-early
  kernels), `temporal.cuh`, `ggml-cuda.cu` (mul_mat_id hook + fusion-disable), `llama-model.cpp`
  (R-slot registration), `models/qwen3moe.cpp` (pre-attention router prime).
- `temporal.cu`, `temporal.cuh` — readable copies of the two wholly-custom files.

## Prefill kernels (2026-07 update)

`systems_bench.patch` now also contains the **prefill** work (expert-major streaming, config-D
control, ubatch handling), not just decode. Apply to `llama.cpp` at base commit **`0badc06`**
(`git apply systems_bench.patch`), sm_86, `GGML_CUDA=ON`, `Release`. Verified to apply cleanly.

Env knobs added by the patch (all default-off; decode path unchanged for `n_tokens==1`):
- `TEMPORAL_PREFILL=expertmajor` — expert-major streaming prefill (counting-sort group by expert,
  per-expert batched GEMM streamed through a ring, scatter). The deployable low-VRAM path.
- `TEMPORAL_PREFILL_RESIDENT=1` — config-D control: same kernel, all experts resident, zero upload
  (isolates kernel effect from streaming cost).
- `TEMPORAL_PREFILL_SKIPSEED=1` — skip re-streaming the R already-resident experts.
- `TEMPORAL_PREFILL=wavemmid` — wave-batched `mul_mat_id` prototype (timing-only; documented NO-GO
  with the stock kernel, see paper Appendix / results/ablations/serving_benchmarks.csv notes).
- `TEMPORAL_PREFILL_COUNT=1`, `_SERIAL=1`, `_NOCOPY=1` — diagnostics.

Measured numbers: `results/ablations/serving_benchmarks.csv`. Fork base = `0badc06`; the a6000 fork's
local commit chain was `2cb4175` (v1) -> `13cc828` (v2) -> `fb0e979` (D) -> `5c2f7b2` (skipseed) ->
`1896afb` (wavemmid) -> `8fa7937` (counter) -> `6094183` (matched-ub sweep).

### Decode/prefill defaults (fork 2447b1a)

`-ncmoe <N> TEMPORAL_UNIFIED=1` alone runs the deployable config: **temporal decode**
(resident-set + <=1 swap/token, ~160 tok/s / 0.79x the full-MoE ceiling) with copy/compute overlap,
and **expert-major streaming prefill** (auto for n_tokens>1). No extra flags.
- <=1-swap temporal decode is DEFAULT-ON. `TEMPORAL_UNIFIED_NOFORCE1=1` -> lazy-full-MoE decode
  (loads all top-k; used only for the full-vs-temporal comparison; bit-identical to the all-resident
  ceiling, which PROVES the load/swap/remap/GEMM infra is exact).
- Overlap DEFAULT-ON (`_NOOVERLAP=1` to disable). `TEMPORAL_PREFILL_RESIDENT=1` = config-D control
  (all-resident expert-major, no upload; paper-table decomposition only).
- Prefill computes the full top-k per token (= full MoE, streamed for memory); the temporal
  <=1-swap mechanism applies at DECODE. Deployment = full-MoE-exact prefill + temporal decode.
