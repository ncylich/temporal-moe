# Cache-Conditional Experts on GPU — Execution Plan

> **⛔ SHELVED — do not execute.** A bandwidth analysis ([`FINDINGS.md`](./FINDINGS.md)) shows
> that at batch-1 on high-bandwidth compute (A6000 VRAM 768 GB/s or M4 Pro unified RAM
> 273 GB/s), a single SSD/RAM-offloaded expert per layer cannot be hidden behind the compute of
> only `k=8` resident experts — you'd need ~20–110 resident experts of compute to mask one miss.
> The method lowers miss *rate* but not the per-miss stall, so it yields no usable speedup toward
> the resident ceiling here. The plan below is retained for reference; the escape is batching
> (Temporal MoE), where the crossover scales with the window `B`.

Replicate **Mixture of Cache-Conditional Experts** (arXiv:2412.00099) faithfully, but for a
GPU with experts offloaded to NVMe SSD instead of mobile Flash, and measure batch-1 decode
speedups on **Gemma 4 26B A4B** (q8). Companion baseline for
[`../docs/research/cache-conditional-experts.md`](../docs/research/cache-conditional-experts.md)
and the Temporal-MoE idea in [`../docs/research/temporal-moe.md`](../docs/research/temporal-moe.md).

## Context / why

The paper measures cache-conditional routing only on mobile (batch-1, Flash↔RAM) and never
benchmarks against a fully-resident model. We re-implement it on this machine's GPU to (a)
reproduce its core claim (soft bias toward cached experts cuts miss rate / extends expert
lifetime vs LRU) and (b) supply the missing **% of fully-resident throughput** number, in the
hardest possible memory hierarchy (VRAM↔SSD). This both validates our understanding of the
method and produces the baseline that motivates Temporal MoE (which fixes the batch-1 ceiling
identified below).

## Verified facts (downloaded HF config — re-asserted at runtime, not assumed)

**Model `google/gemma-4-26B-A4B`** (`text_config`): 30 layers, **all MoE**; `num_experts=128`,
`top_k=8`, `moe_intermediate_size=704`, `hidden=2816`; a **dense `self.mlp`** (intermediate
2112) runs in parallel with the MoE block every layer; sliding-window attention (1024, 6 full /
24 sliding); bf16; Apache-2.0, ungated. **One expert ≈ 5.9M params ≈ 5.9 MB int8.**

Three corrections to the original assumptions (all confirmed from the live config/source):
1. **No literal shared expert.** The always-resident "shared FFN" is the dense `self.mlp`; map
   the shared-resident requirement onto it. There is no 129th expert module.
2. **Experts are fused 3-D `nn.Parameter`s** (`gate_up_proj[128,1408,2816]`,
   `down_proj[128,2816,704]`), not `nn.Linear`s — so bitsandbytes `Linear8bitLt`-per-expert
   does not apply. We use bnb's vector-wise int8 scheme on each expert's 2-D slices, stored as
   plain movable `CB`(int8)+`SCB`(scale) tensors.
3. **Needs `transformers ≥ 5.5`** (gemma4 support); nothing is installed in this env yet.

**Hardware:** RTX A6000 (48 GB VRAM, ~768 GB/s), 503 GB RAM, dual NVMe 3.5 TB (~7 GB/s each,
~14 GB/s striped), PCIe4 x16 (~32 GB/s), CUDA 12.4 / torch 2.4.1.

## Resolved decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Offload tier | **VRAM ↔ system RAM** over PCIe (primary); NVMe SSD selectable via config |
| 2 | Stack | HF Transformers load + custom PyTorch/Triton/CUDA-stream offload. **Milestone 1 = match vLLM single-batch throughput at 100% cache before adding offload** |
| 3 | λ tuning | Matched-quality vs λ=0: single global λ, max hit-rate s.t. quality ≈ λ=0 |
| 4 | q8 weights | bitsandbytes int8 (vector-wise), per-expert movable units |
| — | Eval | 25 WildChat + 25 AceCode, 100 new tokens, greedy, batch 1; cache 50/75/100% |

## The batch-1 roofline reality (sets expectations)

At batch-1 the expert FFN is **VRAM-bandwidth-bound** (each weight read once, hit with one
token), so an offload miss hides behind compute only while `miss_rate ≲ BW_offload/BW_vram`.
With the chosen **RAM offload** that is `32/768 ≈ 4.2%`; with SSD it would be `7/768 ≈ 0.9%`
— RAM is primary because it is ~4–5× more forgiving. Both are still harsher than the paper's
mobile ~8% (its fast tier was only ~50 GB/s RAM, ~8× its Flash; our fast tier is 768 GB/s
VRAM). Implication that the report must state plainly:

- **100% cache** = the resident ceiling → Milestone 1 (match vLLM).
- **50% / 75% cache** = offload-bound; cache-conditional's residual ~7–21% miss rate sits
  well above the ~4% (RAM) / ~1% (SSD) crossover. **Headline there = miss-rate halving + expert-lifetime increase
  vs LRU (the paper's real claim), and % of resident throughput — not "approaches resident."**
- Reaching resident speed needs batch > 1 (crossover scales ≈ `B·BW_ssd/BW_vram`) — that is
  the Temporal-MoE follow-up, deliberately out of scope here.

## Directory layout (`cce/`)

```
cce/
  PLAN.md  README.md                 # this plan; run instructions
  requirements.txt                    # transformers>=5.5, bitsandbytes, triton, datasets, vllm (ref only)
  configs/  a6000.yaml gemma4_26b.yaml # HW bandwidths, cache-frac sweep, λ grid, J; verified arch (asserted at load)
  cce/
    arch.py            # ArchSpec; assert_matches(live hf text_config)
    quant.py           # per-expert int8 unit (CB+SCB); one dequant path for resident==streamed
    expert_store.py    # SSD<->pinned<->VRAM movement; VRAM slab + pinned double-buffer pool
    io/ssd_reader.py    layout.py   # O_DIRECT + posix_fadvise read path (defeats page cache); on-disk offset table
    cache.py           # LruCache (mask m̃_t, touch, admit/evict) + CacheManager (acquire, hit_rate)
    router.py          # CacheConditionalRouter: z' bias, top-J floor, per-layer Δ_avg running mean, weights-from-z
    prefetch.py        # PrefetchPipeline: copy/compute streams, double-buffer, intra-layer + speculative depth overlap
    moe_block.py       # StreamingExperts: drop-in for Gemma4TextExperts.forward (per-expert loop -> cache.acquire)
    patch.py           # install router+experts into Gemma4TextDecoderLayer; cache singletons on model._cce
    kernels/dequant.py # Triton fused int8->bf16 GEMV (reads half the bytes; the high-value batch-1 kernel)
    telemetry.py       # hit/miss rate, experts-loaded/token, expert lifetime, per-token timing, roofline counters
  bench/
    serialize_experts.py  # one-time: dump per-expert int8 weights to NVMe in layout format
    prompts.py            # build+freeze 25 WildChat + 25 AceCode prompts (chat template, frozen indices)
    lambda_sweep.py       # sweep λ; agreement + KL vs λ=0; pick λ*
    bench_decode.py       # tok/s sweep over cache-frac × {LRU λ=0, CCE λ*}; vs vLLM + resident ceiling
    vllm_reference.py     # Milestone-1 parity target (same dtype, apples-to-apples)
    roofline.py           # measure true BW_vram / BW_ssd; compute crossover
  tests/                  # see Test plan
results/cce/runs/<run_id>/  # gitignored; manifest.json per run for reproducibility
```

Reuse from repo: `analysis/router_saturation.py` (per-token `topk(...).indices` trace format →
telemetry), `analysis/expert_coactivation.py` (co-activation prior for optional speculative
prefetch), `analysis/infrastructure.py` (throughput plotting).

## Method core (replicate bit-exactly) — `router.py`

Per layer ℓ, per token, given post-attention hidden `h`:
1. `z = router.proj(h * scale)` — original logits (HF tops-k over `softmax(z)`; monotonic, so
   biasing `z` is equivalent).
2. `update Δ_avg[ℓ]` ← online mean of `max(z)−min(z)`.
3. `m = cache.mask(ℓ)` (∈{0,1}^128, resident experts).
4. `z' = z + λ·Δ_avg[ℓ]·m`  ← **bias for ranking only**.
5. `sel = topk(z', k)`; **top-J floor:** force the top-J experts by *original* `z` into `sel`,
   trim lowest-`z'` extras to keep exactly k.
6. **weights from original `z`:** `w = softmax(z)[sel]; w/=w.sum(); w*=per_expert_scale[sel]`.
7. record `hit_rate = |sel ∩ cache|/k`.

LRU eviction; `λ`, `J` from config. `λ=0` ⇒ `z'==z` ⇒ identical selection to stock HF.

## Prefetch / overlap — `prefetch.py` (step 5)

Two CUDA streams (copy vs compute), pinned-memory double buffer. Per MoE layer:
order `sel` as **[resident-first, missed-last]**; on the **copy stream** fire each miss's
SSD→pinned (O_DIRECT) → pinned→VRAM async upload (evicting the LRU victim's slot); on the
**compute stream** run resident experts; then consume missed experts as their copy-events
signal (compute blocks only on the laggard). Optionally warm layer ℓ+1's likely experts
(co-activation prior) speculatively — measured, never required for correctness (faithful CCE is
reactive). Dense `self.mlp` + attention stay resident; only `experts.{gate_up,down}` move.
`kernels/dequant.py` fuses int8→bf16 into the GEMV so the bandwidth-bound path reads int8 (½
the bytes) — the one kernel worth writing at batch-1.

## Milestones

- **M0 — Setup:** install deps; download Gemma 4; `arch.assert_matches`; `serialize_experts` to
  NVMe; `roofline.py` measures true BW_vram/BW_ssd. ✅ when crossover is empirically confirmed.
- **M1 — Resident parity:** full model in VRAM (100% cache, λ=0), int8; logits match stock HF
  within quant tolerance; **decode tok/s within ~10% of vLLM single-batch at the same dtype**
  (apples-to-apples; also report vs vLLM-bf16 as absolute ceiling).
- **M2 — Offload + LRU:** VRAM↔SSD streaming at 50/75% cache, λ=0, correctness bit-exact vs
  resident; page-cache defeat verified; telemetry (miss rate, lifetime) emitted.
- **M3 — Cache-conditional + λ:** add the bias; `lambda_sweep` picks λ*; show miss-rate drop &
  lifetime rise vs LRU at matched cache, with quality ≈ λ=0.
- **M4 — Kernel optimization (iterate):** prefetch overlap + fused dequant GEMV; push measured
  step time toward `max(T_compute, T_load)`; report speedup vs LRU and % of resident at 50/75%.

## Evaluation methodology

- **Prompts:** WildChat-1M first user turn + AceCode-87K instruction; English/length filters;
  **freeze resolved row indices** in config; `apply_chat_template(add_generation_prompt=True)`;
  exactly 100 new tokens, greedy, EOS ignored (reproducible traces). Cache to disk keyed by hash.
- **λ\*:** reference = λ=0 run. Sweep λ∈[0,1] (coarse→fine, up to 50 pts). Quality = greedy
  top-1 agreement **and** mean next-token KL vs λ=0, primarily **teacher-forced** (re-feed
  reference tokens to isolate routing drift). λ\* = largest λ with **agreement ≥ 99% and KL ≤
  0.01 nats** (≈<1% ppl, matching the paper's +0.1–3%). One global λ; tune at 75%, re-verify at 50%.
- **Speedup:** per cache-frac, three configs — (a) 100% resident ceiling, (b) LRU+λ=0, (c)
  CCE+λ\*. Report `(c)/(b)` and each as **% of (a)**. tok/s over 100-token decode, batch 1;
  `cuda.Event`, prefill/decode separated, 2 warmup + ≥10 repeats, median+IQR.
- **Page-cache defeat (critical):** O_DIRECT reads + `posix_fadvise(DONTNEED)` + `drop_caches`
  between cold runs; a `verify_io` probe **aborts the run** if read bandwidth looks RAM-speed.
- **Logged (paper Table 10):** per-layer hit/miss rate, experts loaded/token, expert lifetime.

## Test plan (faithful replication — `tests/`)

Synthetic 2-layer/8-expert/k=2 model for logic; `@pytest.mark.gpu` for real-model checks.
- `bias_affects_ranking_not_weights` — output weights use original `z`; λ-invariant for fixed `sel`.
- `top_J_always_selected` — J critical experts (by original `z`) always in `sel`; exactly k returned.
- `lambda0_equals_unbiased` / `lambda0_equals_resident` (GPU) — cap changes speed, not outputs.
- `delta_avg_running_mean` — online mean == batch mean, per-layer, per-token update.
- `hit_rate_metric`, `lru_eviction_order`, `expert_lifetime` — formulas vs brute force.
- `cce_reduces_miss_rate` — CCE < LRU miss rate at matched λ.
- `quant_bitexact` + `streamed_vs_resident_logits` — `torch.equal` resident vs SSD-streamed expert.
- `full_resident_equals_hf` — M1 correctness vs stock HF.
- `prefetch_overlap` — instrumented fake-latency SSD → step time ≈ `max(compute, load)`.
- `ssd_true_bandwidth` — cold O_DIRECT read ≈ device spec, not RAM speed (proves cache defeat).
- `shared_mlp_resident` — dense MLP + attention never leave VRAM during decode.
- `exactly_100_tokens`, `greedy_determinism`, prompt-template + selection snapshots.

## End-to-end verification

`pytest cce/tests` green → `bench/roofline.py` confirms crossover → M1 parity table (CCE-int8 vs
vLLM) → `bench/lambda_sweep.py` emits λ\* with the agreement/KL curve → `bench/bench_decode.py`
produces the cache-frac × {LRU, CCE} table with tok/s, %-of-resident, miss rate, and lifetime,
plus box plots. A run is reproducible from its `manifest.json` (promptset hash, λ, cache-frac,
git sha, frozen dataset indices).

## Top risks → mitigations

1. **Page cache inflates SSD throughput** (503 GB RAM) → O_DIRECT + fadvise + drop_caches +
   `verify_io` abort gate.
2. **bnb int8 batch-1 slower than bf16 vs vLLM parity** → compare same-dtype; ship fused int8
   GEMV (½ bytes, where int8 wins at bandwidth bound); bf16-resident toggle to validate offload
   independently of quant.
3. **Streamed≠resident** → freeze quant constants at serialize time; single dequant path;
   deterministic accumulation order; `torch.equal` test.
4. **Fused 3-D expert params** (not Linear8bitLt) → manual per-slice int8 in `quant.py`.
5. **`transformers` churn** (gemma4 attr names) → `arch.assert_matches` + capability-based attr
   resolution in `patch.py`; pin the commit.
6. **Reactive prefetch has little lead at batch-1** → correctness stalls on true miss;
   speculative depth-prefetch (co-activation prior) is a measured add-on, not a dependency.
