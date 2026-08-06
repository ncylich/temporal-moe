# Training throughput optimisation — Qwen3-30B and Qwen3.5

**Goal.** Make one training step faster on both Qwen models, and know by how much. OLMoE is out of
scope: 13,900 tok/s already, 2.2x Qwen3-30B, and no library supports it.

**Target config — fixed. Changing it changes the experiment, not the speed.**

    base bf16 checkpoints only (never the -FP8 repos: those are post-trained, not base)
    trainable  expert LoRA r32 + attn LoRA r32 + router gates + RMSNorm gains
    residency  R = k = 8 on every MoE layer
    step       16,384 tokens, seq 2048

**The baseline number does not exist.** The only measurement of this config on any model is 93 tok/s
(Qwen3.5, mb1, stock Python expert loop) — a 45-hour 15M run. Every estimate is void until measured.

## Guards — a change ships only if all three pass

| guard | threshold | why that number |
|---|---|---|
| BPB delta vs our bf16 | **≤ 1e-4** (≤ 1e-3 for FP8, by explicit decision) | same-kernel bf16 noise floor is 6.26e-05; `grouped_mm` was rejected at 4.93e-04 |
| top-1 agreement | ≥ same-kernel floor − 0.01 | absolute thresholds are meaningless — stock agrees with itself only 0.9779 |
| A/B validity | exactly one variable | every past speedup here failed on this, three times |

## Harness: Unsloth, bf16, in `/workspace/venv_fla` (torch 2.13 + transformers 5.12.1)

Only option covering both models with fused MoE kernels. The live path (verified by loading
Qwen3-30B, not from docs) is **`torch._grouped_mm`** installed by
`unsloth_zoo.temporary_patches.qwen3_moe` — the `grouped_gemm/reference/` tree and
`unsloth/models/qwen3_moe.py` are dead code, and `UNSLOTH_MOE_BACKEND` does not exist. torch 2.13 is
required: 2.4 lacks `_grouped_mm` and the zoo silently degrades to slow fallbacks. Their MoE LoRA
detection auto-enables expert LoRA from the default `target_modules`. grouped_mm's BPB delta is
**recorded, not vetoed at 1e-4** — explicit decision, since their use of it is trusted.

**Residency patches into routing, not kernels** — override
`_make_qwen_moe_sparse_moe_block_forward`'s product, where `[B, S]` is still in scope at the
`self.gate(...)` call. On transformers 5.x the gate is a `TopKRouter` doing top-k *inside*, so the
mask wraps the gate rather than sitting between gate and experts. The same factory serves
`qwen3_5_moe`, so one patch covers both models. Their fused GEMMs never learn residency exists.

## Execution — Qwen3-30B first (headroom, and a validated fallback exists)

| # | run | purpose |
|---|---|---|
| a | Unsloth vs ours, residency **OFF**, step 0 | kernel equivalence. LoRA `B` is zero-init so the adapter is an exact no-op — any logit difference is kernels, not training dynamics |
| b | patch residency into `run_router` | — |
| c | Unsloth vs ours, residency **ON**, step 0 | the patch is correct. Without (a), a mismatch here is ambiguous between bad kernel and bad patch |
| d | throughput at **matched config**, then at **best achievable** | two numbers, always labelled. Conflating them produced the bogus "Qwen3.5 is 24.7% slower" (mb4 vs mb2) |

Then repeat a–d for Qwen3.5. Its baseline stays the Python expert loop — no fused option exists — so
that comparison is unequal but honest, since it is what we would actually run.

**Baseline gets one cheap fix, not a project:** wire `residency_fused` into Qwen3-30B's path. Built,
validated at Δ 9.6e-06, never deployed. Makes (d) a fair fight between two optimised paths.

## Known — not worth re-litigating

- **GEMMs are not a lever**: 451 of 494.7 TFLOP/s, 91% of peak. Time is in gather+scatter (37.5%) and
  the residency scan (14.7% of a block).
- **The scan is ours; Unsloth will not touch it.** Because it's sequential, it barely uses
  parallelization, thus it's effectively O(N) unlike most things which are O(BN) or O(BN^2) in
  training. Therefore, as batch size increases its overhead plummets from as high as ~28% at MB=1 to
  just ~7% at MB=4.
- **Off-the-shelf quantisers are closed**: six surveyed, the only one that quantises 3-D experts has
  no backward. If FP8 is needed, quantise the bf16 base with Qwen's own published recipe
  (fine-grained fp8, block 128) rather than an invented scheme.

## Resolved — all gating questions closed by measurement

Steps a–d are DONE for both models; numbers and verdicts in
`results/ablations/unsloth_parity.md`. Kernels accepted at both residency states.
Qwen3-30B: matched mb2 r32 **10.5×** (5,429 vs 516 tok/s); best mb4 + 8-bit Adam 6,072.
Qwen3.5: **the target config (r32) fits — with bitsandbytes AdamW8bit** (bf16 Adam states
miss the card by ~2–3 GB; 1-byte states return 3.7 GB): matched mb1 **~15.4×** (3,305 vs
215), requiring `UNSLOTH_COMPILE_DISABLE=1` (CUDA-graph pools hold ~7.4 GB VRAM) and warmed
autotune caches. One optimizer everywhere — never difference runs with different
optimizers. All tok/s steady-state; startup is an additive constant.
