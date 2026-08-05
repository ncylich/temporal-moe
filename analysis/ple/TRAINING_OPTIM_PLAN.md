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

## Harness: Unsloth, bf16

Only option covering both models with fused MoE kernels. Kernels on by default
(`UNSLOTH_MOE_BACKEND` selects backend); their default `target_modules` already includes
`gate_proj/up_proj/down_proj`, so expert LoRA is their out-of-the-box recipe.

**Residency patches into routing, not kernels.** Their routing is plain PyTorch:

    calculate_topk(gating_output, top_k, use_sigmoid, renormalize, ...)
        scores = F.softmax(gating_output.to(torch.float32), dim=1)
        topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=1)

Router logits arrive as an argument, so we override `run_router()` and mask them to −inf — the same
shape of patch as `residency_qwen._router_forward`. Their fused GEMMs never learn residency exists.
`renormalize=True` on Qwen3 means masking cannot reintroduce the gate-mass artifact.

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

## Open — gating, in order

1. Does the production fused block use that same `calculate_topk`, or a per-backend variant? The file
   read is under `grouped_gemm/reference/`.
2. Is `[B, S]` recoverable at the router? Our scan is temporal; `calculate_topk` sees `[M, E]` flat.
3. Does `FastModel.from_pretrained` accept our local base checkpoints, or only Unsloth's repos?
4. **Rank on Qwen3.5 is a memory decision.** Unsloth quotes 74 GB for bf16 LoRA but states no config;
   their reference is r=16. r32 + router + norms on 80 GB likely does not fit. If not: drop to the
   highest rank that does, compare Unsloth at that same rank, and let Unsloth spend any memory it
   saves on batch — reported as best-achievable, never as the matched number.
