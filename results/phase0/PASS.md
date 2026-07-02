# Phase 0 — PASS

Stock FLAME-MoE (B=1, s=1) baselines established and validated on a single RTX A6000. **All four
acceptance criteria pass.** This is the temporal-reference baseline for the Temporal-MoE ablation.
Do NOT start temporal phases from here — that needs the within-window design lock (handed back).

All numbers are measured (see `log.md`); metric is **bits-per-byte (BPB)** (tokenizer-invariant;
required because we use a 16k-vocab BPE — see "Adaptations"). FLAME-law CE targets → BPB bars by
÷2.978 (50k bytes/token): **≤1.645 @1e17, ≤2.149 @1e16**.

## Locked hyperparameters

peak-LR **3e-3**, warmup **5%** of iters, cosine→10% (min-lr 3e-4), grad-clip 1.0, weight-decay 0.01,
aux-loss 0.01, z-loss 0.001, global-batch 256, micro-batch 32, bf16, seed 1234. num_experts 64,
top-k 6, 1 shared (intermediate 2·moe_ffn), moe_layer_freq [0]+[1]*(L−1), EP=1, head_dim 16
(heads=hidden/16), 16k-BPE + fused-CE. Tuned at s2 (LR sweep@1e16 → confirm@3e16), re-validated
stable across s2/s4/s6.

## Frontiers (val loss, BPB)

**@1e17 (B=1 IsoFLOP):**  s1 1.284 · **s2 1.269 (min)** · s3 1.289   → parabola, min at s2 (8.1M)
**@1e16 (dedicated annealed, incl. sub-s1 shapes):**  s₋₁ 1.478 · **s0 1.447 (min)** · s1 1.540 ·
s2 1.819 · s3 2.187   → parabola, min at s0 (1.36M); monotone-increasing across s1–s6.

The compute-optimal size shifts s0(1.36M)@1e16 → s2(8.1M)@1e17, matching the law optima (1.48M, 7.74M).

## The (B=1, s=1) temporal-reference baselines (best shape per budget)

| budget | best shape | **BPB** | CE (nats) |
|---|---|---|---|
| 1e17 | s2 (8.12M) | **1.269** | 3.4985 |
| 1e16 | s0 (1.36M) | **1.447** | 3.988 |

## Acceptance — all 4 PASS

1. **Best-shape ≤ bar — PASS.** @1e17 s2 1.269 ≤ 1.645; @1e16 s0 1.447 ≤ 2.149. (Real models beat the
   law's pessimistic 1e16 extrapolation by ~0.8 BPB, so the 1e16 bar clears comfortably.)
2. **Curve shape — PASS.** @1e17 parabolic, min at s2 (within s1–s3); @1e16 parabolic (min at s0) and
   monotone-increasing over s1–s6.
3. **Reproducible — PASS.** s2@1e17 seed-2 CE 3.5075 vs seed-1 3.4985 → |Δ| = **0.0090 nats ≤ 0.03**.
4. **Healthy routing — PASS.** s2@1e17 per-MoE-layer max/mean expert load 1.44–2.07× (worst 2.07× ≪
   8×); balanced load ⇒ aux-loss converged.

## Single-GPU adaptations vs stock FLAME / the plan (documented for fidelity)

- EP=1 (vs FLAME EP=8) + `--moe-grouped-gemm` (batches the 64 local experts; numerically equivalent).
- TransformerEngine impl (FLAME's native path), TE 1.11 from vendored source; `--no-grad-accum-fusion`
  (apex absent; perf-only).
- **head_dim 16 for all shapes** (heads=hidden/16) — fixed 16 heads gave head_dim 12/20/28 for
  s1/s3/s5, breaking TE fused attention (3× slower). Identical params/FLOPs.
- **16k-vocab BPE + fused cross-entropy** (1.69× faster; the 50k logits dominated FLOPs at these tiny
  scales). Metric reported in BPB to stay comparable to the 50k-vocab FLAME law.
- micro-batch capped at 32 by the vocab-logit memory.

## Bonus (beyond Phase 0): s=2 vs s=1 (FLOP-matched, user-requested)

Two constant experts (top-5) vs one shared (top-6), FLOP-matched. Both parabolas reproduce the same
minima (s2@1e17, s0@1e16). **The s-knob is negligible at B=1:** s=2 is ~+0.001 BPB @1e17, ~−0.01 BPB
@1e16 — no meaningful quality change at per-token routing. (Plot: `figures/temporal_vs_dense_and_full_moe_isoflop.png`.) The
constant-vs-routed tradeoff is expected to matter only under windowed routing (B>1) — the next phase.

## Caveats

Budgets are ~2 orders below FLAME's fit range; absolutes are undertrained — the *deltas* (and the
clean parabolas/min-shift) are the signal. The 16k-vocab/fused-CE/head_dim adaptations speed training
without changing N or the FLOP budget, but deviate from the exact stock config; BPB keeps the
comparison to the FLAME law valid.
