# OLMoE Residency-Adaptation — Stage 2 Results (LR sweep + Stage-2b bake-off)

**Model:** `allenai/OLMoE-1B-7B-0125` (7B total / 1.3B active, 16 layers, 64 experts, top-8, no shared expert).
**Question:** how much of the quality lost by imposing FLAME's hard rolling-residency constraint
(**R = k = 8 of 64**, ≤1 expert swap/token, per-4096-sequence cold-fill) can be recovered by adapting the model?

## Metric & references (self-contained)

- **BPB** = bits-per-byte on the Stage-1 **audited held-out slice** (dolmino-mix-1124 `dclm/024*`, 256×4096
  subsample, disjoint from OLMoE pretraining). **Lower is better.** Divisor is byte-derived for THIS
  tokenizer+slice: **D = ln2·bytes/tok = 3.1089** (house rule: never inherit a divisor).
- **base** (residency OFF, free top-8) = **0.6727 BPB**.
- **impose** (R=8 mask, NO training) = **2.7507 BPB** → the residency **gap = +2.078 BPB**.
- **Recovery** = 1 − (adapted − base)/(impose − base). Range (−∞, 1]; **higher is better**; 1.0 = fully
  closes the gap back to free-routing base, 0.0 = no better than the untrained mask.
- **Eval-noise σ** (base re-eval on 3 disjoint 256-pack subsamples) = **0.006 BPB at R=8** → inter-arm BPB
  gaps below ~0.012 (2σ) are noise.

## Stage 2 — router-only LR sweep (pick the learning rate)

Router-only finetune (only the 16 `OlmoeTopKRouter.weight` linears, ~2.10M params), R=8 from step 0,
0.25B tokens/arm, same seed-0 data order. bf16 compute + fp32 master, AdamW(0.9,0.95), grad-clip 1.0,
MB=16 packs of 4096, gradient checkpointing. Loss = LM CE (through the resident-masked softmax) + 0.01·aux
+ 0.001·z-loss on the masked distribution.

| LR | final BPB | recovery |
|------|-----------|----------|
| 3e-5 | 1.3618 | 66.8% (undertrained) |
| 1e-4 | 1.2897 | 70.3% |
| **3e-4** | **1.2825** | **70.7% (winner, converged/flat)** |

Throughput 19.7k tok/s, **compute-bound on OLMoE's Python expert-dispatch loop** (a `for expert_idx in
expert_hit:` over the hit experts), NOT the residency scan (triton accel, verified == torch reference).
Winner **LR 3e-4** used for the whole bake-off.

## Stage 2b — escalation bake-off (what to adapt)

All arms: base-router init, R=8 from step 0, LR 3e-4, same corpus + seed-0 data order, 0.25B tokens
(G/F′ noted), evals at R=8 with telemetry (swap-rate/layer, expert-usage entropy). Telemetry stayed clean
on every arm (swap ≈ 1.0/token at the ceiling, usage-entropy ≈ 0.99 — no collapse, no swap-gaming).

| Arm | Trainable | final BPB | recovery | Role |
|-----|-----------|-----------|----------|------|
| A | router (2.1M) | 1.2825 | 70.7% | routing-only basin floor |
| B | router, anneal R 64→8 | 1.2788 | 70.8% | = A (routing policy from R>24 doesn't transfer) |
| D | router + self-distill | 1.2919 | 70.2% | = A (distillation signal doesn't help router) |
| C | router + RMSNorm gains (133K) | 0.8505 | 91.4% | **calibration** |
| E | router + LoRA r32 all-expert (235M) | 0.8507 | 91.4% | calibration (== C at 1766× the params) |
| G | router+norms+LoRA + distill (150M) | 0.8306 | 92.4% | < CE (distillation *hurts* the rich surface) |
| **CE** | **router + norms + LoRA (237M)** | **0.8149** | **93.2%** | **WINNER (norms & LoRA stack)** |
| F′ | full-FT, all 6.92B params | 0.8106 | 93.4% | = CE within noise (constraint price, not capacity floor) |

Notes: **B** (anneal) wastes its ramp — at R=8 eval it sits at ~impose until the training R actually reaches
8 (~150M), then converges to A. **G** was capped at 0.15B (250M projected ~13h at MB=4 2-forward distill;
the level, not the tail, is what matters — orch fallback). **F′** warm-started from CE via an **exact
LoRA→expert merge** (W′ = W + (α/r)·B·A per expert); identity check reproduced the parent BPB to |Δ|=0.0002
before any training.

## The result — four-part mechanism story

1. **Routing is a shallow lever.** Adapting only the routers plateaus at ~70.7%; neither a constraint
   curriculum (**B**) nor a distillation signal (**D**) breaks it → the router-only basin is
   **capacity-limited, not signal-limited**.
2. **The 70→91% jump is calibration.** Rescaling the 16 RMSNorm gains (**C**, 133K params) recovers as much
   as a 235M-param all-expert LoRA (**E**) — **C == E == 91.4%**. The residency mask shifts the
   activation-scale statistics the router/experts see; re-tuning the norms fixes most of it, cheaply.
3. **Calibration and expert-capacity stack.** Doing both (**CE**) reaches **93.2%**, ~1.8 pts (~6σ) above
   either alone → LoRA's expert adaptation adds a real, separate increment on top of norm calibration.
4. **The remaining ~6.6% is the constraint price.** Full 7B-param finetuning (**F′**) recovers only +0.2 pt
   over CE (below the 2σ bar) → **~93.4% is the irreducible cost** of serving top-8 from a rolling R=8
   resident set. Self-distillation from the base free-routing teacher never helps on any surface: the
   bottleneck is trainable **surface**, not training **signal**.

## Recommendation for Phase C (orchestrator's call — no 1B/5B started without selection)

- **Deployable winner: CE** (router + norm gains + LoRA r32), **93.2%**.
- **Near-free alternative: C** (router + norm gains, **133K** params, **91.4%**) if adapter size/simplicity
  matters — it captures the entire calibration effect at 1/1766 the params of LoRA.
- The hard 1B gate (recovery < 25% AND flat → stop) is cleared by a wide margin.
- Optional-idle rank screens (LoRA r=8 / r=64, adapter-minimality) run while awaiting selection.

## Artifacts & reproducibility

- `results/ablations/olmoe_adapt_bakeoff.csv` — full table, all per-arm curves, σ header.
- `results/ablations/olmoe_adapt_sweep.csv` — LR sweep.
- `results/ablations/olmoe_adapt_impose.csv` — base/impose derivation.
- `results/ablations/olmoe_adapt_corpus_audit.md` — corpus provenance, divisor derivation, throughput root cause.
- `results/ablations/adapt_ckpts/` — router-only deltas (A/B/C/D, 4MB) tracked; LoRA-bearing deltas
  (E/CE/G, 474MB) and the CE full resumable checkpoint (2.85GB) kept pod-local/gitignored (available on request).
- All work isolated in a separate `olmoe-adapt` sub-repo, checked out alongside this one
  (venv/model/data gitignored).

## Optional-idle ablations (ran on otherwise-idle GPU before Phase-C selection)

**LoRA rank sweep** (E-recipe, 50M screens) — rank saturates at r=32:

| rank | trainable | 50M BPB | 50M recovery |
|------|-----------|---------|--------------|
| r=8  | 58.7M  | 0.8956 | 89.3% (undershoots, ~5σ worse than r=32) |
| r=32 | 235M   | 0.8642 | 90.8% (E reference) |
| r=64 | 470M   | 0.8592 | 91.0% (+0.2pt over r=32, within noise) |

r=8 is too small (and even below the 133K-param norm-gain arm C), r=64 buys nothing over r=32 → **r=32 is
the sweet spot**; calibration (norms) remains the parameter-efficient lever.

**Arm H — zone-confined anneal on the E-recipe** (R starts at 24, anneals one expert/rung over the first
50M, holds R=8 for 200M): final **0.8510 / 91.4% == arm E (0.8507) within noise** (Δ=0.0003), nowhere near
the bar. The anneal wastes its ramp exactly as arm B did for routing. **The anneal/curriculum axis is dead
for both routing (B) and expert-capacity (H)** — *what* you adapt (the surface) matters; the *constraint
schedule* does not.

## Bottom line

Adapting OLMoE to the FLAME R=8 rolling-residency constraint is a **calibration** problem, not a routing or
curriculum problem. **Recommended recipe: CE (router + RMSNorm gains + LoRA r32) → 93.2% recovery**, or the
near-free **C (router + norm gains, 133K params) → 91.4%**. ~93.4% is the irreducible constraint price
(full 7B finetuning can't beat it). Neither constraint annealing nor self-distillation helps.
