# Per-Layer Embeddings for OLMoE residency adaptation

**Status: plan of record. Nothing launched.**

## 1. The question

Adapting a released MoE checkpoint to hard rolling residency (`R = k = 8` of 64, at most one expert
swap per token per layer) costs quality that no adaptation of the existing weights recovers. The
bake-off closed at 93.2% of the imposed gap, and unfreezing all 6.92B parameters bought only
+0.2 points more. We called the residual the constraint price.

Two facts make a lookup path the natural next move.

**The constraint de-lexicalizes routing.** The Stage-3 probe found the locus of expert selection
flipping from token identity to context: context-minus-token AUC goes from −0.0041 under the
untrained mask to +0.0493 after adaptation, with context AUC rising 0.603 to 0.673 while token AUC
barely moves. The adapted model routes on context. Whatever token-specific processing the free
model did through expert selection, the constrained model no longer does.

**The constraint-price result does not bound new parameters.** Arm F′ searched the *existing*
parameter space and found no more room. Per-layer embeddings add a token-indexed lookup that does
not route and did not exist in the base model, so F′ says nothing about it. If PLE beats F′'s
0.8106 the claim is that new capacity on a non-routed path breaks a ceiling full finetuning could
not.

**Hypothesis.** Restoring lexical expressivity through a token-indexed lookup recovers part of the
residual constraint price, and does so *because* it supplies the token-specific information the
residency constraint removed. This is testable rather than merely asserted, via the locus probe and the lexical task breakdown.

**Why this fits the paper.** PLE is a flash-resident lookup: per-token fetch is
`n_layers × hidden × dtype` ≈ 64 KB in bf16 (16 KB at 4-bit), two to three orders of magnitude
below the expert-weight traffic the residency schedule already pays. Adding roughly 1.6B parameters
that never enter RAM as a block is the same thesis as the rest of the paper: serving cost tracks
active parameters, not total.

## 2. Architecture

Base model `allenai/OLMoE-1B-7B-0125`: 16 layers, hidden 2048, 64 routed experts, top-8, no shared
expert. Pre-norm, verified during the Cal-0 probe (65 RMSNorm sites = 16 × {`input_layernorm`,
`post_attention_layernorm`, `q_norm`, `k_norm`} + final). Note that `post_attention_layernorm` is
applied *before* the MLP; there is no norm after the MoE inside a layer.

```
h   = x + Attn(LN1(x))
out = h + MoE(LN2(h))
```

**PLE table.** `[vocab, n_layers, hidden]`, full rank, **no bottleneck and no SVD**. A lookup
table is the artifact the memory argument is about, and shrinking it uniformly would destroy the
lexical resolution the experiment exists to restore. bf16 storage, per-row quantization later if
wanted.

**Zero-initialized**, so the model is bit-identical to the no-PLE recipe at step 0. This gives
parity for free and makes never-updated rare-token rows exact no-ops rather than noise.

**Placement.** Two candidates, decided empirically in #1:

| tag | form | effect |
|---|---|---|
| **P-post** (default) | `out = h + MoE(LN2(h)) + g_ℓ ⊙ PLE[tok, ℓ]` | pure bypass. Token information reaches the residual without passing through constrained routing, and the routing distribution is untouched |
| **P-pre** | `h' = h + g_ℓ ⊙ PLE[tok, ℓ]`, then `out = h' + MoE(LN2(h'))` | PLE feeds the router and the residual; more expressive, but re-lexicalizing the router may raise swap pressure and erode the stickiness residency depends on |

`g_ℓ` is a learned per-layer gate, initialized so the branch starts inert. When running P-pre,
watch swap rate and usage entropy: a BPB win bought by destabilizing the residency schedule is not
a win.

**Optimizer.** 8-bit Adam for the PLE table (full-rank fp32 moments would be ~20 GB). Embedding
gradients are sparse, so only rows appearing in the batch update.

## 3. Metrics and bars

`BPB = CE_nats / 3.1089` on the Stage-1 audited held-out slice (dolmino dclm, byte-derived
divisor. **Never inherit a divisor**: re-derive and record it in the CSV header if the slice
changes). Lower is better.

`recovery = 1 − (BPB − 0.6727) / (2.7507 − 0.6727)`, higher better.

| reference | BPB | recovery | note |
|---|---|---|---|
| base, free routing | 0.6727 | 1.000 | unconstrained ceiling |
| impose R=8, untrained | 2.7507 | 0.000 | the gap being recovered, +2.078 |
| A: router only | 1.2825 | 70.7% | routing-only basin |
| C: router + norms (133K) | 0.8505 | 91.4% | **the matched surface for PLE cells** |
| CE: router + norms + LoRA | 0.8149 | 93.2% | bake-off winner (0.8147 once merged) |
| **F′: full finetune (6.92B)** | **0.8106** | **93.4%** | **the bar that matters** |

Eval noise σ ≈ 0.006 BPB on this slice. **Pre-registered: differences below 2σ = 0.012 BPB are
noise.** This program has retracted single-seed wins before; do not report an effect inside the
bar as an effect.

PLE cells train the **C surface (router + norm gains)**, not CE. C and E tie at 91.4%, so LoRA adds
cost without discriminating, and omitting it isolates PLE's contribution instead of confounding it
with adapter capacity.

## 4. Phase 0 — specification, no GPU time

1. Read `config.json` for the exact vocab size, expert intermediate width, and tied/untied
   embeddings. **Report these before building**; the 64 KB/token and ~1.6B-parameter figures above
   are derived from 16 × 2048 and a ~50k vocab and must be confirmed.
2. Implement the PLE module behind a flag, both placements, zero-init.
3. **Parity test before any cell trains**: flag off must reproduce the C recipe's loss curve within
   the run-to-run non-determinism floor. Measure that floor from two identical flag-off runs and
   report both numbers, as the overlap program did (edited-off vs original-off 1.2e-4 against a
   9.7e-4 floor). Do not proceed on "looks the same".
4. Accounting: PLE parameter count, bytes/token fetched at bf16 and 4-bit, and the same figure for
   one expert swap, so the memory claim is measured rather than asserted.

## 5. Experiments

Each cell trains 50M tokens unless stated, on the existing 1B-token adaptation corpus, evaluating
on the audited slice every 10M with telemetry (swap rate, usage entropy, and for P-pre the swap
rate especially). Report each as its own result with its CSV rows committed.

**#1 — placement decision.** PLE + router + norms, 50M, at **P-post** and **P-pre**.
Compare to C (0.8505). Winner carries forward. If they are within 2σ, take P-post, which leaves routing undisturbed and
is the cleaner claim.

**#2 — calibration gate.** Winner of #1 + calibration, 50M, same surface.
Promote calibration into #3/#4 only if it beats #1's winner by >2σ. Expect a null (§6).

> The gate runs at the *deployment surface* (PLE + router + norms), not on a PLE-only cell. The
> overlap program promoted a variant off an L=6 screen into an L=9 cell and the prediction missed
> by 12×. Screen where you deploy.

**#3 — depth.** Best configuration from #1/#2 extended to **100M**, with checkpoints and evals at
50M and 100M so it compares directly against #2 and #4.

**#4 — sequential vs joint.** Router + norms alone for 50M, then add PLE for a further 50M
(100M total). Tests whether introducing new parameters mid-training differs from training them
jointly. Five path variations in this program have come back null, but all of them varied schedule
or initialization. This one changes the *parameter set*, so it is not strictly dominated. Run it
last.

**#5 (optional) — attribution.** PLE only, router and norms frozen at base, 50M. Isolates PLE's
standalone contribution against the 2.7507 impose point. Cheap and clean, but not on the critical
path and not a gate.

## 6. Calibration, specified

The idea: record the per-token per-layer residual between free-MoE and temporal-MoE outputs,
`Δ = MoE(x) − TMoE(x)`, and learn a cheap correction.

**Use reduced-rank regression, not PCA on Δ.** PCA gives the best low-rank reconstruction of
deltas you already know; at inference the delta is exactly what you do not have. The estimator must
*predict* Δ from an available input: minimize `‖Δ − B A x‖`, whose closed form is an SVD of the
whitened cross-covariance `Σ_Δx Σ_xx^{-1/2}`. Same one-pass cost, correct object.

**Fit sequentially on the corrected trajectory.** Recording Δ along the free trajectory and
applying it to the constrained one rebuilds the failure this program measured twice: the O-series
free-versus-drift gap was 0.40 BPB, and the early-router variant failed at depth because staleness
compounds per layer. Fit layer 1, apply it, then record layer 2's deltas under the corrected
activations, and so on.

**Prior.** Cal-0 (closed-form moment matching) recovered 31.5% against 91.4% learned and was
directionally orthogonal to the learned solution (cosine ≈ 0.01). Cal-2 showed a calibrated
initialization is a liability: training undoes it at a 0.047 BPB detour cost. Reduced-rank
regression is strictly more expressive than Cal-0's diagonal scaling, so it earns one test, but
it does not gate PLE, and a null here is the expected outcome rather than a surprise.

## 7. Verification on the best model

Quality numbers alone do not establish the hypothesis. Run all four:

1. **Locus probe** (`analysis/probes/delex_locus_driver.py` protocol). Prediction: token AUC rises
   and context-minus-token moves back toward zero relative to the CE-adapted baseline's +0.0493. If
   BPB improves and the locus does not move, the gain is generic capacity, not lexical restoration,
   and the paper claim must be weakened accordingly.
2. **Downstream, 10 tasks, 0-shot**, same harness and primary-metric convention as
   `olmoe_adapt_downstream.csv` (acc_norm for hellaswag and openbookqa, acc elsewhere).
   **lambada is the sharp test**: the mask drove it 0.706 to 0.000 and adaptation only reached
   0.570. If PLE restores lexical capacity, lambada should move most.
3. **Recovery bucketed by token frequency.** With a ~50k vocab and 50–100M tokens, Zipf means tail
   rows get almost no gradient. If gains concentrate on frequent tokens, say so. It is an honest
   scaling caveat and an argument for longer runs, not a flaw to hide.
4. **Residency telemetry**, every eval: swap rate and usage entropy. PLE must not buy BPB by
   destabilizing the schedule, particularly under P-pre.

## 8. Standing rules

- **Divisors are byte-derived per corpus and never inherited.** 3.1089 for the audited slice.
  Record the divisor and its derivation in every CSV header.
- **Parity before cells.** Flag-off must match baseline within a measured non-determinism floor.
- **Screen at the deployment configuration.** Depth, surface, and geometry must match, or state
  explicitly that they do not and treat the screen as untransferable.
- **2σ or it is noise.** 0.012 BPB on this slice.
- **Report each deliverable separately**, with the CSV committed and pushed before the claim is
  made, and verify the file is on the branch rather than assuming the commit succeeded.
- **Failures go in error messages**, not only in status lines. Do not quote an ETA without
  re-reading the log's terminal line first.
- **No promotion, no long run, and no new experiment beyond this plan without orchestrator
  sign-off.** Report and wait.

## 9. Risks and kill criteria

| risk | handling |
|---|---|
| PLE gains are generic capacity, not constraint recovery | the locus probe and lexical breakdown decide; weaken the claim rather than the caveats |
| P-pre wins on BPB but raises swap rate | not a win; report both and prefer P-post |
| Rare-token rows never train | expected; report the frequency breakdown, do not bottleneck to hide it |
| PLE optimizer state exhausts memory | 8-bit Adam, then reduce micro-batch, then reduce PLE dim only as a last resort with the change recorded |
| Calibration looks strong offline, does nothing live | the sequential-fit requirement exists for this; report offline and live numbers separately |
| No cell beats F′ 0.8106 | a clean negative: the constraint price survives new non-routed capacity, a stronger version of the existing claim |

## 10. Cost

Roughly 350M training tokens total. At the ~19.7k tok/s observed on this model (bound by the
per-expert dispatch loop in the HF implementation, not by the residency scan), that is about 5 GPU
hours plus eval, on the order of $25.

## 11. Not in scope

DeepSeek Engram and other richer lookup schemes are the natural follow-on if PLE works, and the
related-work section already frames the lookup axis (MoLE, Gemma PLE, Engram) as complementary to
residency. PLE first: token-indexed lookup is the minimal test of the whole axis, so a null here
likely kills the axis, and a win gives Engram a proven hypothesis to improve on. Read the Engram
paper while PLE runs; do not build it under this plan.
