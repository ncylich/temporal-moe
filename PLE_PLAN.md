# Per-Layer Embeddings for OLMoE residency adaptation

**Status: plan of record. Nothing launched.**

## 1. The question

Adapting a released MoE checkpoint to hard rolling residency (`R = k = 8` of 64, at most one expert
swap per token per layer) costs quality that no adaptation of the existing weights recovers. The
bake-off closed at 93.2% of the imposed gap, and unfreezing all 6.92B parameters bought only
+0.2 points more. We called the residual the constraint price.

Two facts make a token-indexed lookup the natural next move.

**The constraint de-lexicalizes routing.** The Stage-3 probe found the locus of expert selection
flipping from token identity to context: context-minus-token AUC goes from −0.0041 under the
untrained mask to +0.0493 after adaptation, with context AUC rising 0.603 to 0.673 while token AUC
barely moves. The adapted model routes on context. Whatever token-specific processing the free model
did through expert selection, the constrained model no longer does.

**The constraint-price result does not bound new parameters.** Arm F′ searched the *existing*
parameter space and found no more room. Per-layer embeddings add a token-indexed lookup that does
not route and did not exist in the base model, so F′ says nothing about it. If PLE beats F′'s 0.8106
the claim is that new capacity on a non-routed path breaks a ceiling full finetuning could not.

**Hypothesis.** Restoring lexical expressivity through a token-indexed lookup recovers part of the
residual constraint price, and does so *because* it supplies the token-specific information the
residency constraint removed. Testable via the locus probe and the lexical task breakdown, not
merely assertable.

**Why this fits the paper.** PLE is a flash-resident lookup. Per-token fetch is `r × dtype` bytes in
the factored form (64 B at rank 32, 64 KB unfactored), orders of magnitude below the expert-weight
traffic the residency schedule already pays. Parameters that never enter RAM as a block are the same
thesis as the rest of the paper: serving cost tracks active parameters, not total.

## 2. Architecture

Base model `allenai/OLMoE-1B-7B-0125`: 16 layers, hidden 2048, 64 routed experts, top-8, no shared
expert. Pre-norm, verified during the Cal-0 probe (65 RMSNorm sites = 16 × {`input_layernorm`,
`post_attention_layernorm`, `q_norm`, `k_norm`} + final). Note `post_attention_layernorm` is applied
*before* the MLP; there is no norm after the MoE inside a layer.

```
h   = x + Attn(LN1(x))
out = h + MoE(LN2(h)) + g_ℓ ⊙ PLE[tok, ℓ]        <- PLE enters here, and only here
```

**Placement is settled: after the MoE, added to the layer output.** The correction repairs the
output; it must not perturb the input that produced it. Feeding PLE into the MoE input would change
routing decisions and corrupt the trajectory the norm gains adapted to, making the damage worse
rather than fixing it. There is also no upside available: the O-series showed that at the deployable
`m = 1` swap budget, better routing *information* buys almost nothing (the hindsight-optimal
schedule beat the greedy scan by ~0.023 BPB in replay, and on the adapted model the live scan
actually *beat* a forced offline optimum). Do not implement or test a pre-MoE variant.

**Factored table, layer-shared.** `U[vocab, r]` (the per-token code, flash-resident, token-indexed)
and `V[r, layers, hidden]` (the shared basis, RAM-resident). One code per token receives gradient
from all 16 layers, which concentrates the available signal better than per-layer codes. "Full rank"
means the plain unfactored `[vocab, layers, hidden]` table, not `r = layers × hidden`, because at
that rank the factorization stores both `U` and `V` and is larger than the table it replaces.

| r | total params | flash fetch / token | resident basis |
|---|---|---|---|
| 32 | 2.7M | 64 B | 2 MB |
| 128 | 10.6M | 256 B | 8 MB |
| 512 | 42.5M | 1 KB | 34 MB |
| full (unfactored) | 1.65B | 64 KB | 0 |

**Zero-initialized**, so the model is bit-identical to the C recipe at step 0. Parity comes for free
and never-updated rare-token rows stay exact no-ops rather than noise. `g_ℓ` is a learned per-layer
gate, initialized so the branch starts inert.

**Regularization: rank is the only regularizer. Weight decay on the table is 0 for every rung.**

Wire the table as its own parameter group so the coefficient is settable, then set it to 0 and leave
it there. Do not inherit a value from the C recipe: C trains the router and the RMSNorm gains, and
decaying a norm gain toward zero shrinks activations toward zero, so C has no coefficient that means
anything for a lookup table. An earlier draft of this section said to inherit one; that instruction
was ill-posed and is void.

The reason to hold decay at 0 is that the ladder exists to measure whether constraining rank denoises
underdetermined rare-token rows. Regularize by two mechanisms at once and a null at low rank becomes
unattributable: either rank does not matter, or decay already did that job. At 0 the ladder is
single-axis and measures what it claims, and flag-off parity stays exact by construction.

The cost is real and is the hypothesis, not a bug to pre-empt. With Adam and no decay a row seen once
takes a step nearly the size of a row seen ten thousand times, because after one observation
`v ≈ g²` and the update is ≈ `lr·sign(g)`. Full rank is therefore exposed to fitting noise into rare
rows. That is the stated reason full rank might lose, and losing that way is a result.

**Decay is a contingency with a trigger, not a swept knob.** Report, for each trained cell, mean row
norm bucketed by training-corpus occurrence count. It is a histogram over the trained table and costs
no GPU time. If full rank shows rare-row norms growing to match or exceed frequent-row norms while
eval BPB diverges from train, that is the trigger: stop, report, and we pick a coefficient against
that diagnostic rather than guessing one now. Note also that the targeted fix for the failure above
is on the update rather than the penalty, since it is Adam's normalization that destroys the
frequency signal; do not build that under this plan.

**§9's λ does not supply this coefficient, and cannot.** The two are the same idea — the closed-form
estimator `sum_t/(n_t + λ)` is the ridge solution to `min_p Σ_i ‖Δ_i − p‖² + λ‖p‖²`, so λ-shrinkage
is an L2 penalty, and weight decay is an L2 penalty applied by the optimizer. They are not the same
number and do not convert. λ lives in count space: it is pseudo-observations added to `n_t`. Decay is
a loss coefficient. And because AdamW decouples the decay while Adam normalizes gradients by their own
second moment, a row's equilibrium norm under decay goes roughly as `frequency/wd`, linear in
frequency, not as the `n_t/(n_t + λ)` curve. Do not transfer λ\* into the optimizer; it would
manufacture rigor that is not there. λ stays in §9, which trains nothing.

**Optimizer.** 8-bit Adam for the table; full-rank fp32 moments would be ~20 GB. Embedding gradients
are sparse, so only rows appearing in the batch update.

## 3. Metrics and bars

`BPB = CE_nats / 3.1089` on the Stage-1 audited held-out slice (dolmino dclm). The divisor is
byte-derived. **Never inherit a divisor**: re-derive and record it in the CSV header if the slice
changes. Lower is better.

`recovery = 1 − (BPB − 0.6727) / (2.7507 − 0.6727)`, higher better.

| reference | BPB | recovery | note |
|---|---|---|---|
| base, free routing | 0.6727 | 1.000 | unconstrained ceiling |
| impose R=8, untrained | 2.7507 | 0.000 | the gap being recovered, +2.078 |
| A: router only | 1.2825 | 70.7% | routing-only basin |
| C: router + norms (133K) | 0.8505 | 91.4% | **the surface PLE cells train on** |
| CE: router + norms + LoRA | 0.8149 | 93.2% | bake-off winner (0.8147 merged) |
| **F′: full finetune (6.92B)** | **0.8106** | **93.4%** | **the bar that matters** |

Eval noise σ ≈ 0.006 BPB. **Pre-registered: differences below 2σ = 0.012 BPB are noise.** This
program has retracted single-seed wins before; do not report an effect inside the bar as an effect.

PLE cells train the **C surface (router + norm gains)**, not CE. C and E tie at 91.4%, so LoRA adds
cost without discriminating, and omitting it isolates PLE's contribution rather than confounding it
with adapter capacity.

## 4. Phase 0 — specification, no GPU time

1. Read `config.json` for exact vocab size, expert intermediate width, tied/untied embeddings.
   **Report before building**: the parameter and bandwidth figures in §2 assume 16 × 2048 and a ~50k
   vocab and must be confirmed.
2. Implement the factored PLE behind a flag, post-MoE only, zero-init, with the table as its own
   optimizer parameter group so its weight decay is settable. Set it to 0 (§2) and leave it.
3. **Parity test before any cell trains.** Flag off must reproduce the C recipe within the
   run-to-run non-determinism floor. Measure that floor from two identical flag-off runs and report
   both numbers, as the overlap program did (edited-off vs original-off 1.2e-4 against a 9.7e-4
   floor). Do not proceed on "looks the same".
4. Accounting: parameter count and per-token flash fetch at each rank, the same figure for one
   expert swap, and **coverage** — the fraction of audited-slice tokens that appear in the training
   corpus, and the fraction of eval loss they carry. A headline gain is uninterpretable without it.
5. Verify the zero property: after training, uncovered rows are bit-zero and a forward pass on an
   uncovered token matches the no-PLE model exactly.

## 5. Phase 1 — rank ladder (the main experiment)

Co-train PLE + router + norm gains, 50M tokens per cell, post-MoE placement, evaluating the audited
slice every 10M with telemetry (swap rate, usage entropy). Report each cell as its own result with
CSV rows committed and pushed.

**Always run three cells: full, r=512, r=128.** Then:

- **Skip r=32** if BPB degrades monotonically as rank drops, i.e. `BPB(full) < BPB(512) <
  BPB(128)` with each step worse by more than 2σ. Rank is binding and 32 can only be worse.
- **Run r=32** if 128 is at least as good as the best of full and 512 (within 2σ). That covers both
  the "ties all the way down" case, where low rank is free bandwidth, and the "interior optimum,
  still descending" case.
- Do not extend below 32 under this plan.

Three points rather than stopping at the first degradation, so the result is a reportable curve
rather than a two-point claim, and so one unlucky run is not read as a trend.

**Run full and r=512 first, regardless of what the ladder does next.** Full rank is the
configuration most handicapped by the statistics (32,768 parameters per token row, underdetermined
past the top few hundred tokens), so a null at full rank must **not** be read as "PLE fails". Treat
"does PLE recover anything" as answered by the better of those two cells.

**Pre-registered: the curve may be non-monotonic with an interior optimum.** Truncating or
constraining rank removes noise-dominated directions from underdetermined rare-token rows, so a
middle rank beating full rank is a live possibility, and would be the best of the available outcomes:
better quality *and* a smaller, lower-bandwidth artifact.

| ladder outcome | reading |
|---|---|
| monotone degradation as rank drops | rank binds; ship the highest rank that clears the bar |
| a middle rank beats full by >2σ | interior optimum from denoising; probe its neighbours |
| all rungs tie | rank is not the binding constraint; ship the smallest and say so |
| nothing beats C (0.8505) by >2σ | PLE does not recover the residual; report the negative |

## 6. Phase 2 — depth

Best configuration from Phase 1 extended to **100M** tokens, with checkpoints and evals at 50M and
100M so it compares directly against the Phase-1 cells.

## 7. Phase 3 — sequential versus joint

Router + norms alone for 50M, then add PLE for a further 50M (100M total). Tests whether introducing
new parameters mid-training differs from training them jointly. Five path variations in this program
have come back null, but every one of them varied schedule or initialization within a fixed
parameter set. This changes the parameter set, so it is not strictly dominated. Run it last.

## 8. Phase 4 — verification

Quality numbers alone do not establish the hypothesis. All four are required on the best model:

1. **Locus probe** (`analysis/probes/delex_locus_driver.py` protocol). Prediction: token AUC rises
   and context-minus-token moves back toward zero relative to the CE-adapted +0.0493. **If BPB
   improves and the locus does not move, the gain is generic capacity rather than lexical
   restoration, and the paper claim must be weakened accordingly.**
2. **Downstream, 10 tasks, 0-shot**, same harness and primary-metric convention as
   `olmoe_adapt_downstream.csv` (acc_norm for hellaswag and openbookqa, acc elsewhere). **lambada is
   the sharp test**: the mask drove it 0.706 → 0.000 and adaptation only reached 0.570. If PLE
   restores lexical capacity, lambada should move most.
3. **Recovery bucketed by token frequency.** If gains concentrate on frequent tokens, say so; it is
   an honest scaling caveat and an argument for longer runs, not a flaw to hide.
4. **Residency telemetry** at every eval: swap rate and usage entropy. PLE must not buy BPB by
   destabilizing the schedule.

## 9. Side measurement — the training-free table (gates nothing)

Cheap, independently interesting, and **explicitly not a gate**. Capture `Δ = MoE_free(x) −
MoE_C-adapted(x)` per token per layer, against the **C-adapted** model, since the object of interest
is the residual PLE would have to fix rather than the raw impose damage.

**The fit is an average, with count shrinkage:**

```
p[t, ℓ] = sum_t(Δ) / (n_t + λ)        =  mean_t · n_t/(n_t + λ)
```

Estimate `λ* ≈ σ²_within / σ²_between` from the same capture by accumulating per-token sums of
squares. **Do not fix λ by hand and do not grid it**: it is one estimated value, and it costs nothing
to change afterward because building the table at any λ is arithmetic on the stored sums and counts,
with no re-capture and no training. If the estimate is worth verifying, explore **L-shaped, not as a
grid**: sweep rank at λ\*, then sweep λ at the best rank only, which is `#ranks + #λ` evals rather
than their product. Uncovered tokens have `n_t = 0` and land at exactly zero by construction.

**For the low-rank rungs use precision-weighted SVD**, closed form `L = D⁻¹·SVD_r(D·T)` with
`D = diag(√w_t)`, `w_t = n_t/(n_t+λ)`. This reduces to `L = T·P_Vr`: the weights choose the basis,
each row is projected onto it. So the basis is set by frequent tokens and rare tokens borrow strength
instead of contributing noise, removing every noise component orthogonal to the basis (~98% of the
energy at r=512). What survives is in-basis coefficient noise, roughly `r/d` of the full-rank
problem, which is why **shrinkage is load-bearing at full rank and close to cosmetic at low rank**.
Keep λ in the sweep at every rank anyway and expect `λ*` near zero at the low rungs.

If both are applied, **shrink rows first, then weighted-SVD**: unshrunk noisy rows do not merely add
noise, they distort the basis, which is computed from all rows including the unreliable ones.

**Before averaging anything, check position in sequence.** Residency is cold-filled at the start of
each 4096-token sequence, so early tokens suffer atypically little damage. Report Δ magnitude
bucketed by position and decide whether to exclude the cold-fill region. This is the same class of
error as the windowing inflation the O-series had to retract.

**Optional refinement:** weight the output metric by the RMS gradient of loss with respect to each
activation (a diagonal Fisher approximation, one more accumulator in the same pass). Plain L2 treats
every hidden dimension as equally important when the network is far more sensitive to some; this is
the honest fix for the fact that the fit minimizes activation MSE while we care about loss.

**Two things this measurement cannot do.** It is a **lower bound on PLE's value**, because the
C-adapted model has already spent its plasticity: norms found the best correction available to them
and this measures only the residual conditional on that allocation, whereas under joint training the
allocation is free to differ. So **a null here cannot kill the trained cells.** And it cannot
determine the rank for Phase 1, for the same reason. What it does give: the training-free recovery
number, and a spectrum that can be compared against the trained tables' spectra to read off how
co-adaptation redistributes the work.

## 10. Standing rules

- **Divisors byte-derived per corpus, never inherited.** 3.1089 for the audited slice; record the
  derivation in every CSV header.
- **Parity before cells**, against a measured non-determinism floor.
- **Screen at the deployment configuration.** Depth, surface, and geometry must match, or say
  explicitly that they do not and treat the screen as untransferable.
- **2σ or it is noise.** 0.012 BPB on this slice.
- **Flash attention on, in every training cell, recorded in every CSV.** Its backward is
  non-deterministic and injects roughly 1e-3 relative gradient noise per step, which is understood
  and is not a blocker: it is far below seed-to-seed variance, and seed consistency has been good
  across this program. Do not trade throughput for determinism. If a rank comparison lands inside the
  2σ bar, that is when to add a seed replicate at the rungs in question. Keep flash off only for
  bitwise correctness checks, which are not training cells.
- **Report each deliverable separately**, CSV committed *and pushed*, and verify the file reached the
  branch rather than assuming the commit succeeded.
- **Failures go in error messages**, not only status lines. Never quote an ETA without re-reading the
  log's terminal line.
- **No promotion, no long run, and nothing beyond this plan without orchestrator sign-off.** Report
  and wait.

## 11. Risks

| risk | handling |
|---|---|
| gains are generic capacity, not lexical restoration | locus probe and lexical breakdown decide; weaken the claim, not the caveats |
| full-rank cell nulls and is read as "PLE fails" | run r=512 alongside; the better of the two answers the primary question |
| rare-token rows never train | expected; report the frequency breakdown rather than hiding it behind an aggregate |
| optimizer state exhausts memory at full rank | 8-bit Adam, then smaller micro-batch, then drop the full-rank rung and say so |
| nothing beats F′ 0.8106 | a clean negative: the constraint price survives new non-routed capacity, a stronger form of the existing claim |

## 12. Cost

| phase | content | GPU | cost |
|---|---|---|---|
| 0 | spec, parity, accounting | none | $0 |
| 1 | rank ladder: full, 512, 128 always; 32 conditionally | 3–4 × 45 min | $6–8 |
| 2 | depth to 100M | 85 min | $4 |
| 3 | sequential versus joint | 85 min | $4 |
| 4 | verification battery | ~1 h | $3 |
| side | training-free table, λ and rank sweeps | 30 min | $1.5 |

**Total ≈ 6–7.5 h GPU, ~$19–21**, depending on where the ladder terminates. Throughput assumption
~19.7k tok/s, bound by the per-expert dispatch loop in the HF implementation rather than by the
residency scan.

## 13. Not in scope

DeepSeek Engram and richer lookup schemes are the natural follow-on if PLE works, and the
related-work section already frames the lookup axis (MoLE, Gemma PLE, Engram) as complementary to
residency. PLE first: token-indexed lookup is the minimal test of the whole axis, so a null here
likely kills the axis and a win gives Engram a proven hypothesis to improve on. Read the Engram paper
while PLE runs; do not build it under this plan.

Also out: the pre-MoE placement (§2), and any activation-conditioned linear correction. The latter is
functionally what LoRA on the expert projections already learns, and that surface is measured and
saturated (E = 91.4% at 235M parameters, tying norm gains at 133K).
