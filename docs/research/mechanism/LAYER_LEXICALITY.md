# Layer-wise Lexicality of MoE Routing

**Status: two hypotheses, partial baseline measurements, and the tests to settle them.** Sections
1–2 record what is already measured. Section 3 states what this work exists to prove or disprove.
Sections 4–5 are the tests, split by whether they need training runs. No result here is final.

Battery-wide housekeeping — re-running every mechinterp script across every model and layer — is
tracked separately in [`MECHINTERP_RERUN_PLAN.md`](MECHINTERP_RERUN_PLAN.md). Steps 1–3 of that
plan are prerequisites for several tests below.

## 0. Why

[`delexicalization.md`](delexicalization.md) establishes that unconstrained MoE routers bind
experts to token identity and that rolling residency breaks that binding, and it reports one number
per model — a median pooled over layers. It never asks whether the effect varies with depth.

If the locus of specialization moves toward context as depth increases, then the residency
constraint — which forces routing onto exactly the feature that context provides — should be
cheapest to impose deep in the network and most expensive near the embedding. That would make the
constraint a per-layer decision rather than a global one, and the payoff is a serving model that
spends its fast memory where routing freedom actually matters.

## 1. Locus probes by depth

**Metric.** For each (layer, expert) pair, label `y_e(t) = 1` if expert *e* is among the experts
serving token *t*. Fit two ridge-linear probes on the model's own **input embeddings** (the
`LanguageModelEmbedding` output — a fixed vector per token id, since all configs use RoPE and have
no additive positional term):

- **token probe** — `x_tok(t) = E[x_t]`, the current token alone;
- **context probe (CBOW-like)** — `x_ctx(t) = mean{E[x_t'] : 0 < |t'-t| <= w}`, the surrounding
  window **with the current token excluded**. The exclusion is the control that stops the context
  feature from re-encoding the token itself. Window `w = k`, one residency lifetime.

Fit on the first 70% of a fixed 131k-token evaluation batch, score **held-out AUC** on the last
30%. Chance floor, measured by refitting on permuted and circular-shifted labels, is
0.500 +/- 0.002 everywhere ([`mechinterp_floors.csv`](../../../results/ablations/mechinterp_floors.csv)).

We plot **A_ctx - A_tok**, the median over that layer's experts. Above 0 = routing at that layer is
better predicted by surrounding context than by the token being processed.

![Locus of routing specialization by depth](../../../results/phase0/figures/locus_by_layer.png)

Regenerate with `python3 analysis/plots/plot_locus_by_layer.py [--no-caption]`.

| model | budget | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|
| temporal 18/192 (fine) | 1e16 | +0.126 | +0.140 | +0.177 | — | — |
| temporal 18/192 (fine) | 1e19 | +0.057 | +0.066 | +0.074 | +0.080 | +0.086 |
| temporal 6/64 (coarse) | 1e17 | +0.086 | +0.092 | +0.082 | +0.084 | +0.074 |
| temporal 6/64 (coarse) | 1e19 | +0.032 | +0.039 | +0.060 | +0.075 | +0.093 |
| full MoE 18/192, sigmoid ctrl | 1e16 | −0.324 | −0.304 | −0.280 | — | — |
| full MoE 18/192, softmax-aux (w=32 only) | 1e16 | −0.303 | −0.286 | −0.279 | — | — |
| full MoE 6/64 (coarse) | 1e17 | −0.244 | −0.268 | −0.213 | −0.269 | −0.259 |
| full MoE 6/64 (coarse) | 1e19 | −0.300 | −0.241 | −0.226 | −0.212 | −0.211 |

Per-expert counts: 64/layer at 64E, 192/layer at 192E (165–173 for the 1e16 temporal model, where
some experts fall below the probe's minimum-usage threshold).

**Depth slope, with uncertainty.** OLS slope of the per-layer median against layer index; 95% CI
from 2000 bootstrap resamples of experts within each layer.

| matched pair | temporal slope / layer | unconstrained baseline slope / layer |
|---|---|---|
| fine 18/192 @1e16 | +0.0257 [+0.0133, +0.0336] | +0.0223 [+0.0151, +0.0293] |
| coarse 6/64 @1e17 | −0.0032 [−0.0092, +0.0048] | −0.0030 [−0.0129, +0.0046] |
| coarse 6/64 @1e19 | +0.0157 [+0.0102, +0.0200] | +0.0207 [+0.0171, +0.0248] |

**Readings.**

1. **The regimes never overlap at any depth.** Every temporal series is above 0 at every layer,
   every unconstrained series is below −0.20. The regime gap (~0.3 AUC) dwarfs every depth effect
   (~0.05).
2. **The depth trend is positive in 3 of 4 arms, and appears in both regimes at indistinguishable
   rates.** On current evidence it is a property of network depth — more attention mixing before
   deeper routers see the stream — not of the constraint. This is consistent with the mechanism
   behind H1, but it means any regime-specific claim needs matched baseline arms to be testable.
3. **The coarse 1e17 pair is flat in both regimes**, CI straddling zero. A real exception.
4. **Granularity sets the level more than compute does.**

**Caveats.**

- **Layers 7–9 are missing for every 9-layer model** — `LAYERS = [2,3,4,5,6]` hardcoded at
  [`delex_locus.py:18`](../../../analysis/probes/delex_locus.py), silently skipping the rest. The
  captures contain them.
- **The x-axis mixes models of different total depth.** The fine@1e16 model's L4 is its last layer;
  the 1e19 model's L4 is 3/8 of the way through. Depth claims must be stated against `l/L`.
- **The softmax-aux fine baseline was only ever measured at w=32**, not w=k, and §3 of
  [`delexicalization.md`](delexicalization.md) mislabels that row as w=18. Its sigmoid sibling has
  w=k and sits ~0.02 lower, bracketing the missing measurement.
- Layer 1 is a dense FFN in every config, so there is no layer-1 router to probe.
- The probe is **linear on the input embedding**; `A_tok = 1` is not the ceiling, since routing is
  a function of the post-attention hidden state.

## 2. Cache hit rate by depth

**Metric.** Of the k experts a token's router actually wanted (its *unconstrained* top-k), the
fraction already resident when the token arrived, measured **pre-swap**. Range 0–1, higher =
resident set matches demand better. A random resident set scores **k/E = 6/64 = 18/192 = 0.094**,
identical for both granularities, so the two are directly comparable.

Source: [`e6_per_layer_ranking.csv`](../../../results/ablations/e6_per_layer_ranking.csv).

| MoE layer | 3.9M active, fine 18/192 @1e17 | 8.1M active, coarse 6/64 @1e17 | 38M active, coarse 6/64 @1e18 |
|---|---|---|---|
| 2 | 0.249 | 0.292 | 0.165 |
| 3 | 0.332 | 0.319 | 0.259 |
| 4 | 0.357 | 0.382 | 0.305 |
| 5 | 0.453 | 0.438 | 0.331 |
| 6 | — | 0.479 | 0.362 |
| 7 | — | — | 0.334 |
| 8 | — | — | 0.339 |
| 9 | — | — | 0.338 |

**Depth coverage here is complete** for all three models, which makes its shape important: hit rate
climbs steeply from the first MoE layer through the middle (+82% relative for the fine model, +64%
for the 8.1M coarse) and then **plateaus at 0.33–0.34 through L9 rather than continuing to rise.**
If the locus curve does the same, H1's shape is *increases-then-saturates*, not monotone increase —
a materially different prediction for where a per-layer schedule puts its cutoff.

All three models are temporal; there is no unconstrained baseline hit rate at all (test C4).
Swap rate is unusable for depth work: it is 0.994–1.000 everywhere because at R = k a swap fires
iff at least one demanded expert is missing, so it saturates as "fraction of tokens with >= 1 miss".

## 3. The two hypotheses

### H1 — routing specialization shifts from lexical to contextual as depth increases

**Prediction.** `A_ctx - A_tok` increases with normalized depth `l/L`, and cache hit rate increases
with `l/L`, across budgets and granularities.

**Falsified by:** a flat or negative depth slope in the majority of arms once measured over the
full stack on a normalized axis with matched baselines.

**Refined by:** the shape — monotone versus saturating implies a different cutoff for H2's
schedule, and the one full-depth signal we have currently favours saturation.

**Status:** positive slope in 3 of 4 arms over layers 2–6, CIs excluding zero in 3 of 4; one flat
exception; not yet measured on a normalized axis, over the full stack, or at 1e18.

### H2 — the quality cost of imposing rolling residency decreases with depth

**Prediction.** dBPB from constraining a single layer (R = k there, R = E elsewhere) is largest at
the first MoE layer and falls with depth; if H1 saturates, dBPB flattens at the same depth.

**Falsified by:** a flat or U-shaped per-layer cost curve, or a cost curve uncorrelated with the
per-layer contextual share from H1.

**Why it does not follow from H1.** "This layer routes on context" and "this layer can afford to
lose routing freedom" are different quantities. §5 of
[`delexicalization.md`](delexicalization.md) showed the model co-adapts to the constraint during
training, so per-layer cost must be measured on separately trained models — not by masking a
trained checkpoint (which is test C3's limitation).

**Status:** no evidence either way. The existing dose curve varies R uniformly across all layers.

**Payoff if both hold.** A prefix schedule: leave layers shallower than depth *d* unconstrained,
constrain everything from *d* on. One tunable parameter, with the per-layer curve giving *d*.

## 4. Cheap tests (no training)

Ordered by what they buy. C1–C4 are the critical path.

| id | test | serves | needs |
|---|---|---|---|
| C1 | Replot on normalized depth `l/L`, with bootstrap CI bands | H1 | nothing |
| C2 | Locus at layers 2–9 on the three preserved captures | H1 | existing captures |
| C3 | Per-layer inference-time constraint swap | H2 pre-screen | existing checkpoints |
| C4 | Baseline hit rate by counterfactual replay | H1 control | existing router logs |
| C5 | Per-layer output lens (effective vocabulary) | H1, third view | captures; >L4 needs re-run |
| C6 | Per-layer demand forecastability | H2 mechanism | existing captures |
| C7 | Nonparametric token-id oracle | H1 ceiling | existing captures |
| C8 | Causal token / context substitution | H1, causal | forward passes |
| C9 | Frequency-stratified `A_tok` | H1 refinement | captures + token ids |
| C10 | Cross-layer probe transfer | H1 refinement | existing captures |

**C1 — normalized depth and confidence intervals.** Every H1 statement is about depth, and the
current axis is not comparable across 4-, 6- and 9-layer models. A slope without an interval is not
testable. Minutes.

**C2 — the full stack.** This decides whether H1 is monotone or saturating, which sets what H2
should predict. `LAYERS = range(2, 10)` and a warning in place of the silent skip.

**C3 — per-layer inference-time constraint swap. The cheap pre-screen for H2, and the highest-value
item here.** §5 of [`delexicalization.md`](delexicalization.md) already measures the *global*
cross-regime swap (unmask a trained temporal model, or impose residency on a trained baseline). Do
it **one layer at a time**: impose residency on layer *l* only of a trained unconstrained model,
and separately unmask layer *l* only of a trained temporal model, sweeping *l*. This yields a
per-layer cost profile for the price of `2L` evaluation passes and **no training at all**. It is
deterministic — same checkpoint, same fixed batch, no seed noise — so even small differences are
readable.

Its limitation is exactly the one §5 identified: no co-adaptation, so it measures the cost of
*removing* freedom from a model that was trained expecting it, not the cost of never having had it.
That makes it an upper bound whose *shape* is still informative. If the shape is flat, H2's
training sweep is unlikely to be worth running; if it slopes, we know where to spend.

**C4 — baseline hit rate.** Replay rolling residency over an unconstrained run's router log. Without
it, H1's cache-side evidence has no matched control and cannot separate a tMoE effect from a
generic MoE effect — the same problem the Section 1 slopes exposed on the probe side.

**C5 — per-layer output lens.** An output-side view (what an expert writes) rather than an
input-side view (what makes it fire), from data already on disk for layers 2–4. Independent
evidence for H1 if it agrees; more interesting if it does not.

**C6 — per-layer demand forecastability.** `delex_demand.py` currently pools layers 2–6 into one
probe. Per layer, it tests whether deep-layer demand is more predictable from history — the
mechanism H2 leans on, and directly comparable to the hit-rate curve.

**C7 — nonparametric token-id oracle.** Best achievable AUC from token identity alone via empirical
`P(y_e | token id)`, equivalently `I(expert ; token id) / H(expert)`. Gives the ceiling the linear
probe has been measured against, so a falling `A_tok` can be attributed to genuine
context-dependence rather than probe capacity.

**C8 — causal substitution.** Hold context fixed and substitute the current token; measure how far
the selected expert set moves, per layer. Then the complement: hold the token fixed, shuffle the
context. The ratio is a causal per-layer lexical-versus-contextual measure that does not depend on
a probe being able to express the mapping. Forward passes only, no training. This is the strongest
non-training evidence available for H1.

**C9 — frequency stratification.** Split `A_tok` by token-frequency decile within each layer. If
the lexical shortcut lives entirely on rare tokens, that changes which layers deserve the
constraint and reframes H2.

**C10 — cross-layer probe transfer.** Fit the token probe at layer *l*, evaluate at *l'*.
Off-diagonal collapse would show routing is a qualitatively different function with depth rather
than the same function weakening.

## 5. Training tests

Three, in order. The design is driven by a power calculation, because the first testbed we
considered cannot detect the effect.

**Noise floor.** From [`seed_replicates.csv`](../../../results/ablations/seed_replicates.csv) and
[`flame38m_overnight_seeds.csv`](../../../results/ablations/flame38m_overnight_seeds.csv), test-set
BPB across seeds of the same config:

| cell | seeds | spread | sd |
|---|---|---|---|
| g3_s0_1e16 temporal fine | 1234 / 2 / 3 | 1.4754 / 1.4758 / 1.4737 | 0.0011 |
| g1_s2_1e17 temporal coarse | 1234 / 2 | 1.2821 / 1.2844 | ~0.002 |
| flame38m_g3 1e18 temporal | 2 / 3 | 1.3339 / 1.3323 | ~0.001 |
| flame38m_g3 1e18 full MoE | 2 / 3 | 1.3489 / 1.3483 | ~0.001 |

**Effect sizes available**, from the uniform-R endpoints:

| testbed | MoE layers | total constraint effect | per-layer if uniform | sd | per-layer SNR |
|---|---|---|---|---|---|
| s0 @1e16 | 3 | 0.0231 (cost) | 0.0077 | 0.0011 | **~7** |
| s2 @1e17 | 5 | 0.0131 (cost) | 0.0026 | 0.002 | ~1.3 |
| flame38m @1e18 | 8 | 0.0150 (benefit) | 0.0019 | 0.001 | ~1.9 |

**s2/1e17 — the testbed originally proposed — is the worst of the three and is underpowered at
n=1.** Revised plan:

**T1 — single-layer sweep at s0/1e16. 3 runs, cheapest available, first real H2 evidence.**
Constrain exactly one MoE layer at a time (R = k there, R = E elsewhere); the two reference
endpoints (all-constrained 1.4750, none-constrained 1.4519) already exist. Only 3 MoE layers, so
the depth resolution is coarse — but at SNR ~7 per layer it cleanly answers *"is the per-layer cost
curve flat or sloped?"*, which is the binary H2 turns on.

**T2 — shallow-half versus deep-half contrast at 1e18. 2 arms x 3 seeds = 6 runs.** Constrain
layers 2–5 with 6–9 free, versus 2–5 free with 6–9 constrained: matched layer count, matched
resident-slot budget, so the arms differ *only* in where the constraint sits. The contrast
aggregates four layers of the depth gradient instead of one, which is what makes 1e18 viable
despite a per-layer SNR of ~1.9; with 3 seeds per arm the contrast standard error is ~0.0008.
This is the definitive H2 test, at the budget where the constraint actually wins and across 8 MoE
layers.

**T3 — full per-layer resolution at 1e18, and the schedule-versus-uniform control. 8 + 2 runs, only
if T1 and T2 both point the same way.** Per-layer resolution gives the cutoff *d*; the control
compares the resulting prefix schedule against uniform R at **equal total resident-slot count**.
That control is not optional: the schedule saves memory against the full MoE baseline but *costs*
memory against shipped temporal (1e18 model: 512 slots baseline, 48 uniform R=k), so the only
meaningful comparison is against the existing uniform-R dose curve at equal spend.

**Run C3 before T1.** If the inference-time per-layer profile is flat, T1's prior drops sharply and
the training budget is better spent elsewhere.

**Implementation.** The R knob exists but is global —
[`temporal_router.py:359`](../../../temporal/temporal_router.py) reads `TEMPORAL_RESIDENCY_R` once
and applies it at every layer. `self.layer_number` is already in scope a few lines below, so a
per-layer schedule (an env-var list, or `R=E` sentinel per layer) is a small change. FLOPs are
unchanged at any R, so every arm above is compute-matched to the baseline.
