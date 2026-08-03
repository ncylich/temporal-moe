# What rolling residency does to routing, and what it costs

Results only. History is in [`05-notebook.md`](05-notebook.md), the delta against the published
write-up is in [`02-corrections.md`](02-corrections.md), probe construction is in
[`03-methods.md`](03-methods.md), and what exists over which runs is in
[`04-coverage.md`](04-coverage.md).

## 0. Scope, and how to read the numbers

This document covers what the residency constraint does to a router and what it costs. The adaptation
program's own questions, retrofitting the constraint to a pretrained model and the per-layer embedding
work, are written up in [`../../../results/ablations/ple_RESULTS.md`](../../../results/ablations/ple_RESULTS.md).
Its layer-freeing results appear in section 5 because they bear on layer choice.

Two regimes are compared throughout. **Unconstrained** is an ordinary mixture-of-experts model, every
expert available at every token. **Constrained** is the same shape trained under rolling residency.
Floating-point operations are identical in both.

Measurements, each defined once here and used consistently:

- **Token AUC** and **context AUC.** A ridge probe on input embeddings predicts whether a given expert
  fires at a position, from either the current token's embedding or the mean of a surrounding window
  that excludes the current token. Scored on held-out documents. Chance is 0.5. **Context minus token**
  is positive when routing is better predicted by surroundings than by the token itself.
- **Cost.** Test bits per byte with the constraint changed at one layer, minus the same model in its
  native regime. Positive is worse. For a constrained model this means unmasking one layer; for an
  unconstrained model, imposing residency at one layer.
- **Participation ratio.** Inverse Simpson index of an expert's token distribution, normalised to a
  0 to 1 scale. Higher means the expert spreads over more of the stream.
- **Hit rate.** Share of the unconstrained top-k already resident before any swap. The random floor is
  k/E, which is 0.094 at both granularities used here.
- **Retained mass.** Share of the unconstrained top-k routing mass held by the resident set.
- **Demand AUC.** A causal probe using only routing history predicts whether an expert is demanded at
  the next position.
- **The sham.** The constraint replaced by a Gaussian perturbation of the router logits, matched in
  magnitude and carrying no lexical information. It measures how much of an effect is positional
  rather than about token identity.

Some evidence cannot be regenerated. Eight result files have no producer in any commit on any branch,
and the runs behind them kept neither a router log nor a checkpoint. Claims resting on those say so in
the sentence that makes them. The list is in
[`../../../results/ablations/README.md`](../../../results/ablations/README.md).

## 1. What the router does

Routing changes what it keys on. In the unconstrained regime the current token dominates; under the
constraint, surroundings do.

| regime | arms | token AUC | context AUC | context minus token |
|---|---|---|---|---|
| unconstrained | 14 | 0.884 | 0.640 | −0.242 |
| constrained | 16 | 0.585 | 0.697 | +0.103 |

Medians over per-expert fits at window w = k on document-disjoint splits. Higher AUC means more
predictable; the last column is the one that separates the regimes.

The interesting half of that table is the token column, not the context column. Context AUC is similar
in both regimes, 0.640 against 0.697, so the constraint does not create context sensitivity that was
absent before. What it does is destroy the token signal, from 0.884 to 0.585. Figure:
[`arm_separation.png`](../../../results/phase0/figures/arm_separation.png).

Three checks say the low token number is a fact about routing rather than about the probe.

- The linear probe reaches 99.4% of the nonparametric ceiling from token identity alone in the
  unconstrained regime and 101.8% in the constrained one. There is no token signal left for a better
  probe to find.
- Substituting causally agrees. Holding context fixed and swapping a frequency-matched token, against
  holding the token fixed and swapping context, gives a context-over-token ratio of 0.69 unconstrained
  (range 0.28 to 0.79) and 1.58 constrained (1.25 to 2.18) over 29 measurements each. The ranges do not
  overlap.
- It is not a rare-token effect. Stratifying token AUC by corpus frequency, the unconstrained regime
  runs 0.813 to 0.891 across five strata and the constrained one 0.550 to 0.581. Neither shows the
  signal concentrating in any frequency band.

Two consequences follow at serving time, and they belong here because they describe routing behaviour
rather than a system design.

- Demand becomes predictable from history alone: 0.981 constrained (0.919 to 0.993, n = 103) against
  0.655 unconstrained (0.567 to 0.716, n = 89). This is the largest clean separation in the program and
  the two ranges are nowhere near each other.
- The cache hits far above the 0.094 random floor: 0.317 constrained at coarse granularity and 0.326 at
  fine, against 0.172 unconstrained. Figure:
  [`hitrate_by_layer.png`](../../../results/phase0/figures/hitrate_by_layer.png).

### Depth

Context dominance rises with depth, but not equally in the two regimes, and the difference matters.

| regime | arms | median slope per layer | interval excludes zero |
|---|---|---|---|
| unconstrained | 5 | +0.0139 | 4 of 5 |
| constrained | 6 | +0.0018 | 3 of 6 |

Per-arm ordinary least squares on context minus token against layer index, with bootstrap intervals.
Positive means routing grows more contextual deeper in the stack.

The rise is solid in the unconstrained regime and about eight times smaller and inconsistent in the
constrained one. So the gap between regimes **narrows** with depth rather than widening: it is roughly
0.39 at the shallowest layer and 0.27 at the deepest. A constrained router is already contextual at
layer 2 and has little further to move. Figure:
[`locus_by_layer.png`](../../../results/phase0/figures/locus_by_layer.png).

Pooling layers across arms of different depths hides this, because the eight-layer arms are flat while
the thirteen-layer arms rise. Per-arm slopes are the right statistic.

Finally, the shift is produced by training under the constraint and not by the constraint itself. That
is section 5, and it is the reason this section describes a learned property rather than a mechanical
one.

## 2. What an expert represents

Under the constraint each expert covers more of the stream, and the routing distribution flattens.

| statistic | unconstrained | constrained |
|---|---|---|
| participation ratio | 0.298 | 0.577 |
| generalist fraction | 0.016 | 0.615 |
| router entropy | 0.879 | 0.948 |

Medians over per-layer records. Generalist fraction is the share of experts with participation ratio
above 0.5, so it is the first row thresholded rather than independent evidence. Higher means flatter,
less specialised routing.

- Generalist fraction falls with depth in both regimes, so the flattening is strongest early.
- The inventory is not starved but it is not untouched either. Constrained models use a median 82% of
  their experts per sequence, with a wide spread from 13% to 99.9% across 21 runs. The single
  unconstrained run with a preserved log uses 100%.
- Five different router designs, including auxiliary-loss-free and two momentum variants, land in the
  same structural band, with generalist fractions from 0.578 to 0.698. The effect belongs to the
  constraint and not to any particular router.

Two things do not change. Expert weight geometry is indistinguishable between regimes on centroid
distance and pairwise cosine. The output-side result does not replicate: at 1e18 the constrained model
writes sharper output distributions at 4 of 8 layers on the data-weighted metric and 0 of 8 on the
static one, with no consistent direction in the fine-grained pair.

One structural result runs opposite to the intuition that the constraint fragments the router. Comparing
router subspaces across layer pairs, at gaps of four layers or more the unconstrained router is
**more** self-similar than the constrained one, 2.28 times chance against 1.56. The constrained router
differentiates its subspaces across depth more, not less. The statistic is degenerate on 11 of 26 arms
where the expert count meets or exceeds the hidden width, and those arms are excluded.

## 3. Serving

The demand signal is the lever, and the eviction rule is not.

- Across a matched population of 134 measurements, eviction policies span 0.251 set coverage for
  least-recently-used to 0.510 for a discounted oracle. Two times, from the worst practical policy to
  an oracle.
- Smoothing the demand estimate is worth more. Set coverage goes from 0.310 at no smoothing to 0.854 at
  the strongest, a factor of 2.8 on the same measurements.

A better demand signal is not the same as a perfect one. Replacing the estimate with perfect
next-token foresight helps at coarse granularity, by 6.6 to 10.1 points over six runs, and hurts at
fine, by 2.6 to 11.2 points over fourteen. The split is 20 of 20 by granularity. The natural reading
is that chasing instantaneous demand destroys accumulated locality, so foresight without a retention
objective is a liability. Two caveats belong with the claim: granularity is confounded with expert
count and model family, since every k = 6 run is a 64-expert model and every k = 18 run a 192-expert
one, and **none of the twenty runs kept a router log, so this cannot be re-measured.**

Other properties of the resident set:

- Mass consistency exceeds set consistency, 0.419 against 0.374 constrained. The experts carrying
  routing weight stay resident while marginal ones churn. The unconstrained comparison rests on the
  single preserved baseline log, 0.186 against 0.168.
- Hysteresis can eliminate swapping entirely and the price is steep. Raising the threshold drives swap
  rate from 1.000 to 0.000 while retained mass falls from 0.353 to 0.114.
- A victim cache is cheap and effective. Eight experts recover 16 to 26% of reloads and 32 experts
  recover 44 to 70%, roughly linear at small sizes. Figure:
  [`victim_cache_hitrate_vs_size.png`](../../../results/phase0/figures/victim_cache_hitrate_vs_size.png).
- Recomputing residency in blocks instead of rolling it is a tradeoff, not a loss. At a block length of
  72 tokens it holds 28.55 retained mass against rolling's 45.76, but does 0.12 swaps per token against
  rolling's ceiling of 1. Worth considering when swap bandwidth binds rather than routing quality.
  **Producerless and unreplayable.**
- Swap rate is not a useful statistic. It sits at a median of 1.000 across 112 per-layer records, range
  0.987 to 1.000, because at R = k a swap fires whenever any demanded expert is missing. Use the 95th
  percentile burst length instead.
- Enlarging the resident cache closes the regime gap. At K = k the constrained arm hits 0.236 against
  0.169 unconstrained, and both reach 0.99 by K = 10.5k.
- Locality does not grow with scale. Expert-set overlap runs 19.2% to 32.9% from 1.4M to 186M active
  parameters against a 9.4% random floor, roughly flat, with the one matched unconstrained arm at
  19.2%.
- Document boundaries are close to a non-issue. Hit rate after an end-of-document token is 0.291
  against 0.311 within a document, a deficit of 0.009 over 66 measurements. Real, and too small to
  design around.

## 4. What it costs

Globally the constraint is cheap. Across a 10.7-fold range of resident-set size, test bits per byte
moves from 1.4750 at full constraint to 1.4519 unconstrained, a cost of **0.0231 BPB**.

At 1e18 the sign reverses and the constrained model wins outright, at both granularities: 1.3124
against 1.3175 coarse and 1.3339 against 1.3478 fine, over three to five seeds per cell with seed
standard deviations of 0.0011 to 0.0020. Both gaps are several standard errors wide, so this is not
noise, and no explanation for it is offered here.

That number describes training with the constraint. Imposing it on a finished model is a different
measurement and roughly twenty times more expensive.

| direction | budget | cost, BPB |
|---|---|---|
| trained with, then unmasked | 1e16 | +0.0994 |
| trained with, then unmasked | 1e17 | +0.1242 |
| trained with, then unmasked | 1e19 | +0.2006, +0.2064 |
| trained without, then imposed | 1e16, 1e17 | +0.2403, +0.6099 |
| trained without, then imposed | 1e19 | +0.4314 |

Positive is worse. Neither regime transfers to the other. The unmasking direction rises across its
three budgets; the imposition direction does not, so no budget trend should be claimed for both. Both
files are producerless, though the 1e19 one is re-runnable from surviving checkpoints.

### By layer

The last layer is consistently the most expensive to change. Across all seven per-layer measurements
it costs between 1.61 and 3.22 times the interior mean. The first layer is elevated in two of the
seven, both unconstrained models at 1e18, and sits at or below the interior mean in three others.

| model | budget | direction | first / interior | last / interior |
|---|---|---|---|---|
| g1 moe | 1e18 | impose | 1.78 | 2.02 |
| g3 moe | 1e18 | impose | 1.97 | 3.00 |
| coarse moe | 1e19 | impose | 1.01 | 1.70 |
| g1 temporal | 1e18 | unmask | 1.18 | 1.61 |
| g1 temporal, seed 2 | 1e18 | unmask | 0.93 | 2.00 |
| g3 temporal | 1e18 | unmask | 0.94 | 1.85 |
| coarse temporal | 1e19 | unmask | 1.17 | 3.22 |

Per-layer cost relative to the mean over interior layers. Higher means more expensive to change at
that layer. The seven measurements also disagree about where the minimum sits, so no vertex is claimed.

The reason is positional, not lexical. A magnitude-matched perturbation carrying no lexical information
reproduces most of the endpoint excess, 56 to 63% on the coarse model depending on which of its two
noise calibrations is used and 83% on the fine one. Figure:
[`sham_residual.png`](../../../results/phase0/figures/sham_residual.png). The lexical reading fails on
its own terms as well: in the three imposition arms the last layer ranks most contextual of all,
8 of 8, 8 of 8 and 13 of 13, which is the opposite of what a token-boundness explanation predicts.

Trained from scratch with individual layers constrained, over three seeds and three MoE layers, the
middle layer is cheapest and both ends cost more. Endpoints against middle is +0.0134 CE at 2.4
standard errors. Constraining all three costs +0.0419 CE at 5.3 standard errors, the only contrast in
that sweep that is comfortably resolved.

### Which profile the cost follows

Per-layer cost tracks how stable a layer's demand is, and not how lexical it is.

| cost against | Spearman |
|---|---|
| churn | −0.91 |
| demand forecastability | +0.78 |
| cache hit rate | +0.75 |
| generalist fraction | −0.69 |
| contextual share | +0.19 |

Thirteen layers of one model. Sign convention: cost is higher where churn is lower.

The first three rows are one factor, not three. Churn, hit rate and demand AUC intercorrelate between
0.87 and 0.97, so they measure demand stability three ways. Against that, contextual share at +0.19 is
close to nothing. The layers that lose most when the constraint changes are the ones whose demand was
most predictable, not the ones closest to the token. This is the leading hypothesis rather than a
result: one model, thirteen points.

## 5. Adapting a pretrained model

Everything above is measured on models trained under the constraint from the start. Retrofitting it to
a pretrained 16-layer model separates what the constraint does from what training under it does, and
the two are not the same.

Running the same locus probe on an adapted OLMoE:

| condition | context minus token |
|---|---|
| base model, constraint imposed, no training | −0.0041 |
| adapted on cross-entropy | +0.0493 |
| adapted with per-layer embeddings | +0.093 to +0.096 |

Impose the constraint and there is no contextual shift at all. Train under it and the shift appears,
and grows with more adaptation. Section 1 describes something the model learns, not something the
constraint imposes.

The same asymmetry undoes the obvious way to choose which layers to free. Single-layer damage on this
model is U-shaped, worst at layer 1 and lowest at layer 11, with both ends elevated. Layers 2 and 15
tie on it almost exactly, 0.1408 against 0.1408. Freeing them is not equivalent:

| free set | resident memory | BPB | mean downstream accuracy |
|---|---|---|---|
| {0,1} | +87.5% | 0.814440 | 0.5937 |
| {0,1,2} | +131.2% | 0.808615 | 0.5937 |
| {0,1,15} | +131.2% | 0.797810 | 0.6030 |
| {0,1,14,15} | +175.0% | 0.786275 | 0.6037 |

Lower BPB and higher accuracy are better. The middle two rows cost identical memory and differ only in
which layer is freed third.

The training-free profile predicted layer 2 would be worth 5.8 times layer 15 as the third freed layer.
Trained, layer 15 wins by 0.0108 BPB and takes the better downstream score. Two further cells
contradict the profile the same way, one of them a controlled matched-memory pair. **Do not choose free
sets from single-layer damage.** Freeing both ends is the best configuration in that program, and
training-free it looked dominated.

## 6. What to do with it

- Exempt the first and last MoE layers if you exempt any, on the architectural grounds in section 4,
  not on lexicality. The last layer is among the least token-bound in the stack.
- Do not pick free sets from single-layer ablation. It is wrong on this model in three cells.
- Do not invest in a better demand oracle. A perfect one is worse at fine granularity.
- Spend effort on smoothing the demand estimate, worth 2.8 times, before eviction policy, worth 2.
- A small victim cache is the cheapest remaining win.

## 7. How much of this to believe

- Every locus and lens measurement is one training seed per cell.
- Only one unconstrained run has a preserved router log, so every regime contrast in section 3, and the
  inventory and consistency comparisons in sections 2 and 3, rest on a single baseline.
- No per-layer measurement above 13 layers on models we trained. The 16-layer evidence is section 5's
  and comes from a different program on a model we did not train.
- Eight files cannot be regenerated. Section 3's oracle result rests entirely on two of them and its
  block-wise result on a third, and section 4's cross-regime table on two more.
- Two structural files were measured by a method since superseded but remain the only record of eight
  runs whose checkpoints no longer exist. Section 2's numbers come from the per-layer replacements.
- Four analyses were read and produced nothing load-bearing: momentum smoothing moves set coverage by
  one point across its whole range, anomaly prediction splits by regime exactly as demand
  forecastability does, and a centre-of-mass replay covers two runs of one router variant.

## 8. What is worth doing next

- Does demand forecastability predict which layers are worth *freeing*, or only what constraining one
  *costs*? Section 4 correlates it with solo cost; section 5 shows solo measurements misrank layers for
  set selection. No single model currently carries both measurements, so the question is open by
  construction.
- The 1e19 cross-regime cells are re-runnable and would give the imposition direction a third budget.
- The sham percentages need their producer's definition pinned down before they are quoted again.
