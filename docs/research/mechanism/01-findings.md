# What rolling residency does to routing, and what it costs

Results only. History is in [`05-notebook.md`](05-notebook.md), the delta against the published
write-up is in [`02-corrections.md`](02-corrections.md), probe construction is in
[`03-methods.md`](03-methods.md), and what exists over which runs is in
[`04-coverage.md`](04-coverage.md).

## 0. Scope, and how to read the numbers

**Two regimes throughout.** *Unconstrained* is an ordinary mixture-of-experts model, every expert
available at every token. *Constrained* is the same shape trained under rolling residency. Identical
floating-point operations.

**Out of scope.** Retrofitting the constraint to a pretrained model, and the per-layer embedding work,
belong to the adaptation program and are written up in
[`ple_RESULTS.md`](../../../results/ablations/ple_RESULTS.md). Its layer-freeing results appear in
section 5 because they bear on layer choice.

Every measurement, defined once:

- **Token AUC / context AUC.** Ridge probe asking "does expert *e* fire here?", given either the
  current token's embedding or the surrounding window with that token excluded. Held out on unseen
  documents, chance 0.5.
- **Context minus token.** The difference. Positive means surroundings predict routing better than the
  token does.
- **Cost.** Test bits per byte with the constraint changed at one layer, minus the model in its native
  regime. Positive is worse. Unmasking a layer for a constrained model, imposing residency for an
  unconstrained one.
- **Participation ratio.** Inverse Simpson index of an expert's token distribution, on a 0 to 1 scale.
  Higher means the expert spreads over more of the stream.
- **Generalist fraction.** Share of experts with participation ratio above 0.5. A threshold on the row
  above, not separate evidence.
- **Hit rate.** Share of the unconstrained top-k already resident before any swap. Random floor is
  k/E, which is 0.094 at both granularities here.
- **Set coverage.** Same idea, measured against a serving policy rather than a layer.
- **Retained mass.** Share of the unconstrained top-k *routing mass* the resident set holds. Weights
  experts by how much the router wanted them.
- **Demand AUC.** Causal probe predicting the next position's demand from routing history alone.
- **The sham.** The constraint swapped for a Gaussian perturbation of the router logits, matched in
  average magnitude, carrying no lexical information. Separates positional effects from token-identity
  ones.

**Eight files cannot be regenerated.** No producer in any commit on any branch, and their runs kept
neither a router log nor a checkpoint. Claims resting on them say so in the sentence that makes them.
List in [`results/ablations/README.md`](../../../results/ablations/README.md).

## 1. What the router does

**The constraint destroys the token signal rather than adding a context signal.** Across 34 trained
models the token axis separates the regimes completely, with an empty band 0.184 wide. The context
axis does not separate them at all: several constrained models sit *below* unconstrained ones.

<img src="../../../results/phase0/figures/arm_separation.png" alt="Token AUC against context AUC, one point per trained model" width="66%">

*One point per trained model, lines joining a granularity across budgets. Horizontal: how well the
current token predicts firing. Vertical: how well the surrounding context does. Separation is
horizontal only.*

Three checks that this is routing and not a weak probe:

- **No token signal is left to find.** The linear probe reaches 99.4% of the nonparametric ceiling
  from token identity alone, unconstrained, and 101.8% constrained.
- **Causal, not correlational.** Swapping a frequency-matched token with context held fixed, against
  the reverse: context-over-token ratio 0.69 unconstrained (0.28 to 0.79) and 1.58 constrained (1.25
  to 2.18), 29 measurements each, ranges non-overlapping.
- **Not a rare-token artifact.** Token AUC by corpus-frequency stratum runs 0.813 to 0.891
  unconstrained and 0.550 to 0.581 constrained. No band concentrates the signal.

Two consequences, here rather than in section 3 because they describe the router:

- **Demand becomes predictable from history alone.** 0.981 constrained (0.919 to 0.993, n = 103)
  against 0.655 unconstrained (0.567 to 0.716, n = 89). Largest clean separation in the program, with
  daylight between the ranges.
- **The cache hits far above chance.** 0.317 constrained coarse and 0.326 fine, against 0.172
  unconstrained, on a 0.094 floor.

<img src="../../../results/phase0/figures/hitrate_by_layer.png" alt="Cache hit rate by MoE layer" width="66%">

*Hit rate by layer, both regimes, against the k/E floor. Higher is better. Deeper layers cache
better, so a non-uniform memory budget should favour shallow ones.*

### Depth

**Routing grows more contextual with depth, but far more so unconstrained.** Per-arm slopes of
context minus token against layer index, bootstrap intervals:

| regime | arms | median slope per layer | interval excludes zero |
|---|---|---|---|
| unconstrained | 5 | +0.0139 | 4 of 5 |
| constrained | 6 | +0.0018 | 3 of 6 |

- **The gap narrows with depth**, roughly 0.39 at the shallowest layer down to 0.27 at the deepest.
  A constrained router is already contextual at layer 2 and has little further to move.
- **Use per-arm slopes, not pooled layers.** Pooling hides it, because the eight-layer arms are flat
  while the thirteen-layer arms rise.
- **This is learned, not mechanical.** Imposing the constraint without training produces no shift at
  all. Section 5.

<img src="../../../results/phase0/figures/locus_by_layer.png" alt="Context minus token AUC against layer" width="66%">

*Broken y-axis: the regime gap of about 0.3 dwarfs the depth effect of about 0.05. Unconstrained
curves climb; constrained curves start high and stay flat.*

## 2. What an expert represents

**Each expert covers more of the stream, and routing flattens.** Full range across models, median over
each model's layers. Higher means flatter, less specialised.

| | participation ratio | generalist fraction | router entropy |
|---|---|---|---|
| unconstrained, 14 models | 0.201 to 0.422 | 0.000 to 0.328 | 0.790 to 0.917 |
| constrained, 16 models | 0.292 to 0.801 | 0.036 to 0.914 | 0.886 to 0.974 |
| zero-layer control | 0.328 | 0.036 | 0.886 |

- **These ranges overlap**, unlike the locus result. A strong tendency, not a separator.
- **Flattening is strongest early**: generalist fraction falls with depth in both regimes.
- **The third row is a control.** One arm of the section 4 training sweep runs a constrained schedule
  that constrains no layers. Built, trained and counted as constrained; only the constraint is missing.
  It lands at the bottom of the constrained range, with the unconstrained models, as it must if these
  statistics measure the constraint rather than how constrained runs are configured. Nothing else here
  would catch a bug that inflated every constrained model equally.

**The inventory is not starved.** Union covers 85 to 100% of the pool on shipped configurations:

| model | budget | regime | experts | union, mean | union, share of E | effective experts |
|---|---|---|---|---|---|---|
| `moe_coarse_1e19` | 1e19 | unconstrained | 64 | 63.8 | 0.997 | 59.8 |
| `g3_tmoe_s2_1e17` | 1e17 | constrained | 192 | 160.8 | 0.837 | 187.8 |
| `flame38m_g1_temporal` | 1e18 | constrained | 64 | 62.0 | 0.969 | 63.1 |
| `flame38m_g3_temporal` | 1e18 | constrained | 192 | 163.4 | 0.851 | 187.2 |
| `g1_tmoe_coarse_1e19` | 1e19 | constrained | 64 | 63.9 | 0.999 | 62.5 |
| `temporal_fine_g3_1e19` | 1e19 | constrained | 192 | 184.2 | 0.959 | 183.0 |

- **Union** is the mean count of distinct experts a sequence touches. **Effective experts** weights
  that count by how evenly usage is spread, penalising a long tail of barely-used ones. Higher is
  better on both if the goal is to use the pool you paid for.
- **Not a property of one router.** Five designs, including auxiliary-loss-free and two momentum
  variants, land in the same band, generalist fractions 0.578 to 0.698.
- **An earlier draft said 13 to 99.9% here.** That pooled these six shipped configurations with
  sixteen diversity-suppression screens whose purpose is to collapse the expert set. The 13% floor was
  `ant0p1` behaving as designed.

<img src="../../../results/phase0/figures/expert_residency_distribution.png" alt="Distribution of per-expert residency share" width="66%">

*How often each expert is resident, one curve per regime. Flatter and wider means load spread further.
The same fact the participation-ratio table states numerically.*

What does not change, and one thing that moves the wrong way:

- **Weight geometry is untouched.** Centroid distance and pairwise cosine are indistinguishable
  between regimes.
- **The output side does not replicate.** At 1e18 the constrained model writes sharper distributions
  at 4 of 8 layers on the data-weighted metric, 0 of 8 on the static one, no consistent direction in
  the fine-grained pair.
- **The router does not fragment; it differentiates.** At layer gaps of four or more the
  *unconstrained* router is more self-similar, 2.28 times chance against 1.56. Opposite to the
  intuition. Degenerate on 11 of 26 arms where expert count meets or exceeds hidden width, excluded.

## 3. Serving

**The demand signal is the lever. The eviction rule is not.** Both measured by replaying the same
recorded demand through different policies, so only the cache logic varies. Metric is set coverage,
floor 9.4%.

| eviction policy | set coverage |
|---|---|
| discounted oracle, g = 0.5 | 49.5% |
| discounted oracle, g = 0.9 | 45.8% |
| Belady with prefetch, h = 1 | 41.7% |
| Belady, the offline optimum | 33.1% |
| minimum logit, shipped | 26.2% |
| least recently used | 21.8% |

- **Eviction headroom is 6.5 to 9.7 points**, the gap from the shipped rule up to Belady, which is the
  offline optimum for a pure eviction rule. Medians over 66 measurements, six shipped configurations.
- **Rows above Belady are not eviction rules.** They buy coverage by seeing future demand, a different
  lever.
- **Smoothing the estimate is worth 2.8x**, from 0.310 at no smoothing to 0.854 at the strongest.
  Bigger than every eviction rule combined.

<img src="../../../results/phase0/figures/eviction_policy_headroom_belady_bound.png" alt="Eviction policy headroom against the Belady bound" width="66%">

*Set coverage by policy, per shipped configuration. Higher is better. The practical-to-optimal gap is
small next to what a better demand estimate opens.*

<img src="../../../results/phase0/figures/demand_smoothing_swap_vs_coverage.png" alt="Swap rate against coverage under demand smoothing" width="66%">

*One point per smoothing strength. Up and to the left is better: more coverage for fewer swaps.*

**A perfect demand signal is worse than a good one, at fine granularity.** Replacing the estimate with
perfect next-token foresight:

| granularity | runs | effect of perfect foresight |
|---|---|---|
| k = 6 of 64 experts | 6 | +6.6 to +10.1 points |
| k = 18 of 192 experts | 14 | −2.6 to −11.2 points |

- **Splits 20 of 20** by granularity, no run crossing zero.
- **Likely cause**: chasing instantaneous demand destroys accumulated locality, so foresight without a
  retention objective is a liability.
- **Confounded.** Every k=6 run is a 64-expert model, every k=18 run a 192-expert one.
- **Unrepeatable.** None of the twenty runs kept a router log.

The resident set, briefly:

- **Mass beats set.** Mass consistency 0.419 against set consistency 0.374: the experts carrying
  routing weight stay put while marginal ones churn.
- **Hysteresis is not free.** Raising the threshold drives swap rate 1.000 to 0.000, but retained mass
  falls 0.353 to 0.114.
- **Swap rate is a dead statistic**, median 1.000 across 112 records, because at R = k a swap fires
  whenever any demanded expert is missing. Use 95th-percentile burst length.
- **Block-wise residency is a trade, not a loss.** At block length 72 it holds 28.55 retained mass
  against rolling's 45.76, at 0.12 swaps per token against rolling's ceiling of 1. Worth it if swap
  bandwidth binds. Producerless and unreplayable.

**A bigger cache closes the regime gap**, and the constrained arm leads throughout:

| cache size, K/k | unconstrained | constrained |
|---|---|---|
| 1 | 0.169 | 0.236 |
| 2 | 0.284 | 0.401 |
| 3 | 0.386 | 0.497 |
| 10.5 | 0.990 | 0.993 |

Both saturate near ten times k, so the advantage is largest exactly where memory is scarce.

**Locality does not grow with scale.** Expert-set overlap between neighbouring positions, against a
9.4% random floor:

| active parameters | 1.4M | 8.2M | 12.2M | 184.1M | 185.8M |
|---|---|---|---|---|---|
| constrained overlap | 28.4% | 32.9% | 31.2% | 28.3% | 23.5% |

Near three times the floor across a 130-fold size range rather than growing. The one matched
unconstrained arm sits at 19.2%.

Document boundaries are not a cold start. Over windows of 4, 16 and 64 tokens after a boundary, the
median hit-rate penalty is 0.9 points and is negative on some models, because routing keys on a
surrounding window rather than on document identity.

## 4. What it costs

**Globally, 0.0231 BPB across a 10.7-fold range of resident-set size.** Lower is better:

| resident set, R/k | 1 | 2 | 4 | 7.1 | 10.7 |
|---|---|---|---|---|---|
| test BPB | 1.4750 | 1.4736 | 1.4681 | 1.4580 | 1.4519 |

- **The curve is flat where it matters**, steep only as the constraint is fully released. Most of the
  0.0231 arrives in the last doubling.
- **At 1e18 the constrained model wins outright**, both granularities: 1.3124 against 1.3175 coarse,
  1.3339 against 1.3478 fine, three to five seeds, seed deviations 0.0011 to 0.0020. Several standard
  errors wide, so not noise. No explanation offered.

<img src="../../../results/phase0/figures/residency_dose_curve.png" alt="Quality against resident-set size" width="66%">

*Test bits per byte against resident-set size. Lower is better.*

**Training with the constraint and imposing it later are different costs, about twenty-fold apart.**

| direction | budget | cost, BPB |
|---|---|---|
| trained with, then unmasked | 1e16 | +0.0994 |
| trained with, then unmasked | 1e17 | +0.1242 |
| trained with, then unmasked | 1e19 | +0.2006, +0.2064 |
| trained without, then imposed | 1e16, 1e17 | +0.2403, +0.6099 |
| trained without, then imposed | 1e19 | +0.4314 |

- **Neither regime transfers.** Positive is worse in both directions.
- **Only the unmasking direction rises with budget.** Imposition does not, so claim no trend for both.
- Both files producerless; the 1e19 one is re-runnable from surviving checkpoints.

### By layer

**The last layer is the most expensive to change, in seven of seven measurements**, at 1.61 to 3.22
times the interior mean. The first is elevated in two, both unconstrained models at 1e18.

| model | budget | direction | first / interior | last / interior |
|---|---|---|---|---|
| g1 moe | 1e18 | impose | 1.78 | 2.02 |
| g3 moe | 1e18 | impose | 1.97 | 3.00 |
| coarse moe | 1e19 | impose | 1.01 | 1.70 |
| g1 temporal | 1e18 | unmask | 1.18 | 1.61 |
| g1 temporal, seed 2 | 1e18 | unmask | 0.93 | 2.00 |
| g3 temporal | 1e18 | unmask | 0.94 | 1.85 |
| coarse temporal | 1e19 | unmask | 1.17 | 3.22 |

- **No vertex claimed.** The seven disagree about where the minimum sits.
- **The cause is positional, not lexical.** A lexicality-free perturbation of matched average size
  reproduces 63% of the endpoint excess on the coarse model and 83% on the fine one. Excess is the
  endpoint mean minus the interior mean; the noise scale is the calibration matching the real mean
  cost, to 0.2% coarse.
- **The lexical reading fails on its own terms.** In the three imposition arms the last layer ranks
  *most* contextual of all, 8 of 8, 8 of 8 and 13 of 13. The opposite of what token-boundness predicts.

<img src="../../../results/phase0/figures/sham_residual.png" alt="Real constraint against magnitude-matched sham, and the residual" width="66%">

*Top: real constraint against the sham. Bottom: the residual, near zero across the interior and
positive at both ends. What the sham fails to explain is itself an endpoint effect.*

**Trained from scratch, the middle layer is cheapest and both ends cost more.** Three seeds, three
MoE layers, lower is better:

| constrained layers | test CE | against none |
|---|---|---|
| none | 4.0182 | 0 |
| layer 3, the middle | 4.0208 | +0.0026 |
| layer 4 | 4.0335 | +0.0153 |
| layer 2 | 4.0349 | +0.0167 |
| all three | 4.0601 | +0.0419 |

Endpoints against middle is +0.0134 CE at 2.4 standard errors. All three costs +0.0419 at 5.3, the
only contrast in the sweep comfortably resolved.

### Which profile the cost follows

**Cost tracks how stable a layer's demand is, not how lexical it is.** Thirteen layers, one model:

| cost against | churn | demand forecastability | cache hit rate | generalist fraction | contextual share |
|---|---|---|---|---|---|
| Spearman | −0.91 | +0.78 | +0.75 | −0.69 | **+0.19** |

- **The first three are one factor, not three.** Churn, hit rate and demand AUC intercorrelate 0.87
  to 0.97. They measure demand stability three ways.
- **Lexicality is the odd one out** at +0.19, close to nothing.
- **Reading**: the layers that lose most are the ones whose demand was most predictable, not the ones
  closest to the token. Leading hypothesis, not a result. One model, thirteen points.

## 5. Adapting a pretrained model

Everything above is measured on models trained under the constraint from the start. Retrofitting it to
a pretrained 16-layer model separates what the constraint *does* from what training under it does.

**The contextual shift is learned, not imposed.** Same locus probe, adapted OLMoE:

| condition | context minus token |
|---|---|
| base model, constraint imposed, no training | −0.0041 |
| adapted on cross-entropy | +0.0493 |
| adapted with per-layer embeddings | +0.093 to +0.096 |

Impose the constraint and nothing shifts. Train under it and the shift appears, growing with more
adaptation. Section 1 describes something the model learns.

**Single-layer damage is U-shaped**, worst at layer 1, lowest at layer 11, both ends elevated. It is
also the wrong tool for choosing which layers to free.

<img src="../../../results/phase0/figures/layer_freeing_damage.png" alt="Per-layer damage across sixteen layers" width="66%">

*Cost of constraining each layer alone, all sixteen. Lower is better. The shape is why freeing the
ends looks attractive; the table below is why it cannot be read off this curve.*

Layers 2 and 15 tie on solo damage, 0.1408 against 0.1408. Freeing them is not equivalent:

| free set | resident memory | BPB | mean downstream accuracy |
|---|---|---|---|
| {0,1} | +87.5% | 0.814440 | 0.5937 |
| {0,1,2} | +131.2% | 0.808615 | 0.5937 |
| {0,1,15} | +131.2% | 0.797810 | 0.6030 |
| {0,1,14,15} | +175.0% | 0.786275 | 0.6037 |

- **The middle two rows are a controlled pair**: identical memory, differing only in which layer is
  freed third. Lower BPB and higher accuracy are better.
- **The training-free profile predicted layer 2 at 5.8 times layer 15.** Trained, layer 15 wins by
  0.0108 BPB and takes the better downstream score.
- **Two further cells contradict the profile the same way.**
- **Freeing both ends is the best configuration**, and training-free it looked dominated.

**Do not choose free sets from single-layer damage.**

## 6. What to do with it

- Exempt the first and last MoE layers if you exempt any, on the architectural grounds in section 4,
  not on lexicality. The last layer is among the least token-bound in the stack.
- Do not pick free sets from single-layer ablation. It is wrong on this model in three cells.
- Do not invest in a better demand oracle. A perfect one is worse at fine granularity.
- Spend effort on smoothing the demand estimate, worth 2.8 times, before eviction policy, worth 2.

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
