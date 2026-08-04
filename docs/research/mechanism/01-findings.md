# What rolling residency does to routing, and what it costs

Results only. History is in [`05-notebook.md`](05-notebook.md), the delta against the published
write-up is in [`02-corrections.md`](02-corrections.md), probe construction is in
[`03-methods.md`](03-methods.md), and what exists over which runs is in
[`04-coverage.md`](04-coverage.md).

## 0. Scope, and how to read the numbers

**Two regimes throughout.** *Full MoE* is an ordinary mixture-of-experts model, every expert
available at every token. *Temporal* is the same shape trained under rolling residency. Identical
floating-point operations, and the names match the `regime` column in every CSV and the legends on
every figure here.

**Out of scope.** Retrofitting the constraint to a pretrained model, and the per-layer embedding work,
belong to the adaptation program and are written up in
[`ple_RESULTS.md`](../../../results/ablations/ple_RESULTS.md). Its layer-freeing results appear in
section 5 because they bear on layer choice.

**Cost**, used throughout: test bits per byte with the constraint changed at one layer, minus the
model in its native regime. Positive is worse. Unmasking a layer for a temporal model, imposing
residency for a full MoE one. Everything else is defined where it is used.

**Eight files cannot be regenerated.** No producer in any commit on any branch, and their runs kept
neither a router log nor a checkpoint. Claims resting on them say so in the sentence that makes them.
List in [`results/ablations/README.md`](../../../results/ablations/README.md).

## 1. What the router does

A ridge probe asks "does expert *e* fire here?", given either the current token's embedding
(**token AUC**) or the surrounding window with that token excluded (**context AUC**). Held out on
unseen documents, chance 0.5. Their difference, **context minus token**, is positive when surroundings
predict routing better than the token does.

**The constraint destroys the token signal rather than adding a context signal.** Across 34 trained
models the token axis separates the regimes completely, with an empty band 0.184 wide. The context
axis does not separate them at all: several temporal models sit *below* full MoE ones.

<img src="../../../results/phase0/figures/arm_separation.png" alt="Token AUC against context AUC, one point per trained model" width="66%">

*One point per trained model, lines joining a granularity across budgets. Horizontal: how well the
current token predicts firing. Vertical: how well the surrounding context does. Separation is
horizontal only.*

Three checks that this is routing and not a weak probe:

- **No token signal is left to find.** The linear probe reaches 99.4% of the nonparametric ceiling
  from token identity alone in the full MoE regime, and 101.8% in the temporal one.
- **Causal, not correlational.** Swapping a frequency-matched token with context held fixed, against
  the reverse: context-over-token ratio 0.69 full MoE (0.28 to 0.79) against 1.58 temporal (1.25 to 2.18), 29 measurements each, ranges non-overlapping.
- **Not a rare-token artifact.** Token AUC by corpus-frequency stratum runs 0.813 to 0.891 full MoE
  and 0.550 to 0.581 temporal. No band concentrates the signal.

Two consequences, here rather than in section 3 because they describe the router:

- **Demand becomes predictable from history alone.** A causal probe given only routing history
  predicts the next position's demand: 0.981 temporal (0.919 to 0.993, n = 103)
  against 0.655 full MoE (0.567 to 0.716, n = 89). Largest clean separation in the program, with
  daylight between the ranges.
- **The cache hits far above chance.** **Hit rate** is the share of the unconstrained top-k already
  resident before any swap, on a k/E floor of 0.094. It reaches 0.317 temporal coarse and 0.326
  fine, against 0.172 full MoE.

<img src="../../../results/phase0/figures/hitrate_by_layer.png" alt="Cache hit rate by MoE layer" width="66%">

*Hit rate by layer, both regimes, against the k/E floor. Higher is better. Deeper layers cache
better, so a non-uniform memory budget should favour shallow ones.*

### Depth

**Routing grows more contextual with depth, but far more so in the full MoE regime.** Per-arm slopes of
context minus token against layer index, bootstrap intervals:

| regime | arms | median slope per layer | interval excludes zero |
|---|---|---|---|
| full MoE | 5 | +0.0139 | 4 of 5 |
| temporal | 6 | +0.0018 | 3 of 6 |

- **The gap narrows with depth**, roughly 0.39 at the shallowest layer down to 0.27 at the deepest.
  A temporal router is already contextual at layer 2 and has little further to move.
- **Use per-arm slopes, not pooled layers.** Pooling hides it, because the eight-layer arms are flat
  while the thirteen-layer arms rise.
- **This is learned, not mechanical.** Imposing the constraint without training produces no shift at
  all. Section 5.

<img src="../../../results/phase0/figures/locus_by_layer.png" alt="Context minus token AUC against layer" width="66%">

*Broken y-axis: the regime gap of about 0.3 dwarfs the depth effect of about 0.05. Full MoE
curves climb; temporal curves start high and stay flat.*

## 2. What an expert represents

**Each expert covers more of the stream, and routing flattens.** **Participation ratio** is the
inverse Simpson index of an expert's token distribution on a 0 to 1 scale, higher meaning it spreads
further; **generalist fraction** is the share above 0.5, a threshold on the first rather than separate
evidence. Full range across models, median over each model's layers.

| | participation ratio | generalist fraction | router entropy |
|---|---|---|---|
| full MoE, 14 models | 0.201 to 0.422 | 0.000 to 0.328 | 0.790 to 0.917 |
| temporal, 16 models | 0.292 to 0.801 | 0.036 to 0.914 | 0.886 to 0.974 |
| zero-layer control | 0.328 | 0.036 | 0.886 |

- **These ranges overlap**, unlike the locus result. A strong tendency, not a separator.
- **Flattening is strongest early**: generalist fraction falls with depth in both regimes.
- **The third row is a control.** One arm of the section 4 training sweep runs a temporal schedule
  that constrains no layers. Built, trained and counted as temporal; only the constraint is missing.
  It lands at the bottom of the temporal range, with the full MoE models, as it must if these
  statistics measure the constraint rather than how temporal runs are configured. Nothing else here
  would catch a bug that inflated every temporal model equally.

**The inventory is not starved.** Union covers 85 to 100% of the pool on shipped configurations:

| model | budget | regime | experts | union, mean | union, share of E | effective experts |
|---|---|---|---|---|---|---|
| `moe_coarse_1e19` | 1e19 | full MoE | 64 | 63.8 | 0.997 | 59.8 |
| `g3_tmoe_s2_1e17` | 1e17 | temporal | 192 | 160.8 | 0.837 | 187.8 |
| `flame38m_g1_temporal` | 1e18 | temporal | 64 | 62.0 | 0.969 | 63.1 |
| `flame38m_g3_temporal` | 1e18 | temporal | 192 | 163.4 | 0.851 | 187.2 |
| `g1_tmoe_coarse_1e19` | 1e19 | temporal | 64 | 63.9 | 0.999 | 62.5 |
| `temporal_fine_g3_1e19` | 1e19 | temporal | 192 | 184.2 | 0.959 | 183.0 |

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

**Expert weights come out closer to Gaussian.** Excess kurtosis of the weight matrices, where 0 is
Gaussian and higher means heavier tails, so more outliers:

| matched pair | full MoE | temporal |
|---|---|---|
| coarse @1e18, median | 0.42 | **0.14** |
| fine @1e18, median | 0.62 | **0.24** |
| coarse @1e19, median | 0.10 | **0.07** |
| fine @1e18, 99th percentile | 2.79 | **0.77** |

Lower is better if you intend to quantize. The gap is wider in the tail than at the median, which is
where quantization error is decided.

**And that shows up as quantization robustness.** Test bits per byte under fake quantization, lower
is better:

| model | 16-bit | 8-bit | 4-bit | 3-bit | 16 to 3 |
|---|---|---|---|---|---|
| coarse full MoE @1e18 | 1.3158 | 1.3158 | 1.3218 | 1.3520 | +0.0362 |
| coarse temporal @1e18 | 1.3128 | 1.3128 | 1.3176 | 1.3419 | **+0.0291** |
| fine full MoE @1e18 | 1.3462 | 1.3463 | 1.3505 | 1.3705 | +0.0243 |
| fine temporal @1e18 | 1.3354 | 1.3354 | 1.3390 | 1.3562 | **+0.0208** |
| coarse full MoE @1e19 | 1.0510 | 1.0510 | 1.0536 | 1.0662 | +0.0152 |
| coarse temporal @1e19 | 1.0675 | 1.0675 | 1.0697 | 1.0803 | **+0.0128** |

- **Nothing moves at 8 bits** in either regime.
- **The temporal model degrades less in all three matched pairs**, which is the kurtosis result
  cashed out: fewer weight outliers, less quantization error.
- **The two memory levers do not fight.** Rolling residency and low precision compose, which matters
  because they are the obvious things to reach for together.
- One open flag: gradient norms match on median across regimes, but `temporal_coarse_1e19` records a
  maximum of 12.47 against the full MoE 2.52. A single transient or a real interaction, unexamined
  either way.
- **The output side does not replicate.** At 1e18 the temporal model writes sharper distributions
  at 4 of 8 layers on the data-weighted metric, 0 of 8 on the static one, no consistent direction in
  the fine-grained pair.
- **The router does not fragment; it differentiates.** At layer gaps of four or more the
  *full MoE* router is more self-similar, 2.28 times chance against 1.56. Opposite to the
  intuition. Degenerate on 11 of 26 arms where expert count meets or exceeds hidden width, excluded.

## 3. Serving

**The demand signal is the lever. The eviction rule is not.** Both measured by replaying the same
recorded demand through different policies, so only the cache logic varies. Metric is **set
coverage**: of the experts a token wants, the share the resident set already holds when it arrives.
Floor 9.4%.

Two rows below are bounds, not deployable rules. Both need the future, so neither is available at
serving time.

- **Belady.** Evict `argmax_e next(e)`, where `next(e)` is the position of expert *e*'s next demand.
  Optimal for any pure eviction rule.
- **Discounted oracle.** Evict `argmin_e s(e)` where `s(e) = Σ_{i>0} g^i · d_e(t+i)`, and `d_e` is 1
  when *e* is demanded. Small *g* weights the next token or two; `g = 0.95` reaches roughly 20 ahead.

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
- **Smoothing the demand estimate is worth 2.8x**, 0.310 to 0.854. Bigger than every eviction rule
  combined, and detailed below.

<img src="../../../results/phase0/figures/eviction_policy_headroom_belady_bound.png" alt="Eviction policy headroom against the Belady bound" width="66%">

*Set coverage by policy, per shipped configuration. Higher is better. The practical-to-optimal gap is
small next to what a better demand estimate opens.*

- **Smoothing** replaces the per-token demand `d_t` with `d̂_t = β·d_t + (1−β)·d̂_{t−1}`, so one odd
  token cannot evict an expert the stream still wants.
  - `β = 1` is no smoothing; `β = 0.1` averages over roughly the last 10 tokens.
  - Cuts swaps and raises coverage at once, which no eviction rule here manages.

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
- **Hysteresis is not free.** Raising the threshold drives swap rate 1.000 to 0.000, but **retained
  mass**, the share of top-k routing *weight* the resident set holds, falls 0.353 to 0.114.
- **Swap rate is a dead statistic**, median 1.000 across 112 records, because at R = k a swap fires
  whenever any demanded expert is missing. Use 95th-percentile burst length.
- **Block-wise residency is a trade, not a loss.** At block length 72 it holds 28.55 retained mass
  against rolling's 45.76, at 0.12 swaps per token against rolling's ceiling of 1. Worth it if swap
  bandwidth binds. Producerless and unreplayable.

**A bigger cache closes the regime gap**, and the temporal arm leads throughout:

| cache size, K/k | full MoE | temporal |
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
| temporal overlap | 28.4% | 32.9% | 31.2% | 28.3% | 23.5% |

Near three times the floor across a 130-fold size range rather than growing. The one matched
full MoE arm sits at 19.2%.

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
- **At 1e18 the temporal model wins outright**, both granularities: 1.3124 against 1.3175 coarse,
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
times the interior mean. The first is elevated in two, both full MoE models at 1e18.

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
- **The cause is positional, not lexical.** The **sham** swaps the constraint for a Gaussian
  perturbation of the router logits, matched in average magnitude and carrying no lexical information.
  It reproduces 63% of the endpoint excess on the coarse model and 83% on the fine one. Excess is the
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
| adapted on cross-entropy | +0.0932 |
| adapted on cross-entropy, plus per-layer embeddings | +0.0964 |

- **Imposing the constraint moves nothing.** Training under it produces the whole shift.
- **Per-layer embeddings add nothing on top**, 0.0031 against a spread of 0.0031, and in the opposite
  direction to the one pre-registered. That refuted the premise of the embedding program, which had
  argued it works by restoring token information the constraint strips out.
- **All three rows come from one probe.** The published cross-entropy figure of 0.0493 was measured by
  a different probe on the same surface, so pairing it with the 0.096 would manufacture a shift that
  is not there. `ple_RESULTS.md` flags this trap directly.

- **Five further techniques were tried and none moved anything**: stacking per-layer embeddings on
  LoRA (0.49σ, wrong side), calibrated initialisation in three variants (the strongest is 1.23σ
  *worse*), sequential against joint training (0.47σ), and LoRA rank, which stops binding above 128
  (r = 32 is 1.31σ worse). Together with the two nulls above, eight attempts found one thing that
  matters.

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

### What adaptation strategy to use

Seven strategies were run on the same corpus, seed and learning rate. **Recovery** is
`1 − (adapted − base) / (imposed − base)`, so 1.0 fully closes the residency gap and 0 is the
untrained mask. Differences under about 0.003 are eval noise.

| what is trained | recovery @250M | verdict |
|---|---|---|
| router only | 0.707 | the floor for any adaptation |
| router, annealing R from 64 to 8 over the first 150M | 0.708 | **null**, indistinguishable from no anneal |
| router, self-distilled from the frozen free-routing teacher | 0.702 | **null**, if anything slightly worse |
| router + learnable RMSNorm gains | 0.914 | works |
| router + per-expert LoRA, r = 32 | 0.914 | works, ties RMSNorm gains |
| router + LoRA + zone-confined anneal | 0.914 | **null**, the anneal again adds nothing |
| full fine-tune | 0.934 | ceiling |

- **The jump is capacity, not schedule.** Everything that only trains the router lands at 0.70;
  everything that adds any trainable capacity beyond it lands at 0.91. Nothing in between.
- **Annealing the residency limit does nothing**, tried twice, alone and on top of LoRA.
- **Self-distillation from the free-routing teacher does nothing.** The teacher's routing is exactly
  what the constraint makes unavailable, so there is no signal to transfer.
- **Two very different mechanisms tie at 0.914.** RMSNorm gains are a few thousand parameters, LoRA at
  r = 32 is millions. That they land together suggests the binding constraint is having *any* degree
  of freedom outside the router, not how many.
- **Cheap adaptation gets within 0.02 of a full fine-tune.** LoRA rank matters little: r = 8 gives
  0.893 and r = 64 gives 0.910, against 0.914 at r = 32.

### What the downstream evaluations say

Bits per byte is one number. Ten-task zero-shot accuracy agrees with it and shows what the constraint
costs a model that was never trained under it:

| task | constraint imposed, no adaptation | free routing | {0,1,2} | {0,1,15} | {0,1,14,15} |
|---|---|---|---|---|---|
| lambada | 0.000 | 0.706 | 0.564 | 0.595 | 0.577 |
| arc easy | 0.280 | 0.771 | 0.658 | 0.674 | 0.680 |
| hellaswag | 0.257 | 0.586 | 0.471 | 0.477 | 0.485 |
| sciq | 0.293 | 0.937 | 0.914 | 0.925 | 0.934 |
| winogrande | 0.491 | 0.692 | 0.568 | 0.561 | 0.560 |

- **Imposing the constraint untrained is catastrophic, not merely costly.** Lambada goes to zero and
  arc easy falls to near chance. This is the same fact as section 4's +0.4314 BPB, in a currency that
  makes its size obvious.
- **Adaptation recovers about 70% of the gap**, averaged over all ten tasks: 0.675 for {0,1,2}, 0.698
  for {0,1,15}, 0.699 for {0,1,14,15}. Gap closed is (cell − imposed) / (free − imposed), so 1.0 is
  free-routing quality and 0.0 is the untrained mask.
- **The downstream ordering matches the BPB ordering.** Freeing the last layer beats freeing another
  early one, on the aggregate and on 3 of the 5 tasks above.
- **Recovery is uneven by task.** Sciq comes back almost entirely, at 0.925 against 0.937 free.
  Winogrande barely moves off its 0.491 floor, and it is the one task where the free sets do not
  separate.

## 6. What to do with it

- Exempt the first and last MoE layers if you exempt any, on the architectural grounds in section 4,
  not on lexicality. The last layer is among the least token-bound in the stack.
- Do not pick free sets from single-layer ablation. It is wrong on this model in three cells.
- Do not invest in a better demand oracle. A perfect one is worse at fine granularity.
- Spend effort on smoothing the demand estimate, worth 2.8 times, before eviction policy, worth 2.

## 7. How much of this to believe

- Every locus and lens measurement is one training seed per cell.
- Only one full MoE run has a preserved router log, so every regime contrast in section 3, and the
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
