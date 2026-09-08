# What rolling residency does to MoE routing

**The state of knowledge, by claim.** Every number here was read from a committed CSV at the time of
writing, and each claim names the file it came from. Metric definitions, controls and the identifiers
used by the analysis scripts are in [`03-methods.md`](03-methods.md); what changed against the
published write-up is in [`02-corrections.md`](02-corrections.md); the chronological record of how
this was found, including what turned out to be wrong, is in [`05-notebook.md`](05-notebook.md).

Nothing superseded is preserved here. Where an earlier document said something different, the
difference is recorded in the corrections document or the archive, not in this one.

---

## 0. The setup, so this stands alone

A mixture-of-experts layer routes each token to its top-*k* experts out of *E*. **Rolling residency**
constrains that choice: only *R* experts are resident at a time, the router picks its top-*k* from
among residents, and at most one expert is swapped in per token. `R = k` is the maximal constraint;
`R = E` recovers an unconstrained MoE exactly. FLOPs are identical at every *R*, so the constraint
trades routing freedom against nothing but serving memory.

Two regimes are compared throughout: **temporal** (trained under rolling residency) and
**unconstrained** (an ordinary MoE of identical architecture, data, compute and seed). Both are
trained from scratch; nothing here is a fine-tune.

---

## 1. The constraint moves routing from token identity to context

**Claim.** An unconstrained MoE router implements a near-deterministic token-to-expert lookup. Under
rolling residency that lookup becomes unrepresentable, and expert selection reorganises around the
surrounding context instead.

**Evidence.** For each (layer, expert) pair, two ridge probes predict whether that expert serves a
token, from either the current token's embedding or the mean of its neighbours with the current token
excluded. Held-out AUC, documents disjoint between fit and score halves, window = one residency
lifetime. Measured chance floor 0.500; worst deviation across 1,162 null fits is 0.0030 under iid
permutation (`mechinterp_floors{,_1e19}.csv`).

![Regime separation, one point per trained model](../../../results/phase0/figures/arm_separation.png)

*Regenerate with `$PY analysis/plots/plot_arm_separation.py`.*

Across 34 model arms, 16 unconstrained and 18 temporal, at four compute budgets and three
granularities of 6 of 64, 18 of 192 and 30 of 320 (`mechinterp_locus{,_1e19}.csv`, median over each
arm's experts):

| regime | arms | token AUC | context AUC | experts where context wins |
|---|---|---|---|---|
| unconstrained | 16 | 0.842 to 0.943 | 0.594 to 0.679 | 0 to 3% |
| temporal | 18 | 0.553 to 0.659 | 0.633 to 0.769 | 85 to 97% |

Two of the three statistics separate the regimes completely, with no arm overlapping the other
regime's range. Token AUC leaves a gap of 0.183 between the lowest unconstrained arm and the highest
temporal one; the share of experts better predicted by context leaves a gap of 82 points. Neither
overlaps at any budget, granularity or router recipe, including the sigmoid and aux-free controls.

Context AUC alone does overlap, across 0.633 to 0.679, and four temporal arms sit below the highest
unconstrained one. That is expected rather than awkward. The context probe finds real signal in both
regimes, so its *level* was never the discriminator. What separates the regimes is which feature wins
the comparison, which is why the token probe and the per-expert contrast are the statistics to quote.

**The probe is not underselling lexicality: it saturates the ceiling.** The obvious objection to
§1 is that a *linear* probe on embeddings cannot express an arbitrary token-to-expert lookup, so a low
token AUC might be probe weakness rather than an absent shortcut. The nonparametric ceiling settles it:
score each expert by the empirical `P(fires | token id)` instead of a probe (`mechinterp_oracle.csv`,
30 arms). Median linear-probe AUC as a fraction of that oracle:

| regime | arms | oracle AUC | probe ÷ oracle |
|---|---|---|---|
| unconstrained | 14 | 0.857 to 0.921 | 101.2% |
| temporal | 16 | 0.545 to 0.646 | 103.8% |

The probe reaches the ceiling in both regimes, slightly past it in fact, because the oracle's
per-token-id rates are estimated on the fit half and generalise a little worse than a fitted direction
does. So the regime gap is a real difference in how much of routing token identity determines, not a
limit of the
probe. The same thing stated without any classifier: normalised mutual information
`I(expert ; token id) / H(expert)` is 0.416 to 0.574 unconstrained and 0.095 to 0.153 temporal.

**And it is not a rare-token effect.** Splitting the token probe by token-frequency stratum
(`mechinterp_freqstrat.csv`, 26 arms) gives unconstrained AUC of 0.813 in the rarest stratum rising to
0.891 in the common ones, and temporal flat at 0.550 to 0.581 across all five. The shortcut is a
whole-vocabulary property, and slightly stronger on frequent tokens, which is the opposite of the
plausible guess that routers memorise rare tokens.

**Strength: high.** The largest, most replicated result in the program, and now bounded above as well
as below. Its main untested exposure is that each cell is one training seed. See §5.

### 1.1 The shift is causal, not an artefact of the probes

**Claim.** Substituting the token a position holds moves the constrained model's expert selection
*less* than shuffling that position's context does; in the unconstrained model the ordering reverses.

**Evidence.** Hold context fixed and substitute a frequency-matched token; then hold the token fixed
and shuffle the surrounding window. Score position *t* only, so token substitution cannot leak into
the context arm through its neighbours. The statistic is the ratio of context-driven to token-driven
change in the selected expert set: above 1 means context dominates (`mechinterp_causal.csv`):

| cell | unconstrained | temporal |
|---|---|---|
| 1e18, 6 of 64, layers 2 to 9 | 0.30 to 0.73 | 1.34 to 1.66 |
| 1e18, 18 of 192, layers 2 to 9 | 0.40 to 0.79 | 1.85 to 2.18 |
| 1e19, 6 of 64, layers 2 to 14 | 0.28 to 0.79 | 1.25 to 1.71 |

**Every unconstrained layer sits below 1 and every temporal layer above it**, in all three cells,
and the two populations do not merely straddle the threshold, they are *separated*. Aggregated over
all 58 layer-measurements: unconstrained spans 0.280 to 0.794, temporal spans 1.248 to 2.177, so **the
closest temporal measurement sits 0.453 above the highest unconstrained one**. There is no overlap
across two budgets, two granularities and depths of 9 and 14. The effect decomposes: token
sensitivity falls ~42% while context
sensitivity rises ~35%, which no difference of probe AUCs could have separated.

**Strength: high.** This is the claim that makes §1 a mechanism rather than an association.

### 1.2 The same split shows up in what experts are, and in the geometry of the router

Two more measurements point the same way as the probes, from different directions.

**Experts become generalists** (`mechinterp_structural{,_1e19}.csv`, 30 arms; a *generalist* draws its
usage from more than half the token stream):

| regime | arms | generalist % at first MoE layer | at last | router entropy, first → last |
|---|---|---|---|---|
| unconstrained | 14 | 4.7% | 5.0% | 0.890 → 0.881 |
| temporal | 16 | 75.3% | 65.9% | 0.957 → 0.940 |

A fifteen-fold gap in generalist fraction, roughly stable with depth, on flatter routing throughout.
*One caveat that matters for §2 of the corrections document*: the matched 1e19 coarse pair is not
typical here. Both its regimes are near-total generalists at layer 2 and they diverge only with depth. The
"indistinguishable through layer 4" statement recorded there is true of that pair and not of the
population.

**And the router's own geometry differs.** Each layer's token probe spans a subspace of embedding
directions that layer is sensitive to; comparing those subspaces across layers asks whether routing is
the same function at every depth (`mechinterp_transfer.csv`). Excluding 11 of 26 arms where the
statistic is degenerate (when the number of experts meets or exceeds the hidden size, the column space
spans everything and the measure carries no information), cross-layer overlap relative to chance is:

| regime | arms | overlap ÷ chance | IQR |
|---|---|---|---|
| unconstrained | 15 non-degenerate | 2.18× | 1.30 to 3.34 |
| temporal | | 1.60× | 1.26 to 2.26 |

The *unconstrained* router is the more self-similar one across depth, which is what a token lookup
predicts: the same map at every layer points its probes in the same directions. Constrained routing
uses whatever context is available at each depth, and that differs layer to layer.

## 2. How routing behaves over time, and what that buys serving

**Claim.** Contextual routing is autocorrelated in time, and that shows up directly as cache
behaviour. The resident set matches what the next token wants far more often than chance, and far more
often than it does for an unconstrained router.

**Evidence.** Hit rate is the fraction of a token's *unconstrained* top-k already resident when it
arrives, measured before the swap. A random resident set scores `k/E`, 0.094 at both 6-of-64 and
18-of-192, so the two granularities are directly comparable (`e6_per_layer_ranking.csv`):

![Cache hit rate by MoE layer](../../../results/phase0/figures/hitrate_by_layer.png)

*Regenerate with `$PY analysis/plots/plot_hitrate_by_layer.py`.*

**The comparison is regime-fair, and here is why.** Every arm on that figure is the same measurement:
take *that model's own* raw router logits, replay the identical rolling-residency policy over them,
and ask how often a token's demand is already resident. Neither regime is handed residency for free.
The constrained model's logits happen to come from a model trained under the policy and the
unconstrained model's from one that was not: that is the difference being measured, and it is the
only one. A constrained model evaluated *without* the policy has no residency to hit, so there is no
third arm to add; the two regimes are the whole comparison.

Reading off the matched pair at 1e19 coarse: the one cell where both regimes have a preserved router
log, 13 MoE layers each:

| regime | first MoE layer | last MoE layer |
|---|---|---|
| unconstrained MoE | 0.117 | 0.250 |
| temporal | 0.114 | 0.424 |

The two start at the same place and diverge with depth: by the last layer the constrained model is
1.7× more cacheable. Hit rate rises with depth in *both* regimes, which is what §1's depth trend
predicts, since deeper demand is more temporally coherent regardless of training. It rises much further
under the constraint.

**Strength: medium, limited by sample rather than by method.** Twenty-one constrained arms have a
preserved router log and exactly **one** unconstrained run does, so the vertical gap is indicative
rather than estimated. Replaying more baselines needs no GPU and would fix it. Swap rate is *not*
usable here, because at `R = k` it fires whenever any demanded expert is missing, so it saturates at
0.994 to 1.000 everywhere and carries no signal.

### 2.1 Next-token demand becomes almost perfectly predictable from history

**Claim.** For a constrained model you can tell which experts the *next* token will want using nothing
but the recent demand history, with no embeddings and no hidden states.

**Evidence** (`mechinterp_demand_1e19.csv`, 192 layer-measurements over 30 arms; a causal
history-only probe, AUC):

| regime | measurements | demand AUC |
|---|---|---|
| unconstrained | 89 | 0.567 to 0.716 (median 0.655) |
| temporal | 103 | 0.919 to 0.993 (median 0.981) |

**No overlap, and the ranges are not close.** This is the mechanism behind §2's cache numbers stated
directly: the constraint does not merely make demand *cacheable*, it makes it *forecastable*, which is
what a prefetcher needs. It is also a much larger separation than the 0.85-against-0.64 recorded in the
original write-up, which was a single pooled pair.

### 2.2 The constraint does not starve experts

**Claim.** Restricting each token to a resident set does not collapse expert usage. The model still
touches nearly all of them, just not simultaneously.

**Evidence** (`e2_streamed_diversity.csv`, union of distinct experts touched per sequence): the plain
constrained recipes reach 85 to 100% of E with effective-expert counts of 183 to 188 out of 192 and
62 to 63 out of 64. The low-diversity arms in that file (union 0.13 to 0.66) are all *selection-shaping*
variants (anticipatory, bursty and head-gated losses), which are a different experiment and not the
shipped recipe.

This is the answer to the obvious worry about a one-swap-per-token cache: it does not turn a
192-expert model into a 24-expert one. The resident *set* is small; the expert *inventory* in use is
not.

### 2.3 Eviction policy is nearly exhausted as a lever; demand smoothing is not

Two levers on the same cache, and they behave very differently
(`e5_eviction_policy_headroom.csv`, `e7_demand_smoothing.csv`, set coverage, higher is better):

| lever | setting | set coverage |
|---|---|---|
| eviction rule | LRU | 0.251 |
| | `min_logit` (shipped) | 0.310 |
| | Belady, offline optimal | 0.371 |
| demand smoothing | β = 1 (none, shipped) | 0.310 |
| | β = 0.25 | 0.640 |
| | β = 0.1 | 0.854 |

Changing which expert you evict is worth at most 20%, and the shipped heuristic already sits within
that of an offline oracle that can see the future. Smoothing the demand signal is worth 2.8x, and it
drops swap rate from 1.000 to 0.926, so it costs less bandwidth rather than more.

That is the clearest serving-side result in the program, and it points at the demand signal rather
than the cache policy as the place to spend effort.

*One caveat worth keeping.* The same file reports `discounted-oracle` and `min_logit+tau` variants
scoring above Belady, which should be impossible if Belady were optimal for this objective. The
likely explanation is that Belady is implemented as evict-farthest-next-*demand*, which is not optimal
for set coverage under a changing demand stream. But that is inference from the numbers rather than a
read of the implementation, so treat the exact Belady value as unverified. The headroom conclusion
does not depend on it, since `min_logit` beats LRU and every tau variant above 1.0.

### 2.4 Document boundaries are not a confound

Hit rate barely changes across an end-of-document token: median deficit **+0.009**, range −0.053 to
+0.122 over 66 measurements (`e8_document_boundary.csv`). The temporal locality that makes the cache
work is not an artefact of documents being concatenated in the eval stream. It survives inside them and
across them.

**Strength of §2 overall: medium-to-high on the constrained side, weak on the comparison.** Every
statement above is measured on 21 to 22 arms, but only one unconstrained run has a preserved router log,
so regime *contrasts* in this section rest on a single baseline. Replaying more baselines needs no GPU.

## 3. What the constraint costs

### 3.1 The cost is a single global quantity, and it is small

**Claim.** Loosening the residency cache buys quality monotonically, and the maximal constraint costs
a fraction of a percent of bits-per-byte.

**Evidence** (`rsweep.csv`, 192 experts, k = 18, 1e16, FLOPs identical at every *R*):

| R | 18 (= k) | 36 | 72 | 128 | 192 (= E) |
|---|---|---|---|---|---|
| test BPB | 1.4750 | 1.4736 | 1.4681 | 1.4580 | 1.4519 |

The whole span is 0.023 BPB for a 10.7× change in resident-expert memory. At larger budgets the sign
flips: at 1e18 the temporal model *beats* its matched baseline at both granularities, 3.9094 against
3.9184 test CE coarse, 3.9768 against 4.0087 fine (`flame38m_1e18_cells.csv`).

**Strength: high**, and it predates this program. The dose curve is the original result, reproduced.

### 3.2 Per-layer structure exists at the endpoints; *lexicality* does not explain it

**Claim.** Which layer you constrain matters, and the layers that matter are the first and last MoE
layers. What fails is the *lexical* explanation for why. The endpoint effect is not where routing is
most token-driven, and is largely reproduced by a perturbation carrying no lexical information at all.

#### The endpoint structure is real and is worth memory

Freeing layers from the residency constraint in an adapted OLMoE (`ple_ladder.csv`, R=8 residency,
50M adaptation tokens, BPB lower is better):

| free set | resident memory | BPB | vs full residency |
|---|---|---|---|
| none (CE-adapted, full residency) | baseline | 0.8147 | n/a |
| first two MoE layers `{0,1}` | +87.5% | 0.8144 | −0.0003 |
| `{0,1}` plus the last layer | +131.2% | 0.7978 | −0.0169 |

**Freeing the first two layers buys essentially nothing. Adding the last layer buys 0.0169 BPB.**
That is per-layer structure, it is concentrated at an endpoint, and it is large next to the whole
global dose curve (§3.1 spans 0.023 BPB end to end).

> **Resolved 2026-08-03, and it confirms the claim.** The `{0,1,2}` cell — called above the single
> most valuable missing number here, because it isolates *last layer* from merely *one more layer* at
> matched memory — was run, along with `{0,1,14,15}` and 250M-token variants. All are on
> `ple-adaptation` in `layer_freeing_results.csv`; see its `layer_freeing_RESULTS.md` §7.
>
> | free set | resident memory | BPB | vs `{0,1}` |
> |---|---|---|---|
> | `{0,1}` | +87.5% | 0.814440 | — |
> | `{0,1,2}` — one more *early* layer | +131.2% | 0.808615 | −0.0058 |
> | `{0,1,15}` — the *last* layer | +131.2% | 0.797810 | −0.0166 |
> | `{0,1,14,15}` — both endpoints | +175.0% | 0.786275 | −0.0282 |
>
> At identical memory the last layer is worth **0.0108 BPB more** than an additional early one, so the
> gain really is endpoint structure and not just a third freed layer. Freeing both endpoints buys more
> again. The claim now rests on five cells rather than two.
>
> It also refutes the training-free damage profile that predicted the opposite: that profile made
> layer 2 worth 5.8× layer 15 as the third freed layer. Do not choose free sets from solo damage.

#### At sixteen layers the profile is a clean U

The highest-resolution version of this is a train-free sweep on the adapted OLMoE: impose residency on
exactly one layer and measure the damage in BPB (`layer_damage.csv`, 16 layers, all-free baseline
0.6727):

| layer | 0 | 1 | 2 | 5 | 8 | 11 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| damage (BPB) | 0.218 | 0.259 | 0.141 | 0.099 | 0.081 | 0.070 | 0.075 | 0.122 | 0.141 |

The minimum sits at layer 11, about 0.73 of the way down, and both ends rise away from it. The two
outermost layers at each end average 1.98x the interior mean, and layer 1 alone costs 3.7x the
cheapest layer. Constraining every layer costs 2.078, which is 1.12x the sum of the individual
damages, so the layers interact mildly rather than adding.

This is the strongest version of the shape in the program: sixteen layers rather than eight or three,
on a real pretrained model, and it needs no training at all. It agrees with the free-set ladder above,
which is its complement (that sweep frees layers and measures the gain; this one constrains them and
measures the loss), and with the vertex near two thirds depth seen at 1e18.

*Provenance, resolved 2026-08-03.* The producer was not missing, it was on another branch:
`analysis/residency/layer_ablation.py` on `ple-adaptation`, with `analysis/residency/joint_free.py` for
`joint_free.csv`. Its protocol is documented and audited — the base model evaluated 18 times with no
training (all sixteen layers free, then layer *i* alone constrained for *i* = 0..15, then all
constrained), and both anchors reproduce their published references to six decimals (0.672736 against
0.6727 free, 2.750704 against 2.7507 imposed), which is what licenses the sixteen numbers between
them. **Treat these values as audited.**

`ple_ladder.csv` is the one exception and the caveat still holds for it: its producer,
`analysis/residency/ladder_report.py`, was deleted by `ple-adaptation`'s consolidation commit `2a7fc14`
("one results CSV, 17 scripts from 24"). Its contents are derivable from `ple_results.csv` on that
branch, which keeps the primitives — per-cell `final_bpb` plus the reference rows — from which every
gate and verdict column in it was computed.

#### But lexicality is not the reason

Three separate readings were pre-registered on the lexical explanation and all three failed:

*Perturbing a trained checkpoint one layer at a time* gives a U-shaped cost profile: first and last
MoE layers at 1.4 to 2.5× the interior mean (`swap_sweep.csv`, `swap_shape.csv`). It is not lexical: the
last MoE layer is the **least** lexical in the stack by the §1 probe while being the most expensive to
constrain, which a lexical account predicts backwards.

*A magnitude-matched perturbation carrying no lexical information* reproduces 58% of the endpoint
excess on one model and 85% on the model with the largest effect (`sham_magnitude_matched.csv`), so
most of the endpoint cost is sensitivity to position rather than to what the layer routes on.

![Real versus magnitude-matched sham, and the residual](../../../results/phase0/figures/sham_residual.png)

*Regenerate with `$PY analysis/plots/plot_sham_residual.py`.*

**The residue is not negligible, and it is itself an endpoint effect.** Real minus matched-sham, per
layer: L2 **+0.022**, L3 −0.069, L4 −0.039, L5 −0.019, L6 +0.013, L7 −0.012, L8 +0.012, L9 **+0.095**.
The interior residues are near zero or negative; the two positive outliers are the two endpoints, and
L9's is four times any other. So "58 to 85% positional" does **not** license ignoring the rest, the
15 to 42% that a generic perturbation fails to reproduce is concentrated exactly where the effect is.
What that residue is remains unexplained.

#### From-scratch training on a small testbed agrees, weakly

**Testbed**, shape `s0` at 1e16 (the cheapest cell with a preserved reference): hidden 128,
**4 transformer layers of which layer 1 is dense, so only 3 MoE layers** (2, 3, 4), 192 experts,
top-18, 16k vocabulary, ~65 min per run. Eight arms × three seeds (1234, 2, 3) = 24 runs,
`t1_perlayer_training.csv`. Arm means, sorted by loss:

| constrained layers | mean test CE | sd over 3 seeds |
|---|---|---|
| none | 4.0182 | 0.0128 |
| {3} | 4.0208 | 0.0078 |
| {4} | 4.0335 | 0.0049 |
| {2} | 4.0349 | 0.0108 |
| {3,4}, exempt first | 4.0475 | 0.0044 |
| {2,3}, exempt last | 4.0492 | 0.0036 |
| uniform R=76, matched memory to the two exemption arms | 4.0572 | 0.0204 |
| {2,3,4}, all | 4.0601 | 0.0047 |

**The U survives training too.** Among the three single-layer arms the middle layer is the cheapest
to constrain
and both endpoints cost more, **in all three seeds**. The contrast a U predicts is endpoints against
middle: **+0.0134 CE, se 0.0048, t = 2.80**. Not significant at two degrees of freedom (critical value
4.30), but in the predicted direction and consistent across every seed.

The contrast to avoid here is *first versus last* (+0.0014, 0.2 se). A U-shape predicts that its two
endpoints match, so that particular null confirms the shape's symmetry and says nothing about whether
the shape exists.

**The testbed is still weak evidence about depth.** Three MoE layers is barely a depth axis, "first",
"middle" and "last" are layers 2, 3 and 4 of the same small model at the smallest budget in the
program. What it shows is that the ordering survives from-scratch training in the one cell where that
was affordable to test.

**Strength: the shape is supported three independent ways; the explanation is not.** The endpoint
structure shows up in the inference-time profile, in the PLE free-set ladder on a 16-layer adapted
model, and in the ordering of the from-scratch training arms: three settings, three methods, same
shape. What is refuted is the *lexical* account of it: the endpoint layers are not the token-driven
ones, and a perturbation carrying no lexical information reproduces most of the effect.

The two things that would settle what remains: the missing `{0,1,2}` PLE cell, which separates
"the last layer" from "one more layer" at matched memory, and a from-scratch sweep on a model deep
enough to have a real interior: the 9-layer 1e18 panel rather than three MoE layers at 1e16.

## 4. What the constraint does not do

### 4.1 It does not blunt what experts write

**Claim.** The published finding that constrained experts promote near-uniform vocabulary
distributions does not replicate.

**Evidence.** Data-weighted logit-lens effective vocabulary, re-measured at full depth on captures
taken after a layer-keying defect was fixed (`mechinterp_lens_1e19.csv`; the defect and its blast
radius are in [`02-corrections.md`](02-corrections.md)). At 1e18 the constrained model writes
*sharper* distributions than the unconstrained one at 4 of 8 layers in the coarse pair, and the fine
pair shows no consistent direction. The defensible statement is that **the output-side regime
difference does not replicate**, not that it reverses.

**Strength: the negative is solid; the original positive is withdrawn.** The input side (§1) and the
causal test (§2) are unaffected, they use different fields of the capture.

### 4.2 It does not reduce seed-to-seed variance

**Claim.** An apparent variance reduction under the constraint does not survive scrutiny.

**Evidence.** Within the 24-run training sweep, spread falls with the number of constrained layers
(ρ = −0.857, permutation p = 0.023). But the headline F statistic that suggested it was the single
most extreme of 420 possible groupings, chosen after seeing the data; and the only matched pair at
another budget contradicts it, at 1e18, over three seeds each, the constrained arm spans 0.0031
BPB against the unconstrained arm's 0.0028, so the constrained model is marginally *more* variable
there, not less. (Earlier drafts gave 0.00165 against 0.00061; those were computed over two of the
three seeds, which reversed nothing but overstated the gap as 2.7× where it is 1.1×.)

**Strength: not supported.** Recorded so it is not rediscovered.

## 5. Confidence, and what cannot be measured

**Replicated.** The regime separation in §1 and §2 holds across four budgets, three granularities (6
of 64, 18 of 192, 30 of 320) and
several router recipes. The training result in §3.2 is three seeds per arm.

**Not replicated.** *Every locus and lens measurement is one training seed per cell.* The bootstrap
intervals describe sampling over experts within a model, not variation between training runs. Given
that §3.2 watched four one-seed claims die: three null, one sign-flipped, this is the program's
main untested exposure. Eight seed-replicate checkpoints exist at 1e18 and have never been captured;
closing this costs about two hours of forward passes and no training.

**Depth trends are weaker than levels.** The contextual share rises with depth, but it rises in the
unconstrained baseline too, so part of the trend belongs to the transformer rather than to the
constraint. Curve *shape* statistics, slopes, curvature, vertices, are fragile: on 3 to 5 layer models
the quadratic vertex is unidentified, and a full-range linear slope on a curve that turns over
reverses the comparison it appears to make.

**Permanently unmeasurable.** Four of the five runs behind the published 1e16/1e17 locus numbers, and
the five behind the published replay numbers, are absent from `MANIFEST.csv` and from disk. Those
cells cannot be extended, re-split or re-windowed without retraining. No coarse 6-of-64 checkpoint
survives at either 1e16 or 1e17, so that cell type is missing at the low end entirely.

## 6. Open questions

1. **The missing PLE cell.** `{0,1,2}` at matched memory against `{0,1,15}` is the one number that
   would separate "freeing the *last* layer helps" from "freeing *one more* layer helps". Nine cells of
   that ladder exist outside the repository; committing them is free and closes §3.2's main gap.
2. **A from-scratch per-layer sweep on a model with an interior.** The training sweep ran on three MoE
   layers. Anything
   about depth needs at least the 9-layer 1e18 panel, where "first", "middle" and "last" are distinct.
3. **What is the residue the sham does not reproduce?** It is 15 to 42% of the endpoint cost, concentrated
   at the last layer (+0.095 against +0.022 at the first, near zero in the interior), and unexplained.
4. **Does de-lexicalization explain the loss advantage?** §1 shows the constraint changes routing and
   §3.1 shows it eventually helps, and nothing joins them. Across matched pairs, does a model's
   contextual share predict its advantage over its baseline? Small *n*, correlational, and free: the
data is already committed.
5. **Is the depth curve seed-stable?** See §5. Eight capture passes, two hours, no training.

