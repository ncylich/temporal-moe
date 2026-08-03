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

Across **34 model arms (16 unconstrained, 18 temporal) at four compute budgets and three granularities (6 of 64, 18 of 192, 30 of 320)**
(`mechinterp_locus{,_1e19}.csv`, median over that arm's experts):

| regime | arms | token AUC | context AUC | experts where context wins |
|---|---|---|---|---|
| unconstrained | 16 | **0.842 – 0.943** | 0.594 – 0.679 | **0 – 3%** |
| temporal | 18 | **0.553 – 0.659** | 0.633 – 0.769 | **85 – 97%** |

**Two of the three statistics separate the regimes completely, with no arm overlapping the other
regime's range.** Token AUC leaves a gap of 0.183 between the lowest unconstrained arm and the highest
temporal one; the share of experts better predicted by context leaves a gap of 82 points. Neither
overlaps at any budget, granularity or router recipe, including the sigmoid and aux-free controls.

**Context AUC alone does overlap**, across 0.633–0.679 — four temporal arms sit below the highest
unconstrained one. That is expected rather than awkward: the context probe finds real signal in both
regimes, so its *level* was never the discriminator. What separates the regimes is which feature wins
the comparison, which is why the token probe and the per-expert contrast are the statistics to quote.

**The probe is not underselling lexicality — it saturates the ceiling.** The obvious objection to
§1 is that a *linear* probe on embeddings cannot express an arbitrary token-to-expert lookup, so a low
token AUC might be probe weakness rather than an absent shortcut. The nonparametric ceiling settles it:
score each expert by the empirical `P(fires | token id)` instead of a probe (`mechinterp_oracle.csv`,
30 arms). Median linear-probe AUC as a fraction of that oracle:

| regime | arms | oracle AUC | probe ÷ oracle |
|---|---|---|---|
| unconstrained | 14 | 0.857 – 0.921 | **101.2%** |
| temporal | 16 | 0.545 – 0.646 | **103.8%** |

The probe reaches the ceiling in both regimes — slightly past it, because the oracle's per-token-id
rates are estimated on the fit half and generalise a little worse than a fitted direction does. So the
regime gap is a real difference in how much of routing token identity determines, not a limit of the
probe. The same thing stated without any classifier: normalised mutual information
`I(expert ; token id) / H(expert)` is **0.416 – 0.574** unconstrained and **0.095 – 0.153** temporal.

**And it is not a rare-token effect.** Splitting the token probe by token-frequency stratum
(`mechinterp_freqstrat.csv`, 26 arms) gives unconstrained AUC of 0.813 in the rarest stratum rising to
0.891 in the common ones, and temporal flat at 0.550 – 0.581 across all five. The shortcut is a
whole-vocabulary property, slightly *stronger* on frequent tokens — the opposite of the plausible guess
that routers memorise rare tokens.

**Strength: high.** The largest, most replicated result in the program, and now bounded above as well
as below. Its main untested exposure is that each cell is one training seed — see §5.

### 1.1 The serving-side consequence: routing demand becomes cacheable

**Claim.** Contextual routing is autocorrelated in time, and that shows up directly as cache
behaviour — the resident set matches what the next token wants far more often than chance, and far
more often than it does for an unconstrained router.

**Evidence.** Hit rate is the fraction of a token's *unconstrained* top-k already resident when it
arrives, measured before the swap. A random resident set scores `k/E` — 0.094 at both 6-of-64 and
18-of-192, so the two granularities are directly comparable (`e6_per_layer_ranking.csv`):

![Cache hit rate by MoE layer](../../../results/phase0/figures/hitrate_by_layer.png)

*Regenerate with `$PY analysis/plots/plot_hitrate_by_layer.py`.*

**The comparison is regime-fair, and here is why.** Every arm on that figure is the same measurement:
take *that model's own* raw router logits, replay the identical rolling-residency policy over them,
and ask how often a token's demand is already resident. Neither regime is handed residency for free.
The constrained model's logits happen to come from a model trained under the policy and the
unconstrained model's from one that was not — that is the difference being measured, and it is the
only one. A constrained model evaluated *without* the policy has no residency to hit, so there is no
third arm to add; the two regimes are the whole comparison.

Reading off the matched pair at 1e19 coarse — the one cell where both regimes have a preserved router
log, 13 MoE layers each:

| regime | first MoE layer | last MoE layer |
|---|---|---|
| unconstrained MoE | 0.117 | 0.250 |
| temporal | 0.114 | 0.424 |

The two start at the same place and diverge with depth: by the last layer the constrained model is
**1.7×** more cacheable. Hit rate rises with depth in *both* regimes, which is what §1's depth trend
predicts — deeper demand is more temporally coherent regardless of training — but it rises much
further under the constraint.

**Strength: medium, limited by sample rather than by method.** Twenty-one constrained arms have a
preserved router log and exactly **one** unconstrained run does, so the vertical gap is indicative
rather than estimated. Replaying more baselines needs no GPU and would fix it. Swap rate is *not*
usable here — at `R = k` it fires whenever any demanded expert is missing, so it saturates at
0.994–1.000 everywhere and carries no signal.

## 2. The shift is causal, not a correlation of probes

**Claim.** Substituting the token a position holds moves the constrained model's expert selection
*less* than shuffling that position's context does; in the unconstrained model the ordering reverses.

**Evidence.** Hold context fixed and substitute a frequency-matched token; then hold the token fixed
and shuffle the surrounding window. Score position *t* only, so token substitution cannot leak into
the context arm through its neighbours. The statistic is the ratio of context-driven to token-driven
change in the selected expert set — above 1 means context dominates (`mechinterp_causal.csv`):

| cell | unconstrained | temporal |
|---|---|---|
| 1e18, 6 of 64, layers 2–9 | 0.30 – 0.73 | 1.34 – 1.66 |
| 1e18, 18 of 192, layers 2–9 | 0.40 – 0.79 | 1.85 – 2.18 |
| 1e19, 6 of 64, layers 2–14 | 0.28 – 0.79 | 1.25 – 1.71 |

**Every unconstrained layer sits below 1 and every temporal layer above it**, in all three cells —
and the two populations do not merely straddle the threshold, they are *separated*. Aggregated over
all 58 layer-measurements: unconstrained spans 0.280–0.794, temporal spans 1.248–2.177, so **the
closest temporal measurement sits 0.453 above the highest unconstrained one**. There is no overlap
across two budgets, two granularities and depths of 9 and 14. The effect decomposes: token sensitivity falls ~42% while context
sensitivity rises ~35%, which no difference of probe AUCs could have separated.

**Strength: high.** This is the claim that makes §1 a mechanism rather than an association.

## 3. What the constraint costs

### 3.1 The cost is a single global quantity, and it is small

**Claim.** Loosening the residency cache buys quality monotonically, and the maximal constraint costs
a fraction of a percent of bits-per-byte.

**Evidence** (`rsweep.csv`, 192 experts, k = 18, 1e16, FLOPs identical at every *R*):

| R | 18 (= k) | 36 | 72 | 128 | 192 (= E) |
|---|---|---|---|---|---|
| test BPB | 1.4750 | 1.4736 | 1.4681 | 1.4580 | 1.4519 |

The whole span is 0.023 BPB for a 10.7× change in resident-expert memory. At larger budgets the sign
flips: at 1e18 the temporal model *beats* its matched baseline at both granularities — 3.9094 against
3.9184 test CE coarse, 3.9768 against 4.0087 fine (`flame38m_1e18_cells.csv`).

**Strength: high**, and it predates this program — the dose curve is the original result, reproduced.

### 3.2 Per-layer structure exists at the endpoints; *lexicality* does not explain it

**Claim.** Which layer you constrain matters, and the layers that matter are the first and last MoE
layers. What fails is the *lexical* explanation for why — the endpoint effect is not where routing is
most token-driven, and is largely reproduced by a perturbation carrying no lexical information at all.

An earlier version of this section claimed there was **no** per-layer structure worth exploiting. That
was wrong, and the PLE evidence below contradicts it directly.

#### The endpoint structure is real and is worth memory

Freeing layers from the residency constraint in an adapted OLMoE (`ple_ladder.csv`, R=8 residency,
50M adaptation tokens, BPB lower is better):

| free set | resident memory | BPB | vs full residency |
|---|---|---|---|
| none (CE-adapted, full residency) | baseline | 0.8147 | — |
| first two MoE layers `{0,1}` | +87.5% | 0.8144 | −0.0003 |
| `{0,1}` **plus the last layer** | +131.2% | 0.7978 | **−0.0169** |

**Freeing the first two layers buys essentially nothing. Adding the last layer buys 0.0169 BPB.**
That is per-layer structure, it is concentrated at an endpoint, and it is large next to the whole
global dose curve (§3.1 spans 0.023 BPB end to end).

> **Provenance warning.** Only these two `ce_free_*` cells are committed. A larger ladder exists —
> including `{0,1,2}`, `{0,1,14,15}` and 250M-token variants — and **is not in the repository**. The
> `{0,1,2}` cell is the one that would isolate *last layer* from *one more layer*, at matched memory,
> and it is the single most valuable missing number in this document. Until it lands, the claim above
> rests on two cells.

#### But lexicality is not the reason

Three separate readings were pre-registered on the lexical explanation and all three failed:

*Perturbing a trained checkpoint one layer at a time* gives a U-shaped cost profile — first and last
MoE layers at 1.4–2.5× the interior mean (`swap_sweep.csv`, `swap_shape.csv`). It is not lexical: the
last MoE layer is the **least** lexical in the stack by the §1 probe while being the most expensive to
constrain, which a lexical account predicts backwards.

*A magnitude-matched perturbation carrying no lexical information* reproduces **58%** of the endpoint
excess on one model and **85%** on the model with the largest effect (`sham_magnitude_matched.csv`), so
most of the endpoint cost is sensitivity to position rather than to what the layer routes on.

![Real versus magnitude-matched sham, and the residual](../../../results/phase0/figures/sham_residual.png)

*Regenerate with `$PY analysis/plots/plot_sham_residual.py`.*

**The residue is not negligible, and it is itself an endpoint effect.** Real minus matched-sham, per
layer: L2 **+0.022**, L3 −0.069, L4 −0.039, L5 −0.019, L6 +0.013, L7 −0.012, L8 +0.012, L9 **+0.095**.
The interior residues are near zero or negative; the two positive outliers are the two endpoints, and
L9's is four times any other. So "58–85% positional" does **not** license ignoring the rest — the
15–42% that a generic perturbation fails to reproduce is concentrated exactly where the effect is.
What that residue is remains unexplained.

#### And it did not survive from-scratch training on a small testbed

**Testbed** — shape `s0` at 1e16 (the cheapest cell with a preserved reference): hidden 128,
**4 transformer layers of which layer 1 is dense, so only 3 MoE layers** (2, 3, 4), 192 experts,
top-18, 16k vocabulary, ~65 min per run. Eight arms × three seeds (1234, 2, 3) = 24 runs,
`t1_perlayer_training.csv`. Arm means, sorted by loss:

| arm | constrained layers | mean test CE | sd over 3 seeds |
|---|---|---|---|
| A0 | none | 4.0182 | 0.0128 |
| A3 | {3} | 4.0208 | 0.0078 |
| A4 | {4} | 4.0335 | 0.0049 |
| A2 | {2} | 4.0349 | 0.0108 |
| A6 | {3,4} — exempt first | 4.0475 | 0.0044 |
| A5 | {2,3} — exempt last | 4.0492 | 0.0036 |
| A7 | uniform R=76, matched memory to A5/A6 | 4.0572 | 0.0204 |
| A1 | {2,3,4} — all | 4.0601 | 0.0047 |

**The U survives here too, and an earlier version of this section said otherwise because it tested
the wrong contrast.** Among the three single-layer arms the middle layer is the cheapest to constrain
and both endpoints cost more — **in all three seeds**. The contrast a U predicts is endpoints against
middle: **+0.0134 CE, se 0.0048, t = 2.80**. Not significant at two degrees of freedom (critical value
4.30), but in the predicted direction and consistent across every seed.

What the earlier version reported instead was *first versus last* (+0.0014, 0.2 se) and read its
nullity as evidence against the U. A U-shape predicts first ≈ last. That contrast being null
**confirms the shape's symmetry**; it says nothing against its existence.

**The testbed is still weak evidence about depth.** Three MoE layers is barely a depth axis — "first",
"middle" and "last" are layers 2, 3 and 4 of the same small model at the smallest budget in the
program. What it shows is that the ordering survives from-scratch training in the one cell where that
was affordable to test.

**Strength: the shape is supported three independent ways; the explanation is not.** The endpoint
structure shows up in the inference-time profile, in the PLE free-set ladder on a 16-layer adapted
model, and in the ordering of the from-scratch training arms — three settings, three methods, same
shape. What is refuted is the *lexical* account of it: the endpoint layers are not the token-driven
ones, and a perturbation carrying no lexical information reproduces most of the effect.

The two things that would settle what remains: the missing `{0,1,2}` PLE cell, which separates
"the last layer" from "one more layer" at matched memory, and a from-scratch sweep on a model deep
enough to have a real interior — the 9-layer 1e18 panel rather than three MoE layers at 1e16.

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
causal test (§2) are unaffected — they use different fields of the capture.

### 4.2 It does not reduce seed-to-seed variance

**Claim.** An apparent variance reduction under the constraint does not survive scrutiny.

**Evidence.** Within the 24-run training sweep, spread falls with the number of constrained layers
(ρ = −0.857, permutation p = 0.023). But the headline F statistic that suggested it was the single
most extreme of 420 possible groupings, chosen after seeing the data; and the only matched pair at
another budget contradicts it — at 1e18, over three seeds each, the constrained arm spans **0.0031**
BPB against the unconstrained arm's **0.0028**, so the constrained model is marginally *more* variable
there, not less. (Earlier drafts gave 0.00165 against 0.00061; those were computed over two of the
three seeds, which reversed nothing but overstated the gap as 2.7× where it is 1.1×.)

**Strength: not supported.** Recorded so it is not rediscovered.

## 5. Confidence, and what cannot be measured

**Replicated.** The regime separation in §1 and §2 holds across four budgets, three granularities (6 of 64, 18 of 192, 30 of 320) and
several router recipes. The training result in §3.2 is three seeds per arm.

**Not replicated.** *Every locus and lens measurement is one training seed per cell.* The bootstrap
intervals describe sampling over experts within a model, not variation between training runs. Given
that §3.2 watched four one-seed claims die — three null, one sign-flipped — this is the program's
main untested exposure. Eight seed-replicate checkpoints exist at 1e18 and have never been captured;
closing this costs about two hours of forward passes and no training.

**Depth trends are weaker than levels.** The contextual share rises with depth, but it rises in the
unconstrained baseline too, so part of the trend belongs to the transformer rather than to the
constraint. Curve *shape* statistics — slopes, curvature, vertices — are fragile: on 3–5 layer models
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
2. **A from-scratch per-layer sweep on a model with an interior.** T1 ran on three MoE layers. Anything
   about depth needs at least the 9-layer 1e18 panel, where "first", "middle" and "last" are distinct.
3. **What is the residue the sham does not reproduce?** It is 15–42% of the endpoint cost, concentrated
   at the last layer (+0.095 against +0.022 at the first, near zero in the interior), and unexplained.
4. **Does de-lexicalization explain the loss advantage?** §1 shows the constraint changes routing and
   §3.1 shows it eventually helps, and nothing joins them. Across matched pairs, does a model's
   contextual share predict its advantage over its baseline? Small *n*, correlational, and free —
   the data is already committed.
5. **Is the depth curve seed-stable?** See §5. Eight capture passes, two hours, no training.

