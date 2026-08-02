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

**Strength: high.** The largest, most replicated result in the program. Its main untested exposure is
that each cell is one training seed — see §5.

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

### 3.2 There is no per-layer structure worth exploiting

**Claim.** Neither lexicality nor architectural position identifies layers where the constraint is
cheap enough to justify a per-layer schedule. Three separate hypotheses were pre-registered and all
three failed.

**Evidence, in the order it accumulated:**

*Perturbing a trained checkpoint one layer at a time* produces a U-shaped cost profile — the first and
last MoE layers are 1.4–2.5× the interior mean (`swap_sweep.csv`, `swap_shape.csv`). That profile is
not about lexicality: the last MoE layer is the **least** lexical layer in the stack by the §1 probe
and simultaneously the most expensive to constrain, which a lexical account predicts backwards.

*A magnitude-matched perturbation carrying no lexical information* reproduces **58%** of the endpoint
excess on one model and **85%** on the model with the largest effect
(`sham_magnitude_matched.csv`). So most of it is sensitivity to position in the network, not to what
the layer routes on.

*Training eight arms from scratch at three seeds each* — 24 runs, single layers, endpoint exemptions,
and a uniform schedule at matched memory (`t1_perlayer_training.csv`) — dissolves the rest:

| contrast | Δ test CE | se |
|---|---|---|
| all three MoE layers constrained vs none | **+0.0419** | **5.3** |
| first vs last MoE layer — the endpoint effect | +0.0014 | 0.2 |
| exempt-last vs uniform schedule at matched memory | −0.0080 | 0.7 |
| exempt-last vs exempt-first | +0.0017 | 0.5 |

**Only the global cost survives.** Nothing else clears 2 se.

**The generalisable lesson is larger than the schedule question**: a per-layer cost profile measured by
perturbing a trained checkpoint does not predict what a model trained under the constraint will do.
The endpoint effect is real at inference, mostly positional, and absent entirely after co-adaptation.

**Strength: high, and negative.** Three pre-registered readings, all resolving against the hypothesis.

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

1. **Does de-lexicalization explain the loss advantage?** §1 shows the constraint changes routing and
   §3.1 shows it eventually helps, and nothing joins them. Across matched pairs, does a model's
   contextual share predict its advantage over its baseline? Small *n*, correlational, and free —
   the data is already committed.
2. **Is the depth curve seed-stable?** See §5. Two hours, no training.
3. **What is the ~15–42% of the last MoE layer's inference-time cost that a lexicality-free
   perturbation does not reproduce?** It is the only part of the per-layer profile specific to this
   constraint, and it vanishes under training — which may mean it was never important, or may mean
   co-adaptation routes around it.
