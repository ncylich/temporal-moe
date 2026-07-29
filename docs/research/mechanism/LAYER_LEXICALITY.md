# Layer-wise Lexicality of MoE Routing

**Status: hypothesis + partial baseline measurements + work plan.** Sections 1–3 record what is
already measured. Section 4 states the two hypotheses this line of work exists to prove or
disprove; Sections 5–7 are the plan to test them. No result here is final.

## 0. Why

[`delexicalization.md`](delexicalization.md) establishes that unconstrained MoE routers bind
experts to token identity and that rolling residency breaks that binding, and it reports one
number per model — a median pooled over layers. It never asks whether the effect varies with
depth, and it never states which layers it pooled over.

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
   every unconstrained series is below −0.20 at every layer. The regime gap (~0.3 AUC) dwarfs
   every depth effect (~0.05).
2. **The depth trend is positive in 3 of 4 arms, and is present in both regimes at
   indistinguishable rates.** On current evidence it is a property of network depth — more
   attention mixing before deeper routers see the stream — not of the residency constraint. This
   is consistent with the mechanism behind H1 below, but it means any regime-specific claim needs
   matched baseline arms to be testable at all.
3. **The coarse 1e17 pair is flat in both regimes**, CI straddling zero. A real exception, not
   noise-in-our-favour.
4. **Granularity sets the level more than compute does.** The fine arms bracket the coarse ones.

**Caveats that limit these readings.**

- **Layers 7–9 are missing for every 9-layer model.** `LAYERS = [2,3,4,5,6]` is hardcoded in
  [`analysis/probes/delex_locus.py`](../../../analysis/probes/delex_locus.py), with the note "paper
  convention". For the 4- and 6-layer models this happens to be full coverage; for the 1e18/1e19
  models it is 5 of 8 MoE layers, silently skipped by the `if L not in d["layers"]: continue`
  guard. The captures contain the missing layers.
- **The x-axis mixes models of different total depth.** The fine@1e16 model's L4 is its last layer;
  the 1e19 model's L4 is 3/8 of the way through. Any depth claim has to be stated against
  normalized depth `l/L`.
- **The softmax-aux fine baseline was only ever measured at w=32**, not at w=k. In
  [`mechinterp_locus.csv`](../../../results/ablations/mechinterp_locus.csv) the variants decode as
  `kwin` = w=k/2, `kfull` = w=k, `base` = w=32; `s0_SOFTMAX_BASELINE` has only `base`. §3 of
  [`delexicalization.md`](delexicalization.md) labels that table row "w=18", which is wrong. Its
  sigmoid sibling at the same budget and granularity does have w=k and sits ~0.02 lower, which
  brackets the missing measurement. **The doc needs correcting either way.**
- Layer 1 is a dense FFN in every config (`--moe-layer-freq "[0]*1+[1]*(L-1)"`), so there is no
  layer-1 router to probe.
- The probe is **linear on the input embedding**. `A_tok = 1` is not the ceiling: routing is a
  function of the post-attention hidden state, so the same token id genuinely routes differently in
  different contexts.

## 2. Cache hit rate by depth

**Metric.** Hit rate = of the k experts a token's router actually wanted (its *unconstrained*
top-k), the fraction already resident when the token arrived, measured **pre-swap**. Range 0–1,
higher = the resident set matches demand better. A random resident set scores
**k/E = 6/64 = 18/192 = 0.094** — identical for both granularities, so the two are directly
comparable on this axis.

Source: [`e6_per_layer_ranking.csv`](../../../results/ablations/e6_per_layer_ranking.csv), produced
by [`analysis/probes/probe_replay.py`](../../../analysis/probes/probe_replay.py).

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

**Depth coverage here is complete** for all three models (s1 has 5 layers, s2 has 6, the 38M has 9;
layer 1 is dense in each). This is the one metric that already sees the full stack, which makes its
shape important: hit rate climbs steeply from the first MoE layer through the middle (+82% relative
for the fine model, +64% for the 8.1M coarse) and then **plateaus at 0.33–0.34 through L9 rather
than continuing to rise**. If the locus curve does the same, H1's shape is
*increases-then-saturates*, not monotone increase — a materially different prediction for where a
per-layer schedule should put its cutoff.

**Do not use swap rate for this.** Swap rate is 0.994–1.000 in every cell of every model, and that
is structural, not a finding: at R = k a swap fires **iff at least one demanded expert is missing**,
so swap rate is just "fraction of tokens with >= 1 miss" and saturates. Hit rate is the graded
statistic. The `lifetime_tokens` column is likewise pinned at k (5.99 coarse, 17.85 fine). Of e1's
columns only `p95_burst_len` is independently informative.

**Coverage gaps.** All three models are **temporal**; there is no unconstrained baseline hit rate
at all. Budgets are 1e17 x2 and 1e18 x1. Granularity is coarse x2, fine x1, and the single fine
model is also the shallowest. The cause is a hardcoded run list: `e6()` loops over `HEADLINERS`
(3 runs) while `e1()` loops over `ALL_TEMPORAL` (5), at
[`probe_replay.py:55-57`](../../../analysis/probes/probe_replay.py).

## 3. Coverage audit

| metric | file | per-layer? | layers covered | models |
|---|---|---|---|---|
| locus probes (A_tok, A_ctx) | `mechinterp_locus{,_1e19}.csv` | yes | 2–6 (of up to 9) | 8 |
| output logit lens (effective vocab) | `mechinterp_lens{,_1e19}.csv` | yes | **2–4 only** | 6 |
| logit lens (older) | `mechinterp_logitlens.csv` | yes | 1–3 | 2 |
| cache hit rate | `e6_per_layer_ranking.csv` | yes | **all** | 3 |
| swap rate / burst length | `e1_swap_rate_by_layer.csv` | yes | all | 5 |
| selectivity PR, generalist %, router entropy, weight geometry | `mechinterp_structural{,_1e19}.csv` | **no — pooled** | n/a | 11 |
| demand forecastability | `mechinterp_demand_1e19.csv` | **no — pooled over 2–6** | n/a | 3 |

Against this: **69 runs have preserved checkpoints** in
[`results/MANIFEST.csv`](../../../results/MANIFEST.csv), spanning 1e16 through 1e19, both regimes,
granularities g1/g3/g5, and several seeds. **22 runs have preserved `router_log.pt`** (enough for
all replay/cache metrics with no forward pass). Only **3 runs have a preserved `delex_capture.pt`**,
all at 1e19 — every other locus/lens number needs a fresh capture pass from its checkpoint.

The gap is not data. Every analysis script hardcoded a 3-to-8 model list and a 3-to-5 layer range.

## 4. The two hypotheses

Stated in advance, with what would falsify each. Everything in Sections 5–7 exists to test these;
anything else this analysis turns up is exploratory and labelled as such.

### H1 — routing specialization shifts from lexical to contextual as depth increases

**Prediction.** `A_ctx - A_tok` increases with normalized depth `l/L`, and cache hit rate increases
with `l/L`, in both regimes, across budgets and granularities.

**Falsified by:** a flat or negative depth slope in the majority of arms once measured over the
full stack on a normalized axis with matched baselines; or a slope that vanishes when arms are
pooled with per-arm uncertainty.

**Refined by:** the shape. Monotone increase and increase-then-saturate imply different cutoffs for
H2's schedule, and the one full-depth signal we have (Section 2) currently favours saturation.

**Status:** positive slope in 3 of 4 arms over layers 2–6, CIs excluding zero in 3 of 4; one flat
exception; not yet measured on a normalized axis, over the full stack, or at 1e18.

### H2 — the quality cost of imposing rolling residency decreases with depth

**Prediction.** dBPB from constraining a single layer (R = k at that layer, R = E elsewhere) is
largest at the first MoE layer and falls with depth. If H1's shape is saturating, dBPB should
flatten at the same depth H1 flattens.

**Falsified by:** a flat or U-shaped per-layer cost curve; or a cost curve uncorrelated with the
per-layer contextual share from H1.

**Why it does not follow from H1.** "This layer routes on context" and "this layer can afford to
lose routing freedom" are different quantities. §5 of
[`delexicalization.md`](delexicalization.md) already showed the model co-adapts to the constraint
during training, so per-layer cost has to be measured on separately trained models, not by masking
a trained checkpoint.

**Status:** no evidence either way. Nothing in the repo isolates the cost of constraining one
layer; the existing dose curve varies R uniformly across all layers.

**Payoff if H1 and H2 both hold.** A prefix schedule: leave layers shallower than depth *d*
unconstrained, constrain everything from *d* on. One tunable parameter, and the per-layer cost
curve gives *d* directly instead of assuming where the lexical layers are.

## 5. Plan

Ordered so that every step either sharpens H1 or is a prerequisite for H2. Each lands its own
commit.

### Step 1 — make H1 measurable on what we already have (no GPU)

1. **Replot on normalized depth `l/L`.** Absolute layer index is not comparable across 4-, 6- and
   9-layer models, and every H1 statement is about depth.
2. **Carry per-arm bootstrap CIs into the figure and the table**, as in Section 1. A depth slope
   without an interval is not testable.
3. **Correct §3 of [`delexicalization.md`](delexicalization.md)**: state the layer range, fix the
   `s0_SOFTMAX_BASELINE` window label from w=18 to w=32.

### Step 2 — full stack, on existing captures

4. **Locus at all layers.** `LAYERS = range(2, 10)` in `delex_locus.py`; replace the silent
   `continue` with a warning. Re-run on the 3 preserved captures. **This is the step that decides
   whether H1 is monotone or saturating**, which in turn sets what H2 should predict.
5. **Cache metrics on all captured runs.** Point `e6()` at `ALL_TEMPORAL`, then extend to all 22
   runs with a `router_log.pt`, including the 1e19 models no cache metric currently covers.
6. **Baseline hit rate by counterfactual replay.** Replay rolling residency over the unconstrained
   runs' router logs. Without this, H1's cache-side evidence has no matched control and cannot
   distinguish a tMoE effect from an MoE effect.

### Step 3 — enough arms to test H1 properly

7. **Capture and probe a 1e18 pair, both regimes, both granularities**, plus one seed replicate of
   an existing arm. 1e18 is the budget where the temporal model wins and where the current figure
   has nothing at all. Seed replication gives the between-run variability that the within-run
   bootstrap in Section 1 does not.
8. Regenerate the figure from the full set; the colour/marker encoding already scales.

### Step 4 — H2

9. **Per-layer marginal cost sweep.** Constrain exactly one layer at a time, one run per layer, at
   s2/1e17 (5 MoE layers, cheap runs). Produces dBPB against depth: the direct test of H2 and the
   thing that yields *d*.
10. **Schedule vs uniform at matched memory.** Take the prefix schedule implied by (9) and compare
    against uniform R with the same total resident-slot count. This is the control that decides
    whether a per-layer schedule is worth anything: the proposal saves memory against the full MoE
    baseline but *costs* memory against shipped temporal (1e18 model: 512 slots baseline, 48 for
    uniform R=k), so the only meaningful comparison is against the uniform-R dose curve at equal
    spend.
11. **Confirm at 1e18**, where there are 6 genuinely middle layers and the quality benefit exists.

Implementation note for 9–11: the R knob exists but is global —
[`temporal_router.py:359`](../../../temporal/temporal_router.py) reads `TEMPORAL_RESIDENCY_R` once
and applies it at every layer. `self.layer_number` is already in scope a few lines below, so a
per-layer schedule is a small change. FLOPs are unchanged at any R.

## 6. Additional metrics, only where they serve H1 or H2

The locus probe is correlational and linear. Two additions would materially strengthen H1; the rest
are noted so they are not re-invented, but are not on the critical path.

1. **Causal token substitution** (strengthens H1). Hold context fixed, substitute the current
   token, measure how far the selected expert set moves, per layer; then the complement — hold the
   token fixed, shuffle the context. The ratio is a causal per-layer lexical-vs-contextual measure
   that does not depend on a probe being able to express the lookup. Needs forward passes, no
   training.
2. **Nonparametric token-id oracle** (bounds H1). Best achievable AUC from token identity alone via
   empirical `P(y_e | token id)`, equivalently `I(expert ; token id) / H(expert)`. Gives the
   ceiling our linear probe has been measuring against, so a falling `A_tok` can be attributed to
   real context-dependence rather than probe capacity. Free on existing captures.
3. Not on the critical path: per-layer output lens (free but limited to layers 2–4 until re-run),
   frequency-stratified `A_tok`, cross-layer probe transfer, expert-set autocorrelation length,
   `I(expert ; document id)` to separate topical from local context.

## 7. On recomputing the rest of the mechinterp battery per layer

Cheap, and worth doing once, but not a prerequisite for either hypothesis.

Every per-expert metric — locus, lens, selectivity PR, generalist fraction, router entropy, weight
geometry — is already computed per (layer, expert) and then pooled. Layer is a grouping key that
was discarded at write time, not a measurement we lack. Free-rider / tokens-per-expert is
architecturally fixed and not meaningful per layer.

The one discipline point: a per-layer breakout of the full battery multiplies the available
comparisons by roughly the depth of the network. H1 and H2 above are the claims; anything else the
breakout surfaces is exploratory and should be reported as such.
