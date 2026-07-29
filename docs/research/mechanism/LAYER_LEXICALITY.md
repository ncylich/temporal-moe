# Layer-wise Lexicality of MoE Routing

**Status: partial results + work plan.** Nothing here is a finished claim. The two measurements
below exist and are recorded verbatim; everything in Section 4 onward is what remains to be run.

## 0. The question

[`delexicalization.md`](delexicalization.md) establishes that unconstrained MoE routers bind
experts to token identity and that rolling residency breaks that binding, and it reports one
number per model — a median pooled over layers. It never asks whether the effect is uniform with
depth, and it never states which layers it pooled over.

The motivating hypothesis (raised in advising discussion, not by us): experts in the **first and
last** layers are tied to token identity, while **middle** layers work with more abstract, mixed
context. If that is right, the residency constraint is being applied where it does not belong.
The operational payoff would be a **per-layer residency schedule** — keep the lexical layers fully
resident (R = E) and constrain only the contextual middle (R = k) — which could beat a uniform-R
model at equal serving memory.

This document collects what we can already say, and plans the work to answer it properly.

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

**Readings.**

1. **The regimes never overlap at any depth.** Every temporal series is above 0 at every layer,
   every unconstrained series is below −0.20 at every layer. The regime gap (~0.3 AUC) dwarfs
   every depth effect (~0.05).
2. **Temporal routing becomes more contextual with depth in 3 of 4 arms** — consistent with
   attention having mixed in more context by the time deeper routers see the stream.
3. **The exception is temporal coarse @1e17, which peaks at L3 and declines to L6** — and its L6
   is the final layer of that model. This is the only series consistent with the last-layer half
   of the hypothesis.
4. **Unconstrained routing drifts less lexical with depth but never crosses over.** Even the
   deepest probed layer of a full MoE is overwhelmingly token-driven.
5. **Granularity sets the level more than compute does.** The fine arms bracket the coarse ones;
   moving 1e16 -> 1e19 within a granularity shifts less.

**Caveats that limit these readings.**

- **Layers 7–9 are missing for every 9-layer model.** `LAYERS = [2,3,4,5,6]` is hardcoded in
  [`analysis/probes/delex_locus.py`](../../../analysis/probes/delex_locus.py), with the note "paper
  convention". For the 4- and 6-layer models this happens to be full coverage; for the 1e18/1e19
  models it is 5 of 8 MoE layers, silently skipped by the `if L not in d["layers"]: continue`
  guard. The captures contain the missing layers.
- **The softmax-aux fine baseline was only ever measured at w=32**, not at w=k. In
  [`mechinterp_locus.csv`](../../../results/ablations/mechinterp_locus.csv) the variants decode as
  `kwin` = w=k/2, `kfull` = w=k, `base` = w=32; `s0_SOFTMAX_BASELINE` has only `base`. §3 of
  [`delexicalization.md`](delexicalization.md) labels that table row "w=18", which is wrong. Its
  sigmoid sibling at the same budget and granularity does have w=k and sits ~0.02 lower, which
  brackets the missing measurement. **The doc needs correcting either way.**
- Layer 1 is a dense FFN in every config (`--moe-layer-freq "[0]*1+[1]*(L-1)"`), so there is no
  layer-1 router to probe. "Keep the first layer always resident" is already partly the
  architecture.
- The probe is **linear on the input embedding**. `A_tok = 1` is not the ceiling: routing is a
  function of the post-attention hidden state, so the same token id genuinely routes differently in
  different contexts. We have never measured the nonparametric token-id oracle that would tell us
  how much of the gap to 1.0 is real context-dependence versus probe capacity (Section 5).

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
layer 1 is dense in each). This is the one metric that already sees the full stack.

**Readings.** Hit rate climbs steeply from the first MoE layer through the middle — +82% relative
for the fine model, +64% for the 8.1M coarse — and the 38M, the only model deep enough to show it,
**plateaus at 0.33–0.34 through L9 rather than reverting** toward its L2 value of 0.165. Deep-layer
routing demand stays autocorrelated. This is the only full-depth evidence we have, and it does not
support the last-layer half of the hypothesis.

**Do not use swap rate for this.** Swap rate is 0.994–1.000 in every cell of every model, and that
is structural, not a finding: at R = k a swap fires **iff at least one demanded expert is missing**,
so swap rate is just "fraction of tokens with >= 1 miss" and saturates. Hit rate is the graded
statistic (mean fraction of demand present). The `lifetime_tokens` column is likewise pinned at
k (5.99 coarse, 17.85 fine), confirming the one-swap-per-token dynamics but carrying no depth
signal. Of e1's columns only `p95_burst_len` is independently informative (bandwidth planning).

**Coverage gaps.** All three models are **temporal**; there is no unconstrained baseline hit rate
at all. Budgets are 1e17 x2 and 1e18 x1 — nothing at 1e16 or 1e19. Granularity is coarse x2,
fine x1, and the single fine model is also the shallowest (4 MoE layers). The cause is a hardcoded
run list: `e6()` loops over `HEADLINERS` (3 runs) while `e1()` loops over `ALL_TEMPORAL` (5), at
[`probe_replay.py:55-57`](../../../analysis/probes/probe_replay.py).

## 3. Coverage audit

What each existing mechinterp metric covers today.

| metric | file | per-layer? | layers covered | models |
|---|---|---|---|---|
| locus probes (A_tok, A_ctx) | `mechinterp_locus{,_1e19}.csv` | yes | 2–6 (of up to 9) | 8 |
| output logit lens (effective vocab) | `mechinterp_lens{,_1e19}.csv` | yes | **2–4 only** | 6 |
| logit lens (older) | `mechinterp_logitlens.csv` | yes | 1–3 | 2 |
| cache hit rate | `e6_per_layer_ranking.csv` | yes | **all** | 3 |
| swap rate / burst length | `e1_swap_rate_by_layer.csv` | yes | all | 5 |
| selectivity PR, generalist %, router entropy, weight geometry | `mechinterp_structural{,_1e19}.csv` | **no — pooled** | n/a | 11 |
| demand forecastability | `mechinterp_demand_1e19.csv` | **no — pooled over 2–6** | n/a | 3 |
| free-rider / tokens-per-expert | `mechinterp_freerider.csv` | no | n/a | — |

Against this: **69 runs have preserved checkpoints** in
[`results/MANIFEST.csv`](../../../results/MANIFEST.csv), spanning 1e16 through 1e19, both regimes,
granularities g1/g3/g5, and several seeds. **22 runs have preserved `router_log.pt`** (enough for
all replay/cache metrics with no forward pass). Only **3 runs have a preserved `delex_capture.pt`**,
all at 1e19 — every other locus/lens number would need a fresh capture pass from its checkpoint.

The gap is therefore not data, it is that every analysis script hardcoded a 3-to-8 model list and
a 3-to-5 layer range.

## 4. Plan

Ordered by cost. Each phase is independently useful and lands its own commit.

### Phase A — free re-aggregation (no GPU, minutes)

1. **Per-layer output lens.** `mechinterp_lens{,_1e19}.csv` already carries `layer` and `expert`
   columns; §4 of [`delexicalization.md`](delexicalization.md) only ever reported a pooled median.
   Group by layer and plot effective vocabulary by depth — a third, output-side view of the same
   question, for free. Limited to layers 2–4 until Phase B.
2. **Add the e1 `p95_burst_len` depth curve** to the figure set; drop swap rate from the reporting
   with the saturation argument recorded above.
3. **Correct §3 of [`delexicalization.md`](delexicalization.md)**: state the layer range explicitly
   and fix the `s0_SOFTMAX_BASELINE` window label from w=18 to w=32.

### Phase B — full depth and full model list on existing captures

4. **Locus at all layers.** `LAYERS = range(2, 10)` in `delex_locus.py`. Replace the silent
   `continue` with a warning so future truncation is visible. Re-run on the 3 preserved captures.
5. **Cache metrics on all captured runs.** Point `e6()` at `ALL_TEMPORAL` instead of `HEADLINERS`,
   then extend both lists to every one of the 22 runs with a `router_log.pt`, including the 1e19
   models that no cache metric currently covers.
6. **Baseline hit rate by counterfactual replay.** `moe_coarse_1e19` and the other unconstrained
   runs have router logs; replaying rolling residency over an unconstrained checkpoint's logits
   gives the blue curve that Section 2's table is missing entirely. This is the direct analogue of
   the baseline lines in the Section 1 figure and is needed before any claim that
   "deep layers are more cacheable" is a property of the temporal regime rather than of MoEs.
7. **Per-layer structural stats.** `delex_structural.py` pools experts over all MoE layers. Add
   `layer` as a grouping key so selectivity PR, generalist fraction, and router entropy become
   depth curves. Same captures, no new compute.
8. **Per-layer demand forecastability.** `delex_demand.py` fits one probe over layers 2–6 pooled.
   Fit per layer instead.

### Phase C — capture sweep over the trained fleet

9. **Re-capture and re-probe the best config at every budget, both regimes, both granularities.**
   Selection rule: for each (budget, regime, granularity) cell take the seed-1234 run that the
   isoFLOP analysis treats as the headline, plus one alternate seed where available. Concretely
   this fills the holes the current figure has: **no 1e18 model of any kind** (which is where the
   temporal model actually wins), one lone 1e17 pair, and only the s0 shape at 1e16.
10. **Include the dense control** where one exists at that budget, as a floor.
11. Regenerate the Section 1 figure from the full set. Keep the color/marker encoding
    (hue = regime, shade = granularity, marker = budget); it already scales.

### Phase D — the experiment this is all for

12. **Per-layer marginal cost of the constraint.** Constrain exactly one layer at a time
    (R = k there, R = E everywhere else), one run per layer, at s2/1e17 (5 MoE layers) where runs
    are cheap. Yields dBPB as a function of depth — the direct answer to "where does the constraint
    help and where does it hurt", replacing the assumption of a U-shape.
13. **Schedule vs uniform at matched memory.** Take the best schedule from (12) and compare it
    against uniform R with the same total resident-slot count. **This is the control that makes or
    breaks the idea** — the proposal saves memory against the full MoE baseline but *costs* memory
    against shipped temporal (for the 1e18 model: 512 slots baseline, 164 for
    first-and-last-unconstrained, 48 for uniform R=k), so the only meaningful comparison is against
    the existing uniform-R dose curve at equal spend.
14. Confirm the winner at 1e18, where there are 6 genuinely middle layers and where the quality
    benefit exists.

Implementation note for 12–14: the R knob already exists but is global —
[`temporal_router.py:359`](../../../temporal/temporal_router.py) reads `TEMPORAL_RESIDENCY_R` once
and applies it at every layer. `self.layer_number` is already in scope a few lines below, so a
per-layer schedule is a small change. FLOPs are unchanged at any R.

## 5. Metrics we have not tried

The two measurements above are one correlational input-side probe and one dynamics statistic. Both
are indirect. Candidates, roughly in decreasing value-per-cost:

1. **Nonparametric token-id oracle.** For each expert, the best achievable AUC from token identity
   alone — score with the empirical `P(y_e = 1 | token id)` fit on the train split. This is the
   ceiling the linear probe is measuring against, and its per-layer curve separates "routing
   genuinely depends on context" from "the linear probe cannot express the lookup". Equivalently,
   report normalized mutual information `I(expert ; token id) / H(expert)` per layer. Cheap: it
   reuses the existing captures and needs no model.
2. **Token-type routing determinism.** Across all occurrences of the same token type, the expected
   Jaccard overlap of the selected expert sets, per layer. A probe-free, classifier-free statement
   of "is this a lexical lookup", trivially interpretable and immune to the linearity objection.
3. **Causal token substitution.** Hold the context fixed, substitute the current token, and measure
   how much the selected expert set moves, per layer. Its complement — hold the token fixed and
   shuffle or replace the surrounding context — gives the other half. The ratio of the two
   sensitivities is arguably *the* right per-layer metric for this question, and unlike everything
   above it is causal rather than correlational. Cost: new forward passes, but no training.
4. **Frequency stratification.** Split A_tok by token-frequency decile within each layer. If the
   lexical shortcut lives entirely on rare tokens (plausible — they carry the most identifying
   signal and the least contextual support), that reframes the whole mechanism and changes which
   layers deserve the constraint.
5. **Probe transfer across layers.** Fit the token probe at layer L and evaluate it at layer L';
   an off-diagonal collapse would show routing is a qualitatively different function with depth,
   not the same function weakening.
6. **Expert-set autocorrelation length.** How far apart two tokens can be and still share experts,
   per layer — a continuous, non-saturating version of hit rate.
7. **Document-level mutual information.** `I(expert ; document id)` per layer, separating topical
   specialization from local-context specialization. Both are "contextual" under our current probe
   and they are not the same claim.

## 6. Should everything be recomputed per layer?

Mostly yes, with one discipline requirement.

**Yes, and it is nearly free**, for every per-expert metric: locus, output lens, selectivity PR,
generalist fraction, router entropy, weight geometry. These are all computed per (layer, expert)
and then pooled — layer is a grouping key that was discarded at write time, not a measurement we
lack. Recomputing means re-running existing scripts with an extra key.

**Not meaningful per layer:** free-rider / tokens-per-expert is fixed by the architecture
(top-k assigns k/E of each batch to each expert on average regardless of layer), and the weight
geometry results were regime-invariant and seed-stable, so a depth breakout is unlikely to show
anything — worth computing once to confirm, not worth featuring.

**The discipline requirement:** going from ~10 model-level numbers to ~10 numbers x 8 layers x N
models is a large increase in the number of comparisons available after the fact. Before running
Phase C, write down which curves are the claims (our prediction: the temporal A_ctx − A_tok depth
slope, and the per-layer marginal cost from Phase D item 12) and which are exploratory. Otherwise
"layer 7 of the coarse model behaves differently" becomes findable in any direction.

## 7. Open questions

- Does the last-layer lexicality uptick survive at depth, or is it an artifact of 4- and 6-layer
  models where the last MoE layer is also the last layer feeding the unembedding? Phase B item 4
  answers this.
- Is the deep-layer hit-rate plateau (0.33–0.34 at L7–L9) a temporal-regime property or a generic
  MoE property? Phase B item 6 answers this.
- Do the locus and cache-hit-rate depth curves agree per model? They measure different things and
  have never been plotted against each other on the same models, because their model lists barely
  intersect.
