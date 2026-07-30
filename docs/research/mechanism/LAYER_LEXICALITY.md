# Layer-wise Lexicality of MoE Routing

**Status: H1 measured at full depth and restated. H2 falsified by C3, on its own pre-registered
criterion.** Of the no-training tests, C1, C2, C3, C4, C6, C7, C9 and C10 are done; C5 is blocked on a
re-capture; C8 is not run and is the largest remaining gap. The training tests T1–T3 are not started, and
C3's result says T2 as designed would not test the effect that exists. Sections 1–2 are the measurements,
§3 the hypotheses and what the results did to them, §§4–5 the tests and their status.

**The two-line version.** Routing does move from lexical to contextual with depth, in the *unconstrained*
baseline as well, so part of that trend belongs to transformer depth rather than to rolling residency.
What the constraint adds is a shape: it starts from no measurable effect at the first MoE layer,
accumulates with depth on every metric measured, moves the locus toward context about twice as fast as
the baseline while it is rising, and then **turns over at roughly two thirds depth** while the
unconstrained model is still climbing at its last layer.

**A note on method, because it changed two conclusions in this document.** These curves are not lines.
Summarising them by an OLS slope inverted the regime comparison in §1 (a full-range slope averages the
temporal arm's rise against its own fall) and would have hidden the U in C3's per-layer cost profile
entirely (a symmetric U has near-zero linear trend by construction). Curvature, vertex and
restricted-range slopes are now reported alongside every slope, and `linear_r2`/`quadratic_r2` are in
`mechinterp_locus_slopes.csv` so the adequacy of a line is visible rather than assumed.

Battery-wide housekeeping — re-running every mechinterp script across every model and layer — is
tracked separately in [`MECHINTERP_RERUN_PLAN.md`](MECHINTERP_RERUN_PLAN.md), whose §7 records what that
re-run found, including a null control that turned out to be invalid, a probe split that leaked
documents, and a capture that keyed expert outputs one layer too shallow. Steps 1–2 of that plan are
done, and Step 3 has run at 1e18: seven captures now exist rather than three, including matched
temporal/unconstrained pairs at the budget where the temporal model wins.

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

**The 1e19 rows above are superseded.** They covered layers 2–6 of a **14-layer** model — both plan
documents previously assumed 9 — and were measured with every document present in both the probe's fit
and score halves. Re-measured at full depth on held-out documents (`mechinterp_locus_1e19.csv`, w=k):

| MoE layer | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| temporal 6/64 @1e19 | +.024 | +.041 | +.054 | +.069 | +.087 | +.091 | **+.107** | +.105 | +.095 | +.101 | +.082 | +.076 | +.083 |
| temporal 18/192 @1e19 | +.053 | +.059 | +.061 | +.078 | +.084 | +.071 | +.078 | +.088 | +.096 | +.091 | +.079 | +.081 | +.097 |
| full MoE 6/64 @1e19 | −.305 | −.253 | −.232 | −.214 | −.221 | −.231 | −.225 | −.200 | −.220 | −.219 | −.199 | −.196 | −.176 |

**Depth slope, with uncertainty.** OLS slope of the per-layer median, 95% CI from 2000 bootstrap
resamples of experts within each layer. Reported per unit normalized depth `l/L`, which is comparable
across models of different depth, and per layer index for continuity with the published table.

| matched pair | temporal slope / `l/L` | baseline slope / `l/L` | (per layer index: temporal / baseline) |
|---|---|---|---|
| fine 18/192 @1e16 | +0.1026 [+0.0516, +0.1334] | +0.0893 [+0.0595, +0.1194] | +0.0257 / +0.0223 |
| coarse 6/64 @1e17 | −0.0195 [−0.0534, +0.0289] | −0.0179 [−0.0784, +0.0263] | −0.0033 / −0.0030 |
| coarse 6/64 @1e19 | +0.0590 [+0.0458, +0.0748] | +0.0929 [+0.0773, +0.1148] | +0.0042 / +0.0066 |
| fine 18/192 @1e19 | +0.0408 [+0.0307, +0.0513] | *no baseline capture preserved* | +0.0029 / — |

**A straight line is the wrong summary of these curves, and the full-range slope above inverts the
comparison it appears to make.** The contextual share rises with depth and then turns over, so an OLS
slope across the whole stack averages a rise against a fall. On the coarse temporal arm a line explains
R² = 0.43 while a quadratic explains **0.94**. Fitting the shape instead:

| coarse 6/64 @1e19 | R² lin / quad | slope over the rising region | curvature | vertex (layer) |
|---|---|---|---|---|
| temporal | 0.43 / **0.94** | **+0.1704 [+0.136, +0.197]** | **−0.2739 [−0.334, −0.203]** | **9.5 [9.1, 10.3]** |
| unconstrained | 0.68 / 0.72 | **+0.0929 [+0.076, +0.116]** | −0.1048 [−0.169, −0.001] | 14.2 [11.0, 60.7] |

All three statistics separate, and they say the opposite of the full-range slopes: **while it is rising
the temporal arm moves toward context about twice as fast as the unconstrained one** (+0.170 vs +0.093,
non-overlapping), then reverses at layer 9.5 of 14 — a vertex tightly identified inside the stack — while
the unconstrained arm is still climbing at its last layer, its vertex interval running well past the
network. The full-range slope reads "baseline steeper" only because the temporal arm's own decline is
folded into its average.

**1e18 makes the same point more sharply, and it is the budget where the temporal model wins.** No
capture-based measurement existed here at all before the Step 3 sweep; there are now matched pairs at
both granularities, 8 MoE layers each:

| coarse 6/64 @1e18 | R² lin / quad | full-range slope | rising slope | curvature | vertex |
|---|---|---|---|---|---|
| temporal | **0.01** / 0.40 | +0.0067 [−0.015, +0.031] | **+0.1282 [+0.063, +0.173]** | −0.1529 [−0.234, −0.037] | **5.7 [5.0, 7.5]** |
| unconstrained | **0.94** / 0.94 | +0.1384 [+0.112, +0.169] | +0.1384 [+0.112, +0.169] | −0.0378 [−0.174, +0.079] | 22.0 (outside) |

A line fits the unconstrained arm well (R² 0.94, curvature interval containing zero: it is simply
rising) and fits the temporal arm not at all (R² **0.01**). Reporting only the full-range slope would
have said the 1e18 temporal model has *no depth trend whatsoever* — +0.0067 with an interval straddling
zero — when its rising-region slope is +0.128 and statistically indistinguishable from the baseline's
+0.138.

So at 1e18 the two regimes climb at the **same rate**; the entire difference is that the constrained one
stops and reverses. And the turning point is reproducible across budgets in normalized depth: layer 5.7
of 9 (0.63) at 1e18, layer 9.5 of 14 (0.68) at 1e19 — about two thirds of the way down in both. The fine
arms turn later and less identifiably (12.4 of 14, 7.6 of 9, both intervals wide).

Shape cannot be resolved for the 1e16/1e17 arms, nor for fine 18/192 at 1e18: 3–8 layers give curvature
intervals that straddle zero. All shape statistics, with `linear_r2` and `quadratic_r2`, are in
`mechinterp_locus_slopes.csv`.

**De-lexicalization itself replicates at 1e18 with matched pairs**, which it had never been tested with
at the winning budget (medians over layers 2–9, w=k, documents held out):

| 1e18 arm | median token AUC | median context AUC | % context-dominated |
|---|---|---|---|
| unconstrained coarse | 0.902 | 0.679 | 1% |
| temporal coarse | 0.659 | 0.750 | **93%** |
| unconstrained fine | 0.889 | 0.650 | 2% |
| temporal fine | 0.587 | 0.719 | **97%** |

**Readings.**

1. **The regimes never overlap at any depth.** Every temporal series is above 0 at every layer, every
   unconstrained series below −0.17. The regime gap (~0.3 AUC) still dwarfs every depth effect (~0.05).
2. **The arms differ in the shape of the curve, not in a slope.** Over layers 2–6 the matched pairs were
   statistically indistinguishable. Over 2–14 they separate on curvature (−0.274 vs −0.105) and on where
   the curve turns (layer 9.5 vs 14.2), and over the rising region the *temporal* arm is the steeper one
   (+0.170 vs +0.093). So the depth trend is present in both regimes — that much is a property of
   transformer depth, more attention mixing before deeper routers see the stream — but the constraint
   does not merely fail to cause it and does not dampen it: it **accelerates the move toward context and
   then reverses it in the deepest third**. H1 is restated accordingly in §3.
3. **The shape is rises-then-turns-over, and only the temporal arms turn over inside the network.** The
   coarse temporal locus peaks at layer 9.5 of 14 and declines; the fine temporal arm turns at 12.4; the
   unconstrained arm's vertex sits beyond its last layer. Measuring 2–6 and extrapolating caught only
   the rising part. This settles the question §2 raised from the cache side, and the two sides agree on
   the same model: the coarse temporal locus turns at ~9.5 and its hit rate plateaus from layer 10.
4. **The coarse 1e17 pair is flat in both regimes**, CI straddling zero. Still a real exception, and
   now the only arm that is flat.
5. **Granularity sets the level more than compute does.**

**Caveats.**

- **The 1e16/1e17 rows cannot be fixed.** Those five runs are absent from `MANIFEST.csv` — no capture,
  no checkpoint — so they cannot be extended past layer 6, re-split, or re-windowed by anyone. They
  remain at layers 2–6 on the positional split. The 1e19 rows are the only ones measured properly.
- **Only the iid-permutation floor is trustworthy.** The circular-shift null is inflated by up to
  +0.047 on a single layer and is not a valid control here; see
  [`MECHINTERP_RERUN_PLAN.md`](MECHINTERP_RERUN_PLAN.md) §7.1. The iid floor passes the gate on every
  model: per-model pooled medians are within 0.0005 of 0.500. Resolved to individual
  (layer, feature, window) cells its median deviation is 0.0005 and its 95th percentile 0.0015, with 5
  of 468 cells between 0.002 and 0.0025 — sampling noise in a 64-to-192-expert median, not a floor
  problem.
- **The document-disjoint split moves the AUCs but not their difference.** Token 0.605 vs 0.622 and
  context 0.683 vs 0.702 on the coarse temporal arm, so the difference is +0.078 vs +0.080. Leakage
  inflates both probes similarly and cancels in the reported statistic.
- **The probe is at its ceiling, so the low temporal token AUC is real.** C7's nonparametric oracle —
  the best any function of token identity can achieve, `P(fire | token id)` scored on held-out
  documents — reads 0.833–0.885 for the baseline and 0.540–0.622 for the temporal models, and the
  linear probe matches it in every case. Token identity genuinely carries less routing information
  under the constraint; this is not probe capacity. The oracle is also **flat with depth** in all three
  models, so the entire depth trend in `A_ctx − A_tok` lives on the context side.
- The softmax-aux fine baseline was only ever measured at w=32; its sigmoid sibling has w=k and sits
  ~0.02 lower, bracketing the missing measurement.
- Layer 1 is a dense FFN in every config, so there is no layer-1 router to probe.

## 2. Cache hit rate by depth

**Metric.** Of the k experts a token's router actually wanted (its *unconstrained* top-k), the
fraction already resident when the token arrived, measured **pre-swap**. Range 0–1, higher =
resident set matches demand better. A random resident set scores **k/E = 6/64 = 18/192 = 0.094**,
identical for both granularities, so the two are directly comparable.

Source: [`e6_per_layer_ranking.csv`](../../../results/ablations/e6_per_layer_ranking.csv), now 22 runs
and 112 rows rather than 3 runs and 17 rows.

**The matched pair, and the answer to C4.** Same budget, same shape, same granularity, differing only
in whether rolling residency was imposed during training. The unconstrained arm is a *counterfactual
replay*: an unconstrained model logs no resident set, so the residency policy is replayed over its own
demand to obtain the set it would have held. Random floor is `k/E = 6/64 = 0.094` for both.

| MoE layer | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| temporal (`g1_tmoe_coarse_1e19`) | 0.114 | 0.149 | 0.162 | 0.183 | 0.227 | 0.265 | 0.293 | 0.317 | 0.380 | 0.404 | 0.390 | 0.375 | 0.424 |
| unconstrained (`moe_coarse_1e19`) | 0.117 | 0.128 | 0.126 | 0.140 | 0.161 | 0.173 | 0.175 | 0.168 | 0.172 | 0.182 | 0.198 | 0.199 | 0.250 |
| gap | **−0.003** | +0.021 | +0.036 | +0.043 | +0.066 | +0.092 | +0.118 | +0.149 | +0.208 | +0.222 | +0.192 | +0.176 | +0.174 |

Three readings, and the first two are new:

1. **The depth trend is present in the unconstrained regime too** (0.117 → 0.250), so part of it is a
   property of transformer depth rather than of the constraint — the same conclusion the §1 slope
   table forced on the probe side, now confirmed on the cache side with a matched arm.
2. **But the constraint's contribution is not uniform: it is zero at the first MoE layer and is earned
   with depth.** The two arms are within 0.003 at layer 2 and 0.21 apart by layer 11. Cacheability is
   not a property the constraint confers on a network, it is one it builds up through depth.
3. **The temporal curve saturates**, plateauing at 0.38–0.42 from layer 10 while the baseline keeps
   creeping up. This is the *increases-then-saturates* shape §2 previously suspected from 8 layers of
   one model, now seen over 13 layers with a control.

The 1e18 arms, which had no depth-resolved measurement of any kind before, agree: `flame38m_g1_temporal`
runs 0.171 → 0.266 → 0.317 → 0.342 → 0.331 → 0.331 → 0.362 → 0.380 over layers 2–9, i.e. a steep climb
to layer 5 and a plateau after. The published 38M column above (0.165 … 0.338) was a different run in
the same cell whose log was not preserved; the shape replicates, the values shift by ≤0.04.

Swap rate remains unusable for depth work: 0.994–1.000 everywhere, because at R = k a swap fires iff
at least one demanded expert is missing, so it saturates as "fraction of tokens with ≥ 1 miss".

## 3. The two hypotheses

### H1 — routing specialization shifts from lexical to contextual as depth increases

**Prediction.** `A_ctx - A_tok` increases with normalized depth `l/L`, and cache hit rate increases
with `l/L`, across budgets and granularities.

**Falsified by:** a flat or negative depth slope in the majority of arms once measured over the
full stack on a normalized axis with matched baselines.

**Refined by:** the shape — monotone versus saturating implies a different cutoff for H2's
schedule, and the one full-depth signal we have currently favours saturation.

**Status: measured at full depth, and H1 needs restating in two ways.**

*The prediction holds, and the depth trend is not exclusive to the constraint — but the constraint does
shape it.* `A_ctx − A_tok` increases with `l/L`, and cache hit rate increases with `l/L`, in every arm
except the flat coarse 1e17 pair. Both increase in the **unconstrained baseline too**, so the trend is
partly a property of transformer depth. H1's falsification criterion, "a flat or negative depth slope in
the majority of arms", is not met.

What the matched arms add is that the two regimes differ in the *shape* of the curve rather than in a
slope. While the curve is rising the constrained arm moves toward context about twice as fast
(+0.1704 [+0.136, +0.197] versus +0.0929 [+0.076, +0.116] per unit `l/L`, non-overlapping); it is more
strongly concave (−0.2739 [−0.334, −0.203] versus −0.1048 [−0.169, −0.001]); and it turns over at layer
**9.5 [9.1, 10.3]** of 14 while the unconstrained arm is still climbing at its last layer.

*The shape is rises-then-turns-over, not monotone, and a linear slope hides that.* A line explains
R² = 0.43 of the coarse temporal curve against a quadratic's 0.94, and the full-range linear slopes
invert the regime comparison because they average the temporal arm's rise against its own fall. The
cache side agrees and adds a granularity dependence: the coarse arms saturate hard (deep-over-shallow
half-slope 0.19 at 1e18, 0.62 at 1e19) while the fine arms stay roughly linear (0.88, 1.22).

*What is genuinely constraint-specific is a widening gap, not a slope.* On cache hit rate the matched
1e19 pair is identical at layer 2 (0.114 vs 0.117) and 0.21 apart by layer 11. Selectivity, generalist
fraction and router entropy do the same: indistinguishable through layer 4, totally separated from
layer 6 (`delexicalization.md` §2). The constraint's effect on routing is **near zero at the first MoE
layer and accumulates with depth** — which is a stronger and more useful statement than H1's, and it is
the one a per-layer schedule should be designed against.

The one metric that does *not* follow this pattern is demand forecastability, separated by 0.35 AUC at
layer 2 and staying separated (0.920 vs 0.570 at 1e19 coarse). That is the signature of a mechanical
consequence of residency — the resident set persists by construction — as against the learned
reorganisation that only appears deeper.

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

**Status: C3 has run. The per-layer cost profile is U-SHAPED — the first two and last MoE layers are the
expensive ones — which is the shape H2 named as its own falsifier.**

`flame38m_g1_temporal` at 1e18 — the budget where the temporal model wins — unmasking exactly one MoE
layer of the trained model (R=E there, R=k elsewhere), 8 evaluation passes plus a native reference, same
checkpoint and same fixed eval set throughout, so these differences carry no seed noise
([`swap_sweep.csv`](../../../results/ablations/swap_sweep.csv)). Native CE 3.909461:

| MoE layer | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| ΔCE from unmasking it | +.0425 | +.0463 | +.0350 | +.0337 | +.0326 | +.0300 | +.0374 | **+.0579** |

**The shape is a parabola, and a linear slope is the wrong summary of it.** OLS on layer index gives
+0.00056 per layer at R² = **0.023**, which explains nothing — a symmetric U has near-zero linear trend
by construction, so quoting that slope as "no depth trend" would hide the entire result. A quadratic
fits at R² = **0.706** with curvature **+0.00155** (positive: U-shaped) and a vertex at layer **5.3**.
The honest contrast is ends against middle:

| grouping | mean ΔCE |
|---|---|
| ends — L2, L3, L8, L9 | **+0.0461** |
| middle — L4–L7 | **+0.0328** |
| ends ÷ middle | **1.40×** (min-to-max 1.93×, +0.0300 at L7 to +0.0579 at L9) |

These are deterministic evaluations — one checkpoint, one fixed 33.5M-token eval set, every arm scored
on the identical tokens — so the comparison is paired and exact and differences of 0.003 are real, not
sampling noise.

**The opposite direction, on a different checkpoint, reproduces the same U with the same vertex.**
Imposing residency on exactly one MoE layer of the trained *unconstrained* model (`flame38m_g1_moe`,
native CE 3.9185, R=k there and R=E elsewhere):

| MoE layer | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| ΔCE from constraining it | **+0.411** | +0.273 | +0.253 | +0.219 | +0.241 | +0.184 | +0.215 | **+0.466** |

Linear R² = **0.001**, quadratic R² = 0.762, vertex layer **5.5**, ends ÷ middle **1.52×**. Two different
models, two opposite manipulations, the same shape with vertices at 5.3 and 5.5 — the U is a property of
where in the network routing freedom matters, not of one checkpoint or one direction of swap. In both
directions the two most expensive layers are the **first and the last** MoE layer.

The magnitudes also extend §5 of [`delexicalization.md`](delexicalization.md) from the whole network to
single layers. Imposing the constraint on one layer costs a mean +0.278 CE; removing it from one layer
costs +0.039 — a **7×** asymmetry, in the same direction as the global result ("imposing residency on
lexical routers costs two to five times more than unmasking contextual ones") and larger.

Four things follow.

1. **H2 is falsified by its own pre-registered criterion.** H2 said "Falsified by: a flat or U-shaped
   per-layer cost curve, or a cost curve uncorrelated with the per-layer contextual share from H1." The
   curve is U-shaped, and it is also uncorrelated with H1's contextual share, which rises then saturates
   rather than dipping in the middle. Both clauses fire.
2. **The reversed prediction is refuted too.** §3 argued from H1's re-measurement that cost should
   *rise* with depth, since the constraint's effect on routing is smallest at the first MoE layer. It
   does not rise monotonically either. Per-layer cost is not tracking how much a layer's routing was
   reorganised — the two quantities are genuinely different, as §3 warned, and now measurably so.
3. **T2 as designed would return a null on this profile, and that would be an artefact.** T2 contrasts
   layers 2–5 constrained against 6–9 constrained: a shallow-half versus deep-half contrast. A U-shape
   symmetric about layer 5.3 puts roughly equal cost in each half, so T2 would measure ≈0 and be read as
   "depth does not matter" when the real structure is ends versus middle. The specific training design
   in §5 is mis-specified for the shape that actually exists; testing the U needs an ends-versus-middle
   contrast at matched layer count.
4. **Still sub-additive.** Unmasking all layers costs +0.4795 while the eight single-layer costs sum to
   +0.3155, a ratio of 0.66 — so the layers' contributions overlap, and no per-layer profile predicts a
   multi-layer configuration by addition.

**This shape has been seen before in this project.** The OLMoE adaptation program's per-layer residency
damage was also U-shaped with the ends worst (layers 0–1 and 15), measured a different way on a
different model. Two architectures, two methods, same qualitative answer: routing freedom matters most
at the input and at the layer that forms the output, and least in the middle.

The limitation §4 identified still bounds all of this: no co-adaptation, so this is the cost of removing
freedom from a model trained expecting it — an upper bound whose *shape* is the informative part. A
trained sweep could still find something inference-time swapping cannot see, but H2 as written is
falsified and T2 as written would not test the shape that is there.

**Payoff — not as designed.** The prefix schedule this section was built around needs a monotone
per-layer curve with a knee. The curve has a vertex, not a knee, so the configuration it suggests is
"free the ends, constrain the middle", which is not a prefix and which the existing uniform-R dose curve
does not sweep.

## 4. Cheap tests (no training)

Ordered by what they buy. C1–C4 are the critical path.

| id | test | serves | needs | status |
|---|---|---|---|---|
| C1 | Replot on normalized depth `l/L`, with bootstrap CI bands | H1 | nothing | **done** — `locus_by_layer.png`, `mechinterp_locus_slopes.csv`, with curvature and vertex |
| C2 | Locus at layers 2–**14** on **seven** captures | H1 | existing captures | **done** — H1 rises then turns over at ~2/3 depth in the temporal arms; unconstrained arms never turn |
| C3 | Per-layer inference-time constraint swap | H2 pre-screen | existing checkpoints | **done**, both directions at 1e18 — U-shaped, vertex L5.3/L5.5, falsifies H2 |
| C4 | Baseline hit rate by counterfactual replay | H1 control | existing router logs | **done**, one cell — the only unconstrained router log preserved |
| C5 | Per-layer output lens (effective vocabulary) | H1, third view | captures; >L4 needs re-run | **blocked on a re-capture.** The capture keyed expert outputs one layer too shallow; fixed, but the lens needs captures taken after the fix — see `MECHINTERP_RERUN_PLAN.md` §7.4 |
| C6 | Per-layer demand forecastability | H2 mechanism | existing captures | **done** — 0.920→0.953 temporal vs 0.570→0.698 baseline, separated at every layer |
| C7 | Nonparametric token-id oracle | H1 ceiling | existing captures | **done** — probe is at the ceiling; ceiling is flat with depth |
| C8 | Causal token / context substitution | H1, causal | forward passes | **not run.** The strongest non-training evidence available for H1 and the largest remaining gap |
| C9 | Frequency-stratified `A_tok` | H1 refinement | captures + token ids | **done** — inverted U in both regimes, temporal at a constant 0.72–0.77× the baseline in every stratum; the shortcut is not a rare-token phenomenon |
| C10 | Cross-layer probe transfer | H1 refinement | existing captures | **done** — as subspace overlap, since the literal form is ill-posed; 3–4× chance adjacent, 2.1–2.3× across the full stack |

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

## 5. Optional training tests

**None of these has been started, and none should be until C3 has run.** H1 is now settled by the cheap
tests (§1–§3); only H2 needs training runs, and C3 is a deterministic pre-screen for whether they are
worth starting. C3's driver is committed but the sweep needs a GPU, so the gate on this section is still
closed. Note also that H1's re-measurement reversed the expected *sign* of the per-layer cost profile
(§3), so T2's shallow-half-versus-deep-half contrast is now a genuinely two-sided test rather than a
confirmation. They are listed in order, and the design is driven by a power
calculation, because the first testbed we considered cannot detect the effect.

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

**T1 (optional) — single-layer sweep at s0/1e16. 3 runs, cheapest available, first real H2
evidence.**
Constrain exactly one MoE layer at a time (R = k there, R = E elsewhere); the two reference
endpoints (all-constrained 1.4750, none-constrained 1.4519) already exist. Only 3 MoE layers, so
the depth resolution is coarse — but at SNR ~7 per layer it cleanly answers *"is the per-layer cost
curve flat or sloped?"*, which is the binary H2 turns on.

**T2 (optional) — shallow-half versus deep-half contrast at 1e18. 2 arms x 3 seeds = 6 runs.** Constrain
layers 2–5 with 6–9 free, versus 2–5 free with 6–9 constrained: matched layer count, matched
resident-slot budget, so the arms differ *only* in where the constraint sits. The contrast
aggregates four layers of the depth gradient instead of one, which is what makes 1e18 viable
despite a per-layer SNR of ~1.9; with 3 seeds per arm the contrast standard error is ~0.0008.
This is the definitive H2 test, at the budget where the constraint actually wins and across 8 MoE
layers.

**T3 (optional) — full per-layer resolution at 1e18, and the schedule-versus-uniform control.
8 + 2 runs, only if T1 and T2 both point the same way.** Per-layer resolution gives the cutoff *d*; the control
compares the resulting prefix schedule against uniform R at **equal total resident-slot count**.
That control is not optional: the schedule saves memory against the full MoE baseline but *costs*
memory against shipped temporal (1e18 model: 512 slots baseline, 48 uniform R=k), so the only
meaningful comparison is against the existing uniform-R dose curve at equal spend.

**Run C3 before T1.** If the inference-time per-layer profile is flat, T1's prior drops sharply and
the training budget is better spent elsewhere.

**C3 has run, and it invalidates T2's design rather than T2's premise** (§3). The profile is U-shaped
with a vertex at layer 5.3, so T2's shallow-half-versus-deep-half contrast splits the U near its
minimum and would measure ≈0 whatever the truth. T1 keeps its logic — a per-layer sweep at s0/1e16 is
agnostic about shape — but with only 3 MoE layers it cannot resolve a vertex. Any redesign should
contrast **ends against middle** at matched layer count, which at 1e18 means {2,3,8,9} against
{4,5,6,7}: matched layer count, matched resident-slot budget, and aligned with the shape that exists.
Not started, and not to be started without a decision.

**Implementation.** The R knob exists but is global —
[`temporal_router.py:359`](../../../temporal/temporal_router.py) reads `TEMPORAL_RESIDENCY_R` once
and applies it at every layer. `self.layer_number` is already in scope a few lines below, so a
per-layer schedule (an env-var list, or `R=E` sentinel per layer) is a small change. FLOPs are
unchanged at any R, so every arm above is compute-matched to the baseline.
