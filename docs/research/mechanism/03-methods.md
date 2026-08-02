# Methods: what each probe measures, and the traps

Definitions were previously repeated across three documents and had drifted between them. This is the
authoritative version. If a number elsewhere disagrees with a definition here, the definition wins and
the number is a bug.

The traps in §2 each cost us a wrong table or a wasted window. They are recorded so nobody rediscovers
them.

---

## 1. Probes

### Locus probes (A1–A3, C2) — `mechinterp_locus{,_1e19}.csv`

Ridge-linear probes predicting whether expert *e* fires at token *t*, from two feature sets:

- **`token_AUC`** — from the token's own input embedding.
- **`context_AUC`** — from the mean embedding of a window of preceding tokens, **excluding** *t*.
- **`context_minus_token`** — the difference. This is the statistic of interest.

Range 0–1 per expert per layer; 0.5 is chance. Higher AUC means the feature set predicts firing
better. **A positive `context_minus_token` means routing depends more on context than on the current
token**, which is the de-lexicalisation claim.

*Null:* iid permutation of the label vector (§2.2). *Floor:* 0.500.

*Limitations.* A linear probe failing is not proof the information is absent — that is what the C7
oracle exists to bound. Experts firing fewer than a threshold number of times are unprobeable and are
recorded in `mechinterp_locus_coverage.csv` rather than dropped silently.

### Causal token/context substitution (C8, N6) — `mechinterp_causal.csv`

The only *causal* probe in the set. Three arms on identical batches: `ref` (unmodified), `token`
(substitute the token at position *t* with a frequency-matched one), `context` (shuffle the preceding
context, hold *t* fixed). Scores position *t* only.

- **`token_jaccard_shift`** — how much the selected expert set at *t* changes under token substitution.
- **`context_jaccard_shift`** — the same under context substitution.
- **`context_over_token`** — their ratio. **Above 1 means context matters more than token identity.**

*Control:* the analysis refuses to compare arms whose input-id hashes differ, so a mismatched batch
fails rather than producing a silent comparison across different data.

*Limitations.* Jaccard shift is a set statistic and ignores gate magnitude. Frequency matching
controls token rarity but not syntactic role.

### Output logit lens (C5) — `mechinterp_lens{,_1e19}.csv`

Projects each expert's output through the unembedding to ask what vocabulary it promotes.
**`eff_vocab`** is the effective vocabulary size (entropy-derived) of that promotion. Lower means the
expert promotes a narrower, more lexically specific set.

*Limitations.* One expert's write is a nudge, not a prediction; medians shift modestly even when the
underlying distributions differ a lot. **This is the measurement whose 1e16 result did not replicate at
1e18** — see `02-corrections.md`.

### Structural family (A6–A9) — `mechinterp_structural{,_1e19}.csv`

Gate statistics per layer: selectivity (precision-recall of the gate), generalist fraction, router
entropy, plus **A8 weight geometry** (`dist2centroid_mean`, `pairwise_cos_med/p99`) computed from the
checkpoint rather than from a capture.

*Trap:* A8 needs the checkpoint, so `delex_structural.py` requires `scripts/env.sh` sourced **and** the
Megatron path. Run bare it degrades gracefully — writes the reason into `geometry_note`, blanks the
columns, exits 0. That silently emptied a completed measurement once; `csv_sanity.py`'s EMPTY COLUMN
check is what caught it.

### Cache hit rate (C4, e6) — `e6_per_layer_ranking.csv`

Fraction of demanded experts already resident. Higher is better. **This is the usable per-layer
locality signal.**

### Swap rate (e1) — `e1_swap_rate_by_layer.csv`

**Saturated at R=k and carries no depth signal.** See §2.5. Use hit rate instead.

### Demand forecastability (A10, C6) — `mechinterp_demand_1e19.csv`

Causal, history-only logistic probe: predict *y(t+1)* from the current gate, demand lags, and fast/slow
EMAs. No embeddings. AUC range 0–1, higher means demand is more predictable from its own past — the
property a prefetcher would exploit.

*Split:* document-disjoint (§2.3). Features are the label's own history, so a within-document split
lets the probe exploit that document's base rate.

### Token-id oracle (C7) — `mechinterp_oracle.csv`

Nonparametric ceiling on *any* function of the current token: score each token by its empirical firing
rate on the fit split. **Upper-bounds every token probe**, so it separates "routing is not lexical"
from "the linear probe lacked capacity". Also reports `mi_over_H`, calibration-free.

*Limitation:* ids seen fewer than `MIN_COUNT` times fall back to the global rate;
`frac_score_rows_backoff` reports how binding that is.

### Frequency-stratified token AUC (C9) — `mechinterp_freqstrat.csv`

Token AUC split by token frequency band — does the lexical shortcut live on rare or common tokens?

### Cross-layer probe transfer (C10) — `mechinterp_transfer.csv`

Fit a probe at layer *i*, score at layer *j*. Asks whether routing is the same function of the
embedding at every depth. Mean squared canonical correlation for subspace overlap.

### Constraint swap (C3, X1–X3, N1–N5) — `swap_sweep.csv`, `swap_shape.csv`

Inference-time perturbation of a **trained** checkpoint: impose residency on an unconstrained model
(`impose_one`/`impose_set`/`impose_all`) or lift it from a temporal one (`unmask_one`/`unmask_all`), and
measure test CE. `swap_shape.py` decomposes the per-layer profile into endpoint and interior terms.

**The `perturbation` column distinguishes `real` from `random` (sham) rows.** Reading the file without
filtering on it silently mixes them; that has happened.

*Fundamental limitation, and the reason T1 exists:* this measures the cost of **removing freedom from a
model trained expecting it**. It cannot measure the cost of never having had it, and the two differ —
the endpoint spike that dominates every C3 measurement is absent under co-adaptation
(`01-findings.md`).

---

## 2. Traps

### 2.1 Window variants decode differently in two files

**`base` means w=32 in `mechinterp_locus.csv` and w=k in `mechinterp_locus_1e19.csv`. `kfull` means
w=k in both.** Always read the `window` column rather than assuming from `variant`. This has produced
at least one wrong table.

### 2.2 Only the iid null is valid

| null | floor | verdict |
|---|---|---|
| iid permutation of labels | 0.500, worst deviation **0.0030** over 1,162 fits | **valid — use this** |
| circular shift | up to **+0.017**, scaling with window width | **invalid** |

The shift is invalid for a mechanical reason: the stream is flattened `[S·B]` with batch innermost, so
a circular shift never actually shifted along the token axis. It is retained in the battery as a
diagnostic only.

**Gate tolerance is ±0.002 — the same order as an under-powered estimate of it.** `max_experts=24`
gave the battery ±0.002 of its own sampling noise and flagged four healthy models; the default is now
256. A test whose noise floor equals its threshold flags healthy things forever.

### 2.3 Split semantics

- **`sequence`** — holds out whole documents. **Use this.**
- **`position`** — reproduces the published cut at 70% of the flattened stream, which lands mid-document
  and puts every document in both halves.

Note `mechinterp_locus_1e19.csv` currently holds **only** `sequence` rows; the locus driver last ran
without `--both-splits`.

### 2.4 When a slope is the wrong summary

A full-range OLS slope on a curve that turns over **reverses the comparison it appears to make** — that
error inverted an H1 regime comparison once. Report **curvature, vertex and restricted-range slope**
alongside, and check `linear_r2` against `quadratic_r2`.

On 3–5 layer models the quadratic vertex is **unidentified**: 20 of 24 vertex intervals spanned more
layers than the model has, one covering 550 layers of a 14-layer stack. `quadratic_r2` is the wrong
gate for this — a near-flat parabola fits beautifully and still cannot locate its own turning point.
The gate is identifiability: an interval wider than the stack is not reported, and
`vertex_identified` records the verdict.

### 2.5 Swap rate is saturated

At R=k a swap fires **iff at least one demanded expert is missing**, so `mean_swap_rate` is really
"fraction of tokens with ≥1 miss" and sits near 1.0 everywhere. **It carries no depth signal.** Use
hit rate (e6).

### 2.6 Byte-comparing rendered artifacts is not portable

The same code and data produced a 358,575-byte figure on one machine and 386,169 on another —
matplotlib version and font rasterisation. Last-digit floats differ too (`curvature_hi95` 0.0876 vs
0.0877). **Compare figure *content*** — every series the data supports must be plotted — **and numeric
CSVs with a tolerance.** A gate that cannot go green on both machines stops being run.

### 2.7 Estimating noise from arms other than the one under test

The failure that produced four wrong T1 claims. Arm-level sds spanned 0.0036–0.0204, so no pooled
figure described them; a two-seed spread predicted the three-seed sd badly (A0: 0.0040 → 0.0128).
**Report the contrast standard error from the arms in the contrast**, and treat any effect from one
seed per arm as a hypothesis.

---

## 3. Glossary

Identifiers are deliberately absent from `01-findings.md`. This is where they decode.

### A — descriptive probes over captures

| id | what | where the result lives |
|---|---|---|
| A1 | locus vs normalised depth | `01-findings.md` §1 |
| A2 | locus at full depth | §1 |
| A3 | context-minus-token by layer | §1 |
| A6 | gate selectivity | `04-coverage.md`, structural family |
| A7 | generalist fraction | structural family |
| A8 | expert weight geometry | structural family (needs checkpoints) |
| A9 | router entropy | structural family |
| A10 | demand forecastability | `mechinterp_demand_1e19.csv` |
| A11 | free-rider / tokens per expert | `mechinterp_freerider.csv` |

### C — the no-training test battery (all ten complete)

| id | what | result |
|---|---|---|
| C1 | replot on normalised depth with bootstrap CIs | done |
| C2 | locus at layers 2–14 on all captures | done, 30 runs |
| C3 | per-layer inference-time constraint swap | done; profile does not survive training (§T1) |
| C4 | baseline hit rate by counterfactual replay | done |
| C5 | per-layer output lens | done, 30 runs; 1e16 result did not replicate (`02-corrections.md`) |
| C6 | per-layer demand forecastability | done |
| C7 | nonparametric token-id oracle | done |
| **C8** | **causal token/context substitution** | **done, 6 runs / 3 cells — the strongest result in the program** |
| C9 | frequency-stratified token AUC | done |
| C10 | cross-layer probe transfer | done |

### N — round-2 tests

| id | what | result |
|---|---|---|
| N1 | sham-perturbation control | done; endpoint effect **58–85% positional** across two models |
| N2 | multi-layer schedule and additivity | done; with `impose_all`, exempting endpoints buys 31.9% for 25% of layers |
| N3 | C3 on a second seed at 1e18 | done; endpoint replicates at inference |
| N4 | C3 on fine granularity at 1e18 | done |
| N5 | C3 at 1e19, both directions, layers 2–14 | done; **falsifies H2a** |
| N6 | = C8 | done |
| N7 | per-layer cost vs churn | done |
| N8 | capture gaps | done, 30 captures |
| N9 | document corrections | done, `02-corrections.md` |

### T — training tests

| id | what | status |
|---|---|---|
| **T1** | per-layer constraint sweep with co-adaptation | **done — 24 cells, all three readings negative** |
| T2 | shallow vs deep half at 1e18 | **not run.** Design mis-specified (splits the U at its minimum) and its premise is dead |
| T3 | full per-layer resolution at 1e18 | not run; same premise |
| T4 | dense-final-block at 1e18 | not run; the endpoint effect it targets is 58–85% positional |

### X — cross-regime sweeps

| id | what |
|---|---|
| X1 | sham arm of the constraint swap |
| X2 | per-layer schedule sweep |
| X3 | uniform-R dose curve (`swap_sweep.csv`, `dose_*` arms) |

### e — offline replay family

| id | what | file |
|---|---|---|
| e1 | swap rate / burst length | `e1_swap_rate_by_layer.csv` (saturated — §2.5) |
| e2 | streamed expert diversity | `e2_streamed_diversity.csv` |
| e3–e5 | eviction headroom, retained mass | `e3_*`–`e5_*.csv` |
| e6 | **cache hit rate by layer** | `e6_per_layer_ranking.csv` |
| e7 | demand smoothing | `e7_*.csv` |
| e8 | document-boundary churn | `e8_document_boundary.csv` (needs the EOD masks) |

*The e-family numbers in `archive/probe-replay-e1-e8.md` were computed on runs no longer on disk and
cannot be reproduced; the metrics have since been recomputed over the 22 preserved router logs.*
