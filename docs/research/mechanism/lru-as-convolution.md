# Eviction as a temporal filter: what the convolution reading of rolling residency buys

**Status: analysis + free (CPU, committed-data) replay. Two derived policies implemented and
unit-tested; nothing here is a trained result, so nothing here falsifies anything.** Per the
repo's epistemology (`ablations/decision-time-alignment.md`), free tests may promote ideas and
never kill them.

Prompt: *if the eviction policy is LRU rather than lowest-scoring, rolling residency becomes a
convolution over experts through time.* That is right, and it is exact rather than metaphorical —
but the exactness lands on a different object than expected, and it explains the measured result
(`lru` is worse) instead of overturning it. This note makes the correspondence precise, uses it to
classify every eviction rule as a filter, and reports what falls out.

**What the framing buys, in one table.**

| Claim | Kind | Where |
|---|---|---|
| The shipped `lru` refreshes only on **admission**, so it is FIFO, not LRU — a box kernel of width `k` over the admission stream. Residency is a pure function of admission *order*. | theorem, unit-tested | §1 |
| Corollary: under `lru` every expert is evicted at age **exactly `k` admissions**. Residency time is allocated uniformly, so demand strength cannot buy residency. That is the expressivity gap. | theorem, unit-tested | §1.3 |
| The shipped `min_logit` realises a **heavy-tailed, effectively multi-scale** kernel: median age 3 admissions, 40–45% of live slots held past age `k`, tail to 143. | measured (K1) | §3.1 |
| Repairing `lru` to a real LRU — **refresh on demand, not on admission** (`evict=lrd`) — recovers **73–80%** of its coverage deficit at the same swap rate, and is the cheapest arm on the diversity axis. | measured (K3), promote-only | §4.1 |
| A **width-tuned demand kernel** (`evict=ema`, τ≈8 tokens) reaches parity with `min_logit` using *no logit magnitudes at all* — only the binary demand history. Width has an interior optimum; too narrow and too wide both lose 5–15 pt. | measured (K3), promote-only | §4.2 |
| The eviction kernel is **not** the bottleneck. Demand recurs within 6 tokens 74–77% of the time, yet even Belady reaches only 49–52%: the binding constraint is the 1-admission-per-token sampler. The filter is fine; the input is undersampled. | measured (K2, K3) | §3.3 |
| The one structurally new transfer: a learned nomination kernel with **weight sharing across the expert axis** is permutation-equivariant, so a per-expert popularity table is *unrepresentable* — the exact failure mode that killed the H1/H2 heads and that H3 had to suppress by hand. | proposal | §5.2 |
| Negative: a **multi-scale kernel bank** (fast+slow mixture, the standard CNN prescription) buys nothing over the best single scale — every mixture lands inside the single-scale CI. | measured (K3) | §6 |

Code: `analysis/probes/kernel_replay.py` (all measurements, numpy-only, no GPU, no `router_log.pt`),
`temporal/temporal_router.py` (`evict` ∈ `lru | min_logit | lrd | ema`),
`temporal/tests/test_temporal_router.py` (the theorems of §1 are executable tests).
Data: `results/ablations/k1_*.csv`, `k2_demand_renewal.csv`, `k3_eviction_kernel_replay.csv`,
`k4_signal_bandwidth.csv`.

---

## 1. The correspondence, made exact

### 1.1 `evict="lru"` is FIFO

`compute_resident_mask` keeps a per-expert `refresh` timestamp and evicts the resident with the
smallest one. `refresh` is written in exactly one place:

```python
refresh = refresh.masked_fill(nominee, float(k + t))     # only the newly ADMITTED expert
```

Nothing refreshes an expert for being *used*. So `refresh` is admission time, and "evict the
smallest refresh" is "evict the earliest admitted" — a FIFO queue, or in cache terminology LRI
(least recently *inserted*).

This is not sloppiness; at `K = k` it is forced. Every resident is selected by every token
(top-`k` of `k` residents is all of them), so "recently used" carries zero information and the
implementation had nothing else to key on. The informative recency signal at `K = k` is not *use*
but *demand* — membership of the token's **unconstrained** top-`k`, which the router already
computes to find its nominee. That observation is the whole of §4.1.

### 1.2 Residency is the box convolution of the admission stream

Let `a_1, a_2, …` be the experts admitted, in order (the cold fill counting as `k` pseudo-admissions
ordered by logit), and let `m(t)` be the number of admissions up to token `t`.

> **Proposition.** Under `evict="lru"`, `R_t = {a_{m(t)-k+1}, …, a_{m(t)}}` — the `k` most recent
> admissions, always exactly `k` distinct experts.

*Proof.* `refresh` values are strictly increasing in admission index, and the cold-fill ranks
`k-1 … 0` sit below every subsequent admission value `k + t`. Eviction removes the argmin, i.e. the
oldest admission. An expert is only admitted while non-resident, and once admitted it is evicted
exactly when it becomes the oldest of the `k`, which is at the `k`-th subsequent admission — by
which point its own admission has already left any window of `k` consecutive admissions. Hence no
duplicates in the window and the set is exactly the last `k` admissions. ∎

Writing `n_t ∈ {0,1}^E` for the one-hot admission at step `t`, the residency indicator is

```
    r_t(e)  =  (n * box_k)(m(t))[e]          box_k(j) = 1[0 ≤ j < k]
```

— a **convolution of the admission stream with a rectangular kernel of width `k`**, on the
*admission clock*. Since the measured swap rate is ≈1.0/token (E1), the admission clock and the
token clock coincide, which is why the intuition reads as "a convolution through time".

Pinned as `test_lru_is_a_box_kernel_of_width_k_over_the_admission_stream`.

### 1.3 What that costs: residency time is allocated uniformly

> **Corollary.** Under `evict="lru"` every expert's residency lasts **exactly `k` admission
> events**, regardless of how strongly it is wanted.

Pinned as `test_lru_evicts_at_age_exactly_k_admissions`. This is the precise form of "strictly less
expressive". The box kernel is *linear time-invariant in the admission stream and blind to the
scores*: the only thing an expert's demand can influence is whether it gets admitted, never how
long it stays. Total residency (`k` slots × `T` tokens) is divided equally among admissions.
`min_logit`, by contrast, is a **content-adaptive** rule — the same jump as fixed convolution →
dynamic (input-conditioned) convolution. The measured −7 pt (E5) and +0.004–0.006 BPB
(`FINDINGS.md` §5) are what that jump is worth here.

The diversity side of the same coin: FIFO maximises expert diversity by construction (equal
residency per admission), and it *still* loses on trained quality. That is a small but real
addition to the alignment↔diversity synthesis in `ablations/alignment-program.md`: quality is not
monotone in diversity. Every mechanism in that program moved *up* the alignment axis and paid
diversity; `lru` moves *down* it and gains diversity, and loses too. The shipped policy sits near
an interior optimum, not at the end of a monotone trade.

### 1.4 The general form: eviction as a depthwise temporal filter

Both shipped policies, and everything in §4, are the same object with a different kernel. Let
`d_e(t) = 1[e ∈ unconstrained top-k(t)]` be expert `e`'s demand channel and `s_e(t)` its logit.
Every rule is

```
    evict  argmin_{e ∈ R_t}  (h * x_e)(t)
```

for some causal kernel `h` and per-expert signal `x_e`, with the kernel **shared across experts** —
a depthwise 1-D convolution over `E` channels, exactly the CNN object:

| policy | signal `x_e` | kernel `h` |
|---|---|---|
| `min_logit` (shipped) | logit `s_e` | δ (width 1) |
| `lru` (shipped) | admission indicator | `box_k` |
| `lrd` (§4.1) | demand `d_e` | "time since last 1" (a recency functional, not linear, but the same width knob) |
| `ema` (§4.2) | demand `d_e` | exponential, `γ(1−γ)^j`, τ = 1/γ |
| `box-W` (§4.2) | demand `d_e` | `box_W` |
| Belady | demand `d_e` | acausal δ at the next demand |
| discounted-oracle(γ) | demand `d_e` | **acausal** exponential |

Three things fall straight out of the table:

1. **Width is a free parameter that nobody had swept.** `min_logit` and `lru` sit at opposite,
   untuned extremes (width 1 on the score; width `k` on a signal that ignores the score).
2. **The trigger and the eviction key have separate kernels, but the code ties them.**
   `TEMPORAL_EMA_BETA` smooths the whole trigger stream, so under `min_logit` it widens the
   eviction kernel *and* the nomination kernel together, and under `lru` it widens the nomination
   kernel only (the box on admissions is untouched by score smoothing). Decoupling them is a
   one-line change and is what `evict=ema` does.
3. **The headroom in E5 is the acausal half of the kernel.** Belady and discounted-oracle differ
   from every deployable rule by being non-causal. In filter terms, centring the kernel is worth
   +20–30 pt while improving the causal half is worth +6–10 pt — E5's conclusion, restated as a
   property of the filter rather than of caching.

### 1.5 Where the analogy stops

Worth stating so the frame is not over-extended:

- **There is no expert-axis convolution.** Experts are permutation-symmetric; a convolution over
  the expert index requires an induced topology (e.g. a co-activation graph, making it a graph
  conv). That direction is already benched as the co-activation nomination prior. What the expert
  axis *does* support is **weight sharing** — see §5.2, which is where the real transfer is.
- **The `argmin` is a nonlinearity outside the filter.** Residency is `top-k ∘ filter`, so
  linear-systems reasoning (frequency response, superposition) applies to the key, not to the
  resident set.
- **A separable (time ⊗ expert) approximation is refuted by data.** Separability would mean a
  static per-expert prior times a temporal envelope, i.e. pinning; E2 measured no expert above the
  pinned threshold. The rank-1 factorisation of the residency field is not available.
- **The cap-1 swap is a sampler, not a stride.** A stride subsamples the output; cap-1 limits how
  fast the *state* can change. §3.3 is about that distinction and it is the load-bearing one.

---

## 2. Measurement setup

`analysis/probes/kernel_replay.py` runs off `results/ablations/expert_selection_per_token.csv`: for
three trained temporal models (8.1M / 15M / 38M active, coarse 6-of-64) it holds, per token, the
unconstrained top-`k` demand and the resident set the shipped `min_logit` policy actually served —
**220 tokens, deepest MoE layer, one sequence.** No `router_log.pt`, no GPU, numpy only.

That is a thin slice and every number below carries a moving-block bootstrap 95% CI (block 32; the
per-token series is strongly autocorrelated and an iid bootstrap understates the interval several
fold). Treat 5-pt differences as real, 2-pt differences as not.

Two nomination protocols, because the eviction rule cannot be varied without deciding what gets
admitted:

- **`trace`** — admit exactly what the shipped router admitted. Identical input signal, different
  kernel: the cleanest controlled comparison, but it feeds `min_logit`'s own nominations into
  policies whose state has diverged, which handicaps them.
- **`selfnom`** — each arm nominates from its own state (highest recent-demand EMA among this
  token's demanded non-residents). Self-consistent and the fairer read; it is what §4 quotes.

The harness cross-validates against the published E5 table, which ran the full probe batch over all
layers. Belady beats shipped by **+7.3 / +5.2 / +5.8 pt** here against E5's **+8.6 / +6.0 / +10.0**;
`lru` sits **6.4 / 7.1 / 6.0 pt** below `min_logit` under `selfnom` against E5's **6.7 / 7.9 / 3.9**.
Same sign, same size, on an independent slice. (E5's middle model is the fine-grained G3 3.9M, not
this slice's coarse 15M, so pair the outer two models when comparing cell by cell.)

---

## 3. What the measurements say

### 3.1 K1 — the shipped policy's kernel is heavy-tailed, not box

Age at eviction, in admission events (`k` = 6, so FIFO is a spike at exactly 6):

| model | mean | median | P(age < 2) | P(age = k) | max | live slots older than `k` |
|---|---|---|---|---|---|---|
| 8.1M | 4.73 [4.43, 5.16] | 3 | 0.20 | 0.06 | 28 | 42% |
| 15M | 4.97 [4.65, 5.32] | 3 | 0.20 | 0.07 | 41 | 40% |
| 38M | 5.80 [4.33, 7.80] | 3 | 0.24 | 0.04 | 143 | 45% |

The survival curve (`k1_kernel_survival.csv`) falls to 0.43 by age 3 and then has a long tail —
against FIFO's step from 1 to 0 at 6. Two readings:

- The greedy rule spontaneously runs a **two-population cache**: a churning fraction (a fifth of
  admissions are gone within one further admission — `min_logit` genuinely thrashes, which is what
  recency eviction was supposed to prevent) alongside a persistent core (40–45% of live slots are
  held past the FIFO horizon). FIFO can represent neither population; it holds every slot for
  exactly one horizon.
- The mean age is *close to* `k`, so the two policies agree on the average residency time and
  differ almost entirely in how they **allocate** it. This is the expressivity gap in one number.

### 3.2 K4 — the residency process integrates ~3× the demand correlation time

| model | τ demand (tok) | τ residency (tok) | ratio |
|---|---|---|---|
| 8.1M | 2.54 | 6.13 | 2.4 |
| 15M | 2.96 | 7.85 | 2.7 |
| 38M | 3.13 | 10.15 | 3.2 |

The realised residency has an integrated autocorrelation time of 6–10 tokens against demand's
2.5–3.1. So `min_logit`'s "width-1" kernel is only width-1 on the *score*; because the router's
logits are themselves a smooth function of context, its **effective** integration window is ~8
tokens. That is the same window the explicit demand kernel finds optimal in §4.2 — a satisfying
consistency, and the reason the demand-only kernel can match the logit-based rule.

### 3.3 K2/K3 — the kernel is not the bottleneck; the sampler is

Inter-demand intervals are short: median 1–2 tokens, mean 6.1–7.1, and `F(6) = 0.74–0.77` — three
quarters of all demands recur within one FIFO horizon. A cache that admitted every demanded expert
would therefore cover ~75% at `K = k`. Measured coverage is 37–45%, and **offline-optimal eviction
reaches only 49–52%**.

So the loss is not in choosing victims. It is that each token has `k` demands, ~3 of them missing,
and the cap-1 invariant admits **one**. The residency state simply cannot track the demand process
at the rate the demand process moves. In filter language the input is undersampled, and the two
repairs for an undersampled causal filter are a higher sample rate (more swaps per token — excluded
by the cap-1 design invariant) or **lookahead**, i.e. an acausal kernel, which in deployment means
prediction. This is E5's "invest in anticipation, not eviction" derived rather than observed, and
it bounds everything in §4: kernel work is worth single-digit points because that is all the causal
half of the filter is worth.

---

## 4. Optimizations for the LRU policy

Quoted at the fair (`selfnom`) protocol; `[·]` is the 95% block-bootstrap CI; `eff` is the
effective-expert count (the program's diversity guardrail) over the same window.

| policy | 8.1M | 15M | 38M | eff (8.1 / 15 / 38M) |
|---|---|---|---|---|
| `lru` = FIFO = box_k | 38.6 [35.9, 44.5] | 38.2 [36.1, 41.1] | 37.1 [30.5, 42.2] | 24.3 / 28.0 / 25.9 |
| **`lrd`** (refresh on demand) | **43.5** [41.2, 49.0] | **43.4** [41.2, 46.4] | **41.9** [33.6, 47.9] | 22.5 / 26.1 / 25.2 |
| `ema` demand, τ≈8 tok | 43.5 [41.2, 49.3] | 45.1 [42.2, 48.5] | 44.4 [36.4, 50.4] | 19.6 / 22.1 / 21.5 |
| `min_logit` (shipped) | 45.0 [43.0, 49.8] | 45.3 [42.6, 48.8] | 43.1 [35.0, 49.2] | 23.9 / 27.1 / 26.2 |
| Belady (oracle) | 52.3 [50.7, 56.5] | 50.5 [48.9, 52.6] | 48.9 [41.5, 53.9] | 23.0 / 26.7 / 23.7 |

### 4.1 Refresh on demand, not on admission (`TEMPORAL_EVICT=lrd`) — the headline

One line: also stamp `refresh[e] = t` for every resident in the token's unconstrained top-`k`. The
signal is already computed (the trigger ranks all `E` experts to find its nominee), so the cost is
one `topk` and one `where` in the reference scan.

Recovers **+4.9 / +5.2 / +4.8 pt**, i.e. **76% / 73% / 80%** of the gap from `lru` to `min_logit`,
at an unchanged swap rate. Under the `trace` protocol the effect is larger still (26.3 → 41.8,
24.1 → 41.6, 23.1 → 34.8), because that protocol punishes the box kernel hardest.

Why it works, in the frame: FIFO's kernel is blind to demand, so demand cannot buy residency; `lrd`
restores exactly that channel — the *only* one the box kernel is missing — while keeping the recency
structure. It also inherits an invariant the box kernel lacks: when a swap fires, at least one
resident is outside the token's top-`k` and every demanded resident carries the current timestamp,
so **`lrd` never evicts an expert the current token still wants**. `min_logit` gets the same
invariant for free (the worst resident is by definition below the nominee, hence outside the top-`k`);
`lru` does not, which is the mistake traced in `test_lru_and_min_logit_diverge_then_reconverge`, where
it spends a second swap undoing the first. `lrd` is also the cheapest arm on the diversity axis: its
`eff` sits within ~1 point of `min_logit`'s, whereas the `ema` arms pay 4–5 points of `eff` for the
same coverage.

**This changes what the `lru` ablation measured.** The trained eviction cells
(`tmoe_lru_sh1/sh2_s0_1e16`, +0.0042 / +0.0063 BPB vs `min_logit`) and the E5 replay row labelled
"LRU" both ran the FIFO policy. "LRU is 7 pt worse — do not run"
(`ablations/decision-time-alignment.md` §7) is a correct statement about *insertion-order* eviction;
textbook LRU at `K = k` has never been trained. Whether it closes the +0.004–0.006 BPB gap is an open, cheap
question, and the control pair for it already exists: one s0@1e16 coarse cell with
`TEMPORAL_EVICT=lrd`, read against `tmoe_minlogit_sh1_s0_1e16` (1.4599) and
`tmoe_lru_sh1_s0_1e16` (1.4641) at the same shape, seed and budget. Seed noise on plain temporal at
s0 is ~0.0005, so the 0.0042 gap is 8x noise and a single seed resolves the direction.

### 4.2 Tune the kernel width (`TEMPORAL_EVICT=ema`, `TEMPORAL_EVICT_GAMMA`)

Evict the resident with the smallest causal EMA of its demand indicator; ties break on `lru` order,
so the rule degenerates to FIFO exactly when its kernel is uninformative. Width sweep (`selfnom`,
15M; the other two models have the same shape with the optimum at τ ≈ 8–16):

| kernel | box W=1 | box W=8 | box W=16 | box W=64 | EMA τ≈2 | τ≈8 | τ≈16 | τ≈32 |
|---|---|---|---|---|---|---|---|---|
| coverage | 41.5 | 44.0 | 44.4 | 43.0 | 44.1 | **45.1** | 44.8 | 44.8 |

Two results:

- **Width has an interior optimum at τ ≈ 8 tokens**, which is the residency autocorrelation time
  measured independently in §3.2 and also the `γ_m = 1/8` that the momentum program picked on
  expert-lifetime grounds. Too narrow (W=1) and too wide (W≥32 under the `trace` protocol, which
  resolves the tails better: 28–32%) both lose 5–15 pt. Nobody had swept this axis.
- **The optimum matches `min_logit` while discarding the logits entirely.** The eviction key is a
  filtered binary indicator; it does not see gate magnitude at all, and it ties or beats the
  quality-greedy rule on two of three models. So `min_logit`'s advantage over `lru` is *not* the
  score magnitude — it is having any demand-sensitive kernel at all, which §4.1 supplies more
  cheaply.

Cost: the `ema` arms buy coverage partly with concentration (`eff` 19.6–22.1 against `lru`'s
24.3–28.0). On the alignment↔diversity frontier they sit where every other selection-shaping
mechanism has sat. `lrd` does not, which is why it is the recommendation.

### 4.3 Ranked recommendations for `lru`

1. **`lrd`** — implemented, tested, free, recovers ~³⁄₄ of the deficit, and lands within ~1.4 pt of
   `min_logit` on `eff` (i.e. it buys the coverage without the concentration the other arms pay).
   Run the one training cell.
2. **`ema` at τ≈8** — implemented, tested, reaches `min_logit` parity, but pays `eff`. Worth a cell
   only if `lrd` shows life, and it must report the diversity guardrail.
3. **Per-layer width.** E6 measured locality rising with depth (16–29% shallow vs 36–48% deep) and
   §3.2 measures τ_residency rising with model size; a per-layer τ (wider kernel where demand is
   less predictable) is the natural non-uniform allocation. Not implemented — it needs a per-layer
   knob surface, and the expected size is small.
4. **Do not** pursue: victim caching (already characterised, E1, and a systems freebie rather than a
   policy change), τ-margin as a kernel-width proxy (benched under the cap-1 invariant — a skipped
   swap is free, so a margin buys nothing), or a learned Belady-imitation eviction cache (bounded at
   +6–10 pt by E5, and §3.3 explains why that bound is structural).

---

## 5. Transfer to the lowest-scoring policy

### 5.1 What does not transfer

`min_logit` already realises a heavy-tailed, effectively ~8-token kernel (§3.1, §3.2), so the
width knob has nothing left to give it: no kernel in the sweep beat it outside the CI, and the
multi-scale bank that should have helped did not (§6). Two further collisions worth naming
explicitly, since both are natural CNN-derived moves and both are closed directions:

- **Age-weighted eviction** (`argmin` of `score + λ·f(age)`, the "fixed + dynamic kernel" hybrid
  that dynamic-convolution work converges on) is incumbency bias — closed by the wrong-directions
  header as cache-referential selection pressure.
- **Co-activation-smoothed keys** (a graph convolution on the expert axis) is the benched
  co-activation prior.

Neither is re-opened here. Note the distinction that keeps §4 clear of that header: `lrd` and `ema`
key on the *router's own demand history*, never on cache state — the same demand-referential
discipline as the momentum family, applied to eviction rather than to the trigger.

### 5.2 What does transfer: weight sharing on the expert axis

The one genuinely new thing the frame offers is not a kernel shape but the CNN's *other* structural
commitment. A convolution shares weights across positions, which makes it translation-equivariant
and makes memorising position impossible. The analogue here is sharing a learned kernel across the
**expert axis**, which makes the rule permutation-equivariant and makes memorising *expert identity*
impossible.

That is precisely the failure mode the nomination-head program hit. H1/H2 learned "a static
popularity table … incumbency in disguise" (`ablations/local-global-program.md`); H3 had to make
popularity unlearnable by centring the *labels* on each expert's own running baseline, and only then
produced the program's first genuine A3 gate pass (+4.6 pt of real anticipation, BPB-neutral). A
per-expert head `W_f ∈ R^{E×d}` has one row per expert and can therefore represent popularity; a
kernel shared across expert channels **cannot represent it at all**, because permuting the experts
permutes the outputs identically. H3's hand-imposed fix becomes a structural property.

The concrete proposal, which is a specific form of the program's own named-but-explicitly-unapproved
next rung (freeze a fitted history predictor and use it as the trigger's nomination score, motivated
by the anomaly-predictability probe's AUC 0.70–0.87 from *history* features — "history is more
informative than the hidden state here"):

> **Nomination by a shared causal temporal kernel.** Per MoE layer, one small causal 1-D kernel
> `w ∈ R^L` (`L ≈ 8–16`, optionally dilated) applied depthwise to every expert's demand channel
> `d_e`, plus a shared bias. Score `ŷ_t(e) = σ((w * d_e)(t))`; nominate `argmax` over
> non-residents. `L + 1` parameters per layer, shared across all `E` experts. Train it against the
> discounted future-demand target (`anticipatory_target`, γ ≈ 0.5–0.75) with the gradient stopped at
> the kernel, exactly as the H1 head does — but with popularity structurally out of the hypothesis
> class.

Why this is worth a rung rather than another frontier point:

- It targets **nomination**, where §3.3 and E5 agree the headroom is (+20–30 pt), not eviction
  (+6–10 pt).
- It is Goodhart-immune the same way the stop-grad head is (the loss cannot change the demand
  process) *and* popularity-immune the way H3 is, without needing centred labels.
- The oracle-A3 probe showed a high-alignment, high-diversity policy exists (+17.7–18.7 pt A3 at
  `eff ≈ 176`), and the diagnosis was that the frontier "is where weak predictors land". A
  permutation-equivariant predictor cannot spend its capacity on the popularity shortcut, so it is
  a specific hypothesis about *why* the previous predictors were weak.
- Falsifiable and cheap: fit the kernel offline on preserved router logs first (free, CPU), report
  AUC against the same target as `anomaly_pred.csv`, and only train a cell if the offline fit beats
  the fitted-logistic history baseline. If a shared kernel cannot beat a per-expert one offline, the
  hypothesis is dead before any GPU time.

**Caveat that bounds it.** Demand history alone is content-blind; the oracle's advantage includes
information no causal filter has. Expect the shared kernel to capture the autocorrelated part of
demand and not the content-driven part, so the honest prior is "part of the +18 pt", not all of it.
Composing it with the hidden state (a content term added to the shared temporal kernel) reintroduces
per-expert parameters and with them the popularity channel — so if it is composed, the content term
should be centred, H3-style.

---

## 6. What the framing does not buy (negatives worth recording)

- **Multi-scale kernel bank.** The standard CNN answer to a heavy-tailed response is a bank of
  scales. Two-scale mixtures (fast τ≈2 + weighted slow τ≈32, weights 0.5–4.0) land inside the
  single-scale CI on all three models under both protocols — 44.0 / 44.3 / 43.7 / 42.9 at 8.1M
  against the best single scale's 43.5, on a CI of ±4 pt. No gain outside noise. The heavy tail in §3.1 is apparently a *consequence* of greedy
  scoring on an autocorrelated signal, not something a kernel bank has to be built to produce.
- **Anti-aliasing before decimation.** The DSP prescription (low-pass, then subsample) predicts
  that smoothing the trigger before the cap-1 swap should help. E7 measured the swap-rate cut, and
  B1 measured the quality: eval-time trigger smoothing that changes *which* experts serve tokens
  costs +0.08 BPB off-policy. The prescription is right about the swap count and wrong about the
  quality, because the "aliasing" here is not noise to be removed — the high-frequency component of
  demand is signal.
- **Stride / windowing.** Strided convolution is the block-routing scheme (`LG3`), declined by free
  replay: freezing on stale demand costs 16–35 pt retained mass at every block size.
- **Separability / low-rank residency.** Refuted by E2 (no pinning), as noted in §1.5.
- **Depth as receptive field.** Tempting (stacked small kernels ⇒ large receptive field) but the
  layers' residency states do not compose along time in the way that argument needs; each layer runs
  an independent scan. The E6 depth trend is real but is a property of the demand signal, not of a
  composed filter.

---

## 7. Reproduce

```bash
scripts/setup.sh analysis && . scripts/env.sh
$PY analysis/probes/kernel_replay.py          # K1-K4, ~4 s, writes results/ablations/k*.csv
PYTHONPATH=.:analysis:analysis/probes:analysis/plots $PY -m pytest \
    temporal/tests/test_temporal_router.py analysis/probes -q
```

One incidental fix went in alongside: `probe_replay.replay` stamped admissions with `t` while the
router stamps `k + t`, so for the first `k` tokens of a sequence its `lru` arm could rank a cold-fill
expert as newer than a fresh admission. Corrected to match the router and pinned by a new
cross-framework test over all four policies. The effect is confined to the first `k` of 2048 tokens
— on a 300-token synthetic check it moves `lru` coverage by 0.14 pt and leaves the swap stream
identical, and it does not touch `min_logit`, `belady` or `discounted`, so no published number
moves materially.

With `router_log.pt` pulled (`scripts/artifacts.py`), `probe_replay.replay(..., evict="lrd")` and
`evict="ema"` extend the E5 headroom table to the derived kernels over all layers and the full
probe batch — the right next free measurement, and the one that would turn §4's promote-only
numbers into a proper policy row.

Training-cell form of the §4.1 recommendation:

```bash
# same shape/budget/seed as the published eviction ablation pair (GRAIN=1, one shared expert)
SHAPE=s0 TARGET_FLOPS=1e16 TEMPORAL=1 TEMPORAL_EVICT=lrd bash experiments/run.sh
```

`lrd` and `ema` are reference-path only: `compute_resident_mask_accel` routes them to the eager
scan because the Triton and CUDA-graph kernels take a single `use_lru` boolean and would otherwise
run `min_logit` under an `lrd` label. That is a hard gate with a test
(`test_accel_never_silently_downgrades_a_reference_only_policy`), not a fallback — but it does mean
these policies run at the slow scan speed until someone extends the kernels.
