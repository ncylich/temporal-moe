# Residency as a recurrence: predicted-demand keys, the RNN view, and the guarantee we never had

Third note in the sequence. [`lru-as-convolution.md`](./lru-as-convolution.md) measured the kernel;
[`residency-convolution-theory.md`](./residency-convolution-theory.md) proved the kernel family is
exhausted. This one answers the two questions that follow: **can a predicted-demand key beat it**,
and **what does treating the resident set as a recurrent state buy**.

Short answers. The predicted-demand key is the right shape and has the largest measured ceiling in
the whole program (+20–30 pt), but the theory says four things about how to build it that the three
failed anticipation attempts all got wrong — and one of them, a **robustness guarantee**, is the
reason they failed rather than a detail of how they were built. The recurrence view yields a
policy-independent latency theorem that retrodicts a measured result, locates the missing gradient
path, and — the most valuable item here — shows that **we are already running the right algorithm
for the wrong objective class**.

Tightness grades as before: **[exact]** the theorem is about our object literally; **[structural]**
same structure, different object, transfer is a stated modelling assumption.

---

## Part I — The predicted-demand key

### I.0 The ceiling already exists, and it is the largest number in the program

The proposal — key each expert by an EMA of its *predicted* demand — is not a new object. Replace
"EMA of the past" with "EMA of the future" and you have E5's `discounted-oracle(γ)`:

```
    y_t(e) = Σ_{j≥1} γ^{j−1} · 1[e ∈ top-k(t+j)]
```

which is an **anticausal exponential filter of the demand indicator** — structurally identical to
`evict=ema`, run forwards in time. Measured set coverage: **52.7–66.5%** against `min_logit`'s
30.4–38.2% and Belady's 40.4–46.8%.

Two things follow immediately. First, the ceiling for this idea is the biggest in the measured map,
roughly 3× the eviction-policy ceiling. Second — and this is the detail that tells you the design —
the discounted oracle **beats Belady**, which is supposed to be optimal. It does so because in E5's
implementation `y` drives *nomination as well as eviction*, and because Belady is optimal for a
different problem (minimising faults with unconstrained admission) than ours (maximising pre-swap
coverage under cap-1 admission). So the win is not coming from smarter victim choice.

### I.1 Use it to nominate, not to evict [exact — this is our own measurement]

E5 decomposes the headroom cleanly: better *eviction* is capped at +6–10 pt (offline-optimal), better
*anticipation* is worth +20–30. The previous note derived why — demand recurs within 6 tokens
74–77% of the time, yet even offline-optimal eviction reaches only 49–52%, because each token has ~3
missing demanded experts and cap-1 admits one. **The binding constraint is which expert gets admitted,
not which gets dropped.** The same predictor is worth about three times more on the admission side,
and the proposal as stated ("use it to evict") spends it on the cheap half.

### I.2 There are two γ's doing different jobs; do not fuse them [exact]

"EMA of predicted demand" contains two exponentials with unrelated meanings, and the program has
already been burned once by tuning a coupled pair (Karen's `γ_m` / `γ_q`):

- **The discount `γ_h`** inside the target `y_t(e)`: *what horizon do I care about.* Bounded below by
  actionability (cap-1 means only the next swap or two is preparable) and above by predictability
  decay. E5 measured short horizons winning (γ=0.5 ≈ 2 tokens); the horizon map later re-targeted to
  0.75.
- **The filter gain `γ_f`** applied to the *estimate*: *how much do I trust this number.* This is
  variance reduction on a noisy estimator, not additional lookahead.

Smoothing an already-discounted future estimate with a second EMA does not extend the horizon. If
the two are swept jointly, neither is identifiable.

### I.3 The filter gain is determined by SNR, not swept [exact]

For a signal with autocorrelation time `τ_s` observed through noise, the optimal causal linear
estimator is the Wiener/Kalman filter, and **for an exponentially-correlated (AR(1)) signal that
filter is exactly an EMA**, with gain fixed by the signal-to-noise ratio. Three consequences, and
they tie the whole policy family together:

1. **If demand were AR(1), the EMA family would be optimal and there would be nothing to win from a
   fancier kernel.** The heavy-tailed eviction-age distribution we measured says it is not — which
   is the curse-of-memory / HiPPO point from the previous note, now with a *reason*: the EMA is the
   optimal filter for the wrong process model, not a crude approximation to the right one.
2. **Noisier observation ⇒ slower optimal filter.** This quantitatively predicts what the sweep
   found: the optimal filter time constant was ≈8 tokens against a measured demand autocorrelation
   time of 2.5–3.1 tokens. A ~3× slower filter is what you expect when the observation is a binary
   indicator of a continuous latent demand.
3. **As the predictor improves, smooth less.** `min_logit` is the `γ_f → 1` limit — appropriate for a
   high-SNR signal, which the logit is. `evict=ema` is the low-SNR limit. **The three policies are one
   filter at three SNRs**, and a good learned predictor should sit near the `min_logit` end, not the
   heavily-smoothed end. That is the opposite of the intuition that a prediction needs more smoothing
   than an observation.

### I.4 Train it as a ranking problem — and the sufficient statistic may be one bit [structural]

By the rank-factorisation proposition (previous note §1.3), the policy consumes only the *argmin over
the `k` residents* / *argmax over the `E−k` non-residents*. It never uses a calibrated magnitude.
So a BCE-against-discounted-demand objective — used by Track B, by H1/H2, and by H3 — is solving a
strictly harder problem than the one that matters, and it spends its capacity on the part of the
target that dominates the loss: **the stationary marginal, which is popularity.** That is not a
coincidence of those three cells; it is what a pointwise calibration loss on a heavily-imbalanced
target does. The program diagnosed the symptom ("every A3 gain ever measured is popularity
concentration in some costume") and fixed it per-mechanism; the loss class is the common cause.

Two results from the learned-caching literature say what to do instead:

- **PARROT** ([Liu et al., ICML 2020](https://proceedings.mlr.press/v119/liu20f.html)) imitates
  Belady with a **softmax over the cache lines** — a listwise ranking objective over the resident
  set, imitating the oracle's *choice*, not its reuse distances. It beats LRU by 61% hit rate on a
  web-search workload. This is the direct analogue of what our nomination head should be: a
  distribution over candidates, trained on the oracle's argmax.
- **Paging with Succinct Predictions** ([Antoniadis et al., ICML 2023](https://proceedings.mlr.press/v202/antoniadis23a.html))
  is sharper and, I think, the most useful single result for this question. It proves that **one bit
  per request** — "is it safe to evict this page" — suffices for an algorithm that is simultaneously
  consistent, robust and smooth, **with matching lower bounds**. The sufficient statistic for
  near-optimal paging is a *binary relative judgment*, not a real-valued demand estimate.

That collapses the learning problem in exactly the direction the program needs. A binary
safe-to-evict label is defined relative to the current cache contents, so a globally-popular expert
has no advantage — popularity is not a shortcut to the label, structurally, in the same way that
expert-axis weight sharing makes it unrepresentable. Two independent cures, and they compose.

### I.5 The missing guarantee — and why every anticipation cell failed [structural, and the important one]

[Lykouris & Vassilvitskii](https://arxiv.org/pdf/1802.05399) (ICML 2018 / JACM 2021) prove the
result that should reframe the whole anticipation programme:

> Naively following an oracle's recommendations can lead to very poor performance **even when the
> oracle's average error is low**. Their fix — a Marker algorithm modified to consult predictions —
> achieves a competitive ratio that decreases as oracle error decreases *and is always capped* at the
> `O(log k)` bound achievable with no oracle at all.

The framework names three properties: **consistency** (optimal when predictions are perfect),
**robustness** (never worse than the prediction-free baseline, even under adversarial predictions),
**smoothness** (degrades gracefully in between).

**Not one mechanism in the program has robustness.** Every anticipation cell replaced the base
selection rule wholesale: the anticipatory BCE injected gradient into the shared logits and hurt
monotonically in λ; the H1/H2 head bonus was added unconditionally after warmup and collapsed
diversity; the momentum bonus rode on the trigger at all times. In each case a weak predictor could —
and did — drag quality *below* the policy it replaced. The program's own synthesis is that "the
frontier is where weak predictors land". A robust combiner changes what "landing" means: **a weak
predictor lands at the base policy, not below it.**

Concretely, the combiner is cheap. Keep `min_logit` as the base rule. Let the predictor propose the
nominee. Track a running estimate of the predictor's realised value (e.g. did the admitted expert
appear in the next `h` tokens' demand — a free label, available one token later) and gate the
predictor's influence on that estimate, falling back to the base rule when it degrades. That is a
selection-time, gradient-free, demand-referential mechanism; it reads the predictor's own track
record, not the cache state.

This is the single highest-value structural change available, because it turns anticipation from a
gamble into a monotone improvement, and every other item on this list becomes safe to try once it
exists.

**Honest bound on the size.** E5's +20–30 pt is an *oracle* number. The one mechanism that produced
provably genuine anticipation in training (H3, centred labels) delivered **+4.6 pt of real
anticipation at BPB parity**. So the realistic prize is single-digit coverage points, and the reason
to want robustness is precisely that at that effect size a mechanism that can also lose is not worth
running.

---

## Part II — The recurrence view

Formally the residency scan is a recurrent layer with a hard transition:

```
    h_t = R_t ∈ V( J(E, k) ),        h_t = F(h_{t−1}, s_t),        y_t = mask(s_t, h_t)
```

where `J(E,k)` is the **Johnson graph**: vertices are the `k`-subsets of `E`, edges join subsets
differing by one element. The cap-1 invariant says the policy walks this graph **one edge per token**
(a lazy walk — "no swap" is a self-loop). Everything a policy can do is choose an edge.

### II.1 An adaptation-latency theorem, and a retrodiction [exact]

`J(E,k)` has diameter `min(k, E−k)`. For our geometries that is `k`.

> **Proposition.** Under cap-1, moving from a resident set to a disjoint one takes **exactly `k`
> tokens**, for every policy. Adaptation latency at a demand discontinuity is `Θ(k)` and is
> policy-independent.

This is not a bound that better eviction, better nomination, or better prediction can improve — only
a larger swap budget `s` can, dividing it to `k/s`.

It also **retrodicts a measured result**. E8 found the post-document-boundary coverage deficit is
−4.2 pt at G1 (`k=6`) and −13.0 pt at G3 (`k=18`) — the deficit grows with `k`, and the theorem says
it must, because the latency does. E8 attributed the asymmetry to "finer experts specialize harder";
the graph-diameter argument gives a mechanism that predicts the *ordering* without appealing to
specialisation, and predicts it for any demand discontinuity, not just document boundaries.

Three consequences the program has not drawn:

- **Fine-graining pays a latency tax that scales linearly with `k`.** The roadmap's `k≈32`/`E≈341`
  geometry — chosen so one swap per token is bandwidth-hideable — has an adaptation latency of 32
  tokens, comparable to measured expert lifetimes. That is a regime change worth predicting before
  spending the runs, and it is a *quality* cost of fine-graining that the bandwidth argument for
  fine-graining does not price.
- **Cold fill is the only free set-write there is.** It writes all `k` slots at once, i.e. it buys `k`
  tokens of latency for nothing. Every deployment request starts cold, so a prompt-conditioned warm
  start is worth exactly the adaptation latency, and its value *grows* along the roadmap. E8 rated
  EOD residency reset "a low-priority nicety" because boundary tokens are <2% of the *probe batch* —
  which is a statement about a packed-document artifact, not about deployment, where every request
  begins at the boundary.
- **The cap-1 invariant has a latency price, not just a bandwidth one.** Cap-1 was fixed as a
  device-independent design invariant and the swap-budget question was closed as a bandwidth-framing
  question. The diameter result says `s` is *also* the reciprocal of adaptation latency, which is a
  quality property. Whether a burst allowance (cap-1 amortised, cap-`s` instantaneous at
  discontinuities) buys quality is a question the bandwidth framing cannot ask.

### II.2 Bounded state gives the right "how much headroom is left" diagnostic [structural]

The state holds `log₂ C(64,6) ≈ 26.2` bits; one cap-1 transition can convey at most
`log₂(1 + k(E−k)) = log₂ 349 ≈ 8.4` bits. So re-specifying a fresh resident set takes ≥3 tokens of
channel capacity even ignoring what the policy knows.

Bounded-state models have known hard limits — [Jelassi et al. (ICML 2024)](https://arxiv.org/abs/2402.01032)
prove two-layer transformers copy strings of exponential length while fixed-state models cannot, by a
counting argument on the state. The analogue here: the residency channel cannot track a demand
process whose **conditional entropy rate** `H(D_t | history)` exceeds ~8.4 bits/token, for any policy.

That is worth turning into a measurement, because it decides whether to keep investing in policy at
all. Today the headroom map is bracketed only from below (the oracle's 66.5%). Estimating
`H(D_t | history)` from the preserved router logs — a free CPU job, and the anomaly-predictability
probe already built most of the machinery — would bracket it from above and say whether the residual
gap is *reachable* or *structural*. If the entropy rate is near the channel capacity, the program is
done and the remaining +0.017 BPB is the price of the constraint, full stop.

### II.3 The recurrence locates the missing gradient path [structural]

In an RNN the state carries gradient. Here the transition is `argmax`/`argmin`: the decision "admit
`e` at token `t`", which determines what is available for the next `k` tokens, receives **exactly
zero gradient**. The router is trained only through the current token's gates.

This reframes the auxiliary-loss programme. Coherence BCE, anticipatory BCE and the bursty loss are
all *surrogates for a credit path that is simply cut*, and all three Goodharted — which is the
generic failure mode of a surrogate that is easier to optimise than the thing it stands in for. The
principled repairs are not better surrogates:

- **Stochastic smoothing / perturbed optimizers** ([Berthet et al., NeurIPS 2020](https://arxiv.org/pdf/2002.08676)):
  perturb the scores and take the expectation of the argmax; the perturbed maximiser is differentiable
  everywhere with a non-vanishing Jacobian, and derivatives are simple expectations approximable by
  Monte Carlo. Applied to the residency scan, the LM loss at tokens `t…t+k` credits the swap at `t`
  directly. This is *the* gradient, not a stand-in for it.
- **Policy gradient** on the swap decision with future LM loss as reward — same target, higher
  variance, no relaxation needed.

Caveat, which the repo has already measured: training a relaxed constraint and serving a hard one is a
train/serve mismatch, and the unmask-2×2 result prices mismatch at +0.10 to +0.485. So this needs
annealing from soft to hard — which is the benched "pressure curriculum (loose→tight)". The theory
says the benched idea is the standard cure for exactly this problem class, not a nice-to-have. This is
also the most expensive item here (differentiating through a 2048-step scan) and should be last.

### II.4 Bounded associative memory: evict by interference, not by demand [structural]

[Schlag et al. (ICML 2021)](https://arxiv.org/abs/2102.11174) show that linear-attention memories with
purely **additive** writes saturate in the overcapacity regime, and that a **delta rule** — remove the
old association before writing the new one — fixes it. Our cache is a bounded memory with an explicit
remove-then-write step, so "what to remove" is their question too, and their answer is
*interference-based*: remove what the incoming item overwrites.

Every key we have used — recency, logit, demand-EMA, predicted demand — is **marginal**: each expert
scored independently of the others in the set. But a set of `k` co-activated experts is partly
redundant and holds less than `k` experts' worth of distinct capability. Nothing in the current policy
can see that.

### II.5 The one that matters: we are running the right algorithm for the wrong objective class [exact]

The current policy greedily maximises `Σ_{e∈R} s_t(e)` — a **modular** set function.

- For a modular objective the optimum *is* top-`k`, and single-swap local search finds it exactly.
  The machinery is exact, so there is **no headroom by construction** — which is the real reason the
  kernel sweep flattened, over and above the equivariance-completeness argument.
- **Nemhauser, Fisher & Wolsey (1978)**: single-swap local search — their *interchange heuristic* — is
  a **1/2-approximation for monotone submodular maximisation under a cardinality constraint**
  (extended to matroid constraints by Fisher et al.).
- **The cap-1 swap policy is literally the interchange heuristic.** One interchange per token,
  cardinality constraint `|R| = k`. We built the exact algorithm that submodular maximisation calls
  for, and then handed it a modular objective on which its guarantee is vacuous.

Making the objective submodular is therefore a change of *objective*, not of machinery — the scan,
the cap-1 invariant, the swap semantics and the serving story are all untouched. Natural choices: a
weighted coverage / facility-location function over demand, or `log det` of a demand-weighted expert
Gram matrix (log-determinant is submodular for PSD kernels; this is MAP inference for a determinantal
point process, whose whole point is selecting a maximally-informative subset).

**Why this is the most interesting item in the note.** Every mechanism the program has tried scores
experts marginally, and the alignment↔diversity frontier is what marginal scoring *produces*: raising
alignment means raising the scores of already-favoured experts, which is concentration. A submodular
objective has diminishing returns for redundant experts built into its marginal gains, so it raises
coverage and preserves spread **in the same step**. The oracle-A3 probe proved a high-alignment,
high-diversity policy exists (+17.7–18.7 pt A3 at `eff ≈ 176`) and concluded the frontier "is not
fundamental to the policy space". A submodular objective is a concrete, testable hypothesis for what
the oracle is implicitly doing.

**Adjacency to flag honestly.** This uses co-activation-like structure, and "co-activation nomination
prior" is benched. The benched item was a *prior on nomination*, declined as dominated by
anticipation's ceiling; this is a change of objective class with a different claim (frontier escape,
not coverage gain), and it is scored on the diversity guardrail the benched item was never aimed at.
It is also demand-referential, not cache-referential. Different thing, adjacent shelf — worth an
explicit ruling before anyone spends a run on it.

---

## Ranked, cheapest-first within tiers

| # | Change | Theory | Cost | What failure falsifies |
|---|---|---|---|---|
| 1 | **Robust combiner** around any predictor (consult-and-fall-back, gated on the predictor's realised track record) | Lykouris–Vassilvitskii (§I.5) | small, selection-time, gradient-free | that the anticipation failures were about predictor quality rather than the absence of a floor |
| 2 | **Succinct binary target + listwise ranking loss** for the nomination head, replacing BCE-vs-discounted-demand | Antoniadis et al.; PARROT (§I.4) | retargets an existing head | that popularity capture was a loss-class problem |
| 3 | **Submodular objective** on the existing cap-1 interchange | Nemhauser–Fisher–Wolsey (§II.5) | new key, no new machinery | that the frontier is a property of the objective class |
| 4 | **Kalman-gain-set filter** wherever a key is smoothed; report `γ_f` and `γ_h` separately | Wiener/Kalman (§I.2–I.3) | free; removes a swept knob | that demand is far enough from AR(1) that no scalar gain is right |
| 5 | **Entropy-rate estimate** of the demand process from preserved logs | bounded-state (§II.2) | free CPU | nothing — it is a diagnostic, and it can end the programme early |
| 6 | **Prompt-conditioned warm cold-fill** | Johnson-graph diameter (§II.1) | free at deploy | that adaptation latency is not what costs at request start |
| 7 | **HiPPO-LegS memory** as the key | curse-of-memory + HiPPO (prev. note) | moderate | that the heavy tail is not exploitable |
| 8 | **Differentiable residency** (perturbed optimizers) + pressure curriculum | Berthet et al. (§II.3) | large | that the cut credit path is not what limits the router |

Unchanged and orthogonal from the previous note: the shared-expert × `R` interaction cell (BigBird
global tokens) and train-time trigger bandlimiting (anti-aliasing). Neither is affected by anything
here.

**On the direct question — is a predicted-demand key worth it?** Yes, with the largest measured
ceiling in the program, but rows 1, 2 and 4 are what determine whether it realises any of that, and
row 1 is what determines whether it can lose. Build the floor before the predictor.

---

## Sources

- Lykouris & Vassilvitskii, *Competitive Caching with Machine Learned Advice*, ICML 2018 / JACM 2021 — [PDF](https://arxiv.org/pdf/1802.05399)
- Antoniadis, Boyar, Elias, Favrholdt, Hoeksma & Larsen, *Paging with Succinct Predictions*, ICML 2023 — [PMLR](https://proceedings.mlr.press/v202/antoniadis23a.html), [arXiv](https://arxiv.org/abs/2210.02775)
- Liu, Hashemi, Swersky, Ranganathan & Ahn, *An Imitation Learning Approach for Cache Replacement* (PARROT), ICML 2020 — [PMLR](https://proceedings.mlr.press/v119/liu20f.html), [code](https://github.com/google-research/google-research/tree/master/cache_replacement)
- Nemhauser, Fisher & Wolsey (1978), interchange heuristic / 1/2-approximation for monotone submodular maximisation under cardinality and matroid constraints — [survey treatment](https://theory.stanford.edu/~jvondrak/data/multiple-matroids-MOR.pdf)
- Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021 — [arXiv](https://arxiv.org/abs/2102.11174)
- Berthet, Blondel, Teboul, Cuturi, Vert & Bach, *Learning with Differentiable Perturbed Optimizers*, NeurIPS 2020 — [PDF](https://arxiv.org/pdf/2002.08676)
- Jelassi, Brandfonbrener, Kakade & Malach, *Repeat After Me: Transformers are Better than State Space Models at Copying*, ICML 2024 — [arXiv](https://arxiv.org/abs/2402.01032)
