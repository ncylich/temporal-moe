# Theory-derived run ideas for the residency policy

Candidate experiments derived from the theory notes
([`lru-as-convolution.md`](./lru-as-convolution.md),
[`residency-convolution-theory.md`](./residency-convolution-theory.md),
[`residency-as-recurrence.md`](./residency-as-recurrence.md)), ordered by
(strength of the theoretical case) ÷ (cost to try). Ideas, not conclusions; each carries its own
kill criterion.

**Noah's preference:** #0, #2, #3, #4, #5, #6 (SFT phase), #7, maybe #10 — though #10 seems like
we already tried it and it failed.

---

**Symbols** (used throughout): `E` = experts/layer (64 coarse, 192 fine) · `k` = top-k = resident-set size (6, 18) · `R_t` = resident set at token `t`, `|R_t| = k` · `s_t(e)` = router logit · `d_e(t)` = 1 if `e` is in token `t`'s *unconstrained* top-k ("demand") · coverage/A3 = share of a token's demand already resident on entry.

**`ŷ_t(e)`** = a predictor's estimate, at token `t`, of how much expert `e` will be demanded in the near future, computed from the past only. Target it estimates: `y_t(e) = Σ_{j≥1} γ_h^{j−1} · d_e(t+j)` — discounted future demand, the same quantity E5's `discounted-oracle` uses (the oracle sees the future; `ŷ` has to guess it). #1 is deliberately agnostic to how `ŷ` is produced — the combiner wraps *any* predictor: the existing head (`ŷ_t(e) = σ(W_f · h_t)[e]`), #7's form (`ŷ_t(e) = w · c_e(t)`), etc. Note the base rule is itself a predictor — `min_logit` nominates with `ŷ_t(e) = s_t(e)`, a persistence forecast ("what's wanted now will be wanted next") — so #1 is *learned predictor vs. persistence baseline*, not predictor vs. no predictor.

**"Key"** = the one number per resident expert that decides who gets evicted. Every eviction policy is the same template — when a swap fires, score each of the `k` residents with one number, evict the lowest:

```
evict  argmin_{e ∈ R_t}  key_t(e)
```

The policies differ **only** in what that number is: `min_logit`: `key = s_t(e)` (current logit) · `lru`: `key =` when `e` was admitted · `lrd`: `key =` when `e` was last demanded · `ema`: `key =` running average of `d_e`.

---

# 0. `lrd` — the LRU we never actually ran

* **Problem:** `evict="lru"` writes its timestamp only on admission → it's FIFO. Every expert dies at age exactly `k` admissions, however wanted it is.
* **Background:** Chrobak–Noga 1999 proved `r_LRU(G,k) ≤ r_FIFO(G,k)` for *every* access graph `G` (a graph over pages whose edges constrain what can be requested next — the formal model of locality), strict on some. Locality is the exact property temporal MoE exploits.
* **Solution:** also set `refresh[e] ← t` whenever `d_e(t) = 1`. The signal is already computed — the trigger ranks all `E` experts to find its nominee.
* **Cons/costs:** eager path only (Triton/graph kernels take one `use_lru` bool) → ~10× slower scan until kernels extended. Still content-blind vs `min_logit`.
* **Prior/expected:** replay recovers 73–80% of the `lru`→`min_logit` gap at equal swap rate, `eff` within 1.4 pt of `min_logit`. Control pair exists: `min_logit` 1.4599 / `lru` 1.4641 at s0@1e16, seed noise 0.0005. Expect ~1.460–1.462. **Kill if ≥ 1.4641.**

---

# 1. Robust combiner — advisor, not replacement

* **Problem:** every anticipation mechanism runs unconditionally, so a weak predictor drags *below* baseline. Track B hurt monotonically in λ; H1/H2 dropped `eff` 184 → 103.
* **Background:** Lykouris–Vassilvitskii. Three properties — *consistency* (perfect prediction → optimal), *robustness* (adversarial prediction → no worse than prediction-free), *smoothness* (in between). Key negative: naive oracle-following can be far worse than no oracle **even at low mean error**. Their combiner caps the ratio at the no-oracle bound.
* **Solution:** base pick `b_t = argmax_{e∉R_t} s_t(e)` (computed anyway). Predictor pick `p_t = argmax_{e∉R_t} ŷ_t(e)` (see the `ŷ` definition above).
* Score *both* one token later: `hit(x) = d_x(t+1)`. **Free here** — we observe the full demand vector over all `E` every token, so the candidate we *didn't* admit is still scored. Real caches can't do this.
* Track `A_t`, `B_t` = EMAs of `hit(p)`, `hit(b)`. Admit `p_t` only if `A_t > B_t + m`.
* **Cons/costs:** hard switch is discontinuous — prefer a soft gate. Must be on during training (train/serve mismatch costs +0.10 to +0.485 per unmask-2×2). Competitive-ratio bound is about fault counts on adversarial sequences, not BPB → structural transfer only.
* **Prior/expected:** creates no signal. Worthless predictor → exact parity, cheaply learned. Upside bounded by H3's genuine anticipation (+4.6 pt A3). **Kill if the gated predictor still ≤ base.**

---

# 2. Succinct target + ranking loss — stop asking the head to be calibrated

* **Problem:** heads train on `BCE(head, discounted future demand)`. The policy never reads magnitudes — only `argmax`/`argmin`. So BCE spends capacity on the target's marginal, and the marginal *is* popularity. H1/H2 learned a static popularity table; H3 needed hand-centred labels to stop it.
* **Background math:** per-expert positive rate `p = k/E ≈ 0.094`. Feature-free BCE optimum = predict `p` → loss `H(p) ≈ 0.30` nats, down from `ln 2 = 0.69`. **>50% of achievable BCE reduction comes from the prior alone.** Listwise softmax over the `k` candidates: constant scores → loss `= ln k = 1.79` nats = exactly chance. **Zero reward for the marginal.**
* **Background research:** PARROT (Liu 2020) — softmax over cache lines, cross-entropy on Belady's *choice*, +61% hit rate over LRU. Antoniadis 2023 — **one bit per request** ("safe to evict") suffices for consistency + robustness + smoothness, with matching lower bounds.
* **Solution:** binary relative target (`e`'s next demand beyond the Belady boundary), listwise cross-entropy over the `k` residents against the oracle's victim.
* **Cons/costs:** ranking loss removes the marginal-fitting incentive, **not** per-expert popularity by itself — pair with expert-axis weight sharing or H3 centring. Target needs a reverse scan (`anticipatory_target` already does it).
* **Prior/expected:** H3's decomposition of the head's +8.4 pt A3: ≈ +4.6 genuine, ≈ +3.8 popularity. Expect a ranking-trained head to keep +4.6 and lose +3.8 → smaller A3, diversity-safe. **Kill if centring the labels still moves A3** (popularity still leaking).

---

# 3. Submodular objective — right algorithm, wrong objective

* **Problem:** the policy maximises `f(R) = Σ_{e∈R} s_t(e)`. Marginal gain of adding `e` is `s_t(e)` regardless of what's already in `R`, so the set can hold `k` mutually redundant co-activated experts and the rule cannot see it.
* **Background math:** `f` modular ⇒ optimum is just top-`k` ⇒ single-swap local search is **exact** ⇒ zero headroom by construction. Nemhauser–Fisher–Wolsey 1978: single-swap local search (the *interchange heuristic*) = ½-approximation for monotone **submodular** maximisation under a cardinality constraint. Submodular means `Δ(e|R) := f(R∪{e}) − f(R)` is non-increasing in `R` — diminishing returns.
* **The point: cap-1 swap literally *is* the interchange heuristic** (one swap/token, `|R| = k`). We built the right algorithm and hand it the one objective class where its guarantee is worthless.
* **Solution:** swap `f` for a submodular one. Weighted coverage `f(R) = Σ_c w_c · max_{e∈R} a_{ce}`, or log-det `f(R) = log det(K_R)` with `K` a PSD expert-similarity matrix (co-activation or gate-vector Gram) — submodular for PSD `K`, i.e. MAP-DPP.
* Evict `argmin_{e∈R} Δ(e | R\{e})`, admit `argmax_{e∉R} Δ(e|R)`. Same scan, same cap-1, same serving story.
* **Cons/costs:** needs `K` (`E×E`/layer — trivial storage, but must be estimated). Adjacent to the benched "co-activation nomination prior" → **needs an explicit ruling before anyone spends a run.** One matrix op per swap.
* **Prior/expected:** the only item that could raise coverage *and* `eff` together. Oracle probe showed +17.7–18.7 pt A3 at `eff ≈ 176`, so such a policy exists. **Kill if it trades coverage for diversity like everything else.**

---

# 4. Split the two γ's — two exponentials, two jobs

* **Problem:** "EMA of predicted demand" fuses a *horizon discount* and a *trust weight*. Swept jointly, neither is identifiable. Karen's `γ_m`/`γ_q` pair already burned the program this way.
* **Background math:**
  * horizon: `y_t(e) = Σ_{j≥1} γ_h^{j−1} d_e(t+j)`. Effective lookahead ≈ `1/(1−γ_h)`.
  * filter: `m_t = (1−γ_f)·m_{t−1} + γ_f·ŷ_t`. Effective memory ≈ `1/γ_f`.
  * For an AR(1) latent `x_t = ρx_{t−1} + w` observed as `y_t = x_t + v`, the optimal causal estimator **is** exactly this EMA, with `γ_f` = steady-state Kalman gain — a function of SNR only.
* **Solution:** fix `γ_h` from E5's horizon map (0.5–0.75, ≈2–4 tokens). **Derive** `γ_f` from measured predictor residual variance instead of sweeping it. Report both separately.
* **Cons/costs:** free; needs one extra logged scalar (residual variance).
* **Prior/expected:** measured `τ_demand ≈ 2.5–3.1` tok vs measured optimal filter `≈ 8` tok — a ~3× slower filter, exactly what a noisy binary observation predicts. **Prediction: a better predictor should be smoothed *less*** (`min_logit` is the `γ_f → 1` limit). **Kill if optimal `γ_f` doesn't track SNR** → AR(1) model is wrong → go to #7.

---

# 5. Entropy-rate diagnostic — is the gap even reachable?

* **Problem:** headroom is bracketed only from below (oracle 66.5%). Nobody knows whether the residual +0.017 BPB is winnable or is the price of the constraint.
* **Background math:** state holds `log₂ C(64,6) ≈ 26.2` bits. One cap-1 transition picks among `≤ 1 + k(E−k) = 349` options → `≤ log₂ 349 ≈ 8.4` bits/token of state change. So `≥ 26.2/8.4 ≈ 3.1` tokens to re-specify a set, for *any* policy. Jelassi et al.: bounded-state models provably fail tasks needing more state than they carry.
* **Solution:** estimate `H(D_t | history)` — conditional entropy rate of the demand set — from preserved router logs; compare to 8.4 bits/token.
  * `H ≳ 8.4` → channel-limited, residual gap is structural, stop investing in policy
  * `H ≪ 8.4` → reachable, keep going
* **Cons/costs:** free CPU. Exact entropy on a `C(64,6)` alphabet is infeasible → use a fitted predictor's cross-entropy as an upper bound. The `anomaly_pred` machinery already does most of this.
* **Prior/expected:** not falsifiable — it's a measurement. Both answers are decision-relevant; the "structural" answer ends the policy program cheaply.

---

# 6. Warm cold-fill — the only free set-write

* **Problem:** at every deployment request `R_0` = top-k of the first token. Everything after costs one swap/token.
* **Background math:** state space = Johnson graph `J(E,k)` (vertices = `k`-subsets, edges = one swap). Diameter `= min(k, E−k) = k`. ⇒ **moving to a disjoint set costs exactly `k` tokens, for every policy.** Only a bigger swap budget `s` changes it (→ `k/s`).
* **Solution:** pick `R_0` from a prompt-level summary (mean logits over the prompt, or the first `m` tokens' demand union) instead of token 0's top-k. Prefill sees the whole prompt → this is exact, not predicted.
* **Cons/costs:** inference-side only; one extra router pass over the prompt. Decode-only with a 1-token prompt gains nothing.
* **Prior/expected:** retrodicts E8 — boundary deficit −4.2 pt at `k=6` vs −13.0 pt at `k=18`, growing with `k` as the diameter says it must. Worth `k` tokens of latency *per request*, growing linearly with `k` → biggest payoff on the `k≈32` roadmap. **Kill if warm start doesn't beat `top-k(BOS)` on first-`k`-token loss.**

---

# 7. HiPPO-LegS key — the memory has a heavy tail

*(See the "key" definition above: the one number per resident that decides eviction. "Scalar" is the limitation this idea targets — each existing policy squashes expert `e`'s whole demand bit-string `d_e = 0,1,1,0,0,1,…` into one float before ranking. One float ⇒ one statistic: recency, or a rate at one timescale — never "hot 40 tokens ago, quiet since, likely to return". Measured eviction ages span median 3 to max 143 → the useful history spans many timescales that no single scalar can hold. The fix is to compress later, not earlier: `history → N floats → 1 float → rank` instead of `history → 1 float → rank`.)*

* **Problem:** the eviction key is a scalar summary of demand history. Box (`lru`) truncates hard at `k`; EMA (`ema`) decays exponentially. Measured age-at-eviction: median 3, mean ~5, **max 143** — heavy tail, neither shape.
* **Background:** curse-of-memory (Li/Han/E/Li) — linear recurrent memory approximates *exponentially* decaying memory cheaply, but *polynomially* decaying memory needs exponentially many units. HiPPO (Gu et al.) gives the optimal online projection for a chosen measure: **LegT = our box** ("catastrophically discards context beyond a fixed sliding window"), **LagT = our EMA**, **LegS = scale-invariant, no timescale prior, bounded gradients**.
* **Solution:** replace the scalar key with `c_e(t) ∈ R^N`, `N ≈ 4–8`, updated by the fixed LegS linear recurrence on `d_e(t)` — same cost class as the EMA update (the EMA **is** the `N=1` case). HiPPO's theorem: `c_e(t)` is the *best possible* `N`-number summary of the entire history, with no decay-rate hyperparameter to tune (the difference from stacking `N` EMAs). Key `= w·c_e(t)`, with `w ∈ R^N` **shared across experts** (keeps permutation equivariance → popularity unrepresentable); `w` is learned/fit — it picks *which* statistic of history matters, instead of hard-coding it via a decay rate.
* **Cons/costs:** `N` floats/expert/layer (≈512/layer — trivial). Reference path first, kernels later. `w` must be fit or hand-set.
* **Prior/expected:** my 2-scale EMA bank measured flat — which curse-of-memory predicts (two exponentials ≉ power law). Expect LegS to beat the best single EMA by a few points *if* the tail is exploitable. **Kill if LegS ≈ best single EMA.**

---

# 8. Differentiable residency — the swap decision gets zero gradient

* **Problem:** `R_t` comes from `argmax`/`argmin` — non-differentiable. The decision at `t` sets what's available for `t…t+k`, and **no gradient flows along that path**. The router learns only from the current token's gates.
* **Background:** coherence / anticipatory / bursty losses were surrogates for that cut path, and all three Goodharted — the generic outcome when a surrogate is easier to optimise than its target. Berthet et al. 2020: replace `argmax(s)` with `E_Z[argmax(s + εZ)]` for Gaussian `Z`; the perturbed argmax is differentiable everywhere with non-vanishing Jacobian, and derivatives are expectations estimable from a few MC samples.
* **Solution:** run the residency scan with perturbed selection during training, anneal `ε → 0` so training *ends* at the hard policy. LM loss at `t…t+k` then credits the swap at `t` directly.
* **Cons/costs:** **highest on this list.** Differentiating a 2048-step sequential scan (time + memory). Requires the soft→hard curriculum (the benched "pressure curriculum") because train/serve mismatch costs +0.10 to +0.485. MC gradient variance.
* **Prior/expected:** no prior in this repo. Expect it to matter for the router's *representation*, not the eviction rule (E5 caps eviction at +6–10 pt). Do it last. **Kill if annealed-relaxed ≈ hard training.**

---

# 9. Shared-expert width × R — the global token nobody varied

* **Problem:** shared-expert width and residency size `R` have only ever been measured *separately*. The `s`-knob was declared negligible — but only at `R = E`, which is exactly where theory says it should be.
* **Background:** Yun et al. 2020 — sparse attention stays universal if (a) each token attends to itself and (b) the union of patterns carries a directed information-flow chain. BigBird — sliding window *alone* fails; **global tokens** restore universality. Our always-resident shared expert **is** the global token.
* **Solution:** 2×2 at fixed total FLOPs — shared width ∈ {1×, 2×} crossed with `R` ∈ {`k`, `E`}. Widen shared, narrow routed to compensate so routed FLOPs stay constant.
* **Prediction:** `∂quality/∂(shared width)` is larger at `R = k` than at `R = E`.
* **Cons/costs:** 4 training cells. Architecture allocation, not a selection mechanism → clear of every closed direction. Needs a careful FLOP-match or the widening confounds with optimisation.
* **Prior/expected:** measured null at `R = E` (±0.003 ≈ seed noise) — theory *explains* that null rather than contradicting it. `R = k` costs +0.0275 BPB vs full. Expect a wider shared expert to recover part of that at `R = k` and nothing at `R = E`. **Kill if the interaction is flat** → drop the sparse-attention frame entirely.

---

# 10. Train-time bandlimiting — anti-aliasing has to be trained in

* **Problem:** `top-k` is a pointwise nonlinearity applied to a non-bandlimited logit stream, then sampled by the cap-1 state update. Aliased by construction.
* **Background:** Karras et al. 2021 — a pointwise nonlinearity generates frequency content above the input bandlimit; sampling without a low-pass aliases and breaks equivariance; fix is to filter *before* sampling. Zhang 2019 (BlurPool) — same fix in CNNs, and it **improves accuracy** (acts as a regulariser), but only when **trained with**.
* **Solution:** train with `TEMPORAL_EMA_BETA` on. Trigger runs on `s'_t = (1−β)s'_{t−1} + β s_t`; gates still see raw logits. Already implemented — only the training cell is missing.
* Set `β` from measured demand autocorrelation: `τ ≈ 2.5–3.1` tokens ⇒ `β ≈ 1/τ ≈ 0.25–0.5`.
* **Cons/costs:** one training cell. Do **not** reuse `β = 0.1` — that came from the swap-rate framing and is over-smoothed on this reasoning.
* **Prior/expected:** B1 measured eval-time smoothing at +0.08 BPB — exactly what BlurPool/StyleGAN3 predict for a post-hoc filter, and consistent with unmask-2×2. The training cell was benched on a *bandwidth* argument that the quality claim doesn't depend on. Judge on BPB + `eff`, not swap rate. **Kill if trained-in smoothing ≥ L0 (1.4750 ± 0.0009)** → the E7/B1 retirement stands, for the right reason.

---

**If you only run three:** 0 (free, theorem-backed), 9 (cheapest untested prediction), 3 (only candidate frontier escape).
