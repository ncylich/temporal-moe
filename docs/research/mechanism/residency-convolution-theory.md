# Rolling residency as convolution: the theory, and what the CNN literature already proves

Companion to [`lru-as-convolution.md`](./lru-as-convolution.md), which established the
correspondence and measured it, and to
[`residency-as-recurrence.md`](./residency-as-recurrence.md), which takes the same policy as a
recurrent state machine and covers predicted-demand keys, the learning-augmented guarantees, and the
submodular-objective result. This note is the theoretical half: what is *provable* about the
policy family, which results from the convolution literature transfer, and — the point — which
of them prove something we have not tested.

**Reading the tightness grades.** Analogies are cheap and this note leans on other people's
theorems, so each is graded:

- **[exact]** — the theorem is about our object literally (a `k`-slot cache, a permutation-equivariant
  score, a memory kernel). It applies without translation.
- **[structural]** — the theorem is about a different object with the same structure; the transfer
  is a modelling assumption, stated explicitly, not a proof about us.
- **[heuristic]** — a suggestive parallel. Listed for completeness, load-bearing on nothing.

---

## 0. Two correspondences, and which one carries the theory

The prompt's reading — LRU eviction makes rolling residency a convolution over experts through time
— is exact on the **time axis**: residency is the box-kernel convolution of the admission stream
(proved and unit-tested in the companion note, §1.2). That is where the intuition starts, and it is
also where it stops being productive, because a single fixed kernel is a two-parameter object
(shape, width) and §2 below shows the whole family is small.

The theory lives on the **expert axis**, which the convolution reading opens up and which nothing in
the program has used. Routing is a selection over `E` experts; rolling residency restricts that
selection's *support* to a `k`-of-`E` set that slides. That makes three distinct literatures apply:
competitive paging (the cache is literally a cache), equivariant deep learning (the expert axis is
literally a set with no order), and sparse-attention universality (support restriction is literally
what those theorems are about). The first two are exact. The third is structural and is where the
untested prediction comes from.

---

## 1. The expressivity hierarchy, with the theorems that pin it

### 1.1 The two shipped policies are LightConv and DynamicConv [structural]

Wu et al., *Pay Less Attention with Lightweight and Dynamic Convolutions* (ICLR 2019), separate
exactly the two things we ship. **LightConv** is a depthwise kernel, shared across channels, with
*the same weights at every time step*. **DynamicConv** predicts a fresh kernel at each time step
*from the current input alone*. Our `lru` is a LightConv (fixed box, reused every step, blind to the
scores); our `min_logit` is a DynamicConv (the kernel is the current logit vector, recomputed per
token, a function only of the current position).

Their measured result is the one we reproduced without knowing it: DynamicConv matches self-attention
at slightly fewer parameters, LightConv sits below it. Our −7 pt is the expected sign, and the
expected size class, for that substitution. Note also their negative: DynamicConv *diverges* without
softmax normalisation of the kernel. That prescription does not transfer to us, and §1.3 says why —
which is itself informative.

### 1.2 The containment is formal [structural]

Cordonnier, Loukas & Jaggi, *On the Relationship between Self-Attention and Convolutional Layers*
(ICLR 2020), prove that a multi-head self-attention layer with enough heads **can express any
convolutional layer**. Read our router as attention over experts: content-adaptive selection
contains fixed-kernel selection, strictly. `lru ⊊ min_logit` is a containment, not a horse race,
and no amount of tuning the box can close it — the best a fixed kernel can do is *coincide* with
the adaptive one on a particular demand distribution.

That is the formal content of your "strictly less expressive". The companion note's corollary
(`lru` evicts at age exactly `k`, always) is the same statement in the concrete: the kernel has no
input to condition on.

### 1.3 A completeness result about our own system: the rule factors through the ranking [exact]

> **Proposition (rank-factorisation).** With `τ = 0`, the residency update at token `t` depends on
> the score vector `s_t ∈ R^E` only through the ordering of its entries. For any strictly increasing
> `φ: R → R`, replacing `s_t` by `φ ∘ s_t` leaves the resident set unchanged at every token.
>
> *Proof.* The trigger compares `max_{e∉R} s_t(e)` against `min_{e∈R} s_t(e)`; the nominee is an
> `argmax`; the `min_logit` victim is an `argmin`. All are order statistics of `s_t`, and strictly
> increasing maps commute with order statistics and preserve strict inequalities. ∎
> (With `τ ≠ 0` the invariance group shrinks to the positive affine maps.)

Three consequences, and they are the reason so many natural transfers are dead on arrival:

1. **Per-token normalisation is inert.** Softmax over experts, temperature, per-token mean-centring,
   any per-token additive constant: all no-ops. So Wu et al.'s softmax-normalisation prescription —
   the thing that made *their* dynamic kernel trainable — cannot help here. Likewise the whole
   BatchNorm/LayerNorm/weight-standardisation family, insofar as it acts per token.
2. **It generalises the program's own no-op lemma.** `local-global-program.md` notes that "a
   zero-sum-per-token variant is a provable no-op; the residency trigger only compares scores within
   a token". That is the `γ`-term special case of the proposition. §3.2 shows the lemma has a
   loophole the program did not take.
3. **It bounds the design space.** Anything that changes behaviour must be **non-monotone across
   experts within a token**, i.e. must inject per-expert-differential information not already in
   `s_t`. There are exactly three such sources: the expert's **own history** (a temporal filter), the
   expert's **content** (what `min_logit` reads), and the **future** (what the oracles read). Every
   mechanism in the whole program is one of those three or a no-op.

---

## 2. Why the box kernel is the worst available choice — three independent proofs

### 2.1 Paging theory proves FIFO is dominated [exact — this one is literally about our algorithm]

This is the closest theoretical hit in the whole note, and it needs no translation: our `lru` policy
is a `k`-slot FIFO cache, and the question "is LRU better than FIFO" is a solved problem in
competitive analysis.

Classical competitive analysis is useless here — LRU and FIFO are both `k`-competitive, which is why
the theory long failed to explain LRU's practical dominance. Borodin, Irani, Raghavan & Schieber
introduced the **access-graph model** precisely to fix that: a graph whose vertices are pages and
whose edges constrain which page may be requested next, so admissible request sequences are walks in
the graph. That is a formalisation of *locality of reference* — the exact property rolling residency
exists to exploit. They conjectured LRU dominates FIFO on every access graph, and **Chrobak & Noga,
*LRU is better than FIFO* (Algorithmica 23(2), 1999), proved it**: `r_LRU(G, k) ≤ r_FIFO(G, k)` for
every access graph `G` and every `k`, with strict separation on some graphs.

So the ranking we measured is a theorem, not a finding. More usefully, it says *where* the separation
comes from: it exists only under a locality model, and it is exactly the locality structure that
makes temporal MoE work at all. The `lrd` policy (refresh on demand rather than on admission) is not
a tweak — it is the move from the dominated algorithm to the dominating one, and the theorem says
the direction cannot go the other way under any locality assumption.

Caveat worth stating: the theorem bounds *competitive ratio* against an offline optimum on
worst-case walks, not expected coverage on our demand distribution, and our variant admits at most
one page per request. So it establishes direction, not magnitude. Magnitude is the companion note's
73–80%.

### 2.2 Receptive-field theory says depth will not fix the kernel for us [structural]

Luo et al., *Understanding the Effective Receptive Field in Deep CNNs* (NeurIPS 2016), prove that
stacking convolutions gives an **effective** receptive field that is asymptotically **Gaussian** (a
central-limit argument over paths through the stack), with radius growing as `O(√n)` in the number of
layers while the nominal receptive field grows as `O(n)` — so the effective/nominal ratio *shrinks* as
`O(1/√n)`.

Two ramifications, one reassuring and one not:

- The reason box kernels are tolerable in CNNs is that **depth launders them**: repeated convolution
  of a box converges to a Gaussian, so no individual layer's crude shape survives. Our residency
  filter gets no such laundering — each MoE layer runs an independent residency scan, and the scans
  do not compose along the time axis. We are stuck with the raw box and its hard truncation.
  **Therefore kernel shape must be chosen explicitly here in a way it need not be in a CNN.** This is
  the theoretical reason the width/shape knob is real rather than a detail.
- It also predicts that the *measured* integration window will be shorter than the nominal one, which
  is what the companion note found (`min_logit`'s nominal width-1 kernel realises a ~8-token
  effective window because the logits themselves are autocorrelated).

### 2.3 Memory theory says both of our kernel families are the wrong family [exact]

Li, Han, E & Li, *On the Curse of Memory in Recurrent Neural Networks* (2020/2021), prove for linear
recurrent models that targets with **exponentially decaying** memory are efficiently approximable,
while **polynomially decaying** (long) memory requires a number of hidden units growing exponentially
in the memory length. Exponential decay is the easy case; power-law decay is the curse.

Now put that next to the measured eviction-age distribution (companion §3.1): median 3 admissions,
mean ~5, tail to 143, with 40–45% of live slots held past the box horizon. That is a heavy tail, not
an exponential one. So:

- **`lru`** is a box: hard truncation, zero memory past `k`. Worst case for a heavy tail.
- **`ema`** is a single exponential: the curse-of-memory theorem says one exponential cannot represent
  polynomial decay at any rate `γ`.
- **A two-scale bank** — which I tried and which showed nothing outside noise — is two exponentials.
  The theorem says two is not meaningfully different from one against a power law; you would need
  a number of scales growing with the memory length.

Gu et al., **HiPPO: Recurrent Memory with Optimal Polynomial Projections** (NeurIPS 2020), give the
resolution and, unusually, give it as an optimality result: given a measure specifying how much each
past time step matters, HiPPO produces the *optimal* online projection of the history onto a
polynomial basis. Their two classical measures are exactly our two policies:

| HiPPO measure | what it is | our policy |
|---|---|---|
| **LegT** (translated Legendre) | uniform weight on a fixed sliding window | `lru` — and the paper's own words: it "catastrophically discards context beyond a fixed sliding window" |
| **LagT** (translated Laguerre) | exponentially decaying weight | `ema` |
| **LegS** (scaled Legendre) | scale-invariant, all history, no timescale prior | *not in our family* |

HiPPO-LegS is proved to have timescale robustness (no `γ` to tune), bounded gradients, and fast
updates. **The theoretically correct eviction key is an `N`-coefficient LegS memory of each expert's
demand channel, not a scalar EMA.** That is a fixed linear recurrence — `N` floats per expert per
layer, `N ≈ 4–8` — and it is the principled version of "widen the kernel" that my sweep could only
approximate by guessing `γ`. It also predicts *why* the guess mattered so much (LagT has a timescale
prior; LegS does not) and why the two-scale bank failed.

---

## 3. The expert axis: what permutation equivariance proves

### 3.1 A completeness theorem for the whole history-based family [exact]

Two results compose here. Kondor & Trivedi, *On the Generalization of Equivariance and Convolution in
Neural Networks to the Action of Compact Groups* (ICML 2018), prove that a linear map is equivariant
to a group action **if and only if** it has group-convolutional structure — convolution is necessary,
not merely sufficient. Zaheer et al., *Deep Sets* (NeurIPS 2017), Lemma 3, give the finite
permutation case explicitly: a linear layer `R^M → R^M` is permutation-equivariant **iff** its matrix
is `Θ = λI + γ11ᵀ`.

Apply both to the eviction key. Require it to be (i) causal, (ii) linear in the per-expert demand
channels `d_e`, and (iii) equivariant to relabelling the experts — condition (iii) being forced,
since expert indices carry no meaning. Then

```
    key_t(e)  =  Σ_{j≥0} [ λ_j · d_e(t−j)  +  γ_j · d̄(t−j) ]        d̄ = mean over experts
```

and nothing else. By the rank-factorisation proposition (§1.3) the `γ` terms are per-token uniform
and therefore **no-ops**. Hence:

> **Corollary.** The only permutation-equivariant, causal, linear, single-layer eviction key is a
> **depthwise temporal convolution of each expert's own demand history** — `key_t(e) = (λ * d_e)(t)`.

`lru`, `lrd`, `box-W` and `ema` are all members, differing only in `λ`. There is nothing else in the
class. That retrospectively explains the companion note's flat result: the kernel sweep was, up to
the choice of `λ`, *exhaustive* over the equivariant linear history-only family, so its ceiling is the
family's ceiling — and §2.3 says the ceiling is set by how well a single fixed `λ` matches a
heavy-tailed memory.

It also names the only two escapes: **break equivariance**, or **add depth**.

### 3.2 The loophole in the program's own no-op lemma [exact]

The program retired zero-sum-per-token mechanisms on the grounds that "a zero-sum-per-token variant is
a provable no-op". That is true, and §1.3 proves it in general — **for a single linear layer**. It
fails as soon as a nonlinearity sits between two equivariant layers, because the cross-expert term
`γ_j d̄(t−j)` is uniform only *before* the nonlinearity; afterwards it is not, and the second layer
sees a signal the first could not produce.

So the minimal non-trivial permutation-equivariant key is **depth 2**:

```
    key_t(e)  =  (λ² * σ( λ¹ * d_e  −  γ¹ * d̄ ))(t)
```

which is a two-layer DeepSets/Set-Transformer network in disguise, with cross-expert competition
entering at the hidden layer. This is a genuine hole in the closed-direction reasoning: the no-op
lemma was applied to the whole family when it only covers the linear-readout case.

### 3.3 The anti-Goodhart corollary [exact]

Read Kondor–Trivedi backwards: a map with per-expert free parameters is *not* equivariant, and
conversely, an equivariant map **cannot represent a per-expert popularity table** — permuting the
experts permutes the outputs identically, so "expert 17 is generally hot" is not in the hypothesis
class.

That is exactly the failure mode the nomination-head program hit. H1/H2 (`W_f ∈ R^{E×d}`, one row per
expert) learned "a static popularity table … incumbency in disguise"; the controlled triplet showed
centring the bonus removed both the diversity collapse and the entire A3 gain. H3 recovered genuine
anticipation (+4.6 pt, BPB-neutral) only by re-engineering the *labels* so popularity was unlearnable.
Equivariance gets the same guarantee **structurally, with no label surgery** — and the parameter count
collapses from `E·d` to `O(L)`.

The caution, also from the CNN literature: equivariance leaks at the boundary. Islam et al. showed
CNNs encode absolute position through padding despite nominal translation equivariance. Our analogues
are the **cold fill** (a specific expert set at `t=0`) and, more insidiously, **tie-breaking on expert
index** — `argmin` returns the lowest index on ties, which is a systematic symmetry break in favour of
low-numbered experts. Any equivariant design should break ties on a symmetric quantity or randomly.

---

## 4. Support restriction: what the sparse-attention theorems predict [structural]

Read routing as attention over experts. Rolling residency restricts that attention's support to a
sliding `k`-of-`E` set. This is the setting of the sparse-attention universality results — with the
translation being from token-token patterns to token-expert patterns, which is a modelling assumption
and not a theorem about us. Graded structural, and the predictions below are the reason it is worth
making.

Yun et al., *O(n) Connections are Expressive Enough* (NeurIPS 2020), give sufficient conditions for a
sparse attention pattern to retain universal approximation: (a) every token attends to **itself**, and
(b) there is a **directed information-flow chain** — a permutation of the tokens such that consecutive
elements are connected through the *union of the patterns across layers*. Zaheer et al., **BigBird**
(NeurIPS 2020), instantiate this: sliding-window attention **alone** is not enough; universality (and
Turing completeness) is recovered by adding **global tokens** and **random attention**.

Three consequences, and the program has half-discovered all three without the theory that ties them
together.

### 4.1 The shared expert is the global token — and its capacity should scale with the restriction

BigBird's global tokens are what carry universality when the local window cannot. Our architecture
already has one: the always-resident **shared expert**, which every token uses regardless of
residency. The theory says its role is *load-bearing under support restriction specifically* — it is
the component that keeps the restricted model in the same function class as the unrestricted one.

**What we have measured, and the gap.** `FINDINGS.md` §4 records the shared-expert knob as
"negligible at B=1" (FLOP-matched `s=2`/top-5 vs `s=1`/top-6, within seed noise) and adds the
prediction that "the constant-vs-routed tradeoff is expected to matter only under windowed routing
(B>1)". Separately, the R-knob sweep varied residency size `R ∈ {18, 36, 72, 128, 192}` and measured a
monotone quality cost (`R=k` costs +0.0275 BPB against full) — **with the shared expert held fixed
throughout**.

So the two knobs have never been varied *together*, and the theory says their interaction is the whole
point: **the marginal value of shared capacity should rise as `R` falls.** At `R = E` the global token
is redundant (the local support is already complete) — which is exactly why the `s` knob measured as
negligible in the unrestricted setting. At `R = k` it is doing the most work it will ever do.

> **The missing cell.** Shared-expert width × `R`, at fixed total FLOPs. Concretely: at `R = k`,
> widen the shared expert and narrow the routed experts to compensate, and compare against the same
> trade at `R = E`. Prediction: the optimal shared fraction is a decreasing function of `R`, and at
> `R = k` a wider shared expert recovers a measurable part of the +0.0275 BPB constraint cost at zero
> routed-FLOP change. This is cheap, orthogonal to every closed direction (it is an architecture
> allocation, not a selection-pressure mechanism), and if the prediction fails it falsifies the
> global-token reading cleanly.

### 4.2 Streamed diversity is the connectivity hypothesis, not a nice property

Condition (b) — information flow through the union of patterns — translates to: the union of resident
sets over time must reach the whole expert pool. E2 measured exactly that (97% of the coarse pool, 83%
fine-grained, touched per sequence, with effective-expert counts near the full pool) and reported it
as evidence that "streamed diversity is real".

The theory upgrades that from an observation to a **hypothesis of the universality argument**, and in
doing so *derives* the program's no-permanence principle rather than asserting it: pinning shrinks the
reachable union, which is precisely the condition that fails. The `eff-experts ≥ 170` gate has been
enforced as a norm ("permanence is a degeneracy"); this gives it a reason.

### 4.3 Random admission has a structural role, not a rescue role

BigBird's random edges exist because random sparse graphs are expanders — short diameter, good spectral
gap — which is what makes the information-flow chain short. Our nomination is a deterministic `argmax`.
The repo benches Gumbel nomination noise as something to reach for "only if dead experts appear",
i.e. as a rescue mechanism. The theory says its role is **mixing**: an ε-random admission bounds the
number of steps for any expert to become reachable, independent of whether any expert looks dead.
That reframing makes it worth a cheap replay (ε-random admission vs coverage and union growth) rather
than a contingency.

---

## 5. Aliasing: where the theory says we ran the wrong experiment [structural]

Karras et al., **Alias-Free GAN / StyleGAN3** (NeurIPS 2021), give the cleanest statement of a fact
that applies to us: a **pointwise nonlinearity applied to a bandlimited signal generates frequency
content above the bandlimit**, and sampling that signal without first low-pass filtering aliases —
destroying equivariance. Their prescription is structural: upsample, apply the nonlinearity, low-pass,
downsample. Zhang, *Making Convolutional Networks Shift-Invariant Again* (ICML 2019), makes the same
point for downsampling in ordinary CNNs, and reports that anti-aliasing **improves accuracy**, not just
artifacts — it behaves as a regulariser.

Our system has that structure exactly: the router's logit stream is not bandlimited, `top-k`/`argmin` is
a pointwise nonlinearity applied at token rate, and the cap-1 swap is the sampler that carries the
result forward. The residency schedule is therefore aliased by construction, and the DSP prescription
is: **low-pass the trigger before the nonlinearity** — which is exactly where `TEMPORAL_EMA_BETA` sits.
The mechanism is already implemented.

**Why the experiment that retired it does not settle it.** B1 measured eval-time trigger smoothing at
+0.08 BPB off-policy and concluded that "eval-time trigger changes that alter WHICH experts serve
tokens hurt". That is precisely the outcome the anti-aliasing literature predicts for a *post-hoc*
filter: BlurPool improves accuracy **when trained with**, and StyleGAN3 required retraining from
scratch — inserting the filter into a model that co-adapted to the aliased signal changes the function
the model was fitted to. The repo's own unmask 2×2 experiment says the same thing in its own terms
(temporal advantage is *serving co-adaptation*; imposing or removing a constraint post-hoc costs
+0.10–0.485).

The training cell that would test this (C1: train with the smoothed trigger) was scheduled, then
benched — but benched under the cap-1 **framing correction**, on the grounds that "under cap-1 the
single swap is free, so a margin that skips it buys nothing". That is a bandwidth argument. The
anti-aliasing claim is not a bandwidth argument: it says the smoothed trigger produces a *better-posed
selection problem*, and predicts a quality gain that the swap-rate framing never asked about.

> **The second missing cell.** Train with `TEMPORAL_EMA_BETA` at a bandlimit matched to the demand
> autocorrelation time (measured: `τ_demand ≈ 2.5–3.1` tokens, so `β ≈ 0.25–0.5`, *not* the `β = 0.1`
> the swap-rate framing favoured). Judged on BPB against the L0 anchor, with the diversity guardrail —
> not on swap rate. If anti-aliasing is a regulariser here as it is in CNNs, this is where it shows.

---

## 6. What the literature says to stop doing

- **Do not expect depth to fix residency locality.** ERF theory (§2.2): the laundering effect that
  makes crude kernels acceptable in CNNs is unavailable to us, because residency scans do not compose
  across layers. The E6 depth trend is a property of the demand signal, not a composed filter.
- **Do not add more EMA scales.** Curse-of-memory (§2.3): against a heavy tail, two exponentials are
  not meaningfully better than one. Either go to a structured basis (HiPPO-LegS) or accept the
  single-scale ceiling. This is the theoretical reason my two-scale bank measured flat.
- **Do not port normalisation tricks.** Rank-factorisation (§1.3): softmax normalisation, temperature,
  per-token centring are provably inert, however essential they are in the source papers.
- **Do not invest further in eviction.** Nothing in §§1–3 changes the fact that the deployable causal
  policy space tops out near `min_logit`; the exhaustiveness corollary (§3.1) now explains why.

---

## 7. Ranked consequences

| # | Action | Grounded in | Status | What failure would tell us |
|---|---|---|---|---|
| 1 | **Shared-expert width × `R` interaction cell** at fixed FLOPs | BigBird global tokens (§4.1) | never run; both knobs measured separately | the global-token reading of the shared expert is wrong; support restriction is not the right frame |
| 2 | **Train-time trigger bandlimiting** at `β ≈ 0.25–0.5`, judged on BPB | anti-aliasing (§5) | scheduled, then benched on a bandwidth argument that does not apply | aliasing is not a quality effect here; the E7/B1 retirement stands for the right reason |
| 3 | **`lrd`** (refresh on demand) | Chrobak–Noga (§2.1) | implemented + tested; one cell pending | the access-graph dominance does not survive the cap-1 admission variant |
| 4 | **Depth-2 permutation-equivariant nomination kernel** | Kondor–Trivedi + DeepSets (§3.1–3.3) | proposal; offline-fit first | history-only nomination is exhausted even with depth — content is required |
| 5 | **HiPPO-LegS memory as the eviction key** | HiPPO + curse-of-memory (§2.3) | proposal | the heavy tail is not exploitable; the single-scale ceiling is the real ceiling |
| 6 | **ε-random admission**, measured on union growth | BigBird random edges (§4.3) | benched as a dead-expert contingency | mixing is already adequate; determinism costs nothing |

Rows 1 and 2 are the answer to "something we're missing": both are cheap, both are predicted by
theory the program has not applied, and both were left undone for reasons that the theory says were
the wrong reasons — row 1 because the two knobs were never crossed, row 2 because it was judged on
swap rate instead of on quality.

---

## Sources

- Chrobak & Noga, *LRU is better than FIFO*, Algorithmica 23(2), 1999 — [ACM](https://dl.acm.org/doi/pdf/10.5555/314613.314655); access-graph model: Borodin, Irani, Raghavan & Schieber (see [survey](https://arxiv.org/pdf/1204.4047))
- Kondor & Trivedi, *On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups*, ICML 2018 — [arXiv](https://arxiv.org/abs/1802.03690), [PMLR](https://proceedings.mlr.press/v80/kondor18a.html)
- Zaheer et al., *Deep Sets*, NeurIPS 2017 (Lemma 3) — [PDF](http://papers.neurips.cc/paper/6931-deep-sets.pdf)
- Gu, Dao, Ermon, Rudra & Ré, *HiPPO: Recurrent Memory with Optimal Polynomial Projections*, NeurIPS 2020 — [arXiv](https://arxiv.org/abs/2008.07669), [PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/102f0bb6efb3a6128a3c750dd16729be-Paper.pdf)
- Li, Han, E & Li, *On the Curse of Memory in Recurrent Neural Networks*, 2020 — [arXiv](https://arxiv.org/pdf/2009.07799)
- Luo, Li, Urtasun & Zemel, *Understanding the Effective Receptive Field in Deep CNNs*, NeurIPS 2016 — [PDF](https://proceedings.neurips.cc/paper/2016/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf)
- Wu, Fan, Baevski, Dauphin & Auli, *Pay Less Attention with Lightweight and Dynamic Convolutions*, ICLR 2019 — [PDF](http://www.nlpir.org/wordpress/wp-content/uploads/2019/04/Pay-Less-Attention-with-Lightweight-and-Dynamic-Convolutions.pdf)
- Cordonnier, Loukas & Jaggi, *On the Relationship between Self-Attention and Convolutional Layers*, ICLR 2020 — [arXiv](https://arxiv.org/abs/1911.03584)
- Yun, Chang, Bhojanapalli, Rawat, Reddi & Kumar, *O(n) Connections are Expressive Enough: Universal Approximability of Sparse Transformers*, NeurIPS 2020 — [proceedings](https://proceedings.neurips.cc/paper/2020/hash/9ed27554c893b5bad850a422c3538c15-Abstract.html)
- Zaheer, Guruganesh et al., *Big Bird: Transformers for Longer Sequences*, NeurIPS 2020 — [arXiv](https://arxiv.org/pdf/2007.14062)
- Karras et al., *Alias-Free Generative Adversarial Networks*, NeurIPS 2021 — [project](https://nvlabs.github.io/stylegan3/)
- Zhang, *Making Convolutional Networks Shift-Invariant Again*, ICML 2019 — [arXiv](https://arxiv.org/pdf/1904.11486)
