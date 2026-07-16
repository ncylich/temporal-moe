# Mechanistic background: *why* does rolling-residency temporal MoE work?

Curated prior work for the mechanistic investigation. Our method works **iff expert usage is locally
coherent** — a small resident set (K of E experts) covers a span of consecutive tokens with few swaps.
So "why it works" reduces to established, *measurable* properties of MoE routing. Papers below are
grouped by the mechanism they reveal, each annotated with its **signature graph** (the reason to read
it) and what it implies for us. ★ = highest mechanistic payoff / best figures.

Complements `temporal-moe.md §6` (systems/offloading prior work) and `cache-conditional-experts.md`.
Existing probes in-repo that already mirror these methods: `analysis/router_saturation.py`,
`analysis/expert_coactivation.py`, `empirical_analysis/expert_specialization.*`.

### Exact figures to open (verified against the papers)
- **Mixtral** [2401.04088] §5: **Table 5** (% consecutive expert-assignment repetitions ← the mechanism),
  **Fig 8** (tokens colored by expert — the visual), **Fig 7** (no domain specialization).
- **Local Routing Consistency** [2505.16056]: **Fig 4** (SRP vs segment len), **Fig 8** (SCH vs len×cache),
  **Table 1** (20 models ranked by SRP vs architecture), **Fig 7** (SRP vs specialization).
- **OLMoE** [2409.02060] §5: **Fig 20** (router saturation, §5.1), **Fig 21** (co-activation, §5.2),
  **Fig 22** (domain spec., §5.3), **Fig 23** (vocab spec., §5.4).
- **DeepSeekMoE** [2401.06066] §4.5: **Fig 4** (Pile loss vs top-experts disabled ← explains our G3 dip),
  **Fig 5** (loss vs #activated), **Fig 6** (half-activated comparison).
- **Cache-Conditional** [2412.00099]: expert-lifetime result (~19–26 → ~55–76 tokens), results section.
- **Only three?** Mixtral **Table 5** (mechanism) · OLMoE **Fig 20** (stability) · DeepSeekMoE **Fig 4** (fine-graining).

---

## A. The core mechanism — temporal/local routing coherence (*directly why a fixed set covers a window*)

★ **Mixtral of Experts** — Jiang et al., 2024, [arXiv:2401.04088](https://arxiv.org/abs/2401.04088).
- **Signature graph:** "proportion of repeated consecutive expert assignments" per layer vs. the
  random baseline (their Fig. in the routing-analysis section).
- **Finding (our mechanism, stated by others):** routing is **not** domain-specialized (no ArXiv/bio/
  philosophy clustering) — instead it's **positional/syntactic and temporally local**: consecutive
  tokens hit the *same* expert far above chance, **increasingly in deeper layers**; structural tokens
  (indentation, Python `self`, repeated words) route to fixed experts. This is precisely the locality
  rolling residency exploits — and it says the win should grow with depth.

★ **Not All Models Suit Expert Offloading: On Local Routing Consistency of MoE** — Zhu et al., 2025,
[arXiv:2505.16056](https://arxiv.org/abs/2505.16056). *(already cited in our docs)*
- **Signature graphs:** per-model / per-layer **SRP** (Segment Routing best Performance — how well one
  fixed expert set serves a segment) and **SCH** (Segment Cache best Hit-rate under a cache budget with
  look-ahead) — plus the ranking of which models are "offloadable."
- **Why:** these two metrics *are* the direct quantitative test of "can K resident experts cover a
  B-token window," and they show it **varies a lot by model/layer** (shared-expert & fine-grained models
  are more locally consistent) — the measurement to run on our own checkpoints to pick K, B.

**Mixture of Cache-Conditional Experts** — Skliar et al., 2024,
[arXiv:2412.00099](https://arxiv.org/abs/2412.00099). *(summary: `cache-conditional-experts.md`)*
- **Signature graph:** **expert lifetime** (avg consecutive tokens an expert stays cached before LRU
  eviction), rising 19–26 → 55–76 tokens with a cache-affinity bias.
- **Why:** independent evidence the locality is real *and* exploitable — a soft, per-token, batch-1
  version of the same target; the closest "reactive cache" baseline our predictive-window method must beat.

---

## B. Do experts specialize, and how *stably*? (*why the coherence exists and is predictable*)

★ **OLMoE: Open Mixture-of-Experts LMs** — Muennighoff et al., 2024,
[arXiv:2409.02060](https://arxiv.org/abs/2409.02060).
- **THE mechanistic-analysis paper.** Defines and plots four routing properties, each a great figure:
  **(1) Router saturation** — % of an intermediate checkpoint's routing that matches the *final* router,
  vs. training step & layer (routing **locks in early**, more so in later layers); **(2) Expert
  co-activation** — pairwise simultaneous-activation heatmap (redundancy structure); **(3) Domain
  specialization**; **(4) Vocabulary specialization** (per-token-id expert bars).
- **Why:** saturation ⇒ residency patterns are *stable across training* (a fixed set is learnable);
  co-activation ⇒ how much K>k headroom buys; specialization ⇒ the source of coherence. Our
  `router_saturation.py` / `expert_coactivation.py` reproduce these directly.

★ **DeepSeekMoE: Towards Ultimate Expert Specialization** — Dai et al., 2024,
[arXiv:2401.06066](https://arxiv.org/abs/2401.06066).
- **Signature graphs:** the **specialization ablation** — disable the top-N routed experts and watch
  perplexity spike (steeper = more specialized, less redundant) — and the **shared-expert ablation**;
  plus fine-grained-segmentation scaling.
- **Why:** explains our **G=3 fine-grained** result mechanistically — finer experts (18/192) specialize
  harder and hold less redundancy, which is *exactly* why they'd recover marginally less under rolling
  residency (less slack to tolerate a churned expert). Directly interrogates the G-knob we swept.

**Part-of-Speech Sensitivity of Routers in MoE** — 2024,
[arXiv:2412.16971](https://arxiv.org/abs/2412.16971).
- **Signature graph:** router-decision vs. POS-tag mutual information / per-tag expert distributions.
- **Why:** pins the coherence to a concrete, *predictable* signal (syntax) — supports a cheap
  macro-router and explains the code/indentation locality Mixtral saw.

---

## C. Working-set / co-activation structure (*why K>k headroom and swap-1/token suffice*)

- **OLMoE** co-activation heatmap (above) + our `analysis/expert_coactivation.py` (already run on
  FLAME-MoE 290M/1.7B per `scripts/empirical_analysis/`): the empirical distribution of *distinct experts
  per window* is what sets the required K and the swap rate.

---

## D. Prefetch across other axes (*composable, not competing*)

- **Pre-gated MoE** — Hwang et al., 2023 (cited in `temporal-moe.md §6`): compute a layer's gate *early*
  to prefetch across the **depth** axis. Signature: latency-hiding pipeline diagram. Orthogonal to our
  **time**-axis window sharing — composable.

---

## E. Learned segment-level expert reuse (*the trained version of our fixed-window heuristic*)

- **Lory** — Zhong et al., 2024 (cited in §6): fully-differentiable MoE with **causal segment routing** +
  soft expert merging; shows segment/domain-level specialization trains well. Signature: segment-routing
  quality vs. segment length.
- **Temporally Extended Mixture-of-Experts** (cited in §6): a controller that learns *when to keep vs.
  switch* the expert mask — the learned form of "reuse experts over spans."

---

## What to run on OUR checkpoints (the mechanistic experiments these motivate)

The papers above are all *measurements* we can reproduce on FLAME-MoE + our temporal/dense/G=3 runs to
explain **why** temporal held up (and why fine-graining cost a few points of recovery):

1. **Consecutive-token expert reuse vs. layer** (Mixtral-style) — the single most direct "why it works"
   plot; expect it to rise with depth. Overlay G1 (6/64) vs G3 (18/192).
2. **SRP / SCH vs. window length B and budget K** ([2505.16056]) — quantify how well K=6 / K=18 resident
   covers a span; this is the go/no-go and the B,K picker (temporal-moe.md §8 step 1).
3. **Router saturation & expert co-activation** (OLMoE-style; probes already in-repo) — is residency
   stable across training, and how much K>k headroom the co-activation demands.
4. **Specialization ablation** (DeepSeekMoE-style) — disable top-routed experts, G1 vs G3, to test the
   "finer = more specialized = less residency slack" hypothesis that would explain the G3 recovery dip.


## Adjacent: swap avoidance by training loss (read as a counterpoint)

- **CoSMoEs** [2503.00245] (Meta, 2025) — on-device MoE. Three overlaps with this work:
  (1) fair FLOP-aligned MoE-vs-dense at small scale (MoE wins by >=2% absolute) — independently
  replicates our dense-floor finding one scale band up; (2) **weight-decomposed (LoRA-style)
  experts** (M ~ L x R, r = hidden/2, +1.1%) — untried here, would shrink per-swap bytes b and
  compose with fine-graining for streaming; (3) the **BIES loss** (block-wise expert selection):
  hard count x soft L1 of consecutive-token router-prob deltas, penalizing expert replacement —
  the published member of the loss family our negative results reject. Their own Table 3 shows the
  trade: 6x fewer replacements, +54% tok/s, but a consistent LM-eval regression (up to ~1.9 avg pts
  at wearable scale). Signature comparison: they avoid swaps by loss incentive and pay quality; we
  make the single swap affordable by training-time constraint + fine-graining and pay ~nothing at
  matched granularity (see paper Appendix B + the tau-hysteresis replay, where deploy-time swap
  dropping is nearly free without touching training).
