# Probe-replay results — Tier-1 experiments E1–E8

> **Provenance.** Every per-model number in this document was computed on runs that are **no longer on
> disk** — they are absent from `MANIFEST.csv` and from every current `e1`–`e8` CSV (see `TODO.md` §2b).
> The metrics themselves have since been recomputed over the 22 preserved router logs, so the *findings*
> stand, but the specific values here cannot be reproduced from committed data and should not be quoted
> as if they could. Where a same-cell run survives under a new name the values differ slightly — e.g.
> the 38M hit-rate quoted here as 16.5% → 36.2% is 17.1% → 38.0% for `flame38m_g1_temporal` in
> `e6_per_layer_ranking.csv`.

Follow-up to [`probe-results.md`](probe-results.md). **Everything here is offline CPU replay of
the already-saved router-probe logs — zero training, no GPU.** Each experiment replays a *selection
policy* over the logged per-token gating logits; the trained weights are never touched. Code:
`analysis/probes/probe_replay.py` (regenerates every number and figure); tests:
`analysis/probes/test_probe_replay.py`.

### Setup and definitions (read once — every metric below uses these)

- **Rolling-residency temporal MoE ("temporal")**: a Mixture-of-Experts (MoE) variant that keeps only
  `K = k` routed experts *resident* per layer (k = the top-k width) and swaps **at most one** expert
  in per token. Shipped swap policy: swap iff the best non-resident expert's gating logit exceeds the
  worst resident's; evict the lowest-logit resident (`min_logit`). Reference: `temporal_router.py`.
- **Probe log**: per MoE layer, per token, on one fixed 16×2048 batch (seed 1234), the raw pre-mask
  gating logits and (for temporal models) the resident set actually used. `S=2048` tokens, `B=16`
  sequences (`B=8` for the 38M model). Logits are stored fp16; the logged resident mask was computed
  in fp32 during the probe, so replaying the policy on the fp16 logits reproduces the fp32 resident
  set on **98.6 %** of tokens (the rest is fp16 rounding in a chaotic sequential scan — immaterial to
  aggregates). All replay comparisons use the same fp16 logits, so they are internally exact.
- **Models** (active non-embedding parameters; expert grain): **8.1 M coarse (6 of 64 experts)** [G1,
  the reference], **3.9 M fine-grained (18 of 192 experts)** [G3], **38 M coarse (6 of 64)** [the
  paper's 10^18-FLOP budget, 50k-vocab]. E1–E3 also cover **1.4 M** and **15 M** coarse.
- **Hit-rate / coverage (set)** — fraction of a token's *unconstrained* top-k demand that is **already
  resident on entry** to the token (before that token's own ≤1 swap). Range 0–1, **higher = better**.
  This is exactly the "A3 self-consistency" of the prior doc; the replay reproduces it to the digit
  (temporal G1 = 38.2 %, full MoE = 21.3 %).
- **Mass coverage** — the same, but weighting each demanded expert by its softmax **gate mass**
  (softmax over the k selected logits) instead of counting it 1. Higher = the *heavy* experts are
  resident. **Retained mass** in E4 is this quantity.
- **Swap rate** — fraction of tokens (t ≥ 1) at which the policy fires a swap. **Lower = cheaper to
  stream.** A swap fires iff ≥1 demanded expert is non-resident on entry.
- **Bandwidth budget `s_max`** — swaps/token that can be hidden behind the same layer's resident
  compute, `s_max = (k−1)·r/r_ram` with SSD→RAM ratio `r_ram/r ≈ 32`: **G1 (k=6) = 0.16, G3 (k=18) =
  0.53, k=32 = 0.97**. Computing the router *early* (before attention) roughly doubles these.

---

## E1 — Swap-rate telemetry (is the cap-1 policy bandwidth-feasible?)

Figure: `results/phase0/figures/swap_rate_vs_bandwidth_budget.png`,
`results/phase0/figures/victim_cache_hitrate_vs_size.png`.

| model | k | realized swaps/token | budget `s_max` | margin (realized / budget) |
|---|---|---|---|---|
| 1.4 M coarse | 6 | 0.999 | 0.16 | 6.4× over |
| 8.1 M coarse (G1) | 6 | 0.998 | 0.16 | 6.4× over |
| 15 M coarse | 6 | 0.999 | 0.16 | 6.4× over |
| 3.9 M fine-grained (G3) | 18 | 1.000 | 0.53 | 1.9× over |
| 38 M coarse | 6 | 0.999 | 0.16 | 6.4× over |

The realized swap rate is **pinned at ~1.0 swap/token** on every model: at K=k the entering resident
set almost never equals the token's global top-k, so the single allowed swap fires almost every step.
Per-layer p95 burst length is the full run (swaps are near-continuous, not bursty). This is **6.4×
over** the coarse budget and **1.9×** over the fine-grained budget (≈3.2× and ≈0.95× against the
doubled router-early budgets).

**Re-reference / victim cache.** 93–97 % of swap-ins re-load an expert that was resident before
(finite re-reference distance), i.e. the stream *oscillates*. A small RAM victim cache of the most
recently-evicted experts absorbs a large share of that traffic at zero quality cost:

| victim-cache size (experts) | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| G1 (8.1 M) swap-ins served | 14 % | 26 % | 46 % | 74 % |
| G3 (3.9 M) swap-ins served | 7 % | 14 % | 26 % | 45 % |

**Decision.** The shipped cap-1 policy is **not bandwidth-feasible as-is** at K=k — realized demand is
~1 swap/token, several× the SSD→RAM budget. Feasibility must come from *reducing* demand (τ margin,
E4; demand-smoothing, E7; or K>k headroom), not from the cap alone. A 16–32-expert victim cache is a
cheap, orthogonal win (halves G1 re-load traffic).

---

## E2 — Streamed-diversity attribution (is streamed diversity real? any pinning?)

Figures: `results/phase0/figures/streamed_expert_diversity_per_sequence.png`,
`results/phase0/figures/expert_residency_distribution.png`.

| model | E (total experts) | union used / seq | effective experts (exp-entropy) | max per-expert residency | experts >0.8 resident |
|---|---|---|---|---|---|
| 1.4 M coarse | 64 | 62.8 (98 %) | 58.9 | 20.5 % | 0 |
| 8.1 M coarse (G1) | 64 | 62.2 (97 %) | 61.2 | 20.2 % | 0 |
| 15 M coarse | 64 | 62.8 (98 %) | 61.6 | 20.5 % | 0 |
| 3.9 M fine (G3) | 192 | 158.8 (83 %) | 176.6 | 25.3 % | 0 |
| 38 M coarse | 64 | 62.3 (97 %) | 53.7 | 39.1 % | 0 |

Token-service concentration (share of tokens each expert serves), Gini 0–1 (higher = more
concentrated): **temporal G1 = 0.167 vs matched full MoE = 0.148** — temporal is marginally *more*
concentrated but both are near-uniform.

**Decision — streamed diversity is REAL, pinning is NOT motivated.** Over a single 2048-token
sequence the temporal model touches **97 % of the coarse pool (83 % fine-grained)**, with an effective
expert count near the full pool — it genuinely uses far more experts over time than fit in RAM, so a
static small-E MoE would *not* match it. But residency is **near-uniform** (aux-loss load-balancing):
the most-resident expert sits at only 20 % (G1/G3) to 39 % (38M) of tokens, and **no expert anywhere
exceeds the 0.8 "de-facto pinned" threshold**. The single pinned-looking streak in the prior doc's
raster was a per-sequence artifact, not an aggregate property. → **Do not pursue an explicit
pinned-slot architecture**; the quality comes from streamed access to the whole pool, not from a hot
subset.

---

## E3 — Mass-weighted consistency and coverage (is self-consistency actually low?)

Figure: `results/phase0/figures/gate_mass_vs_set_self_consistency.png`.

| model | temporal set | temporal mass | full-MoE set | full-MoE mass |
|---|---|---|---|---|
| 1.4 M coarse | 33.2 % | 35.9 % | 19.4 % | 20.4 % |
| 8.1 M coarse (G1) | 38.2 % | 42.1 % | 21.3 % | 22.8 % |
| 15 M coarse | 33.5 % | 36.7 % | 20.2 % | 21.3 % |
| 3.9 M fine (G3) | 34.7 % | 39.1 % | — | — |
| 38 M coarse | 30.4 % | 34.4 % | — | — |

**Hypothesis tested:** that gate mass is so concentrated in the top-1/2 experts that 36 % *set*
coverage would be 60–80 % *mass* coverage.

**Decision — self-consistency really is low; gate mass does NOT rescue it.** Mass coverage is only
**+3 to +5 points** above set coverage (G1: 38.2 → 42.1 %), nowhere near 60–80 %. The heavy top
experts are resident slightly more often than the tail, but the "~40 % self-consistent" reading of the
temporal router **stands**. Raising real self-consistency (via anticipation or smoothing) remains a
genuine bottleneck — it cannot be reframed away as a mass-vs-set accounting artifact.

---

## E4 — Trigger-margin (τ) replay (a free deploy-time knob)

Figure: `results/phase0/figures/swap_rate_vs_retained_mass_tradeoff.png`. Swap iff
`best_nonresident_logit > worst_resident_logit + τ` (logit units), K=k, cap-1, min_logit evict.

| τ (logits) | 0 | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 |
|---|---|---|---|---|---|---|
| **G1** swap / retained-mass | 1.00 / 42.1 % | 0.96 / 42.1 % | 0.82 / 41.6 % | 0.40 / 36.6 % | 0.04 / 20.2 % | 0.00 / 10.1 % |
| **G3** swap / retained-mass | 1.00 / 39.1 % | 1.00 / 39.1 % | 0.99 / 39.1 % | 0.84 / 38.2 % | 0.29 / 28.9 % | 0.00 / 12.0 % |
| **38 M** swap / retained-mass | 1.00 / 34.4 % | 0.99 / 34.4 % | 0.94 / 34.4 % | 0.62 / 33.0 % | 0.09 / 23.7 % | 0.00 / 11.8 % |

**Decision — τ is a genuinely concave, near-free knob for the *first* chunk of swaps, but cannot reach
the SSD budget alone.** For G1, τ≈1 cuts swaps 1.0 → 0.82 for only −0.5 pt mass; τ≈2 halves-and-more
(→0.40) for −5.5 pt. But reaching the bandwidth budget (0.16 swaps/token) needs τ≈3–4, which collapses
retained mass to ~20 %. Recommended **τ\*** (max retained mass at swap ≤ 0.5): **G1 τ=2 (swap 0.40,
mass 36.6 %); G3 τ=4 (swap 0.29, mass 28.9 %); 38 M τ=4 (swap 0.09)**. τ\* feeds a later single
train-with-τ cell, but on its own it trades too much quality to hit budget — pair it with smoothing
(E7).

---

## E5 — Belady replay (the policy-headroom bound; is eviction learning worth it?)

Figure: `results/phase0/figures/eviction_policy_headroom_belady_bound.png`. All at K=k, cap-1, same
logged demand. **Belady** = offline-optimal eviction (evict the resident whose next demand is farthest
ahead). **discounted-oracle(γ)** = nominate/evict by exact discounted future selection mass
`y_t(e)=Σ_{j≥1} γ^{j−1}·1[e∈top-k(t+j)]` (the exact upper bound for a *learned* discounted-lookahead
head). **Belady+prefetch(h)** = let the swap fire h tokens early. Set hit-rate (%):

| policy | G1 (8.1 M) | G3 (3.9 M) | 38 M |
|---|---|---|---|
| LRU (evict oldest) | 31.5 | 26.8 | 26.5 |
| **min_logit (shipped)** | **38.2** | **34.7** | **30.4** |
| min_logit + τ\* | 32.9 | 24.5 | 19.9 |
| **Belady (optimal eviction)** | **46.8** | **40.7** | **40.4** |
| discounted-oracle (γ=0.5) | 66.5 | 52.7 | 60.6 |
| discounted-oracle (γ=0.9) | 51.7 | 52.6 | 47.7 |
| Belady + prefetch(h=1) | 60.3 | 45.1 | 54.1 |
| Belady + prefetch(h=4) | 52.8 | 44.7 | 47.0 |

Two clean findings: **(a)** LRU is *worse* than the shipped min_logit (−7 pt) — cache-recency is the
wrong instinct here, quality-greedy eviction already beats it. **(b)** Better *eviction alone* has
**limited headroom**: offline-optimal Belady beats shipped by only **+8.6 pt (G1), +6.0 pt (G3), +10.0
pt (38 M)**. But **anticipation** — nominating experts by *future* demand (discounted-oracle /
prefetch) — buys **+20 to +30 pt** (G1: 38 → 66 %).

**Decision — eviction-policy learning is NOT the place to invest; anticipation is.** A learned
Belady-imitation eviction cache (Parrot-style) is bounded at ~+8 pt over the shipped policy — small.
The large residual gap lives in **lookahead/prefetch**: a learned discounted-lookahead *nomination*
head (or demand smoothing, E7) is the high-value training-side direction. This redirects the
research plan from "learn a better eviction rule" to "learn to anticipate demand."

---

## E6 — Per-layer ranking

Figure: `results/phase0/figures/per_layer_routing_locality_ranking.png`. Pre-swap hit-rate by MoE
layer (shallow → deep):

| model | shallowest MoE layer | … | deepest |
|---|---|---|---|
| G1 (8.1 M) | 29.2 % | 31.9 / 38.2 / 43.8 | 47.9 % |
| G3 (3.9 M) | 24.9 % | 33.2 / 35.7 | 45.3 % |
| 38 M | 16.5 % | 25.9 / 30.5 / 33.1 | 36.2 % (peaks mid, plateaus ~34 % after) |

**Decision — locality grows with depth; shallow MoE layers are the least cacheable** (consistent with
Mixtral-style depth trends and the per-layer variance reported by Zhu et al. 2505.16056). Any
non-uniform budget (per-layer K, per-layer pinning, per-layer τ) should spend most of its headroom on
the **shallow** layers. The 38 M model plateaus/dips slightly after its middle layers rather than
rising monotonically.

---

## E7 — EMA-logit smoothing replay (is demand-smoothing worth a training cell?)

Figure: `results/phase0/figures/demand_smoothing_swap_vs_coverage.png`. Because the router is linear,
an EMA over hidden states equals an EMA over logits: `logits'_t=(1−β)logits'_{t−1}+β·logits_t`.
**β=1.0 reproduces the baseline exactly (harness identity check: PASS).**

| β (weight on current token) | 1.0 (none) | 0.5 | 0.25 | 0.1 |
|---|---|---|---|---|
| **G1** swap / hit-rate | 1.00 / 38.2 % | 0.97 / 57.4 % | 0.87 / 74.6 % | 0.57 / 88.7 % |
| **G3** swap / hit-rate | 1.00 / 34.7 % | 1.00 / 50.2 % | 1.00 / 67.1 % | 0.92 / 87.3 % |
| **38 M** swap / hit-rate | 1.00 / 30.4 % | 0.99 / 49.8 % | 0.93 / 68.1 % | 0.69 / 84.8 % |

**Caveat (important):** replay evaluates the *selection policy* on a model trained **without** it, and
the hit-rate is measured against the *smoothed* demand — which is self-consistent by construction. So
the coverage gains are **indicative, not a quality measurement**; only the swap-rate reduction is a
clean quantity.

**Decision — demand-smoothing IS worth one training cell.** At β=0.1 the swap rate drops to 0.57 (G1)
/ 0.69 (38 M) — a 30–43 % cut — while self-coverage rises sharply. This clears the doc's "smoothing
looks strong" bar. **Flagged follow-up (optional GPU, not run here):** one `EVAL_ONLY=1` pass with the
EMA patched into the temporal router on an existing checkpoint, to convert the indicative coverage gain
into a real bits-per-byte (BPB) delta — no training required.

---

## E8 — Document-boundary attribution

Figure: `results/phase0/figures/document_boundary_churn.png`. The probe packs several dclm documents
per 2048-token sequence; residency only cold-fills at t=0, so end-of-document (EOD) boundaries inject
topic shifts absent at deployment (each request starts with a fresh cold fill). **Method:** the fixed
probe batch was reconstructed deterministically from the logged data blend + seed 1234 via Megatron's
`GPTDataset` (CPU, no model), and EOD positions (token id 0) cached to
`results/phase0/probe_batch_cache/eod_{16k,50k}.npy` (19 boundaries in the 16k batch, 18 in the 50k).
The swap rate saturates near 1.0 everywhere, so the **graded pre-swap hit-rate** is used instead of a
binary miss rate.

| model | hit-rate ≤4 tok after EOD | hit-rate mid-document | boundary deficit | within-document-only headline | all-token headline |
|---|---|---|---|---|---|
| G1 (8.1 M) | 34.0 % | 38.2 % | **−4.2 pt** | 38.2 % | 38.2 % |
| G3 (3.9 M) | 21.8 % | 34.8 % | **−13.0 pt** | 34.8 % | 34.7 % |
| 38 M | 33.9 % | 30.4 % | +3.5 pt (noise; ~72 tokens) | 30.3 % | 30.4 % |

The boundary dip is real (independently confirmed: the G1 post-EOD coverage drop is z = −2.8 vs a
random-position null) and largest for **fine-grained G3** (−13 pt — finer experts specialize harder, so
a topic shift costs more). But boundary-adjacent tokens are only **0.9–1.8 % of the batch**, so
**within-document-only headline numbers are identical to all-token** (G1 38.2 % = 38.2 %).

**Decision — document boundaries do NOT materially bias the reported locality statistics; "reset
residency at EOD" is a low-priority nicety, not needed for deployment-faithful numbers.** It is worth a
one-line probe change *only* for cleaner G3 measurements, where boundary sensitivity is largest.

---

## Summary of decisions

| exp | question | decision |
|---|---|---|
| E1 | is cap-1 bandwidth-feasible? | **No** — ~1 swap/token, 6.4× over the coarse budget; needs τ+smoothing to shed demand. A 16–32-expert victim cache halves re-load traffic for free. |
| E2 | is streamed diversity real? pinning? | **Diversity real** (uses 83–98 % of experts/seq); **no pinning** (max residency ≤ 20–39 %) → skip pinned-slot architecture. |
| E3 | is self-consistency actually low? | **Yes, genuinely ~30–42 %** — gate mass adds only +3–5 pt, does not reach 60–80 %. |
| E4 | is τ a useful free knob? | **Partially** — near-free for the first ~20 % of swaps, but cannot reach budget without heavy mass loss. |
| E5 | is eviction-policy learning worth it? | **No** — Belady only +6–10 pt over shipped; the +20–30 pt headroom is in **anticipation** (learned lookahead/prefetch). |
| E6 | which layers to prioritize? | **Shallow MoE layers** (least cacheable); allocate any non-uniform budget there. |
| E7 | is demand-smoothing worth a training cell? | **Yes** — EMA cuts swaps 30–43 % with rising coverage; flag the eval-only GPU BPB pass. |
| E8 | do packed-doc boundaries contaminate the stats? | **No** — <2 % of tokens; within-doc headline == all-token. G3 most sensitive (−13 pt local dip). |

**Flagged GPU follow-ups (not run — offline-only task):** (1) E7 `EVAL_ONLY=1` EMA-router BPB pass on
an existing checkpoint; (2) any re-probe (e.g. to log token ids directly, or to add EOD residency
reset for cleaner G3 boundary stats).
