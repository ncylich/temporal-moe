# Temporal MoE: amortizing expert I/O over a window of tokens

**One line.** Standard MoE re-selects experts *per token*, so SSD-offloaded experts must
be streamed on essentially every step. Temporal MoE fixes a single resident expert set for
a contiguous **window of `B` tokens**, chosen *predictively* at the window boundary, so the
SSD→RAM transfer of an expert is amortized over `B` tokens of compute instead of one. This
shrinks the resident (RAM/VRAM) footprint to a small fraction of the experts and hides SSD
latency behind compute, at the cost of forcing `B` tokens to share an expert pool.

---

## 1. Problem

In a sparse MoE, the expert FFNs hold the overwhelming majority of parameters but only a
top-`k` slice is active per token. Concretely, for **FLAME-MoE-1.7B** (1.7B active / 10.3B
total): `E = 64` experts/layer, top-`k = 6`, `L = 17` MoE layers, hidden `d = 2048`, expert
intermediate `d_ff = 1408`, gated (SwiGLU) experts.

- One expert ≈ `3·d·d_ff = 8.65M` params ≈ **17.3 MB** at bf16.
- All experts ≈ `E·L·8.65M = 9.4B` params ≈ **18.8 GB** — **≈ 91%** of the model.
- Per-token active experts ≈ `k·L = 102` experts ≈ **882M** params ≈ **1.76 GB**.

So if RAM/VRAM is the binding constraint, the experts are the thing you want on cheaper,
slower storage (NVMe SSD: ~3–7 GB/s) and stream in on demand. The obstacle: **standard
routing changes the active set every token.** A naive offload pays an SSD round-trip for
(worst case) `k·L` experts *per generated token*, and SSD bandwidth — not FLOPs — becomes
the throughput ceiling. (Attention, embeddings, the router, and the always-on *shared*
expert are small and stay resident; only the routed experts are offloaded.)

## 2. Core idea

Partition the token stream into contiguous windows of length `B`:

```
window m = [ m·B , (m+1)·B − 1 ]
```

For each MoE layer `ℓ`, every token in window `m` is served by a single **resident set**
`S(m,ℓ) ⊆ {1..E}` with `|S(m,ℓ)| = K`, where `k ≤ K ≤ E` is the **RAM budget** (experts a
layer may hold at once). Two rules:

1. **Within a window**, each token still routes normally — ordinary top-`k`, but the
   selection is *masked to the resident set* `S(m,ℓ)` (top-`k` of the `K` loaded experts).
   No behavioral change beyond the restriction to `K` experts.
2. **At the window boundary**, a small **macro-router** predicts the *next* window's
   resident set `S(m,ℓ)` from information available at the boundary, and issues an async
   SSD prefetch so the experts are in RAM before window `m` is processed. The boundary
   carrier is a dedicated `[ROUTE]` token (below); window 0 is bootstrapped from the
   leading `[ROUTE]` token that precedes `[BOS]`.

Experts are reused along the **time/sequence axis** — hence *temporal* MoE — as opposed to
the per-token (spatial) reuse of ordinary routing.

### Why prediction is required (decode) vs. not (prefill)

- **Prefill** (all tokens present): the window's working set can be computed *exactly* —
  run the router for all `B` tokens at layer `ℓ`, take the union of their top-`k`, keep the
  top-`K` by mass. No speculation; the only cost is the `K`-restriction. Temporal MoE in
  prefill is essentially free locality.
- **Decode** (autoregressive, one token at a time): future tokens don't exist yet, so the
  resident set for window `m` must be *predicted* before its tokens are generated — this is
  the speculative part, and the main source of both novelty and risk.

### Macro-router placement

Routing is per layer, so to prefetch a whole window's experts at the boundary we predict
`S(m,ℓ)` for **every** layer from the boundary token. When token `m·B−1` finishes its
forward pass, its pre-router hidden state `h_ℓ` is available at each layer `ℓ`; a tiny
per-layer head maps `h_ℓ → ` (a distribution over `E` experts, take top-`K`). Prefetches
for all layers fire at once; deeper layers naturally get more lead time. A cheap baseline
with no new parameters: reuse the boundary token's own top-`k` per layer and **grow it to
`K`** by expert co-activation priors (Section 5).

### The `[ROUTE]` token: a dedicated carrier for macro-routing (experiment direction)

The macro-router needs an input at each window boundary. The obvious choice — reuse the
last real token's hidden state `h_ℓ` — **overloads that position with two conflicting
jobs.** In training it receives gradients from (a) its own next-token loss and (b) the
routing loss of *all `B` tokens* in the window its decision gated. These pull the
representation in different directions: *be a good local predictor* vs. *be a good
`B`-token-ahead routing query*. Mean-reducing the window's routing loss fixes the **scale**
of that second gradient but not its **direction** — the boundary token still trains unlike
every other token, and the routing objective (which governs `B` tokens) is the more
important one to get right there.

The clean fix is to stop overloading a prediction-bearing token: insert a dedicated
**`[ROUTE]` token** at each window boundary whose *only* job is to emit `S(·,ℓ)` for every
layer. It carries no next-token target, so its hidden state is free to specialize entirely
for routing — no competing gradient — and the macro-router gets undivided priority at that
position by construction.

It also gives the macro-router a **better** input, not just a cleaner one. The `[ROUTE]`
token attends over the preceding window, so each per-layer `h_ℓ` is an attention-pooled
summary of the whole span rather than one token's features — in effect an **attention
router** for expert prefetch, distributing the routing signal across the past instead of
cramming it into a single token. (Ordinary MoE routers are linear on one token's state;
this is strictly more expressive for a window-level decision.)

**Sequence layout.** Place `[ROUTE]` *before* `[BOS]` at the very start, then one `[ROUTE]`
at every window boundary:

```
[ROUTE] [BOS] t₁ … t_{B-1} | [ROUTE] t_B … t_{2B-1} | [ROUTE] t_{2B} …
```

The leading `[ROUTE]` bootstraps window 0; each later `[ROUTE]` opens window `m` and
predicts `S(m,ℓ)`. Cost is one extra token per window (`≈1/B`, e.g. 1.6% at `B=64`).

**Masking — make `[ROUTE]` a pure side-channel** (real-token behavior stays
baseline-identical except through `S`; `[ROUTE]` is trained only by the window it routes):

- **Loss:** keep `t_{B-1}→t_B` scored (override the label to skip the marker) and set
  `[ROUTE]`'s own target to `ignore_index` — full token coverage, zero prediction-vs-routing
  overload on `[ROUTE]`.
- **Attention:** mask `[ROUTE]` out of real tokens' incoming attention (it keeps its own
  causal view of the prefix) — routing flows through `S`, not the residual, so this removes
  contamination without blocking the macro-router. *(Off the fused-causal fast path — use a
  FlexAttention `mask_mod`.)*. Note: although this is mathematically sound, it should be
  tested whether this makes a real difference as it will have a substantial performance slowdown.
- **Positions:** give `[ROUTE]` a non-consuming `position_id` (e.g. shared with `t_B`) so it
  doesn't shift real tokens' RoPE distances — the silent one; skipping it breaks the
  baseline-identical guarantee even with the mask correct.

> **Experiment — re-add router z-loss for the macro-router.** Baseline FLAME-MoE dropped
> z-loss, but the macro-router solves a harder, noisier problem (predict a `B`-token-ahead
> set from one carrier), so its logits are more prone to blow-up. Worth re-trialing
> `--moe-z-loss-coeff` *on the macro-router specifically* to bound logit magnitude during the
> noisy early/transition phases.

## 3. What it buys: footprint and the latency-hiding condition

**Resident footprint.** RAM holds `K` experts/layer instead of `E`:

```
resident = K·L·(expert bytes)        (×2 if double-buffered for prefetch)
```

For FLAME-MoE-1.7B at `K = k = 6`: `6·17·17.3MB ≈ 1.76 GB` (single) / `3.5 GB`
(double-buffered), vs **18.8 GB** for all experts — a **5–10×** reduction. Choosing `K`
larger (better coverage, Section 4) trades footprint back: `K = 16` → 4.7 GB single.

**Latency hiding.** Let `r` = SSD read bandwidth (B/s), `Δ(m,ℓ)` = experts newly loaded for
window `m` (those in `S(m,ℓ)` not already resident). The per-window SSD transfer is
`Σ_ℓ Δ(m,ℓ) · (expert bytes)`; amortized over `B` tokens it is fully hidden when

```
   B · t_tok  ≥  ( Σ_ℓ Δ(m,ℓ) · expert_bytes ) / r
                 └──────── per-window SSD load ────────┘
```

where `t_tok` is per-token compute time. **Worked example** (FLAME-MoE-1.7B, CPU decode):
`t_tok ≈ 34 ms` (≈ `2·1.7e9` FLOP/token at ~100 GFLOP/s ≈ 30 tok/s); full reload of `K=6`
across 17 layers = 1.76 GB; NVMe Gen4 `r ≈ 5 GB/s` → 0.35 s. Hidden when `B ≥ 0.35 / 0.034
≈ 11` **even if every window fully reloads**. With temporal locality (`Δ ≪ K·L`), a much
smaller `B`, or a much larger model, still hides the SSD. The same inequality says: pick the
smallest `B` that keeps SSD off the critical path, because `B` also sets the quality cost.

**Roofline view (why `B>1` is the whole point).** At batch-1 decode the expert FFN is itself
RAM-bandwidth-bound (each weight read once, hit with one token), so with overlapped prefetch
per-token time `= max(T_compute, T_load)`, and storage stops masking compute only once the
per-token miss rate drops below `BW_ssd / BW_ram` (≈ 5–15% on commodity hardware). Per-token
caching (e.g. Cache-Conditional Experts, [summary](./cache-conditional-experts.md)) is stuck
at that stingy `B=1` crossover. A shared window multiplies the compute side by `B`, moving the
crossover to `miss_rate* ≈ B · BW_ssd/BW_ram`; at `B≈16` storage is hidden almost regardless of
working-set turnover, provided it fits in `K`. Temporal MoE *is* this crossover shift.

A concrete bandwidth study ([`../../cce/FINDINGS.md`](../../cce/FINDINGS.md)) confirms the
corollary: at `B=1` on high-bandwidth compute (GPU VRAM 768 GB/s, Apple-Silicon unified RAM
273 GB/s) a single offloaded expert costs ~20–110 resident-experts of compute to hide but only
`k≈8` run per layer, so per-token cache-conditional offload is *disqualified* there. Batching
(this section's `B`) is the only lever that reopens it — independent motivation for Temporal MoE.

## 4. The central tradeoff

Two knobs, opposed:

- **`B` (window length):** larger `B` → better amortization (footprint/bandwidth) but more
  tokens forced to share `S`, and a *longer prediction horizon* (the boundary token must
  anticipate experts up to `B` tokens ahead). Quality falls as `B` grows.
- **`K` (RAM budget):** larger `K` → the resident set covers more of each token's true
  top-`k` (less quality loss, easier prediction) but a larger footprint and more I/O.

The research deliverable is the **shape of the quality(`B`, `K`) surface**: perplexity /
downstream accuracy vs. `(B, K)`, against the `K=E` (unrestricted) ceiling. Success =
a regime (e.g. `B≈16, K≈12`) with near-baseline quality and multi-× footprint reduction.

## 5. Why it should work — and how to check before training

The bet is **temporal locality of routing**: adjacent tokens reuse experts, so a modest
resident set covers a window's needs. This repo already ships the exact measurements:

- **`analysis/router_saturation.py`** — how the *union* of selected experts grows as tokens
  accumulate. Directly bounds `K`: if `B` consecutive tokens touch ≤ `K` distinct experts,
  the `K`-restriction is lossless for that window.
- **`analysis/expert_coactivation.py`** — pairwise co-activation, i.e. which experts fire
  together. Gives the prior for *growing* a seed to a resident set of size `K`, and tells
  you whether a small `K` can cover correlated experts.

**Zero-training feasibility probe** (do this first): on a trained FLAME-MoE checkpoint,
slide a window of `B`, form the per-layer working set, and measure the fraction of each
token's true top-`k` mass that survives restriction to the top-`K`. This upper-bounds
achievable quality with *no* retraining and costs only a forward pass over eval data. The
locality metrics of *Local Routing Consistency* (SRP/SCH; Section 6) formalize this probe —
adopt them so results are comparable to that paper's per-model findings.

## 6. Relationship to prior work

*Closest — learned/segment-level expert reuse:*
- **Temporally Extended Mixture-of-Experts** (closest prior art): argues MoEs switch experts
  too often and adds a controller that learns *when to keep vs. switch* the expert mask —
  the learned form of "reuse experts over spans." Temporal MoE differs by fixing the span
  length `B` for an *I/O* objective and predicting the set for *prefetch*, not just stability.
- **Lory** (Zhong et al., 2024): fully-differentiable MoE with *causal segment routing* and
  soft expert merging; shows segment/domain-level specialization. Validates that
  sequence-segment-level routing trains well — but it merges experts (dense) rather than
  keeping a small *resident* set for offload.

*Diagnostic — does a fixed set cover a span?:*
- **Not All Models Suit Expert Offloading: On Local Routing Consistency of MoE Models**:
  defines locality metrics (SRP/SCH) measuring whether a fixed expert set covers a token
  segment. This is precisely the measurement Section 5/8 needs to pick `B` and `K`, and to
  decide which checkpoints are even viable for window-sharing — adopt its metrics directly.

*Routing-side I/O awareness (alternative to freezing the set):*
- **Mixture of Cache-Conditional Experts** ([summary](./cache-conditional-experts.md);
  Skliar et al., 2024): training-free, calibration-free per-token logit *bias* toward
  already-cached experts — but **batch-1 only** and demonstrated at a **large 50–75% cache**.
  Same expert-locality target with less architectural change; the closest soft baseline to
  beat, and composable as a *within-window tiebreaker* among our `K` resident experts. Its
  measured 55–76-token expert lifetimes are independent evidence the locality we exploit is
  real. **Caveat that bounds its usefulness:** it is never benchmarked against a
  fully-resident model (only LRU / Belady's oracle, both under offload), so its gap to
  no-offload speed is unquantified; at batch-1 the regime stays Flash-bound, so its residual
  7–21% miss rate keeps it ~2–3× below resident throughput — exactly the ceiling we must
  measure and beat (Section 8).
- **Pre-gated MoE** (Hwang et al., 2023): computes a layer's gate *early* to prefetch across
  the **depth** axis; Temporal MoE prefetches across the **time** axis and shares one set
  over `B` tokens — orthogonal and composable.

*Systems baselines (router unchanged; cache/prefetch from traces):*
- **MoE-Infinity** and **Mixtral-offloading** (Eliseev & Mazur, 2023), **EdgeMoE / SwapMoE /
  Fiddler**: activation-trace caching and speculative prefetch with per-token routing. These
  are the bar to beat — Temporal MoE makes the stronger commitment (fix the set for a
  window) for a *hard* footprint/bandwidth bound and a simple double-buffer pipeline instead
  of a probabilistic cache. Justifying a model-side change requires outperforming them.
- **Mixture-of-Depths**: sequence-axis sparsity for *compute*, not expert *residency* —
  different objective.

The novel, falsifiable claim: **predictive window-shared routing** beats per-token caching
on the footprint↔quality frontier under SSD offload.

## 7. Training options (increasing cost / quality)

1. **Zero-shot restriction** — no training. Run a pretrained MoE with prefill-exact /
   decode-predicted resident sets. Quality = whatever Section 5 reports. Free; the baseline.
2. **Light adaptation** — fine-tune router + macro-router (experts frozen) so the model
   *expects* the `K`-restriction and the macro-router learns to predict windows. Recovers
   most of the gap cheaply.
3. **Train-from-scratch** with window-shared routing + an auxiliary loss that rewards
   *low working-set growth* within a window (encourages temporal expert locality). Highest
   quality at a given `(B,K)`; aligns the model's inductive bias with the deployment.

## 8. Minimal experiment plan (on FLAME-MoE)

1. **Locality probe** (no training): extend `router_saturation.py` to report working-set
   size and retained-top-`k`-mass as functions of `(B, K)` on a held-out set. → Go/no-go.
2. **Zero-shot eval:** implement masked-to-`S` routing at inference; plot perplexity vs.
   `(B, K)` against the `K=E` ceiling.
3. **Macro-router:** add the per-layer boundary predictor; report prediction hit-rate
   (predicted `S` ⊇ realized top-`k`) and the perplexity it costs vs. prefill-exact.
4. **Systems:** wire NVMe-backed experts with double-buffered async prefetch; report decode
   throughput as **% of the fully-resident (all-experts-in-RAM) ceiling** — the comparison the
   cache-offload papers omit — against (a) Mixtral-style LRU offload, (b) Cache-Conditional
   per-token bias, (c) Belady's oracle, all at matched RAM. Headline = footprint vs. % of
   resident speed.
5. **Adaptation:** fine-tune router + macro-router (experts frozen); re-plot the frontier.

## 9. Open questions / risks

- **Prediction at long horizon.** Accuracy of `S(m,ℓ)` likely degrades with `B`; the
  horizon, not just sharing, may dominate the quality loss. Mitigation: `K > k` headroom,
  co-activation-expanded sets, or shorter `B` with locality covering the gap.
- **Working-set blowups.** Topic shifts / rare tokens can spike distinct-expert demand
  past `K`. Need a fallback (on-demand load + stall, or temporary `K` bump) and a measured
  worst-case tax.
- **Load imbalance across windows** complicates a fixed `K`; a small variable slack helps.
- **Where the win is real.** At 10.3B total, all experts already fit in commodity RAM — the
  case is for *large* total-param MoEs (100B+) on memory-constrained hosts. FLAME-MoE is the
  cheap prototype to characterize the frontier; the economics land at scale.
- **Interaction with quantization** (e.g. 4-bit experts) shrinks every figure here
  proportionally and is fully composable — report both axes.

---

### Notation

`E` experts/layer · `k` top-k per token · `K` resident budget/layer (`k ≤ K ≤ E`) ·
`L` MoE layers · `B` window length · `S(m,ℓ)` resident set for window `m`, layer `ℓ` ·
`Δ(m,ℓ)` newly-loaded experts · `r` SSD bandwidth · `t_tok` per-token compute time.
