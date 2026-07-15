# Cache-Conditional Experts (summary)

Skliar et al., *Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference*,
arXiv:2412.00099 (Dec 2024; Contextual AI · Qualcomm AI Research · Tivaro; preprint, under
review). Companion to [temporal-moe.md](./temporal-moe.md) — this is the nearest *routing-side*
baseline to Temporal MoE: same locality target, but a **soft, reactive, per-token** nudge
instead of a hard window commitment.

**One line.** At inference, bias each token's expert *ranking* toward experts already in
cache — so once an expert is loaded it tends to be reused, cutting Flash→RAM reloads —
*without retraining and without changing the experts' output weights.*

## Setup

On-device decode, **batch size 1**: experts live in slow Flash, a small RAM **cache** holds
a subset; loading an uncached expert is the bottleneck. Standard top-`k` routing re-picks
experts every token, thrashing the cache. Goal: raise cache reuse at fixed quality.

## Mechanism (per MoE layer, per token `t`)

Symbols:

| Symbol | Meaning | How it's determined |
|---|---|---|
| `z ∈ ℝ^E` | router logits over the `E` experts for this token/layer | model forward pass (unchanged) |
| `k` | experts selected per token (top-`k`) | model architecture |
| `m̃_t ∈ {0,1}^E` | cache mask: `1` for experts resident in cache at token `t` | current cache state (LRU) |
| `λ ∈ [0,1]` | cache-bias strength: `0` = original routing, `1` = max bias | **swept** (see below) |
| `Δ_avg ∈ ℝ` | per-layer average logit *range*, scales the bias to this layer | **online running mean** (see below) |
| `J` | number of top "critical" experts always kept regardless of cache | hyperparameter (accuracy floor) |

Step 1 — biased logits, **used for ranking only**:

```
z' = z + λ · Δ_avg · m̃_t
```

In-cache experts get their logit lifted by `λ·Δ_avg`; this can pull a cached expert into the
top-`k` ahead of a marginally-better uncached one.

Step 2 — select experts by the new ranking `r'` derived from `z'`, **but** with a guarantee
that the **top-`J` experts by the original `z`** are always included (so the most important
experts are never dropped for cache reasons — this bounds the quality loss).

Step 3 — compute the gate/output weights from the **original, unmodified `z`** (softmax over
the selected set). The bias changes *which* experts run, never *how much* each contributes.

## How each knob is set (calibration status)

- **`Δ_avg`** — defined as the expected gap between the largest and smallest router logit:

  ```
  Δ_avg = E_{x∈X} E_{t∈1..T} [ max(z) − min(z) ]
  ```

  Estimated **per layer** as a **running average during inference** — no calibration corpus.
  (Appendix D: this online estimate matches a full-dataset estimate and is *more* robust
  out-of-domain.) Its role: normalize the bias to each layer's logit scale so a single global
  `λ` acts consistently across layers of different magnitudes.

- **`λ`** — a **single global scalar**, identical across all layers and models. Not learned
  or calibrated: you **sweep** it over `[0,1]` (paper uses 50 points) and pick the operating
  point on the accuracy↔cache-hit Pareto curve that meets your budget.

- **Net:** **training-free *and* calibration-free**; the only tuning is a one-time global `λ`
  sweep, and the only data-derived statistic (`Δ_avg`) self-estimates online.

## Findings

- **2× on-device speedup** (Snapdragon), achieved while caching **30/60 and 45/60 experts**
  (i.e. **50–75%** of all experts resident — a *large* cache regime).
- **Cache miss rate roughly halved**: Qwen1.5-MoE 35%→16%, DeepSeek-V2 28%→7%,
  Phi-3.5-MoE 22%→9%, Mixtral 40%→21%.
- **Negligible quality cost**: perplexity +0.1–3%, downstream accuracy <0.1%.
- **Expert lifetime** — average consecutive tokens an expert stays cached before LRU
  eviction — rises from **19–26 → 55–76 tokens** (Qwen 26→58, DeepSeek 19→76, Phi 22→55).
- **Batch-size-1 only**, by design: the bias depends on the cache state left by the *previous*
  token, an inherently sequential feedback loop; no larger-batch evaluation.

## Caveat: never benchmarked against a fully-resident model

This materially limits how useful the result is. Their baselines are **LRU** (the 2× is over
this) and **Belady's oracle** — both *under the same offloading constraint*. They **never
compare to the all-experts-in-RAM, zero-streaming ideal**, and report only relative speedups
(no ms/token), so the gap to the no-offload ceiling — the comparison that reveals the
technique's true cost — is unquantified. (Structurally, that config doesn't fit on their
target phones, which is their premise; but it leaves "how much do you give up vs. resident?"
unanswered.)

What we *can* infer. They state the regime stays **Flash-bound** ("loading from flash is the
major bottleneck"; throughput ∝ hit rate), so residual misses still cost real time. At
batch-1 the expert FFN is itself RAM-bandwidth-bound, so with overlapped prefetch
per-token time `= max(T_compute, T_load)`, and **Flash stops masking compute once**

```
miss_rate ≲ BW_flash / BW_ram        (≈ 5–15% on mobile; higher with always-on shared experts)
```

i.e. the crossover miss rate is just the **storage-to-RAM bandwidth ratio**. Their post-method
rates straddle it: **DeepSeek 7% / Phi 9%** sit near the compute-bound plateau (close to
resident speed), while **Qwen 16% / Mixtral 21%** stay Flash-bound, ≈ `miss_rate/miss_rate*`
→ **~2–3× slower than a fully-resident model** (estimate; the paper gives no bandwidth
numbers). **New experts loaded per token** = `miss_rate · k` per layer ≈ **0.2–0.6/layer**
(~half the LRU baseline), order ~6–15/token across all layers.

The batch-1 ceiling is the point: with a window of `B` tokens the compute side scales by `B`,
so the crossover becomes `miss_rate* ≈ B · BW_flash/BW_ram` — at `B≈16` Flash is hidden almost
regardless of turnover. That is exactly the corner this method, fixed at `B=1`, cannot reach.

**It gets worse on faster hardware.** The per-miss cost is `BW_fast/BW_offload` resident-experts
of compute. The paper's mobile fast tier (~50 GB/s RAM) is only ~12× its Flash, so one miss
costs ~12 resident experts — borderline hideable. On an A6000 (VRAM 768 GB/s ↔ SSD ~7 GB/s) one
miss costs ~110; on an M4 Pro (unified RAM 273 GB/s ↔ SSD ~5 GB/s) ~55 — but only `k` experts
(e.g. 8) run per layer to hide behind. So at batch-1 the method is **disqualified for
single-machine GPU/Apple-Silicon offload**: it lowers miss *rate* but cannot make any single
miss cheap. Full derivation and per-platform numbers in
[`./cce/FINDINGS.md`](./cce/FINDINGS.md).

## Relation to Temporal MoE

- **Same locality, opposite commitment.** They *softly bias* per-token routing toward a
  *large* cache; Temporal MoE *hard-commits* a *small* resident set for a window of `B`
  tokens, chosen predictively for prefetch. Their 55–76-token lifetimes are independent
  evidence that the tens-of-tokens locality Temporal MoE exploits is real.
- **Different regime.** Their gains come at 50–75% cache and batch 1; Temporal MoE targets
  ~10% resident and batched windows — the corner their sequential, large-cache method does
  not enter. Where caches are large and storage fast, this method likely Pareto-dominates us
  at near-zero quality cost; concede that.
- **Composable.** Their `λ·Δ_avg·m̃` bias is a natural *within-window tiebreaker* among
  Temporal MoE's `K` resident experts, inheriting its calibration-free `Δ_avg` and single
  global `λ`. Their `λ`-sweep is also the template for reporting our own `(B, K)` Pareto curve.

Source: [arXiv:2412.00099](https://arxiv.org/abs/2412.00099) ·
[HTML](https://arxiv.org/html/2412.00099v2) ·
[OpenReview](https://openreview.net/forum?id=ul4W26KEKz)
