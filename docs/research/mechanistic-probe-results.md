# Mechanistic probe results — the cheap graphs (A, B, C)

*Why* rolling-residency temporal MoE works, measured post-hoc on trained checkpoints (no training).
Probe: `scripts/phase0/router_probe.py` (via `run.sh PROBE=1`) logs, per MoE-layer per-token on **one
fixed 16×2048 batch** (seed 1234 → identical tokens across models), the raw gating logits (pre-mask) and,
for temporal models, the resident set used. Analysis + figures: `scripts/phase0/plot_probe.py`.

Checkpoints probed (matched temporal+MoE pairs, 16k vocab, plus G3): `tmoe_minlogit_sh1_s2_1e17` +
`v16k_sweep_s2_1e17` (s2@1e17, 64 experts, k=6); `tmoe_minlogit_sh1_s0_1e16` + `v16k_d_s0_1e16`;
`g3_tmoe_s1_1e17` (192 experts, k=18).

## A — experts chosen per token (`results/phase0/figures/expert_selection_per_token_8M_model.png`)
Token × expert raster, s2@1e17, deepest MoE layer, three panels:
- **full MoE (top-k)** — scattered dots, switches experts nearly every token.
- **temporal (resident set used)** — clear horizontal bands (held experts).
- **temporal (unconstrained preference)** — the model's free top-k with the mask removed — **also banded**,
  i.e. it *wants* temporally-stable experts, unlike the vanilla MoE.

**A3 scalar — did the model learn to want what's resident?** Overlap of the model's current top-k with the
*previous* active set (resident set for temporal, previous top-k for MoE):
- **temporal 38.2%** vs **vanilla MoE 21.3%** → training under the residency constraint made the router
  **~1.8× more temporally self-consistent**. The constraint became *partly self-enforcing*: the model
  internalized locality rather than fighting the mask.

## B — rolling-policy hit-rate vs resident budget K (`results/phase0/figures/routing_coverage_vs_resident_cache_size.png`)
Replay each model's demand through *our* policy (K resident, ≤1 swap/token, min_logit evict); mean top-k
coverage vs K/k:
- **temporal routing is ~2× more cacheable than the MoE** at every budget (s2: 36% vs 18% at K=k).
- A modestly bigger cache pays off fast: **K=k → 2k ≈ +15 points** coverage; → 1.0 (= full MoE) at K=E.
- **G3 (18/192) sits just below G1 (6/64)** — the same fine-graining penalty seen in the quality results.

## C — expert lifetime vs K (`results/phase0/figures/expert_lifetime_vs_resident_cache_size.png`)
Mean consecutive tokens an expert stays resident grows super-linearly with K; temporal > MoE at moderate K;
G3 experts live longer per swap (more experts, each churned less often).

## Takeaways / design implications
1. **The model learns the locality** (A3, and the banded unconstrained-preference panel): the residency
   constraint is largely internalized, not imposed against the router's will. → an explicit
   churn/coherence loss or straight-through swap has *less* to fix than expected; worth testing whether a
   little extra temporal-coherence pressure closes the residual gap (esp. for G3).
2. **A larger resident cache is a cheap, high-leverage lever** (B, C): K=k→2k roughly halves the routing
   miss — motivating the one remaining *training* experiment, **D (BPB vs K)**: does K>k headroom buy back
   the MoE-quality gap (and close the G3 dip)?
3. **G3 is consistently a touch less cacheable/self-consistent than G1**, matching the quality recovery dip
   — finer experts specialize harder and tolerate residency slightly worse.

All A/B/C come from the same 5 probe passes; only D needs new training.

## Larger models (#1 s3@1e17 14.8M, #2 38M@1e18 real budget) — does it hold at scale?

Added probes: `tmoe_minlogit_sh1_s3_1e17` + `v16k_sweep_s3_1e17` (s3, ~14.8M active), and
`flame38m_temporal_minlogit` (38M active, the paper's 1e18 budget, 50k vocab). Figures:
`results/phase0/figures/expert_selection_per_token_15M_model.png`, `results/phase0/figures/expert_selection_per_token_38M_model.png`, `results/phase0/figures/learned_temporal_locality_vs_model_size.png` (+ B/C now overlay all).

**A3 overlap (top-k(t) vs previous active set), by scale:**

| model | active | temporal | vanilla MoE | random |
|---|---|---|---|---|
| s0 @1e16 | 1.4M | 33.2% | 19.4% | 9.4% |
| s2 @1e17 | 8.1M | 38.2% | 21.3% | 9.4% |
| s3 @1e17 | 14.8M | 33.5% | 20.2% | 9.4% |
| 38M @1e18 | 38M | 30.4% | (not trained) | 9.4% |

**Finding: the learned-locality effect is robust but roughly flat, not a scaling law.** Temporal stays
**~1.5–1.8× the vanilla MoE and ~3–4× random at every size** (incl. 38M at the paper's real 1e18 budget,
50k vocab). The 38M raster still shows clearly banded resident + unconstrained-preference panels. It does
**not** strengthen with scale in our range (peaks at s2, slight decline at 38M — though 38M uses a
different tokenizer/data, so it's not a clean same-family comparison; the clean 16k family s0/s2/s3 is
33→38→33%). B/C: all temporal models cluster ~2× more cacheable than the MoE regardless of size.

**Caveat:** these are still small (≤38M active) vs frontier MoEs; the effect could scale differently at
100B+. The definitive test is a released FLAME-MoE (290M–1.7B) probe (#3) or a larger temporal train (#4).
