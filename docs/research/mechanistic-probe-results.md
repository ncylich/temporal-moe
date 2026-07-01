# Mechanistic probe results — the cheap graphs (A, B, C)

*Why* rolling-residency temporal MoE works, measured post-hoc on trained checkpoints (no training).
Probe: `scripts/phase0/router_probe.py` (via `run.sh PROBE=1`) logs, per MoE-layer per-token on **one
fixed 16×2048 batch** (seed 1234 → identical tokens across models), the raw gating logits (pre-mask) and,
for temporal models, the resident set used. Analysis + figures: `scripts/phase0/plot_probe.py`.

Checkpoints probed (matched temporal+MoE pairs, 16k vocab, plus G3): `tmoe_minlogit_sh1_s2_1e17` +
`v16k_sweep_s2_1e17` (s2@1e17, 64 experts, k=6); `tmoe_minlogit_sh1_s0_1e16` + `v16k_d_s0_1e16`;
`g3_tmoe_s1_1e17` (192 experts, k=18).

## A — experts chosen per token (`probe_A_raster.png`)
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

## B — rolling-policy hit-rate vs resident budget K (`probe_B_coverage_vs_k.png`)
Replay each model's demand through *our* policy (K resident, ≤1 swap/token, min_logit evict); mean top-k
coverage vs K/k:
- **temporal routing is ~2× more cacheable than the MoE** at every budget (s2: 36% vs 18% at K=k).
- A modestly bigger cache pays off fast: **K=k → 2k ≈ +15 points** coverage; → 1.0 (= full MoE) at K=E.
- **G3 (18/192) sits just below G1 (6/64)** — the same fine-graining penalty seen in the quality results.

## C — expert lifetime vs K (`probe_C_lifetime_vs_k.png`)
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
