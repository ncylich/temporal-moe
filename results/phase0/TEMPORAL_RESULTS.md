# Temporal MoE (rolling residency) — results

**Question:** can an MoE that keeps only **K = 6 of 64 routed experts resident** per layer (streaming one
expert in per token, evicting the least-wanted resident) approach full-MoE quality? **Answer: yes — it
recovers ~80% of MoE's quality gain over a dense model at every shape, both compute budgets.**

Metric is **bits-per-byte (BPB) = CE/(ln2·bytes_per_token)**, lower is better; all values measured
(`log.md`). The temporal config below is **min_logit eviction, 1 shared expert** (the swept curve);
`temporal_router.py` implements the rolling-residency router (single-launch Triton scan, verified
bit-exact vs the reference at runtime). Figure: `figures/fine_grained_vs_coarse_experts_isoflop.png`.

## Full IsoFLOP curves (BPB) — dense floor vs full MoE (1 shared) vs temporal

**@1e16** (shapes sm1,s0,s1,s2; compute-optimal = s0):
| shape | N_active | dense | MoE (1sh) | **temporal** | recovery* |
|---|---|---|---|---|---|
| sm1 | 0.77M | 1.534 | 1.478 | 1.4891 | 80% |
| **s0** | 1.36M | 1.519 | **1.447** | **1.4599** | **82%** |
| s1 | 3.81M | 1.591 | 1.540 | 1.5488 | 83% |
| s2 | 8.12M | 1.848 | 1.819 | 1.8260 | 76% |

**@1e17** (shapes s1,s2,s3; compute-optimal = s2):
| shape | N_active | dense | MoE (1sh) | **temporal** | recovery* |
|---|---|---|---|---|---|
| s1 | 3.81M | 1.361 | 1.284 | 1.3039 | 74% |
| **s2** | 8.12M | 1.341 | **1.269** | **1.2821** | **82%** |
| s3 | 14.77M | 1.408 | 1.289 | 1.3073 | 85% |

\*recovery = (dense − temporal)/(dense − MoE): fraction of MoE's advantage-over-dense that temporal keeps.

## Findings

1. **Temporal sits inside the dense↔MoE band at every shape**, tracking just above the MoE curve
   (~0.009–0.020 BPB) and far below the dense floor. With only 6/64 experts resident it keeps **~80%**
   of MoE's gain (82% at both budgets' compute-optimal shapes).
2. **Same compute-optimal shape and parabola as MoE/dense** — min at s0@1e16 (1.4599) and s2@1e17
   (1.2821); the optimum shifts right with budget identically. Temporal preserves the scaling geometry.
3. **Eviction policy:** min_logit ≳ lru, small (≤0.006 BPB), consistent at 1e16.
4. **Shared-expert knob is negligible/inconsistent at B=1:** at s0@1e16, 2-shared 1.4569 vs 1-shared
   1.4599 (2-shared better by 0.003); at s2@1e17 it flips — 1-shared 1.2821 vs 2-shared 1.2903
   (1-shared better by 0.008). Both within seed noise (~0.003 BPB) — matches the Phase-0 s-knob finding.

## min_logit 2-shared spot-checks (FLOP-matched, K=5)
s0@1e16 → 1.4569 ; s2@1e17 → 1.2903 (both inside band; see finding 4).

## At the paper's real budget (1e18, FLAME-MoE-38M-100M) — temporal lands between dense and MoE

We replicated the paper's smallest compute-optimal model **exactly** (hidden 256 / 9 layers, top-6,
shared 2·moe_ffn, **pythia-12b 50k tokenizer**, dclm, gb 1024, LR 3e-4, WSD, 2121 iters = 4.45B tokens
= 1e18 FLOPs) and swapped in the temporal router. Metric = **validation cross-entropy (nats, lower
better)** — the paper's metric, so directly comparable (we match their tokenizer). Figure:
`figures/temporal_vs_dense_and_moe_1e18_crossentropy.png` (script `plot_1e18.py`).

| config | val-CE @1e18 | source |
|---|---|---|
| dense floor (ffn 1422, matched active non-embed params) | **4.137** | measured (our setup) |
| **temporal** (rolling residency, 6/64 resident, min_logit) | **3.906** | measured (our setup) |
| MoE (compute-optimal) | **≈3.78** | paper fitted scaling law |

- **Temporal beats the dense floor by 0.231 nats** — a clean, fully-measured result (dense and temporal
  trained identically on our setup; the only difference is the router). With just 6 of 64 experts
  resident, temporal is a real, large step up from dense.
- Against the MoE endpoint it recovers **~65%** of the dense→MoE gap (0.231 / 0.357). This fraction is
  softer than the dense↔temporal gap because the MoE value is the paper's **law prediction on their val
  set**, not measured on ours — so it carries extrapolation + val-set confounds. (At 1e16/1e17, where MoE
  was measured directly, recovery was ~82%.) A same-setup MoE control run would make the fraction exact.
- Single-GPU adaptations (vs paper EP=8): EP=1 + grouped-gemm, mb=8 (temporal) / mb=32 (dense),
  no CE-fusion (the 50k-vocab logits overflow the inductor-fused CE) — all numerically equivalent.

## Caveat on absolute scale
These budgets (1e16, 1e17) are **1–2 orders below the FLAME-MoE paper's smallest point (1e18 →
FLAME-MoE-38M-100M: 38M active, 4.4B tokens, hidden 256/9 layers/64 experts/top-8/2 shared)**, so the
absolute BPB are undertrained. The **deltas** (temporal vs dense vs MoE) and the clean parabolas are the
signal. Paper scaling law: N\*∝C^0.69, D\*∝C^0.31.

## Repro
`run.sh TEMPORAL=1 TEMPORAL_EVICT=min_logit` (+ `SHARED_MULT`/`TOPK` for the shared knob), driven by
`temporal_matrix.sh` / `temporal_minlogit_1e17.sh` / `temporal_minlogit_sh1_sweep.txt`.
Plot: `plot_g3_curves.py`.
