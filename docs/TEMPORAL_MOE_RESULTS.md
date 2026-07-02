# Temporal MoE (rolling residency) — results

Complete results for the temporal Mixture-of-Experts (MoE) variant: a from-scratch MoE that keeps only
**K = top-k routed experts resident** per layer (of E = 64), streaming one expert in per token and
evicting the least-wanted resident, so resident RAM is a small fraction of all experts. Goal: **not** to
match full MoE, but to be a tunable point on the RAM-footprint↔quality frontier **between a dense model
and a full MoE**, for RAM-constrained deployment.

**Bottom line:** across three compute budgets, temporal (with only **6 of 64 experts resident**) lands
**inside the dense↔MoE quality band at every model size**, recovering **~80%** of the MoE-over-dense
quality gain at 1e16/1e17 and **beating the dense floor by 0.231 nats** at the paper's real 1e18 budget.
The routing mechanism is validated; it does not collapse toward dense.

Design/implementation handoff: `docs/TEMPORAL_HANDOFF.md`. Methodology (metric, harness, stop rules):
`docs/EVALUATION_METHODOLOGY.md`. Measured ledger: `results/phase0/log.md`.

---

## Metrics

- **1e16 / 1e17** (our sweeps): **bits-per-byte (BPB) = CE/(ln2·bytes_per_token)**, lower is better —
  tokenizer-invariant because these runs use a custom 16k-vocab BPE (chosen for speed). Divisor 2.7568.
- **1e18** (paper-replication): **validation cross-entropy (CE, nats)**, lower is better — the paper's
  metric, using the paper's exact 50k pythia tokenizer so the number is directly comparable.
- The two metrics are **not** comparable across budgets (different vocab); compare *within* a budget.

The temporal config reported below is **min_logit eviction, 1 shared expert** (the swept configuration);
`scripts/phase0/temporal_router.py` is the router.

---

## Results at 1e16 and 1e17 (our IsoFLOP sweeps, BPB)

Full curves: dense floor vs full MoE (1 shared) vs temporal, over dense-matched shapes. Figure:
`results/phase0/figures/temporal_vs_dense_and_full_moe_isoflop.png` (combined single-axes, both budgets).

**@1e16** (compute-optimal shape s0):
| shape | N_active | dense | MoE | **temporal** | recovery* |
|---|---|---|---|---|---|
| sm1 | 0.77M | 1.534 | 1.478 | 1.4891 | 80% |
| **s0** | 1.36M | 1.519 | **1.447** | **1.4599** | **82%** |
| s1 | 3.81M | 1.591 | 1.540 | 1.5488 | 83% |
| s2 | 8.12M | 1.848 | 1.819 | 1.8260 | 76% |

**@1e17** (compute-optimal shape s2):
| shape | N_active | dense | MoE | **temporal** | recovery* |
|---|---|---|---|---|---|
| s1 | 3.81M | 1.361 | 1.284 | 1.3039 | 74% |
| **s2** | 8.12M | 1.341 | **1.269** | **1.2821** | **82%** |
| s3 | 14.77M | 1.408 | 1.289 | 1.3073 | 85% |

\*recovery = (dense − temporal)/(dense − MoE): fraction of MoE's advantage-over-dense that temporal keeps.

**Findings:** temporal tracks ~0.01–0.02 BPB above MoE at every shape, well inside the band; it has the
**same compute-optimal shape and parabola** as dense/MoE (min s0@1e16, s2@1e17). Eviction: min_logit ≳
lru (≤0.006 BPB). Shared experts: negligible/inconsistent at B=1 (2-shared better @1e16 by 0.003,
1-shared better @1e17 by 0.008 — both within seed noise), matching the Phase-0 s-knob finding.

---

## Result at 1e18 (paper's real budget — FLAME-MoE-38M-100M, CE)

We replicated the paper's smallest compute-optimal model **exactly** (hidden 256 / 9 layers, top-6,
shared 2·moe_ffn, **pythia-12b 50k tokenizer**, dclm, global-batch 1024, LR 3e-4, WSD schedule, 2121
iters = 4.45B tokens = 1e18 FLOPs) and swapped in the temporal router. Figure:
`results/phase0/figures/temporal_vs_dense_and_moe_1e18_crossentropy.png`.

| config | val-CE @1e18 | source |
|---|---|---|
| dense floor (ffn 1422, matched active non-embed params) | **4.137** | measured, our setup |
| **temporal** (6/64 resident, min_logit) | **3.906** | measured, our setup |

- **Temporal beats the dense floor by 0.231 nats** — fully measured (dense and temporal trained
  identically; only the router differs). A real, large step up from dense with 6/64 experts resident.
- Measured full-MoE controls at 1e18 (coarse 64-expert and fine-grained 192-expert) are in
  `results/phase0/G3_RESULTS.md`; temporal lands inside that measured dense↔MoE band.
- Model size follows the paper's compute-optimal rule (N\* ∝ C^0.69, D\* ∝ C^0.31); we replicate the
  paper's published FLAME-MoE-38M shape exactly.

Single-GPU adaptations vs the paper's 32-GPU (EP=8) setup — all numerically equivalent: EP=1 +
`--moe-grouped-gemm`; TransformerEngine impl; mb 8 (temporal) / 32 (dense) + no CE-fusion (the 50k-vocab
logits overflow the inductor-fused cross-entropy → OOM); metric on our 90/5/5 dclm val split.

---

## Router performance (why this was runnable)

The rolling-residency selection is a per-token sequential scan (2048 steps × every MoE layer × every
micro-batch) — originally ~10× slower than the baseline router (kernel-launch-bound). We fused the whole
scan into a **single-launch Triton kernel** (grid over batch, E-wide state in registers): ~91 ms →
**~1 ms per call**, taking training from 3.5 → **~1.35 s/iter** (≤50% over the dense-router baseline).

**Correctness:** the reference `compute_resident_mask` is unchanged and unit-tested (11 tests). The
Triton path is verified **bit-exact vs the reference once at runtime**; on any mismatch or kernel error
it **raises hard** (no silent fallback) so a bug crashes the run rather than degrading quietly. Every
1e16/1e17/1e18 run logged `scan path: triton (verified == reference)` with zero mismatches.

---

## Reproduce

- Router + tests: `scripts/phase0/temporal_router.py`, `test_temporal_router.py`; entrypoint
  `pretrain_temporal.py` (monkeypatches `TopKRouter.forward`).
- 1e16/1e17: `run.sh TEMPORAL=1 TEMPORAL_EVICT=min_logit` via `temporal_matrix.sh` /
  `temporal_minlogit_1e17.sh` / `temporal_minlogit_sh1_sweep.txt`. Plot: `plot_temporal_final.py`.
- 1e18: `flame38m_temporal.sh` (temporal) and `flame38m_temporal.sh` with `DENSE=1` (floor).
  Plot: `plot_1e18.py`.
