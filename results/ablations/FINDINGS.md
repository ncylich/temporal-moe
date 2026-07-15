# Phase-0 findings — FLAME-MoE baselines, dense floor, temporal MoE, fine-graining

One consolidated narrative of the Phase-0 ablation program (single-GPU A6000/H100). Combines the
former `results/phase0/{RESULTS,DENSE_BASELINES,PASS,TEMPORAL_RESULTS,G3_RESULTS}.md` (verbatim
numbers; duplication removed; stale status lines fixed — those originals live in this branch's
history). Per-point data: `phase0_isoflop_points.csv` (finals for every §2–§6 run, extracted from the raw
run log `results/phase0/log.md` on `temporal-moe-impl@66f786b7`) + `phase0_lr_tuning.csv` (§1) +
`flame38m_1e18_cells.csv` (§7); later 1e18/1e19 programs have their own CSVs — see
[README.md](README.md). **Eval protocol:** all §2–§6 quality numbers below are the
**end-of-training test-set eval** (`ce_test_final`) — final weights, held-out split, identical
protocol for every run. The original write-ups sometimes quoted during-training val evals ~0.004 CE
higher (20-batch subsample, occasionally pre-final iteration); both series are preserved per point
in `phase0_isoflop_points.csv`. One value (temporal s2@1e17, 1.2821) has no surviving train.log and
keeps its published figure. Figures: `../phase0/figures/`.

**Metrics.** `BPB = CE_nats / (ln2 · bytes_per_token)` — bits-per-byte, tokenizer-invariant, lower
is better (needed because Phase-0 sweeps use a custom 16k-BPE; divisor 2.7568 for bpe-16k,
2.978 for pythia-50k). 1e18 results report raw **validation cross-entropy in nats** (the paper's
metric; we match their pythia-50k tokenizer there). `recovery = (dense − temporal)/(dense − MoE)`:
the fraction of MoE's quality gain over dense that temporal keeps (higher is better).

---

## 1. Setup — locked hyperparameters & single-GPU adaptations

All 1e16/1e17 sweep runs: peak-LR **3e-3**, warmup **5%** of iters (cosine → 10%, min-lr 3e-4),
grad-clip 1.0, weight-decay 0.01, aux-loss 0.01, z-loss 0.001, global-batch 256, micro-batch 32,
bf16, seed 1234. MoE: num_experts 64, top-k 6, 1 shared expert (intermediate 2·moe_ffn),
moe_layer_freq [0]+[1]*(L−1). Tuned at s2 (LR sweep @1e16 → confirm @3e16: lr3e-3 beat lr1e-3 at
the longer budget), re-validated stable across the 14× size range (s2/s4/s6 @3e16, no divergence).

Single-GPU adaptations vs stock FLAME (documented for fidelity; all FLOP/param-neutral):

- **EP=1** (vs FLAME's EP=8) + `--moe-grouped-gemm` to batch the 64 local experts (numerically
  equivalent to FLAME's per-GPU sequential experts).
- **TransformerEngine impl** (FLAME's native path), TE 1.11 built from the vendored source;
  `--no-gradient-accumulation-fusion` (apex absent; perf-only).
- **head_dim = 16 for every shape** (heads = hidden/16). Fixed 16 heads gave head_dim 12/20/28 for
  s1/s3/s5 → TE fused-attention fell back to a slow path (3× slower); identical params/FLOPs.
- **16k-vocab BPE** instead of the 50k pythia tokenizer + **fused cross-entropy** — 1.69× faster
  (the 50k logits dominated FLOPs at these tiny scales). Reported in BPB to stay comparable.
- micro-batch capped at 32 by the vocab-logits memory.

**Caveat on absolute scale.** The 1e16/1e17 budgets are 1–2 orders below the FLAME paper's smallest
point (1e18 → FLAME-MoE-38M-100M), so absolute BPB are undertrained — the **deltas** (temporal vs
dense vs MoE, coarse vs fine) and the clean parabolas/min-shift are the signal. Paper scaling law:
N\*∝C^0.69, D\*∝C^0.31.

Repro: `scripts/phase0/run.sh` (env-parametrized launcher), `drive.sh` (driver), `shapes.py`,
`parse_run.py`, `summarize.py`; data = dclm-baseline tokenized with the 16k BPE
(`train_tok16k.py` + `fast_tokenize.py`), 5.55B-token corpus.

## 2. G=1 FLAME-MoE baseline — IsoFLOP parabolas match the scaling law

At fixed compute, validation loss vs model size is a **parabola**, and its **minimum shifts right
as the FLOP budget grows** — the compute-optimal model size scales with compute, exactly as the
FLAME scaling law predicts.

| budget | parabola minimum | min loss (BPB) | law-predicted optimal N |
|---|---|---|---|
| 1e16 | **s0 (1.36M active)** | 1.447 | 1.48M |
| 1e17 | **s2 (8.12M active)** | 1.269 | 7.74M |

Measured frontiers (test BPB): **@1e17** s1 1.2803 · **s2 1.269 (min)** · s3 1.289.
**@1e16** (dedicated annealed runs incl. sub-s1 shapes) s₋₁ 1.4766 · **s0 1.447 (min)** · s1 1.540 ·
s2 1.819 · s3 2.187 (monotone-increasing across s1–s6).

These are **the (B=1, s=1) temporal-reference baselines**: 1e17 → 1.269 BPB / CE 3.4985 nats (s2),
1e16 → 1.447 BPB / CE 3.988 nats (s0).
Plot: `../phase0/figures/fine_grained_vs_coarse_experts_isoflop.png`.

## 3. Dense IsoFLOP floor — MoE beats an equal-cost dense model everywhere

For each shape we trained a plain dense (no-MoE) SwiGLU transformer with `ffn_hidden` enlarged so
its total non-embedding params equal the MoE's **active** non-embedding params — same FLOPs, same
tokens, same locked HPs (dense ffn rounded to even; odd ffn crashes the fused-swiglu JIT warmup).
The only difference is dense-vs-sparse: the honest "what if you spent this compute on a dense model
instead" floor.

**@1e16 (BPB):**
| shape | N_active | dense | MoE | MoE − dense |
|---|---|---|---|---|
| sm1 (=s₋₁) | 0.77M | 1.534 | 1.4766 | **−0.057** |
| **s0** | 1.36M | **1.519 (min)** | **1.447 (min)** | **−0.072** |
| s1 | 3.81M | 1.591 | 1.540 | −0.051 |
| s2 | 8.12M | 1.848 | 1.819 | −0.029 |

**@1e17 (BPB):**
| shape | N_active | dense | MoE | MoE − dense |
|---|---|---|---|---|
| s1 | 3.81M | 1.361 | 1.2803 | −0.081 |
| **s2** | 8.12M | **1.341 (min)** | **1.269 (min)** | **−0.072** |
| s3 | 14.77M | 1.408 | 1.289 | −0.119 |
| s4 | 24.29M | 1.485 | — | — |

(MoE @1e17 was only swept s1–s3, the parabola bracket; dense added s4 to bracket its own min.)

Findings: (1) **MoE wins everywhere** (−0.03 to −0.12 BPB; no shape where dense catches up).
(2) **Same compute-optimal shape** — both parabolas bottom out at s0@1e16 / s2@1e17 and the optimum
shifts right identically: MoE lowers the curve without changing the dense scaling geometry.
(3) **The MoE advantage grows with size** (@1e17: −0.081 at s1 → −0.119 at s3); the 1e16 right arm
is far past optimum and noisier. Headline: **~0.072 BPB MoE gain at the compute-optimal shape of
both budgets.** Repro: `run.sh` with `DENSE=1`, configs `dense_1e16.txt`/`dense_1e17.txt`
(+ `dense_ext_*.txt`); note `DENSE=1` needs absolute `TOKENIZER_MODEL`/`DATA_DIR` paths (run.sh
`cd`s into Megatron-LM first).

## 4. Acceptance criteria — all 4 PASS

FLAME-law CE targets → BPB bars (÷2.978): **≤1.645 @1e17, ≤2.149 @1e16** (law optima 1.578/2.048).

1. **Best-shape ≤ bar — PASS.** @1e17 s2 = 1.269 ≤ 1.645; @1e16 s0 = 1.447 ≤ 2.149 (real models
   beat the law's pessimistic 1e16 extrapolation by ~0.8 BPB).
2. **Curve shape — PASS.** @1e17 parabola, min at s2 (within s1–s3); @1e16 parabola, min at s0,
   monotone-increasing s1–s6.
3. **Reproducible — PASS.** s2@1e17 seed-2 test CE 3.5004 vs seed-1 3.4985 → |Δ| = 0.0019 nats ≤ 0.03.
4. **Healthy routing — PASS.** s2@1e17 per-MoE-layer max/mean expert load 1.44–2.07× (worst
   2.07× ≪ 8×); aux-loss converged.

**Bonus — the shared-expert (s) knob is negligible at B=1.** FLOP-matched s=2 (two constant
experts, top-5) vs s=1 (one shared, top-6): both parabolas reproduce the same minima; s=2 is
~+0.001 BPB @1e17, ~−0.01 BPB @1e16 — within seed noise (~0.003 BPB). The constant-vs-routed
tradeoff is expected to matter only under windowed routing (B>1).

## 5. Temporal MoE (rolling residency, G=1) — ~80% of the MoE gain with 6/64 resident

**Question:** can an MoE that keeps only **K = 6 of 64 routed experts resident** per layer
(streaming one expert in per token, evicting the least-wanted resident) approach full-MoE quality?
**Answer: yes — ~80% recovery at every shape, both budgets.** Config: min_logit eviction, 1 shared
expert; `temporal_router.py` (single-launch Triton scan, verified bit-exact vs the reference).

**@1e16 (BPB; compute-optimal = s0):**
| shape | N_active | dense | MoE (1sh) | **temporal** | recovery |
|---|---|---|---|---|---|
| sm1 | 0.77M | 1.534 | 1.4766 | 1.4872 | 82% |
| **s0** | 1.36M | 1.519 | **1.447** | **1.4599** | **82%** |
| s1 | 3.81M | 1.591 | 1.540 | 1.5473 | 85% |
| s2 | 8.12M | 1.848 | 1.819 | 1.8248 | 79% |

**@1e17 (BPB; compute-optimal = s2):**
| shape | N_active | dense | MoE (1sh) | **temporal** | recovery |
|---|---|---|---|---|---|
| s1 | 3.81M | 1.361 | 1.2803 | 1.3027 | 72% |
| **s2** | 8.12M | 1.341 | **1.269** | **1.2821** | **82%** |
| s3 | 14.77M | 1.408 | 1.289 | 1.3061 | 86% |

Findings: (1) temporal sits **inside the dense↔MoE band at every shape**, tracking ~0.006–0.022 BPB
above the MoE and far below dense. (2) **Same compute-optimal shape and parabola as MoE/dense**
(s0@1e16, s2@1e17) — temporal preserves the scaling geometry. (3) Eviction: min_logit ≳ lru, small
(≤0.006 BPB), consistent at 1e16. (4) Shared-expert knob negligible/inconsistent at B=1 (2-shared
K=5 spot-checks: s0@1e16 1.4569 — 2-shared better by 0.003; s2@1e17 1.2903 — it flips, 1-shared
better by 0.008; both within seed noise; matches §4's s-knob finding). Repro: `run.sh TEMPORAL=1 TEMPORAL_EVICT=min_logit`, driven by `temporal_matrix.sh` /
`temporal_minlogit_1e17.sh`.

## 6. Fine-graining (G=3: 192 experts, top-18) @1e16/1e17 — quality-neutral for MoE, small temporal penalty

`GRAIN=3`: routed experts 64→192, top-k 6→18, shared expert unchanged (BPB divisor 2.7600, mb=64;
mb≥128 OOMs; `--moe-permute-fusion` tested and rejected — 2–2.5× slower at these sizes). All 12
runs complete (6 MoE + 6 temporal, split H100/A6000).

**Both G3 MoE parabolas — min unchanged vs G=1:**
| budget | left | **min** | right | vs G=1 baseline min |
|---|---|---|---|---|
| 1e16 | sm1 1.4786 | **s0 1.4585** | s1 1.5352 | G1 s0 1.447 (+0.012) |
| 1e17 | s1 1.2846 | **s2 1.2708** | s3 1.2815 | G1 s2 1.269 (+0.002) |

→ Fine-graining 6→18 is **quality-neutral for the full MoE** at these budgets and preserves the
scaling geometry (same compute-optimal shape per budget).

**Temporal (G3) vs the dense↔MoE band** (dense floor = G=1 dense; active params identical):
| budget | shape | N_active | dense (G1) | MoE (G3) | **temporal (G3)** | recovery | vs G1 temporal |
|---|---|---|---|---|---|---|---|
| 1e16 | sm1 | 0.81M | 1.534 | 1.4786 | **1.4976** | 66% | 1.4872 |
| 1e16 | **s0** | 1.42M | 1.519 | 1.4585 | **1.4753** | 72% | 1.4599 |
| 1e16 | s1 | 3.91M | 1.591 | 1.5352 | **1.5861** | 9% | 1.5473 |
| 1e17 | s1 | 3.91M | 1.361 | 1.2846 | **1.3065** | 71% | 1.3027 |
| 1e17 | **s2** | 8.23M | 1.341 | 1.2708 | **1.2873** | 77% | 1.2821 |
| 1e17 | s3 | 15.09M | 1.408 | 1.2815 | **1.3129** | 75% | 1.3061 |

At the compute-optimal shapes temporal lands solidly inside the band (**72% @1e16-s0, 77%
@1e17-s2**, costing only ~+0.017 BPB over the full MoE) — below G=1's ~82% at the same shapes: a consistent **~5–10-pt hint that finer experts
(18/192) recover marginally less under rolling residency** than coarse (6/64). The off-optimal
right-arm s1@1e16 (9%) is noisy — the dense→MoE gap there is tiny (0.056) and the model is
oversized/undertrained for 1e16. Net: the headline holds at the shapes that matter.

## 7. 1e18 (paper scale, FLAME-MoE-38M-100M replica) — fine-graining hurts MoE; temporal is robust

Paper's smallest compute-optimal model replicated exactly (hidden 256 / 9 layers, gb 1024, WSD,
2121 iters = 4.45B tokens, **pythia-50k**, dclm; metric = validation CE in nats, lower better).
Two measurement panels — note they use **different val splits**, so compare within a panel:

**(a) A6000 panel** (dense + temporal-G1 trained identically; only the router differs):
| config | val-CE @1e18 |
|---|---|
| dense floor (ffn 1422, matched active non-embed params) | **4.137** |
| **temporal** (rolling residency, 6/64 resident, min_logit) | **3.906** |

Temporal beats dense by **0.231 nats** — clean, fully measured. Against the paper's law-predicted
MoE (≈3.78, *their* val set — extrapolation + val-split confounds) that is ~65% recovery
(0.231/0.357); the measured panel below is the sharper comparison.

**(b) Local self-consistent panel** (freshly-tokenized 50k dclm, one shared val split;
`flame38m_run.sh`; dense not re-run locally — the A6000 4.137 cross-checks within ~0.01 nats):
| config | val-CE @1e18 |
|---|---|
| MoE (G1, 6/64, full) | **3.9209** — best MoE |
| temporal (G3, 18/192) | **3.9768** |
| MoE (G3, 18/192, full) | **4.0087** |

Two clean findings:

1. **Fine-graining HURTS the full MoE at 1e18** — coarse MoE-G1 (3.921) beats fine MoE-G3 (4.009)
   by 0.088 nats: 192 experts is over-fine for 4.45B tokens (experts undertrained). Opposite of
   1e16/1e17, where the coarse MoE is *also* token-starved, hiding the effect.
2. **Temporal is robust to over-fine-graining** — temporal-G3 (3.977) beats its own-granularity
   full MoE (4.009) by 0.032 (train loss agrees: 3.975 vs 4.011): rolling residency concentrates each span on a
   small resident set, so those experts see more tokens and train better than the full MoE's
   thinly-spread 192. Against the true MoE quality (G1 3.921) and the dense floor (4.137),
   temporal-G3 recovers **~74%** — consistent with the ~66–77% at 1e16/1e17.

Perf: ~7–13s/iter locally (G1 ~7s, G3 ~13s) — the un-fused 50k-vocab LM head is the fixed cost at
this 12M-hidden-dim scale. Plot: `../phase0/figures/temporal_vs_dense_and_moe_1e18_crossentropy.png`.

**Superseded/extended:** these single-shape 1e18 points were later expanded into the full 1e18
**3-point isoFLOP program** (h192/h256/h512 × {dense, MoE, temporal} × {G1, G3}, all 15 cells, one
shared corpus/split per flank) — see `flame192_leftflank_1e18.csv`, `flame512_1e18_rightflank.csv`,
the 38M-middle training curves in `t18_1e18_curves.csv`, and the 1e19 extension in
`t19_1e19_curves.csv`.
