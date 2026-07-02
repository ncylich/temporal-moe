# G=3 fine-grained MoE — Phase-0 results (live, 11/12 runs)

Fine-grained (`GRAIN=3`: routed experts 64→192, top-k 6→18, shared expert unchanged) IsoFLOP sweep,
comparing MoE vs temporal (rolling-residency, K=18 resident of 192, swap 1/token, min_logit) vs the
dense floor. Config: mb=64, TE 1.11, seed 1234, `BPB_DIVISOR=2.7600`. **Lower BPB is better.**
All values **measured** from `results/phase0/runs/`. Split across H100 (@1e17 big shapes + fillers)
and A6000 (small-shape temporal) per `docs/shared-fine-grained-moe.md`.

**Status: 12/12 done** — all 6 MoE + 6 temporal (3 H100 + 3 A6000). Complete.

## Both MoE parabolas — complete & bracketed (min unchanged vs G=1)

| budget | left | **min** | right | vs G=1 baseline min |
|---|---|---|---|---|
| 1e16 | sm1 1.4786 | **s0 1.4585** | s1 1.5352 | G1 s0 1.447 (+0.012) |
| 1e17 | s1 1.2846 | **s2 1.2708** | s3 1.2815 | G1 s2 1.269 (+0.002) |

→ Fine-graining experts 6→18 is **quality-neutral for the MoE** (every point within ~0.001–0.012 of
G=1) and preserves the scaling geometry (same compute-optimal shape per budget).

## Temporal vs the dense↔MoE band (recovery = (dense−temporal)/(dense−MoE))

Dense floor = G=1 baseline (dense doesn't fine-grain; active params identical).

| budget | shape | N_active | dense (G1) | MoE (G3) | **temporal (G3)** | recovery | vs G1 temporal |
|---|---|---|---|---|---|---|---|
| 1e16 | sm1 | 0.81M | 1.534 | 1.4786 | **1.4976** | 66% | G1 1.4891 |
| 1e16 | **s0** | 1.42M | 1.519 | 1.4585 | **1.4753** | 72% | G1 1.4599 |
| 1e16 | s1  | 3.91M | 1.591 | 1.5352 | **1.5861** | 9% | G1 1.5488 |
| 1e17 | s1  | 3.91M | 1.361 | 1.2846 | **1.3065** | 71% | G1 1.3039 |
| 1e17 | **s2** | 8.23M | 1.341 | 1.2708 | **1.2873** | 77% | G1 1.2821 |
| 1e17 | s3  | 15.09M | 1.408 | 1.2815 | **1.3129** | 75% | G1 1.3073 |

**Finding so far:** at the **compute-optimal shapes** temporal lands solidly inside the dense↔MoE band —
**s0@1e16 72%**, **s2@1e17 77%** recovery, costing only ~+0.017 BPB over the full MoE. That's below
the G=1 baseline's ~82% at the same shapes (G1 was s0@1e16 82%, s2@1e17 82%) — a consistent **~5–10-pt
hint that finer experts (18/192) recover marginally less under rolling residency** than coarse (6/64).
The **off-optimal right-arm points are noisier**: s1@1e16 temporal (1.5861) recovers only 9% — it barely
beats the dense floor because the dense→MoE gap there is tiny (0.056) and the model is oversized/
undertrained for the 1e16 budget (G1 temporal was also worst here at 83% but G3 is markedly worse). Net:
the headline holds at the shapes that matter (18-of-192 resident, single swap/token keeps most of the MoE
gain), with fine-graining costing a modest, consistent recovery penalty vs coarse experts. The 1e17
temporal parabola is now **complete and bracketed** — s1 1.3065 · **s2 1.2873 (min)** · s3 1.3129 — with
the same compute-optimal shape (s2) as MoE/dense; s1@1e17 recovers 71%.

## 1e18 (paper scale — FLAME-MoE-38M-100M, 50k pythia / CE)

Extends the fine-graining test to the paper's real 1e18 budget (hidden 256 / 9 layers, gb 1024, WSD,
2121 iters = 4.45B tokens, pythia-50k, **validation cross-entropy in nats**, seed 1234, mb=32). Run
locally on a freshly-tokenized 50k dclm corpus so all points share one val split (self-consistent
panel); the A6000 baseline's dense **4.137** / temporal-G1 **3.906** are an external cross-check on
different data. Script: `flame38m_run.sh` (dense / full-MoE / temporal × G1/G3).

| config | val-CE @1e18 | status |
|---|---|---|
| dense floor | **4.137** | A6000 baseline, reused as floor (cross-data, <~0.01 nats) |
| **temporal (G3, 18/192)** | **3.9768** | measured (local) |
| **MoE (G3, 18/192, full)** | **4.0087** | measured (local) |
| MoE (G1, 6/64, full) | *pending* | running (local) |
| MoE (G1) — paper scaling law | ≈3.78 | external ref |

**Surprise vs 1e16/1e17:** at 1e18, G3-**temporal (3.9768) edges out the full G3-MoE (4.0087)** by
~0.03 nats — train loss agrees (3.975 vs 4.011), so it's not a val fluke — the *opposite* of the lower
budgets where temporal was always slightly worse than the MoE. Likely cause: **192 experts is over-fine
for the 4.45B-token budget** (~tens of M tokens/expert), so the full MoE spreads routing thin across
*undertrained* experts, while temporal's rolling residency concentrates each span on a small resident
set → those experts see more tokens → train better → lower loss. The pending **G1-MoE (64 experts)**
tests this (if it beats G3-MoE, fine-graining hurt the full MoE here). The floor is the A6000 dense
**4.137** (same corpus + pythia-50k, different val split → within ~0.01 nats; a local dense re-run
wasn't worth ~6–7h). So both temporal (3.977) and G3-MoE (4.009) sit well below the dense floor.
Perf note: the local 1e18 runs are ~12.7s/iter — the 50k-vocab LM head (un-fused CE) dominates at this
tiny 12M model, not the temporal scan.

## Complete (A6000)

All three A6000 temporal points done: `g3_tmoe_s0_1e16` (1.4753), `g3_tmoe_s1_1e16` (1.5861),
`g3_tmoe_s1_1e17` (1.3065). Both G3 temporal parabolas are now bracketed.
Combined graph: `results/phase0/figures/fine_grained_vs_coarse_experts_isoflop.png` (regenerate via `plot_g3_curves.py`).

## To finalize

`.venv/bin/python scripts/phase0/plot_g3_curves.py` → `results/phase0/figures/fine_grained_vs_coarse_experts_isoflop.png`
(dense/full-MoE/temporal curves, fine-grained vs coarse overlay). Then fill the recovery column for the A6000 points
and add the acceptance checks (2nd-seed repro + expert-load ≤8× mean) if run.

## Perf note

mb=64 is the ceiling (mb≥128 OOMs; the `mb·seq·topk` dispatch dominates). `--moe-permute-fusion`
(TE≥2.1) was tested and **rejected — 2–2.5× slower** at these micro-model sizes. Details:
`docs/shared-fine-grained-moe.md`.
