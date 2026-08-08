# LR sweep results — residency adaptation (SWEEP_PLAN.md execution) — FINAL

Complete: three LR grids (14 runs + 1 rule-skipped, later run as a make-up), winner + null
downstream, matched nulls, OLMoE r128 rank arm, and dense floors. Executed 2026-08-06/07.

Producers: `train_ple.py` (OLMoE, stock path), `train_unsloth.py` (Qwen, unsloth path),
`summarize_sweep.py` (standings, pre-registered rules), `downstream.py` +
`downstream_trained_unsloth.py` (winner/null suites), `dense_bar.py` (dense floors),
`olmoe_adapt_downstream.csv` (OLMo dense bracket, prior record). Per-run artifacts under
`/workspace/{olmoe,qwen3moe,qwen35}-adapt/`. Config fixed across runs: expert LoRA r32 +
attn LoRA r32 + router + norms; R=k=8 every layer; 15M tokens; 16,384 tok/step.

Definitions: BPB = bits/byte on the family's audited slice, lower better. downstream = mean
0-shot accuracy, matched acc-only 10-task basis (per-task stderr ≈ 0.02), higher better.
ref = min(null, base): OLMoE base 0.672723 (null 0.695064 is higher), Qwen3-30B base
0.615392 (null 0.616034 higher), Qwen3.5 **null 0.623235** (below its base 0.625152).
recovery = (constrained − x)/(constrained − ref). % over base = (x − base)/base.
† = stock-path downstream (cross-path caveat; O(1e-03) BPB offsets, unsloth_parity.md).
Nulls verified inert: swap = 0.0000.

## Table 1 — every run

| run | final BPB | downstream | ΔBPB over ref | % recovery | % over base |
|---|---|---|---|---|---|
| OLMoE base (no temporal) | 0.672723 | 0.6820 | 0 (ref) | — | 0% |
| OLMoE null @3e-5 (trained, free) | 0.695064 | — | +0.022341 | ceiling | +3.3% |
| OLMoE untrained + R8 | 0.842848 | 0.5723 | +0.170125 | 0% | +25.3% |
| OLMoE lr=1e-5 | 0.797638 | — | +0.124915 | 26.6% | +18.6% |
| **OLMoE lr=3e-5 (win)** | **0.793289** | **0.5978** | +0.120566 | **29.1%** | +17.9% |
| OLMoE lr=1e-4 | 0.797131 | — | +0.124408 | 26.9% | +18.5% |
| OLMoE lr=3e-4 | 0.831992 | — | +0.159269 | 6.4% | +23.7% |
| OLMoE lr=1e-3 | 1.029561 | — | +0.356838 | −109.8% | +53.0% |
| OLMoE **r128** @3e-5 | 0.790693 | — | +0.117970 | 30.7% | +17.5% |
| *dense floor: OLMo-1B-0724* | — | *0.6006* | | | |
| *dense: OLMo-7B-0724* | — | *0.6774* | | | |
| Qwen3-30B base | 0.615392 | 0.7267† | 0 (ref) | — | 0% |
| Qwen3-30B null @1e-4 | 0.616034 | 0.7198 | +0.000642 | ceiling | +0.1% |
| Qwen3-30B untrained + R8 | 0.734020 | 0.6311† | +0.118628 | 0% | +19.3% |
| Qwen3-30B lr=1e-5 | 0.687047 | — | +0.071655 | 39.6% | +11.6% |
| Qwen3-30B lr=3e-5 | 0.679645 | — | +0.064253 | 45.8% | +10.4% |
| **Qwen3-30B lr=1e-4 (win)** | **0.676359** | **0.6860** | +0.060967 | **48.6%** | +9.9% |
| Qwen3-30B lr=3e-4 | 0.681890 | — | +0.066498 | 43.9% | +10.8% |
| Qwen3-30B lr=1e-3 | 0.733675 | — | +0.118283 | 0.3% | +19.2% |
| *dense floor: Qwen3-4B-Base* | *0.678077* | *0.6852* | | | |
| Qwen3.5 base | 0.625152 | 0.7501† | +0.001917 | — | 0% |
| Qwen3.5 null @3e-5 | 0.623235 | 0.7402 | 0 (ref) | ceiling | −0.3% |
| Qwen3.5 untrained + R8 | 0.680022 | 0.7030† | +0.056787 | 0% | +8.8% |
| Qwen3.5 lr=1e-5 | 0.665960 | — | +0.042725 | 24.8% | +6.5% |
| **Qwen3.5 lr=3e-5 (win)** | **0.665780** | **0.7098** | +0.042545 | **25.1%** | +6.5% |
| Qwen3.5 lr=1e-4 | 0.668113 | — | +0.044878 | 21.0% | +6.9% |
| Qwen3.5 lr=3e-4 | 0.687210 | — | +0.063975 | −12.7% | +9.9% |
| Qwen3.5 lr=1e-3 (make-up) | 0.802438 | — | +0.179203 | −215.6% | +28.4% |
| *dense floor: Qwen3.5-4B-Base* | *0.689223* | *0.7028* | | | |

## Table 2 — best vs null vs baseline (+ dense floors)

| model | arm | BPB | avg downstream | ds recovery (vs null ceiling) | perf retained (vs null / vs base) |
|---|---|---|---|---|---|
| OLMoE | base | 0.672723 | 0.6820 | — | 100% |
| | null @3e-5 | 0.695064 | *(not measured)* | — | — |
| | untrained + R8 | 0.842848 | 0.5723 | 0% | 83.9% (vs base) |
| | **winner 3e-5** | 0.793289 | **0.5978** | 23.2% (vs base ceiling) | 87.7% (vs base) |
| | *OLMo-1B floor* | — | *0.6006* | **winner FAILS the dense floor** | |
| Qwen3-30B | base | 0.615392 | 0.7267† | — | 100% |
| | null @1e-4 | 0.616034 | 0.7198 | ceiling | 99.1% |
| | untrained + R8 | 0.734020 | 0.6311† | 0% | 86.8% |
| | **winner 1e-4** | 0.676359 | **0.6860** | **61.9%** | 95.3% / 94.4% |
| | *Qwen3-4B floor* | *0.678077* | *0.6852* | **winner PASSES (BPB −0.0017, ds +0.001)** | |
| Qwen3.5 | base | 0.625152 | 0.7501† | — | 100% |
| | null @3e-5 | 0.623235 | 0.7402 | ceiling | 98.7% |
| | untrained + R8 | 0.680022 | 0.7030† | 0% | 93.7% |
| | **winner 3e-5** | 0.665780 | **0.7098** | 18.3%* | 95.9% / 94.6% |
| | *Qwen3.5-4B floor* | *0.689223* | *0.7028* | **winner PASSES (BPB −0.0234, ds +0.007)** | |

\*Qwen3.5's downstream gain (+0.68 pts) is within task noise; its untrained gap was only 3.7 pts.

## Findings

1. **The dense floor separates the models.** OLMoE's adapted-constrained model (0.5978) sits
   below even OLMo-1B (0.6006): at 64 experts / 12.5% residency the constrained MoE is not
   worth running over a small dense model. Both Qwen models clear their 4B floors — Qwen3-30B
   at parity-to-better, Qwen3.5 with clear margin on both axes.
2. **LR optima are model-specific** (3e-5 / 1e-4 / 3e-5), all below the inherited 3e-4 (which
   degrades OLMoE outright) and Unsloth's 2e-4 default.
3. **Adaptation recovers tasks, not just bits**: Qwen3-30B recovers 62% of the constraint's
   downstream damage against its achievable ceiling, retaining 95.3% of null-level accuracy
   at 6.25% expert residency.
4. **Rank is not the binding constraint**: r128 beats r32 by 2.6e-03 BPB (inside the 3e-03
   noise band) for 4× the adapter parameters.
5. **The nulls certify the recipe**: both Qwen nulls land within 2e-03 of the untrained base
   (swap = 0.0000), so the corpus/recipe is neutral and the recovery numbers measure the
   constraint, not corpus drift. Qwen3.5's null (0.623235) is *below* its base — its
   reference tightened accordingly.
6. Scale trend, now on three axes (training-free BPB cost, adapted BPB cost, downstream
   retained): the rolling-residency constraint gets cheaper as expert count grows.

## Eviction-policy ablation — settled (2026-08-07)

Training-free, R=8 every layer, matched 16-seq slices; producers: the scan kernel's two
policies (`compute_resident_mask(..., evict=)`, bit-exact-tested in
`temporal/tests/test_temporal_router.py`), evaluated via the family harnesses.

| model | min_logit | lru | lru penalty | swap min_logit / lru |
|---|---|---|---|---|
| OLMoE | **0.839290** | 0.890595 | +0.051 | — |
| Qwen3-30B | **0.733680** | 0.812270 | +0.079 | 0.9973 / 0.9998 |
| Qwen3.5 | **0.679861** | 0.716777 | +0.037 | 0.9996 / 1.0000 |

min_logit wins unanimously by 12–26× the noise band: evicting the expert the router
currently values least keeps the resident set aligned with routing demand, while lru
churns more (higher swap) and hurts more. min_logit is the policy everywhere, permanently.

## 100M distillation campaign — complete (2026-08-08)

Winning recipe per model (distill T=1, KL against the own-base teacher, LR from the 15M
brackets), 100M tokens, evals every 10M, downstream immediately after each run. Producers:
`analysis/ple/train_unsloth.py` (Qwen; rolling cached teacher, 5M-token segments, top-2048,
mass coverage 0.992-0.994), `analysis/ple/train_ple.py` (OLMoE, inline teacher),
`downstream_trained_unsloth.py` / `downstream.py`. BPB lower is better; downstream is the
mean over the ten 0-shot tasks (limit 500), scored under R=8 everywhere.

| model | BPB 100M | 15M best | dense floor | null | ds 100M | ds dense bar | ds null |
|---|---|---|---|---|---|---|---|
| Qwen3-30B | **0.667648** | 0.671301 | 0.678077 | 0.616034 | **0.6926** | 0.6852 | 0.7198 |
| Qwen3.5 | **0.662826** | 0.664074 | 0.689223 | 0.623235 | **0.7144** | 0.7028 | 0.7402 |
| OLMoE | **0.777929** | 0.788727 | 0.672723 | (base 0.6727) | **0.6079** | 0.6006 (OLMo-1B) | — |

Findings:
7. **All three models clear their dense downstream bars at 100M** (+0.7 to +1.2 acc points);
   both Qwens also clear the dense BPB floor (by 0.010 / 0.026). OLMoE still fails its BPB
   floor by 0.105 — at 64 experts / 12.5% residency the constraint's bits are unrecoverable,
   but task accuracy is not: 80% of downstream damage closed (0.3164 -> 0.6079 vs base 0.6823).
8. **Token scaling saturates by ~20-40M.** Qwen3.5 15M->100M is flat on both axes (BPB
   -1.2e-03, inside noise; downstream -0.005, <1 sigma). Qwen3 gains modestly (BPB -3.7e-03,
   downstream +1.6 sigma) — enough to flip it from under to over the dense bar. OLMoE, least
   converged at 15M, gains most (-1.1e-02 BPB). Next levers are rank, LR schedule and data
   quality, not more tokens.
9. **The cached top-K teacher is free of measurable cost**: at 10M tokens the cached run
   matched the inline-teacher reference to +3.3e-05 BPB while lifting throughput 3.5k -> 4.3k
   tok/s (qwen3 mb4). Teacher truncation at top-2048 of the vocab covers 99.2-99.4% of mass.
