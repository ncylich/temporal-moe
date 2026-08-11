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
   but task accuracy fares better: 32% of downstream damage closed on the correct-convention
   basis (untrained R8 floor 0.5723 -> 0.6079 vs base 0.6820), edging the OLMo-1B bar. (An
   earlier draft said 80%, computed against the renorm-era floor 0.3164 — see the cross-era
   note below; that basis is invalid.)
8. **Token scaling saturates by ~20-40M.** Qwen3.5 15M->100M is flat on both axes (BPB
   -1.2e-03, inside noise; downstream -0.005, <1 sigma). Qwen3 gains modestly (BPB -3.7e-03,
   downstream +1.6 sigma) — enough to flip it from under to over the dense bar. OLMoE, least
   converged at 15M, gains most (-1.1e-02 BPB). Next levers are rank, LR schedule and data
   quality, not more tokens.
9. **The cached top-K teacher is free of measurable cost**: at 10M tokens the cached run
   matched the inline-teacher reference to +3.3e-05 BPB while lifting throughput 3.5k -> 4.3k
   tok/s (qwen3 mb4). Teacher truncation at top-2048 of the vocab covers 99.2-99.4% of mass.

## Cross-era reconciliation: why old OLMoE "recovery" numbers looked near-floor

The Stage-2b program (olmoe_adapt_RESULTS.md, 250M tokens/arm) reported 91-93% recovery,
which reads as "nearly at the free-routing floor". Those percentages were computed against
that era's impose reference: an untrained R=8 mask with per-sequence COLD-FILL and no
eviction policy, BPB 2.7507 — a denominator of +2.078 over base. In absolute terms its
best arms were: CE (router+norms+LoRA) 0.8149, full 7B finetune F' 0.8106, and that doc
called ~0.81 "the irreducible constraint price".

More fundamentally, the Stage-2 era used gate_mass=RENORM — expert weights renormalized
to sum to 1 over the selection — which on a norm_topk_prob=False model is the WRONG
convention: it raises top-k gate mass from ~0.40 to 1.0 and scales every MoE block output
~2.5x over 16 layers (olmoe_gatemass_remeasure.csv, which measures the same untrained R=8
cell at 2.6717 renorm vs 0.8393 preserve). Every Stage-2 number — impose 2.7507, the
1.28 router arms, CE 0.8149, full-finetune 0.8106 — trained AND evaluated that different
intervention. Therefore NO cross-era BPB comparison is valid, absolute or percentage.
The renorm-era files (olmoe_adapt_*, olmoe_cal*, olmoe_scratch_ladder, minflow captures,
adapt_ckpts, the old plan doc) are quarantined in results/archive/olmoe_wrong_renorm/ —
its README states the error and forbids any use of those numbers. The correct-convention
record is entirely current-era: untrained impose 0.839 -> best adapted (distill, 100M)
0.7779, never near the 0.6727 free base.

Housekeeping (08-08): before archiving, the fuller historical CSVs (olmoe_adapt_bakeoff
per-arm curves, olmoe_adapt_impose wikitext derivation) and the two router-parity
checkpoints were recovered from FLAME-MoE, which remains archived in its original state.

## Granularity program — complete (2026-08-10)

Seven models, five labs, training-free: base checkpoints on the byte-identical audited
slice (BPB) and ten 0-shot tasks (downstream); instruct-only gpt-oss on downstream only.
Producers: granularity_ladder.py, frontier_*.py, downstream_ladder.py,
gptoss_downstream.py; figures from plot_scaling.py (figures/damage_law.png,
figures/downstream_scaling.png). Cells in the named CSVs; % = degradation over free.

| % resident | LFM(32) | OLMoE(64) | gpt-oss-20b(32) | qwen3(128) | gemma4(128+sh) | gpt-oss-120b(128) | q35(256+sh) |
|---|---|---|---|---|---|---|---|
| 25 (BPB / ds) | 4.2/2.7 | 14.7/12.4 | -/2.9 | 4.1/4.0 | 1.6/0.4 | -/3.1 | 3.5/2.7 |
| 12.5 | 7.8/6.1 | 25.1/16.1 | -/6.2 | 9.2/7.3 | 4.1/1.8 | -/5.0 | 4.7/3.4 |
| 6.25 | floor | floor | floor | 19.3/13.2 | 8.4/3.5 | -/4.8 | 6.2/4.7 |

10. **Within-model law**: degradation = C·(k/R)^0.81, fixed-effects R² 0.91 over 22 BPB
    rungs; C ranges 6.7% (gemma4) to 25.1% (OLMoE) at R=k, ordered by shared expert
    (~2x cushion, gemma4-vs-qwen3 controlled pair) and router lexicality (OLMoE).
11. **At fixed memory fraction, sparser models pay less** (direction robust to dropping
    any model; slope magnitude uncertain ~2x with 3 x-positions). The direct granularity
    penalty is ~zero: the benefit is mediated entirely by slots-per-active-expert.
12. **Correct-convention re-runs**: free-set {14,15} is the best OLMoE cell (BPB 0.7600,
    ds 0.6119 at 15M); PLE dead: zero-init 0.8104, calibrated 0.8061 (preserve-recaptured
    table; the 0.8182 arm had loaded a renorm-era table) vs LoRA 0.7887.
Protocol notes: MXFP4 (gpt-oss) accs shift ~2pts/task across batch shapes (kernel
numerics, measured at logit level) — every delta uses same-bs arms; downstream cells
are the batched pad-warmed protocol of all prior tables.

13. **Generative benchmarks, four instruct models** (`instruct_genbench*.csv`; batch-fair,
    chat template, greedy, prefill free, stateful rule on generated tokens — the first
    decode-regime measurement of the constraint): at R = 12.5% of E, OLMoE-Instruct loses
    across the board (gsm8k 0.69→0.41, humaneval 0.37→0.29); LFM2.5 and gemma4-IT pay only
    on their strongest generative skill (humaneval −0.13; gsm8k −0.01 at 12.5%, −0.09 at
    R=k); Qwen3.5-35B is within noise of free on three of four benchmarks even at R=k = 3%
    residency. Floor-censored cells (LFM mmlu, gemma4 humaneval) are extraction artifacts.
