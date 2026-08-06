# LR sweep results — residency adaptation (SWEEP_PLAN.md execution)

**Living document**: updated as the endgame lands (winner downstream → nulls → final tables).
This edition: all grids complete; OLMoE winner downstream running; nulls pending.

Producers: `analysis/ple/train_ple.py` (OLMoE, stock path) and `analysis/ple/train_unsloth.py`
(Qwen, unsloth path) for the runs; `analysis/ple/summarize_sweep.py` on the run logs for
standings and verdicts (pre-registered rules); reference rows from
`olmoe_downstream_naive_preserve.csv`, `qwen3_30b_downstream_naive.csv`,
`qwen35_downstream_naive.csv` and `unsloth_parity.md`. Per-run JSONs/adapters under
`/workspace/{olmoe,qwen3moe,qwen35}-adapt/`. Executed 2026-08-06.

Config, fixed across all runs: expert LoRA r32 + attn LoRA r32 + router gates + RMSNorm
gains; R = k = 8 on every MoE layer; 15M tokens, evals 5/10/15M; 16,384 tok/step matched.
Qwen: AdamW8bit + cut_cross_entropy, aux from shipped config. OLMoE: fp32-master AdamW.
Qwen3.5's 1e-3 arm was skipped by user rule (its 3e-4 finished above 1e-4).

BPB = bits per byte on each model's held-out audited slice, lower better. downstream = mean
over the ten-task 0-shot suite's 17 metric rows, higher better. ΔBPB = final − min(null,
untrained-free base); nulls pending, so the reference is currently the base alone.
recovery = (constrained − trained)/(constrained − base): share of the constraint's damage
removed. % over base = (final − base)/base. † = stock-path downstream measurement
(unsloth-path re-runs scheduled; cross-path BPB offsets are O(1e-03) under the constraint —
see unsloth_parity.md — so † cells are perspective, not precision).

## Table 1 — every run, with reference rows

| run | final BPB | downstream | ΔBPB over min(null*, base) | % recovery | % over base |
|---|---|---|---|---|---|
| **OLMoE base (no temporal)** | 0.672723 | **0.6883** | 0 (reference) | 100% (ceiling) | 0% |
| **OLMoE untrained + temporal R8** | 0.842848 | **0.5993** | +0.170125 | 0% (floor) | +25.3% |
| OLMoE lr=1e-5 | 0.797638 | — | +0.124915 | 26.6% | +18.6% |
| **OLMoE lr=3e-5 (win)** | 0.793289 | *running* | +0.120566 | **29.1%** | +17.9% |
| OLMoE lr=1e-4 | 0.797131 | — | +0.124408 | 26.9% | +18.5% |
| OLMoE lr=3e-4 | 0.831992 | — | +0.159269 | 6.4% | +23.7% |
| OLMoE lr=1e-3 | 1.029561 | — | +0.356838 | −109.8% | +53.0% |
| **Qwen3-30B base (no temporal)** | 0.615392 | **0.7203**† | 0 (reference) | 100% | 0% |
| **Qwen3-30B untrained + temporal R8** | 0.734020 | **0.6438**† | +0.118628 | 0% | +19.3% |
| Qwen3-30B lr=1e-5 | 0.687047 | — | +0.071655 | 39.6% | +11.6% |
| Qwen3-30B lr=3e-5 | 0.679645 | — | +0.064253 | 45.8% | +10.4% |
| **Qwen3-30B lr=1e-4 (win)** | 0.676359 | *queued* | +0.060967 | **48.6%** | +9.9% |
| Qwen3-30B lr=3e-4 | 0.681890 | — | +0.066498 | 43.9% | +10.8% |
| Qwen3-30B lr=1e-3 | 0.733675 | — | +0.118283 | 0.3% | +19.2% |
| **Qwen3.5 base (no temporal)** | 0.625152 | **0.7402**† | 0 (reference) | 100% | 0% |
| **Qwen3.5 untrained + temporal R8** | 0.680022 | **0.7036**† | +0.054870 | 0% | +8.8% |
| Qwen3.5 lr=1e-5 | 0.665960 | — | +0.040808 | 25.6% | +6.5% |
| **Qwen3.5 lr=3e-5 (win)** | 0.665780 | *queued* | +0.040628 | **26.0%** | +6.5% |
| Qwen3.5 lr=1e-4 | 0.668113 | — | +0.042961 | 21.7% | +6.9% |
| Qwen3.5 lr=3e-4 | 0.687210 | — | +0.062058 | −13.1% | +9.9% |
| Qwen3.5 lr=1e-3 | *skipped* | — | — | — | — |

Reading: the LR optimum is model-specific (3e-5 / 1e-4 / 3e-5), always below both the
inherited 3e-4 and Unsloth's 2e-4 default; the inherited 3e-4 actively degrades OLMoE. The
constraint's residual cost after adaptation falls with expert count (+17.9% / +9.9% / +6.5%
over base at the winners), extending the training-free scaling result. The untrained
constraint's downstream cost follows the same order: −8.9 / −7.7 / −3.7 points of average
accuracy.

## Table 2 — winners vs null vs baseline (PENDING)

Filled when winner downstream + nulls land: per model best run vs matched null vs baseline —
BPB increase, % recovery, avg raw performance, avg raw / avg base (performance retained).
