# Instruct MoE under rolling residency: the benchmark grid, thinking, and adaptation

Week of 2026-08-11 to 2026-08-18. Three pieces of work: (a) a clean re-measurement of
four benchmarks across thinking modes and residency arms for five instruct MoE models,
(b) what thinking and generation length do under the constraint, (c) adaptation recipes
for gemma4-26B-IT and Qwen3.5-35B-A3B. Numbers below are accuracy percentage points;
negative = worse than the same model unconstrained. Data: `results/ablations/`
(`instruct_genbench_vllm.csv` authoritative, `think_ablation_summary.csv`,
`screening_genbench.csv`); adaptation detail in
[`gemma_adapt_RESULTS.md`](../../../results/ablations/gemma_adapt_RESULTS.md).

## Setup

- **Constraint**: rolling residency at serve time. R experts resident per layer, at most
  **1 swap per token**, min-logit eviction, prefill free, rule applies to generated tokens.
- **Arms**: `free` (no constraint), `R=k` (residency = the model's top-k; the tight arm),
  `R=12.5%` of the expert count.
- **Models**: gemma4-26B-IT (k8/E128), Qwen3.5-35B-A3B (k8/E256), gpt-oss-120b and -20b,
  LFM2.5-A1B.
- **Benchmarks**: GSM8K, IFEval, HumanEval, MMLU. 200-item (HumanEval full), single boot
  per model, both thinking modes where the model has them.
- **Measurement fixes this week**, both load-bearing:
  - **Proper lengths**: the truncate-and-retry ladder retired for single-pass-at-cap
    generation; caps sized per task (thinking IFEval needs 8192). The ladder had been
    worth **+2.6 points on average** (n=59 paired cells, `reroll_delta_record.md`), so
    all pre-cutover numbers were inflated.
  - **Proper parsing**: MMLU dual-scored from the same generations. The stock strict
    "The answer is (X)" filter measures few-shot format imitation, not knowledge;
    relaxed extraction is the reported metric. HumanEval scored channel-aware.

## a) The baseline grid

- **Headline: the constraint costs little unless the model thinks or the residency
  fraction is small.** Mean damage over the four benchmarks at the tight arm:

  | model | thinking off / low | thinking on / high |
  |---|---|---|
  | gemma4-26B-IT (R8 = 6.25%) | **−2.5** | −7.7 |
  | Qwen3.5-35B (R8 = 3.1%) | **−7.2** | −9.6 |
  | gpt-oss-120b (R4 = k) | +0.2 | −1.3 |
  | gpt-oss-20b (R4 = 12.5%) | −1.2 | −1.0 |
  | LFM2.5-A1B (R4) | (no off mode) | −8.5 |

- At **R=12.5%** the off-mode cost is near zero for every model (gemma +0.8,
  qwen −2.2, gpt-oss-120b −0.2 at medium).
- Residency **fraction, not k, sets the difficulty**: qwen's R8 is 8-of-256 (3.1%) and
  bleeds 7 points where gemma's R8 (6.25%) loses 2.5. Confirmed causally in (c): at
  matched 6.25% qwen behaves like gemma.
- Per-benchmark spread is wide (per-cell SE 2 to 4 points): IFEval and HumanEval carry
  the worst single cells (qwen thinking IFEval **−16.5**, HumanEval −14.6).
- Figures: [`think_tax.png`](../../../results/ablations/figures/think_tax.png)
  (the grid in one plot),
  [`instruct_bench_damage.png`](../../../results/ablations/figures/instruct_bench_damage.png).

## b) Thinking and length under the constraint

- **Thinking amplifies constraint damage** in the two dense-thinking models: gemma
  −2.5 to −7.7, qwen −7.2 to −9.6 when thinking turns on. gpt-oss barely moves
  (+0.2 to −1.3 across low/medium/high): effort-controlled thinking is cheaper to
  constrain than free-form thinking.
- **The constraint lengthens thinking, and the extra tokens buy nothing back.**
  Think-token ratio constrained/free at R=k: gemma **1.22x**, gpt-oss-120b high
  **1.24x**, gpt-oss-20b low 1.23x, LFM 1.23x, qwen 1.13x. Dose-dependent for
  gpt-oss (low 0.95, medium 1.11, high 1.24). Same models, thinking off:
  ratio 0.92 to 0.99, i.e. **no lengthening without a thinking channel**.
- Full generations follow the same pattern (gemma on: 1.33x at R=k, 1.16x at 12.5%).
- Reading: residency perturbs routing mid-chain, the model backtracks and re-derives,
  chains grow, accuracy still lands below free. Length is a symptom of damage here,
  not a compensation mechanism that works.
- Figures:
  [`think_length_shift.png`](../../../results/ablations/figures/think_length_shift.png)
  (constrained vs free think tokens, points above the diagonal),
  [`length_vs_damage.png`](../../../results/ablations/figures/length_vs_damage.png),
  [`length_story.png`](../../../results/ablations/figures/length_story.png)
  (per-item percentile shifts and flip accounting).

## c) Adaptation

### gemma4-26B-IT: solved to ~1 point (D12)

- **Recipe**: cross-entropy on the model's own think-off responses (9,173
  benchmark-free prompts), trained **with the constraint active on response tokens**
  (prefill free, per-row enforcement), expert-tensor LoRA r16 + attention LoRA r32,
  **KL-to-base anchor at weight 0.05** on free-routing top-50 logprobs, **3.4M response
  tokens**, lr 3e-5, micro-batch 2.
- **Result** (authoritative 200-item runs, MMLU multi-run means), damage vs
  unconstrained base at R8:

  | | GSM8K | IFEval | HumanEval | MMLU |
  |---|---|---|---|---|
  | base under R8 | −6.0 | 0.0 | −6.1 | −0.2 |
  | **D12 under R8** | **0.0** | −1.0 | −1.2 | −1.8 |

- What the ladder established on the way:
  - **Constraint-aware training is the active ingredient**: identical data with the
    constraint off during training gives nothing.
  - **KL weight is a dial**: 0 keeps constrained MMLU but leaves the free arm damaged;
    0.1 repairs the free arm and costs constrained MMLU; **0.05 is the operating
    point**.
  - **More tokens hurt**: 10M with the anchor collapses constrained GSM8K by 14
    points; without the anchor 10M is neutral. 3.4M stands.
  - **Benchmark lineage is a trap**: an Orca-Math lane (GSM8K-train-seeded) faked +8
    GSM8K via style matching; removed, and the pool rule is benchmark-free by
    construction plus 8-gram screens against all four test sets.
- Figure: [`d12_adapt_final.png`](../../../results/ablations/figures/d12_adapt_final.png).

### Qwen3.5-35B-A3B: recipe transfers, residual gap is the tight fraction

- **Porting cost was real**: 70GB of bf16 weights on an 80GB card forced documented
  accommodations (paged 8-bit Adam, chunked-LM-head CE/KL that never materialises
  full 248k-vocab logits, plain-HF stack after unsloth's batched constrained path
  drifted 4.9% where HF shows 0.0 to 0.3%, per-row KL forward). The chunked head cut
  the training watermark **81 to 72.8GB** and later enabled r16 adapters and 4.6k
  sequences.
- **Fraction-matched, the gemma result reproduces**: at R16 (6.25% resident) the
  adapted model reaches **base-free parity on GSM8K, HumanEval and MMLU**. R8-of-256
  (3.1%) is a structurally harder ask than anything gemma faced.
- **Per-model tuning matters and is not additive**: a 2x2 over {pool with 11%
  truncated rows, truncation-free pool} x {KL 0.05, 0.1} showed the KL-0.1 repair of
  IFEval and MMLU only materialises on the clean pool, while dropping long rows costs
  2 to 4 GSM8K points. A unification run (all lengths kept, zero truncation, r16,
  KL 0.1 = r5) then took the free arm to **GSM8K 0.0 / HumanEval +1.8 / MMLU +0.4**
  vs base-free.
- **Committed result**: r2 (clean pool, KL 0.1, worst cell −6.0) by the max-min
  criterion; r5 challenges it on every arm except R8, where its IFEval/MMLU deficits
  sit at the edge of single-run noise. A two-run confirmation of both is finishing;
  this line updates with the verdict.
- **Open**: constrained IFEval is the one metric no qwen config fixes (best −6.0 at
  R8); candidate lever is compliance-filtering the existing self-generated format
  lane (no new benchmark-styled prompts, per the lineage rule).
- Figure: [`qwen_d12r_adapt.png`](../../../results/ablations/figures/qwen_d12r_adapt.png).

## Where everything lives

- Grid and thinking: `results/ablations/instruct_genbench_vllm.csv` (authoritative),
  `think_ablation_summary.csv`, per-item dumps in `results/ablations/genbench_samples/`.
- Adaptation: `screening_genbench.csv` (relative screens; candidates compare only
  against same-batch base references), `gemma_adapt_RESULTS.md` (recipes, ladder,
  measurement discipline), adapters under `/workspace/olmoe-adapt/data/`.
- Producers are committed next to every cited number; figures regenerate from
  `analysis/residency/d12_final_figure.py` and `qwen_d12r_figure.py`.
