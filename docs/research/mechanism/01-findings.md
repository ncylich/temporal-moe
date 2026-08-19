# What rolling residency costs, and what fixes it

Rewritten from scratch 2026-08-19, replacing the deprecated 2026-08-14 document
(now `archive/01-findings-deprecated-20260814.md`; its router-probe era lives in
`archive/probe-results.md` and neighbors). Every number below comes from a
committed source with a committed producer: `results/ablations/
instruct_genbench_vllm.csv` (authoritative), `think_ablation_summary.csv`,
`reroll_delta_record.md`, the per-item dumps in `genbench_samples/`,
`gemma_adapt_RESULTS.md`, and `writingbench/{summary,cell_stats}.csv`.

## 0. Scope, and how to read the numbers

- **Rolling residency**: at decode time only R experts per layer are resident;
  the resident set changes by at most **one swap per generated token**,
  min-logit eviction. Prefill is unconstrained. A router request for a
  non-resident expert is served by the resident approximation of its mixture.
- **Arms**: `free` (unconstrained), `R=k` (residency equals the model's own
  top-k, the tightest servable cache), `R=12.5%` of the expert count.
- **Damage**: constrained accuracy minus free accuracy, percentage points,
  negative is worse. Cells are 200-item runs (HumanEval full); per-cell
  binomial SE is 2 to 4 points. Deltas quoted between arms of the same boot
  only (see section 5).
- **Models**: gemma4-26B-IT (k8/E128), Qwen3.5-35B-A3B (k8/E256),
  gpt-oss-120b and -20b (effort-controlled thinking), LFM2.5-A1B.
  **Benchmarks**: GSM8K, IFEval, HumanEval, MMLU (dual-scored; the relaxed
  extraction is the reported metric, the strict one measures few-shot format
  imitation).

## 1. What the constraint costs

Mean damage over the four benchmarks at the tight arm (`think_ablation_summary.csv`):

| model | think off / low | think on / high |
|---|---|---|
| gemma4-26B-IT (R8 = 6.25%) | **−2.5** | −7.7 |
| Qwen3.5-35B (R8 = 3.1%) | **−7.2** | −9.6 |
| gpt-oss-120b (R4 = k) | +0.2 | −1.3 |
| gpt-oss-20b (R4) | −1.2 | −1.0 |
| LFM2.5-A1B (R4) | no off mode | −8.5 |

- **Relaxing to 12.5% erases most of it**: gemma +0.8, qwen −2.2,
  gpt-oss-120b −0.2 (medium), thinking off.
- **Residency fraction, not k, sets the difficulty.** Qwen at R8 is 3.1%
  resident and bleeds 7 points where gemma's 6.25% loses 2.5. Causal check in
  section 4: at matched 6.25%, adapted qwen behaves like adapted gemma.
- Worst single cells sit in IFEval and HumanEval (qwen thinking IFEval −16.5).
- Figures: `results/ablations/figures/think_tax.png`,
  `instruct_bench_damage.png`.

## 2. Thinking under the constraint

- **Free-form thinking amplifies damage** (gemma −2.5 to −7.7, qwen −7.2 to
  −9.6 with thinking on); **effort-controlled thinking barely moves**
  (gpt-oss-120b +0.2 to −1.3 across low/medium/high).
- **The constraint lengthens thinking, and the extra tokens buy nothing
  back.** Think-token ratio constrained/free at R=k: gemma **1.22x**,
  gpt-oss-120b high **1.24x**, gpt-oss-20b low 1.23x, LFM 1.23x, qwen 1.13x.
  Thinking off: 0.92 to 0.99, no lengthening. Dose-dependent for gpt-oss
  (low 0.95, medium 1.11, high 1.24).
- Reading: residency perturbs routing mid-chain; the model backtracks and
  re-derives; chains grow; accuracy still lands below free. Length here is a
  symptom of damage, not a compensation that works.
- Figures: `think_length_shift.png`, `length_story.png`, `length_vs_damage.png`.

## 3. Fluency is the robust surface

WritingBench (official queries and critic model, run locally; 3 disjoint
50-query subsets per cell, paired deltas; `writingbench/cell_stats.csv`):

- Constraint cost at R=k, critic points out of 10: LFM **−0.31**, oss-20b
  −0.17, qwen −0.15, gpt-oss-120b −0.08, gemma −0.07. Small everywhere, and
  inversely related to model size. Compare the 6 to 12 point accuracy costs.
- Repetition (3-gram loops in MMLU generations) rises under the constraint for
  base qwen (3.5% to 5.7% of responses) and is removed by adaptation.
- Absolute writing quality: gpt-oss-120b 8.52, qwen 7.97, oss-20b 7.63,
  gemma 7.53, LFM 7.38.

## 4. Adaptation closes most of the gap

Full recipes, ladder, and caveats: [`gemma_adapt_RESULTS.md`](../../../results/ablations/gemma_adapt_RESULTS.md).

- **gemma4 (D12)**: CE on the model's own think-off responses (9,173
  benchmark-free prompts), **constraint active on response tokens during
  training**, expert-tensor LoRA r16 + attention r32, **KL-to-base anchor
  0.05**, 3.4M tokens. Damage vs unconstrained base at R8 goes from
  −6.0 / 0.0 / −6.1 / −0.2 (GSM8K/IFEval/HE/MMLU) to **0.0 / −1.0 / −1.2 /
  −1.8**. The R8 penalty is essentially erased on math and ~1 point elsewhere.
- **Load-bearing ladder facts**: constraint-aware training is the active
  ingredient (identical data with the constraint off gives nothing); the KL
  weight is a dial (0 leaves the free arm damaged, 0.1 repairs it but costs
  constrained MMLU, 0.05 is the operating point); 10M tokens with the anchor
  collapses constrained GSM8K by 14 points while 10M without it is neutral;
  benchmark-lineage data fakes gains by style matching (Orca-Math +8 GSM8K,
  removed) and pools are benchmark-free by construction plus 8-gram screened.
- **Qwen3.5 (r2)**: the recipe transfers with documented accommodations
  (r8 adapters, paged 8-bit Adam, HF stack, chunked-head CE/KL). At the
  fraction-matched R16 arm the adapted model reaches base-free parity on
  GSM8K/HE/MMLU; at R8-of-256 (3.1%, harder than anything gemma faced) the
  committed r2 config holds every cell within −6.0, and a pool x KL 2x2 showed
  the knobs interact (the KL-0.1 IFEval/MMLU repair needs the truncation-free
  pool; dropped long rows cost math). Constrained IFEval (−6.0) is the one
  unfixed qwen metric.
- **Adaptation is fluency-free** (section 3): D12 at-or-above base in every
  WritingBench cell; r2 within ±0.06.
- Figures: `d12_adapt_final.png`, `qwen_d12r_adapt.png`, `qwen_attrib_square.png`.

## 5. Measurement corrections that gate everything above

- **Truncate-and-retry inflated scores +2.6 points on average** (n=59 paired
  cells, `reroll_delta_record.md`). All numbers here are single-pass-at-cap,
  caps sized per task (2048 plain, 4096 thinking, 8192 thinking IFEval).
- **Strict MMLU parsing measures format imitation, not knowledge** (base
  knowledge ~0.92-0.95 across arms under relaxed extraction). MMLU is
  dual-scored from the same generations; relaxed is reported.
- **Thinking must be specified explicitly**; template defaults differ by model
  (early qwen references silently ran think-on).
- **Constrained arms are batch-composition sensitive** (8.6 points on
  constrained IFEval between 70-item and 200-item batches): deltas are valid
  only against same-batch references. Hard-item screening subsets amplify
  deltas (a +4 screening GSM8K compressed to +1 authoritative): screens are
  relative instruments and winners get full confirmation grids.
- **Free-arm MMLU at temperature 1.0 spreads up to 3.5 points across runs**
  (multi-run means used); seeded same-batch runs replay exactly, so repeats
  there test stability, not variance. Constrained decoding requires eager
  execution; CUDA graphs silently disable the constraint (engagement checks
  are mandatory on every new harness).

## 6. Open

- Qwen constrained IFEval (−6.0 at R8): compliance-filtering the existing
  self-generated format lane is the identified lever; no benchmark-styled
  prompts, per the lineage rule.
- Gemma free-arm MMLU (−2.8): untested KL bracket 0.03/0.07.
- Think-on adaptation: needs ≥6k generation caps (35.7% of think responses cap
  at 3072); the chunked-head trainer removed the memory blocker.
- Think-on evaluation of the adapted models: not yet measured; the section-2
  lengthening under adaptation is an open mechanistic test.
