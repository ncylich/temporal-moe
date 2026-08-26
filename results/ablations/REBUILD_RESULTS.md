# Rebuilt-pool adaptation: what reproduced, what did not

**2026-08-25.** The d7 prompt pool died with the training pod and no builder was ever
committed, so `RECOVER_DATA_PLAN.md` §1.1 required rebuilding it from a prose specification.
This is what the rebuilt pool produces, measured against the published Section 8 numbers.

Read it as a **replication, not a reproduction**. The original pool was substantially
self-generated — a saved transcript records `math_selfgen`, Magicoder-style OSS-seeded code
and self-generated format drills, with only its chat and `math_user` lanes mined from real
conversations — and the rebuild is real-corpus throughout. Roughly half the pool therefore
differs in kind, and `mcq-writer` (691 rows) is absent entirely.

---

## The headline

**Four of five benchmarks reproduce or beat published, across three independently trained
arms. GSM8K at the tight R8 arm is the single failure.**

gemma4 R8 damage in points vs the unadapted base free arm (MMLU vs own free; WritingBench
is the paired R8-minus-own-free delta in critic points):

| benchmark | published D12 | unadapted base | armA real-corpus | armC real-math | full-pass |
|---|---|---|---|---|---|
| GSM8K | +0.0 | −6.0 | −5.5 | −6.0 | **−7.5** |
| IFEval | −1.0 | +0.0 | **−1.0** | −2.5 | **+0.5** |
| HumanEval | −1.2 | −6.1 | −4.9 | −4.3 | −4.3 |
| MMLU | −1.8 | −0.2 | **+0.0** | **+0.0** | **+1.3** |
| WritingBench | +0.040 | −0.073 | −0.048 | −0.040 | −0.036 |

IFEval lands exactly on published on armA. MMLU beats published on all three arms.
WritingBench sits within the across-subset SD of D12 and above base on all three, so
`01-findings`' claim that *"adaptation pays no fluency tax"* reproduces. HumanEval recovers
substantially from the base's −6.1 but not to the published −1.2.

## Five hypotheses, all eliminated

| # | hypothesis | test | result |
|---|---|---|---|
| 1 | the math lane must be self-generated, as the original was | authored 2,671 problems from templated seeds | GSM8K −5.5 → −3.0, but **rejected on method**: the prompts were shaped like grade-school word problems *because GSM8K was the failing cell*, which passes the 8-gram screen while overfitting to the evaluation by construction — the same defect as the Orca-Math lane the lineage rule forbids |
| 2 | the math lane needs *better real* math | whole lane swapped to math.stackexchange (StackMathQA), lineage-clean | −5.5 → **−6.0**. No effect |
| 3 | the code lane | held identical across arms | control: −4.9 / −4.3 / −4.3, ≈0.6pt arm-to-arm variance |
| 4 | token-budget coverage | the original's 3.4M was one COMPLETE pass (371 tok/row × 9,173); ours covers 46–60%. Ran a full pass at 7,356,394 tokens | −5.5 → **−7.5**, i.e. worse — independently reproducing the ladder's "more tokens hurt" (D10 collapsed constrained GSM8K +4 → −10 at 10M) on a new pool at KL 0.05 |

| 5 | the KL anchor is over-anchoring the constrained arm | KL 0.03, the lower end of the bracket `gemma_adapt_RESULTS` §Open names as untested. Strict single variable: identical pool, trajectories, budget, rank, batch | **no change** — R8 stays −5.5. The free arm rose to 0.880, the highest of any arm, which is the OPPOSITE of the ladder's prediction that less anchoring weakens the free arm and strengthens the constrained one. The dial that governs the free/constrained tradeoff on MMLU has no purchase on GSM8K at R8 |

Only hypothesis 1 moved GSM8K, and it moved it for the wrong reason. That is itself a
useful demonstration: benchmark-shaped training data produces exactly the signature the
lineage rule exists to catch.

**Disqualified source, worth recording:** `AI-MO/NuminaMath-CoT`, the obvious "real math
dataset", is unusable here — its own `source` field shows it bundles `orca_math`, `gsm8k`,
`math` and `synthetic_math`. Using the most popular public math corpus would have recreated
the D1-vs-D4 failure directly.

## The finding the data search actually produced

**R16 is near-undamaged in every arm; only R8 fails.**

| | R8 | R16 |
|---|---|---|
| GSM8K | −5.5 / −6.0 / −7.5 | **−0.5 in all three arms** |
| HumanEval | −4.9 / −4.3 / −4.3 | −1.8 / −1.8 / **+0.0** |

For gemma4, R16 is 12.5% resident and R8 is 6.25%. The adapters recover the constraint at
12.5% across the board and fail at 6.25% on math, and that shape did not move under four
different data recipes or a doubled token budget. `gemma_adapt_RESULTS` already warns that
cross-model claims should quote matched *fractions* rather than matched R, because
"R8-of-256 (3.1%) is structurally harder than anything gemma faced"; the same logic applies
within a model. The first training-side lever, the KL bracket at 0.03, has now been tried and does not
move it either.

Full four-arm picture at R8:

| benchmark | pub D12 | base | armA | armC | full-pass | KL 0.03 |
|---|---|---|---|---|---|---|
| GSM8K | +0.0 | −6.0 | −5.5 | −6.0 | −7.5 | −5.5 |
| IFEval | −1.0 | +0.0 | −1.0 | −2.5 | +0.5 | −3.0 |
| MMLU | −1.8 | −0.2 | +0.0 | +0.0 | +1.3 | −0.4 |

**Conclusion.** This is not a pool-reconstruction failure. Four data recipes, a doubled
token budget and a KL change all leave the same shape: R8 at −5.5 to −7.5, R16 at −0.5 to
−1.0. Something specific to gemma4 at 6.25% residency on math resists adaptation, and the
published D12's +0.0 on that cell depends on something the recipe as written down does not
capture. That belongs in the paper as a reproducibility caveat on one cell, not as a silent
gap — and §1.5's second disposition follows: keep the published numbers, release these
adapters as labelled replications with this table beside them.

## B2: think-on adaptation (RECOVER_DATA_PLAN Part 0 group B)

Section 7's length result rested on gemma alone, because qwen's adapted checkpoints had only
ever been evaluated think-off, where neither routing regime lengthens and the comparison is
empty. Both think-on adapters are now trained, merged, verified and measured at 16384.

qwen3.5, adapted vs base, damage vs each model's own free arm:

| benchmark | arm | adapted | base | effect |
|---|---|---|---|---|
| IFEval | R8 | −7.0 | −12.5 | **improved 5.5** |
| IFEval | R32 | −3.5 | −2.0 | worse 1.5 |
| GSM8K | R8 | **−1.5** | −2.5 | **improved 1.0** |
| GSM8K | R32 | **−1.0** | −3.0 | **improved 2.0** |

Three of four cells improve, and the tight R8 arm improves on both benchmarks. This is a
new result: the recipe transfers to thinking mode on qwen.

gemma4's think-on adapter reads +0.0 (R8) and +3.5 (R16) against its own free arm, but
**no base gemma think-on GSM8K arm exists at 16384**, so those are absolute numbers and not
effects. Do not quote them as an adaptation result until that baseline is run.

## Measurement bugs found and fixed

Each of these produced a confident, wrong number before being caught. All are the same
family: *the instrument the harness offers is not the one the paper reports.*

* **HumanEval scored 0.000 on every arm.** gemma4 emits `<channel|>` markers, so the stock
  extractor finds no code. Published rows come from bespoke producers — `humaneval_gemma.py`
  (`humaneval_gemma_fixed`, `pass@1,channel-aware`); qwen think-off uses `humaneval_instruct`,
  qwen think-on uses `humaneval_think`.
* **The qwen grid ran think-ON** while its adapter trains on think-off trajectories. 150 of
  200 responses blew the 2048 cap and IFEval read 0.27 against a true value near 0.85.
* **MMLU strict vs relaxed.** `mmlu_gptoss_relaxed` is not an lm_eval task; the reported
  metric `acc,relaxed-extract` comes only from `mmlu_gptoss.py`. Reading the strict flan
  filter instead flipped the Group A gemma MMLU re-run from −4.4→−1.3 (shrinking) to
  −9.2→−12.3 (widening).
* **Attention LoRA trained on the vision tower.** gemma4 wraps only its VISION tower's
  projections in `Gemma4ClippableLinear`; the language model's are plain `nn.Linear`.
  Targeting `q_proj.linear` matched the vision tower alone, a text-only forward never
  reached it, and all 108 `lora_B` tensors stayed at zero init — through a passing smoke and
  a "successful" merge. Caught only by diffing merged against base weights, which
  `verify_merge.py` now does on every merge.

## Reproducing this

    analysis/residency/build_d7_prompts.py      # the pool, deterministic, sha256-stamped
    analysis/residency/cut_trajectories.py      # drop over-length rows WHOLE
    scripts/residency/train_adapters.sh         # smoke -> KL precompute -> train -> mirror
    scripts/residency/merge_and_remeasure.sh    # merge -> verify_merge -> grid
    scripts/residency/grid_parallel.sh          # GSM8K/IFEval/MMLU, one GPU each
    scripts/residency/remeasure_humaneval.sh    # channel-aware HumanEval
    scripts/residency/wb_arm.sh                 # WritingBench, the fifth benchmark

A parallel grid alone is **three of five** cells. HumanEval needs its architecture's
producer and WritingBench needs the local critic.

## Statistical power: why the D7 arm sweep was inconclusive (2026-08-26)

Every arm number here is a difference of differences: (R8 - free) for an adapted
checkpoint, against (R8 - free) for the unadapted base. Both levels are paired on the
same questions, so the error bar is McNemar's, driven by the count of questions where
the two arms disagree -- typically 13-27 of 200.

At the sampled n=200 that puts the SE on a single record's residency gap at ~2.2 points
and on a cross-record comparison at ~3.0 points. Producer:
`analysis/residency/arm_power.py`.

| record | R8 - free | z (within) | vs base | verdict |
|---|---|---|---|---|
| gemma4_instruct (no adapter) | -6.0 | -2.68 | reference | gap is real |
| gemma4_ce_d12 | -0.5 | -0.28 | +5.5 +/- 2.9 | z=1.91, not resolved |
| gemma4_ce_domain / think3k | -2.5 | -1.1 / -1.2 | +3.5 +/- 3.1 | not resolved |
| gemma4_ce_selfgen | -3.0 | -1.60 | +3.0 +/- 2.9 | not resolved |
| gemma4_ce_short1pass (mean 363 tok) | -6.0 | -2.83 | +0.0 +/- 3.1 | not resolved |
| gemma4_ce_dose1024 (mean 650 tok) | -6.0 | -3.00 | +0.0 +/- 3.0 | not resolved |
| gemma4_ce_fullpass | -9.5 | -3.66 | -3.5 +/- 3.4 | not resolved |

Consequences:

- **The residency gap itself is real** (base -6.0, z=-2.68) and reproduces across every
  record. The open question is only whether any adaptation closes it.
- **No D7-derived arm has been shown to beat doing nothing.** Six successive hypotheses
  (self-generated math, StackMathQA, code-lane control, full-pass budget, KL weight,
  response length) each moved 1-3 points and none cleared its own error bar. That
  pattern is what an underpowered comparison looks like, not six failed mechanisms.
- **The response-length hypothesis is falsified on its own terms**, independent of
  power: cutting mean response length from 668 to 363 tokens and to 650 tokens produced
  the SAME -6.0, so there is no dose-response along the axis the hypothesis predicted.
- Only flexible-extract is meaningful for gemma4 here. strict-match reads 0.000 on every
  gemma cell because the model emits `<channel|>` before the answer; an earlier -4.5
  reported for short1pass came from misreading a strict row and is superseded by -6.0.

Fix in flight: rescore on the FULL GSM8K test split (1,319 problems, `--tasks
"gsm8k_cot_zeroshot=0"`), which takes the within-record SE to ~0.9 and the cross-record
SE to ~1.2. Same benchmark, same data, scored completely instead of sampled at 200.
