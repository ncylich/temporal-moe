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

## Full-split GSM8K settles it: the adapter works (2026-08-26)

Scored on all 1,319 GSM8K test problems (`--tasks "gsm8k_cot_zeroshot=0"`), flexible-extract.

| record | free | R8 | R16 | R8-free | R16-free |
|---|---|---|---|---|---|
| gemma4_instruct (no adapter) | 87.8 | 78.8 | 86.6 | -9.0 (z=-9.26) | -1.2 (z=-2.26) |
| gemma4_ce_rebuild (D7 published recipe) | 86.8 | 81.9 | 87.5 | -4.9 (z=-5.81) | +0.7 (z=+1.24) |
| improvement vs base | | | | **+4.1 +/- 1.3, z=+3.17** | **+1.9 +/- 0.8, z=+2.46** |

The D7 reconstruction closes 45% of the R8 residency gap and eliminates the R16 gap
entirely (its R16 arm scores above its own free arm). Both clear |z| > 1.96.

The n=200 sample was not merely noisy, it was unrepresentative: it put the base R8 gap at
-6.0 where the full split puts it at -9.0. **Every conclusion drawn from the n=200 sweep is
void in both directions** -- the arms recorded above as "not resolved" were being compared
against a base estimate that was off by 3 points, so none of them is actually falsified
except the length family, whose internal comparisons (363 vs 650 tokens, 0.6M vs 3.4M
budget, all exactly -6.0 on identical questions) were self-consistent. Those arms need
re-measurement at n=1319 before any of them is ranked.

Practical rule going forward: residency-gap claims are made at n=1319, never at n=200.
A full-split 3-arm gemma cell costs ~25 min on one H200.

## The rebuild arm on all five benchmarks (2026-08-26)

`gemma4_ce_rebuild` = the faithful D7 published-recipe reconstruction, expert-LoRA r32 on
the 3D expert tensors + attention LoRA r32 + router/RMSNorm, 3.4M response tokens.
All cells are base-vs-adapted at MATCHED settings, paired by question, R8 = 6.25% resident.

| benchmark | n | base R8-free | rebuild R8-free | rebuild vs base | verdict |
|---|---|---|---|---|---|
| GSM8K | 1319 | -9.0 | -4.9 | +4.1 +/- 1.3 (z=3.17) | **REAL** |
| HumanEval | 164 | -5.5 | -5.5 | +0.0 +/- 2.9 | unresolvable at n=164 |
| IFEval | 541 | -1.8 | -2.6 | -0.7 +/- 1.6 | no change |
| MMLU | 228 | -0.4 | +1.8 | +2.2 +/- 1.8 | not resolved |
| WritingBench | 150 | -0.073 | -0.080 | -0.007 +/- 0.143 | no change |

Reading: **residency damage is concentrated on GSM8K** (-9.0) and HumanEval (-5.5). MMLU
(-0.4) and WritingBench (-0.073) are barely touched by the constraint, so there is almost
nothing there for adaptation to recover, and IFEval loses under 2 points. The adapter
closes 45% of the GSM8K gap and eliminates the R16 gap entirely (-1.2 -> +0.7, z=2.46),
while costing nothing on any other surface -- the "adaptation pays no fluency tax" claim
reproduces (WritingBench unchanged within +/-0.14).

Two cells cannot be strengthened by more data: HumanEval has only 164 problems total
(SE 2.9, so it resolves nothing below ~5.7 points) and WritingBench's 150 queries are
critic-scored. Report them as unresolved rather than as nulls.

Matched-settings notes, each of which had to be fixed before the comparison was valid:
- base IFEval had only ever run at n=200 while the adapted arm ran the full 541;
- base MMLU had only `mmlu_flan_cot_fewshot` while the adapted arm used the dual producer;
- gemma4 strict-match is identically 0.000 (channel markers) -- flexible-extract only.

## Code: MBPP shows the damage HumanEval was too small and too ceilinged to see (2026-08-26)

HumanEval put base R8 code damage at -5.5 with a detection floor of +/-5.7 -- unable to
resolve its own effect. Its free arm also sits at 98.2, near ceiling, which compresses how
much damage is even expressible. MBPP (500 problems, free arm 86.8) has room to move.

| surface | n | base R8-free | rebuild R8-free | rebuild vs base |
|---|---|---|---|---|
| HumanEval | 164 | -5.5 | -4.3 | +1.2 +/- 2.7 (z=0.45) |
| MBPP | 500 | **-14.6** | -13.4 | +1.2 +/- 2.6 (z=0.46) |
| pooled code | 664 | **-12.3** | -11.1 | +1.2 +/- 2.1 (z=0.58) |

Two findings, both important:

1. **Code is the most residency-damaged surface, not math.** Pooled code loses 12.3 points
   at R8 against GSM8K's 9.0. HumanEval understated it by ~7 points because its free arm is
   at ceiling. Any claim about where rolling residency hurts that rests on HumanEval alone
   is understating the code cost.

2. **The adapter does not fix code.** +1.2 +/- 2.1 pooled, z=0.58. This is now a real null,
   not an absence of measurement: at n=664 the surface resolves +/-4.1, so a math-sized
   recovery (+3.1) would have been near-detectable and the point estimate is well below it.
   Against the same adapter's +3.1 +/- 1.0 on GSM8K, the honest statement is that the
   recipe recovers math and leaves the worst-damaged surface essentially untouched.

D7 carries 431 code rows out of 8,482 (5.1%). Whether code damage is fixable by a
code-weighted mix is untested and is the obvious next experiment.

Correction recorded: the arm labelled `realmath` was NOT trained on StackMathQA. Its
adapter metadata reads traj=gemma4_d7_seq4096 and its chain log builds only the standard
D7 trajectory -- build_realmath_lane.py never ran. It is a second run of the D7 recipe, so
it is a run-to-run replicate (+3.3 vs rebuild's +4.1), not independent-data evidence.

## Swap rate: quality is flat over a 14x bandwidth reduction, then falls off a cliff (2026-08-26)

BASELINE_METHODS_COMPARISON.md #3 (cache-conditional experts, Skliar et al.,
arXiv:2412.00099), implemented on the serving path as a swap deadband: evict only when the
best non-resident logit beats the worst resident one by more than RHO. RHO=0 is the
published min_logit rule, verified bit-identical, so no existing row moves. Producer:
`TEMPORAL_RHO` in `temporal/temporal_router.py` (applied in both `_step` and
`_minlogit_step` so the CUDA-graph path and its eager reference stay equal).

gemma4, R8 (6.25% resident), full GSM8K test split, free arm 87.8:

| RHO | swaps/token | GSM8K R8 | paired vs RHO=0 |
|---|---|---|---|
| 0.00 | 0.99 | 78.8 | reference (published rule) |
| 0.25 | 0.79 | 79.4 | +0.6 +/- 1.1 |
| 0.50 | 0.42 | 80.0 | +1.2 +/- 1.0 |
| 1.00 | 0.07 | 78.3 | -0.5 +/- 1.1 |
| 2.00 | 0.00 | **71.7** | **-7.1 +/- 1.3 (z=-5.59)** |

**Quality is unchanged across a 14x reduction in swap traffic** (0.99 -> 0.07 swaps per
token; every point within +/-1.1 of the published rule), and then collapses by 7.1 points
when swaps reach exactly zero. So the swaps matter, but only a handful of them do: roughly
one swap per fourteen tokens preserves full quality, and none at all is catastrophic.

Consequences:

- The bandwidth cost of rolling residency as currently served is ~14x higher than the
  quality needs. This is a free serving win, available with no training and no memory cost,
  since R stays pinned at 8 throughout.
- Skliar's method is **complementary, not competitive**. It buys bandwidth, which is what
  their paper claims; it does not bound the resident set, which is the memory claim. Appendix
  E can now make that argument from a measurement on our own system rather than from their
  reported table.
- The zero-swap point is the honest control for "does the rolling constraint do any work at
  all". It does: freezing the resident set at whatever prefill left costs 7.1 points.

Two predictions of mine were wrong here and are recorded because the sequence matters:
quality would FALL as RHO rose (it did not, until the cliff), and the 0.25/0.50 rise was a
trend worth reporting (RHO=1.0 flattened it; it was noise). The curve only became
interpretable at five points.
