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

## Swap rate: the swaps are necessary. min_logit sits near the efficient frontier (2026-08-26)

BASELINE_METHODS_COMPARISON.md #3 (cache-conditional experts, Skliar et al.,
arXiv:2412.00099), implemented on the serving path as a swap deadband: evict only when the
best non-resident logit beats the worst resident one by more than RHO. RHO=0 is the
published min_logit rule, verified bit-identical, so no existing row moves. Producer:
`TEMPORAL_RHO` in `temporal/temporal_router.py`; swap rates measured with
`TEMPORAL_COUNT_SWAPS=1` (`swap_stats()`), reported per cell by the eval driver.

gemma4, R8 (6.25% resident), full GSM8K test split, free arm 87.8. Swap rate is MEASURED on
the same generations as the accuracy, not simulated:

| RHO | swaps/token | reduction | GSM8K R8 | paired vs RHO=0 |
|---|---|---|---|---|
| 0.00 | 0.9987 | 1.00x | 78.8 | reference (published rule) |
| 0.25 | -- | -- | 79.4 | +0.6 +/- 1.1 |
| 0.50 | 0.9806 | 1.02x | 80.0 | +1.2 +/- 1.0 |
| 1.00 | 0.8985 | 1.11x | 78.3 | -0.5 +/- 1.1 |
| 1.25 | 0.8270 | 1.21x | 79.7 | +0.9 +/- 1.1 |
| 1.50 | 0.7405 | 1.35x | 77.9 | -0.8 +/- 1.1 |
| 1.75 | 0.6390 | 1.56x | 77.0 | -1.7 +/- 1.1 |
| 2.00 | 0.5281 | **1.89x** | **71.7** | **-7.1 +/- 1.3 (z=-5.59)** |

**About 36% of swap traffic is removable for free; removing 47% costs 7.1 points.** There is
no order-of-magnitude headroom -- there is barely a factor of two. Nearly every swap
min_logit performs is doing work, so the eviction rule operates close to the efficient
frontier and the mechanism cannot be cheapened much without breaking.

Quality is flat over RHO 0-1.75 on FIVE benchmarks and two models, base and adapted, so the
free region is general even though it is narrow:

| surface | n | RHO=0 | RHO=1.0 | paired |
|---|---|---|---|---|
| GSM8K | 1319 | 78.8 | 78.3 | -0.5 +/- 1.1 |
| MBPP | 500 | 72.2 | 72.2 | +0.0 +/- 2.3 |
| HumanEval | 164 | 92.7 | 93.3 | +0.6 +/- 2.2 |
| IFEval | 541 | 86.9 | 87.6 | +0.7 +/- 1.1 |
| MMLU | 228 | 92.5 | 94.3 | +1.8 +/- 1.4 |

Also: gemma R16 +0.0 +/- 0.5, qwen R8 (3.1% resident) -0.2 +/- 1.1, adapted gemma R8
-0.9 +/- 1.1 -- the deadband composes with the adapter at no cost.

**RHO does not transfer across models.** At RHO=1.0 gemma runs 0.8985 swaps/token and qwen
0.9468, because their router logit scales differ. Any deployment setting must be expressed
as a target swap rate and tuned per model, never as a shared RHO.

Consequence for Appendix E: Skliar's method is not refuted, it is simply operating where
there is little to win. Under a hard R=k bound the swaps are load-bearing, and their bonus
can buy ~36% before quality breaks. That is now a measurement on our own system across five
benchmarks rather than an argument from their reported table.

**Correction, recorded because the error was mine and it was large.** This section first
claimed a 14x (then 65x) free bandwidth reduction. That came from a SIMULATED swap rate
computed on a synthetic correlated routing signal whose logit gaps are far tighter than the
real router's, so a deadband that suppressed 93% of swaps in simulation suppresses 10% in
practice. The tell was visible before the measurement -- RHO=1.75 and RHO=2.0 both showed
0.000 simulated swaps while differing by 5.3 quality points, which is impossible. Never
divide a measured numerator by a simulated denominator: run with TEMPORAL_COUNT_SWAPS=1 so
both axes come from the same generations.

## The R x swap-rate frontier: memory and bandwidth substitute (2026-08-26)

Falls out of the RHO sweep once swap rates are measured rather than simulated. All cells
gemma4/qwen3.5 GSM8K, swap rate from `TEMPORAL_COUNT_SWAPS=1` on the same generations.

**Resident fraction sets where the cliff is.** At comparable swap rates the tighter arm is
destroyed while the looser one is untouched:

| arm | resident fraction | swaps/token | GSM8K | vs its own RHO=0 |
|---|---|---|---|---|
| gemma R16 | 12.5% | 0.575 | 86.0 | -0.6 |
| gemma R8 | 6.25% | 0.528 | 68.5 | -10.3 |
| qwen R8 | 3.1% | 0.673 | 48.5 | -28.1 |

Approximate rate at which quality breaks: 12.5% -> ~0.28/token, 6.25% -> ~0.53,
3.1% -> ~0.75. The ordering is by resident FRACTION, not model identity. The product
(fraction x critical rate) is ~0.035 / 0.033 / 0.023 -- suggestive of a conserved expert
throughput, but these are three points interpolated off a coarse RHO grid at n=200 and the
qwen point is the least constrained. Knee sweeps at finer resolution are the test; do not
state the invariant until they land.

**Adaptation does not move the cliff.** Base vs adapted at R8, same swap rates:

| RHO | swaps/token | base | adapted | delta |
|---|---|---|---|---|
| 0 | 0.999 | 78.8 | 81.9 | +3.1 |
| 2.0 | ~0.525 | 68.5 | 71.5 | +3.0 |
| 2.5 | ~0.295 | 51.5 | 51.0 | -0.5 |
| 3.0 | ~0.086 | 12.5 | 16.5 | +4.0 |

A flat vertical offset of roughly +3 at every swap rate, with the collapse in the same
place. The adapter learns to be more accurate under the constraint, not to need fewer swaps.

So there are three separable levers, where the paper currently plots one:
- resident fraction R/E: moves the cliff horizontally (memory buys bandwidth tolerance)
- adaptation: moves quality vertically (~+3, cliff unchanged)
- swap rate: the axis previously held fixed at ~1/token as if it were a property of the
  method rather than a tunable

Deployment consequence: RHO is NOT portable. At RHO=1.0 gemma runs 0.8985 swaps/token and
qwen 0.9468 because their router logit scales differ. Tune to a target swap RATE per model.

Note on the frontier's low-fraction end: gemma4 cannot be run below R8. Its top-k is 8, so
the resident set must hold at least the experts a token routes to (`instruct_genbench_vllm.py`
asserts R >= k). gemma's minimum resident fraction is therefore 6.25% and qwen's is 3.1%,
and the 3.1% cell cannot be reproduced on gemma. The fraction-vs-model confound is instead
tested at E/R=8, where gemma R16 (12.5% of 128) and qwen R32 (12.5% of 256) coincide.

## Memory-for-bandwidth substitution is model-specific, not a design rule (2026-08-26)

Critical swap rate = the measured swaps/token at which GSM8K falls 10 points below the
same arm at RHO=0. Rates measured with `TEMPORAL_COUNT_SWAPS=1` on the same generations as
the accuracy. Screening cells are n=200 (+/-3-4 points), adequate for locating a cliff.

| E/R | resident | gemma4 (E=128) | qwen3.5 (E=256) |
|---|---|---|---|
| 32 | 3.1% | unreachable (R >= k = 8) | 0.82 |
| 16 | 6.25% | 0.53 | 0.86 |
| 8 | 12.5% | 0.25 | 0.79 |
| 4 | 25% | 0.16 | 0.81 |
| 2 | 50% | 0.07 | 0.30 |

**gemma substitutes memory for bandwidth smoothly**: 0.53 -> 0.07 as resident memory goes
6.25% -> 50%, well fit by rate ~ 0.042*(E/R)^0.85 (five points, 16x range, +/-15%, and the
E/R=2 point was predicted before it was run). At 50% resident it needs 0.084 swaps/token --
a 12x bandwidth saving.

**qwen does not.** Its requirement is FLAT at ~0.8 swaps/token from E/R=32 down to E/R=4 --
eight-fold more resident memory buys nothing -- and drops only at 50% resident. It is a step,
not a curve.

So there is no transferable law. At matched E/R=8 gemma needs 0.25 and qwen 0.79, 3x apart;
at E/R=2, 0.07 vs 0.30. Any deployment must measure its own model's curve. That negative is
the durable result here, and it is worth more than the law would have been, because it tells
a practitioner to measure rather than to apply a constant.

**Three false starts, recorded because the sequence is the lesson.** (1) "Quality falls as
RHO rises" -- it does not, until the cliff. (2) "fraction x critical rate is conserved
(~0.03)" -- the product drifts 0.026-0.040 and the true exponent is ~0.85, not 1. (3) "E/R
sets the cliff" -- fit five gemma points across a 16x range within 15% and correctly
predicted an extrapolated point, then missed qwen R32 by 3x. A within-model fit, however
good, is not evidence of a mechanism. Every one of these was called from two or three points
and killed by the next one.

Architectural floor: gemma4 cannot be served below R8 because R >= top-k = 8, so its minimum
resident fraction is 6.25% against qwen's 3.1%. Expert count sets how aggressive residency
can be, which the memory-quality frontier does not currently express.

## The code conclusion was a generation-budget artifact (2026-08-26)

Every code cell in this document above was produced at the 1536-token default. Re-running
HumanEval at 8192 (paper/TODO.md line 189, which asked for this for an unrelated
bookkeeping reason) reverses the conclusion:

| budget | model | free | R8 | R8-free | adapted vs base |
|---|---|---|---|---|---|
| 1536 | base | 98.2 | 92.7 | -5.5 | -- |
| 1536 | adapted | 97.6 | 92.1 | -5.5 | +0.0 +/- 2.9 (z=0.00) |
| 8192 | base | 99.4 | 94.5 | -4.9 | -- |
| 8192 | adapted | 97.0 | 97.0 | **+0.0** | **+4.9 +/- 2.4 (z=2.00) REAL** |

At 8192 the adapted model's constrained arm exactly equals its own free arm: residency
damage on HumanEval is gone. **The adapter repairs code damage; it needs generation budget
to show it.** This is consistent with the documented gemma failure mode -- non-convergent
deliberation, constrained generations ruminating until truncated -- and with
halfgrain_RESULTS.md already recording gemma code as budget-saturating
(0.524@1536 / 0.634@3k / 0.628@6k). That was in the repo and was not connected to the cap
in use here.

Consequences, stated plainly:
- "The adapter recovers math and leaves code untouched" was wrong, and was repeated several
  times before the budget was varied.
- Every MBPP cell here used 1536, including all three d7code arms, so
  "code damage is structural, not a data-mix problem" is unsupported until MBPP is re-run
  at 8192. That re-run is in flight.
- A benchmark's generation cap is part of its measurement, not an implementation detail. The
  n=200-vs-n=1319 lesson has a twin: sample size AND budget both have to be right before a
  null means anything.

### Frontier operating points, paired at n=1319 (supersedes the screening table)

The critical-rate table above uses a 10-POINT drop as its threshold. That is adequate for
locating a cliff and far too permissive for a deployment claim. Paired against the same arm
at RHO=0, both cells at n=1319:

| arm | resident | reduced rate | saving | accuracy | vs full rate |
|---|---|---|---|---|---|
| R16 | 12.5% | 0.574 | 1.7x | 87.6 | **+1.1 +/- 0.6 (free)** |
| R32 | 25% | 0.194 | 5.2x | 84.8 | -2.9 +/- 0.7 (z=-4.30) |
| R64 | 50% | 0.084 | 11.9x | 83.2 | -4.1 +/- 0.8 (z=-5.40) |

**Only ~1.7x is free.** Larger savings buy bandwidth with accuracy along a smooth curve;
they are not a free lunch. The earlier framing of R64 as a "12x saving" quoted the -4.0 cost
from screening and still presented the point as attractive -- the number was right and the
framing oversold it. Quote the pair, not the bandwidth factor alone.

### CORRECTION to the budget-artifact section above (2026-08-26, same day)

The section above concluded from ONE run (rebuild) that raising the cap to 8192 lets the
adapter fully repair HumanEval. A second D7 run at the same budget contradicts it:

| benchmark @8192 | base gap | rebuild | seed3 | D7 mean |
|---|---|---|---|---|
| HumanEval n=164 | -4.9 | +0.0 (recovery +4.9) | -7.3 (recovery -2.4) | +1.2 |
| MBPP n=500 | -14.0 | -11.4 (+2.6) | -12.4 (+1.6) | +2.1 |

The two HumanEval runs sit ~3 sigma apart. At n=164 that surface cannot support a
single-run claim in either direction -- which was already established earlier in this same
document and was ignored when the 8192 number looked decisive.

What survives, across both budgets and all runs:
- **MBPP damage is real, large and largely unrepaired**: -14.6 at 1536, -14.0 at 8192, with
  a consistent but small D7 recovery of ~+2 that never approaches significance against a
  14-point gap. Budget does NOT explain MBPP.
- HumanEval is too small (n=164, SE 1.7 per gap, ~2.4 cross-run) to settle anything; its
  runs scatter from -7.3 to +0.0.
- The code-mix arm (d7code, 26.7% code) is within run scatter of the D7 arms on both
  surfaces at 8192, as it was at 1536. The hypothesis remains unsupported.

Sequence of claims made about code tonight, all from under-measurement:
"no movement" (unmeasurable at n=164) -> "a real null" (one MBPP run, sd 3.5) ->
"a budget artifact, adapter repairs code" (one HumanEval run at 8192) -> the above.
Each correction came from adding a replicate, never from new reasoning.

## The recipe sits at a sharp optimum: four knobs, four failures (2026-08-26)

`gemma_adapt_RESULTS.md` asserts "all settings load-bearing" without showing the ablation.
This is that ablation, on the rebuilt pool. GSM8K n=1319, absolute accuracy; the adapter's
job is to raise the R8 arm above the unadapted base's 78.8.

| arm | free | R8 | vs base R8 |
|---|---|---|---|
| base (no adapter) | 87.8 | 78.8 | -- |
| **rebuild (published settings)** | 86.8 | **81.9** | **+3.1** |
| d7code_s2 (published settings + code lane) | 87.7 | 81.1 | +2.4 |
| A2: KL weight 0.05 -> 0.02 | 87.6 | 79.2 | +0.5 |
| A1: lr 3e-5 -> 5e-5 | 87.0 | 78.1 | -0.7 |
| A3: expert-LoRA r32 -> r64 | 87.6 | 77.3 | -1.5 |
| A4: 3.4M -> 5M tokens | -- | 80.1 | +1.4 |

All four variants lose to the published settings, and they degrade in order of distance
from them. Every lever that lets the model adapt HARDER makes it worse: more capacity, bigger steps,
weaker anchoring. With the earlier budget result (3.4M beats both 0.6M and 7.36M) that is
four independent knobs all showing the published settings at a local optimum, and the
KL anchor at 0.05 doing real work rather than acting as a safety margin.

Consequence for the D12 gap: the rebuild reaches +3.1 where D12 reported +6.0, and none of
these knobs closes it -- they all point the wrong way. The remaining difference is most
likely the PROMPT POOL, which is a reconstruction from a prose spec because the original
was lost with the pod. That is the honest disposition: the recipe is reproducible, the
data is not.

Caveat: each variant is ONE training run against a run-to-run sd of 0.9, so no single
comparison here is airtight (A4 vs rebuild is ~1.9 sigma). The conclusion rests on the
monotone ordering across four independent knobs, not on any one cell.

## The rebuild on qwen3.5: it does not transfer at 3.1% resident (2026-08-27)

Same recipe, same pool, qwen's own trajectories (`qwen35_d7_seq4096`), adapter
`qwen_ce_rebuild_adapter.pt`. Same-arm delta = adapted arm minus base arm at the SAME
constraint, so a moving baseline cannot flatter it. Published d12r figures are the
same-arm gains stated in gemma_adapt_RESULTS.md.

**R8 = 8/256 = 3.1% resident** (twice as tight as gemma's R8):

| benchmark | base | d12r published | rebuild |
|---|---|---|---|
| GSM8K (n=1319) | 76.6 | +6.5 | +2.1 |
| IFEval (n=541) | 82.6 | +3.5 | +0.2 |
| HumanEval (n=164) | 90.9 | +3.0 | -1.8 |
| MMLU (n=228) | 92.1 | -2.2 | -2.2 |
| MBPP (n=500) | 75.2 | n/a | +1.6 |
| **MEAN (4 published cells)** | | **+2.7** | **-0.4** |

**R32 = 12.5% resident** (the fraction matching gemma's R8):

| benchmark | base | rebuild |
|---|---|---|
| GSM8K | 79.8 | +3.2 |
| HumanEval | 89.0 | +0.6 |
| MMLU | 93.4 | -1.8 |
| MBPP | 76.4 | +2.6 |
| **MEAN (3 published cells)** | | **+0.7** |

Read against gemma's +1.6 mean, the recipe **transfers poorly**: net-negative at 3.1%
resident and only mildly positive at 12.5%. MMLU was the cell most likely to rescue the
mean, since it is where the published arm LOST -- it came in at exactly -2.2, matching
published, so it pulls our mean down rather than up.

The one signal consistent across both models and both bounds is MATH: gemma +3.1,
qwen +2.1 (R8) / +3.2 (R32). Everything else is within noise or negative. The defensible
claim from this rebuild is a gemma math-recovery result with a partial qwen replication at
matched residency fraction -- not a general adaptation method.

Instrument note: HumanEval is n=164 with 10-17 arm disagreements, so neither the gemma
+2.4 nor the qwen -1.8 is individually resolvable; their difference is ~1.4 sigma. Do not
read a model contrast into that cell.

## Baseline #2 (ReMoE) measured: router-only reuse buys nothing under a hard R (2026-08-27)

BASELINE_METHODS_COMPARISON.md #2, faithful remake: router projections ONLY trainable
(attention LoRA, expert LoRA and norms frozen), recency-reuse objective at lambda 1.0 /
gamma 0.9, residency constraint OFF during training, same 3.4M-token budget and pool as
every other arm. Producer: `--router-only --no-constraint --remoe-lambda` in
train_gemma_ce.py; objective in temporal/ablation_mechanisms.py::remoe_reuse_loss.

| model | free | R8 | R8-free | R8 vs base R8 |
|---|---|---|---|---|
| base (no adapter) | 87.8 | 78.8 | -9.0 | -- |
| ours (rebuild) | 86.8 | 81.9 | -4.9 | +3.1 +/- 1.0 |
| **ReMoE** | 87.6 | 78.7 | -8.9 | **-0.1 +/- 1.1** |

The router genuinely trained (all 30 router projections moved by ~3.8e-4, loss 0.345 ->
0.225), so this is a null from the method and not from a no-op. verify_merge reports the
expert and attention surfaces as unchanged, which is CORRECT for a router-only arm.

Reading: ReMoE improves expert reuse without ever bounding the resident set, so at a fixed
R=8 it recovers none of the residency damage. This is the same conclusion baseline #3
(Skliar, serving-side deadband) reached from the other direction. Both competitors buy
bandwidth; neither buys memory; our constraint-aware CE recovers +3.1 where they recover
nothing. Appendix E can now make that argument from two measurements on our own ladder.

Prediction registered before the run: "little or no constrained-quality gain". Held.

## EXP B: KL anchor on the constrained arm -- worse (2026-08-27)

Hypothesis: the rebuild anchors KL to the base's FREE-routing logprobs, pulling the adapter
toward behaviour it cannot exhibit under R=8; anchoring to the base's own constrained
behaviour should free it to move. Same pool, same trajectories, same budget, `--kl-arm
constrained` for both the reference precompute and training.

| model | free | R8 | R16 | R8 vs base | R16 vs base |
|---|---|---|---|---|---|
| base | 87.8 | 78.8 | 86.6 | -- | -- |
| rebuild (free-arm KL) | 86.8 | 81.9 | 87.5 | +3.1 +/- 1.0 | +0.9 +/- 0.6 |
| **EXP B (constrained KL)** | 86.7 | 81.0 | 86.5 | +2.2 +/- 1.1 | -0.1 +/- 0.6 |

Falsified. The free-arm reference is the TARGET the adapter recovers toward, not a
distraction from it; anchoring to constrained behaviour anchors to the damage. Fifth recipe
knob (after lr, KL weight, rank, budget) confirmed at its published setting.

## EXP A: StackMathQA lane doubled -- no gain (2026-08-27)

Math is the only signal consistent across both models, so the obvious data lever is more
of it. The realmath lane went 2306 -> 4700 rows (27% -> 55% of the pool) with total pool
size held at 8482; other lanes shrank proportionally. Same recipe, same budget.

| model | free | R8 | R16 | R8 vs base | R16 vs base |
|---|---|---|---|---|---|
| base | 87.8 | 78.8 | 86.6 | -- | -- |
| rebuild (math 27%) | 86.8 | 81.9 | 87.5 | +3.1 +/- 1.0 | +0.9 +/- 0.6 |
| **EXP A (math 55%)** | 87.4 | 81.4 | 86.3 | +2.7 +/- 1.1 | -0.3 +/- 0.7 |

No gain on R8, a loss on R16. The recovery is not data-hungry along the math axis; the
published lane proportion is at least as good as doubling it. Sixth lever confirmed at the
published setting (lr, KL weight, rank, budget, KL arm, math share).

## EXP C: 3x unique prompts -- worse (2026-08-27). The overnight sweep is closed.

The pool went 8,482 -> 25,446 prompts (published lane ratios, realmath lane scaled and
spliced correctly), yielding 22.1M response tokens. At the same 3.4M budget the run sees
~15% of the pool (vs ~46% for the rebuild), so this isolates diversity from repetition.

| model | free | R8 | R16 | R8 vs base | R16 vs base |
|---|---|---|---|---|---|
| base | 87.8 | 78.8 | 86.6 | -- | -- |
| rebuild (8.5k prompts) | 86.8 | 81.9 | 87.5 | +3.1 +/- 1.0 | +0.9 +/- 0.6 |
| B: constrained-arm KL | 86.7 | 81.0 | 86.5 | +2.2 +/- 1.1 | -0.1 +/- 0.6 |
| A: math lane x2 | 87.4 | 81.4 | 86.3 | +2.7 +/- 1.1 | -0.3 +/- 0.7 |
| **C: 3x unique prompts** | 87.6 | 79.8 | 86.3 | **+1.0 +/- 1.1** | -0.3 +/- 0.6 |

C is the worst arm of the night. More unique prompts at fixed budget means LESS repetition
per prompt, and that costs recovery: the adapter appears to need several passes over a
prompt to learn its constrained trajectory, and 3x diversity starves it of them. Taken with
fullpass (7.36M tokens over the same 8.5k prompts, also worse), the budget-to-pool ratio
of the published recipe (~half a pass) is itself a tuned quantity.

**Seven levers, seven failures**, all landing at or below the published settings:
lr, KL weight, expert-LoRA rank, token budget, KL anchor arm, math share, pool size.
Every direction is downhill. The +3.1 vs +6.0 gap to D12 is not in any hyperparameter or
in any data-composition or data-scale knob available to us. It is in D12's SPECIFIC
prompts, which were lost with the pod and have no committed builder.

Disposition: the recipe reproduces; the data does not. The paper should report +3.1 (five
runs, z=7.5) as the reproducible result and state that the published +6.0 rests on an
unrecoverable pool. This is the second of the two dispositions RECOVER_DATA_PLAN.md
section 1.5 anticipated, and it is now established by exhaustion rather than assumed.

### qwen R32 completed with IFEval (2026-08-27) -- the fraction-matched table at 4 of 4

| benchmark | base R32 | rebuild |
|---|---|---|
| GSM8K | 79.8 | +3.2 |
| IFEval | 85.2 | -1.3 |
| HumanEval | 89.0 | +0.6 |
| MMLU | 93.4 | -1.8 |
| MBPP | 76.4 | +2.6 |
| **MEAN (4 published cells)** | | **+0.2** |

IFEval at R32 is -1.3, so the fraction-matched mean falls from the +0.7 quoted on three
cells to +0.2 on four. gemma at the same 12.5% fraction is +1.6. Final qwen position: the
rebuild's math recovery transfers (+3.2 GSM8K, +2.6 MBPP); nothing else does, and IFEval
and MMLU go slightly negative. Net ~0 at both residency fractions tested.

## Two full runs of the published recipe on the five-benchmark surface (2026-08-27)

seed3 (the median D7 seed) now has every cell. Same-arm R8 delta vs matched base.
**This table uses same-arm framing throughout** (adapted R8 minus base R8). The earlier
five-run GSM8K mean of +3.1 used gap-closure framing (R8-free vs base's R8-free), which
credits the adapter for its own free-arm sag; same-arm is the honest number and is used
from here on. In same-arm framing the five GSM8K runs are +3.1/+2.5/+0.9/+3.1/+2.0,
**mean +2.3, sd 0.9, z=5.6**.

| benchmark | D12 published | rebuild | seed3 | two-run mean |
|---|---|---|---|---|
| GSM8K (n=1319) | +6.0 | +3.1 | +2.0 | +2.5 |
| IFEval (n=541) | -1.0 | -0.7 | +0.0 | -0.4 |
| HumanEval (n=164, @8192) | +4.9 | +2.4 | -3.0 | -0.3 |
| MMLU (n=228) | -1.1 | +1.8 | +0.9 | +1.3 |
| MBPP (n=500, @8192) | n/a | +1.6 | -0.4 | +0.6 |
| WritingBench (n=150) | n/a | +0.020 | +0.008 | +0.014 |
| **MEAN (4 published cells)** | **+2.2** | **+1.6** | **-0.1** | **+0.8** |

The second run halves the headline. seed3 is a worse run than rebuild on every cell but
IFEval, and its HumanEval is -3.0 (150/164 vs base 155/164, zero unfinished -- real, not
an artifact). HumanEval's two runs differ by 5.4 points on 164 problems; that cell cannot
carry a claim in either direction and should be reported as unresolved.

Honest position for the paper: the published recipe recovers GSM8K under residency
(+2.3 same-arm, five runs, z=5.6) and is neutral-to-slightly-positive elsewhere, with a
four-cell mean of roughly +0.8 across two full runs against the published +2.2. The
published number rests on an unrecoverable pool; ours reproduces.

## THE GAP TO D12 IS THE SELF-GENERATED MATH LANE (2026-08-27)

D12's pool used a `math_selfgen` lane: the model writing its own multi-step arithmetic word
problems ("Maya went to the grocery store... purchased 12..."). The rebuild replaced it with
StackMathQA (real math.stackexchange questions) after the lane was flagged as overfitting
to GSM8K. An adapter trained on the self-generated lane (`gemma_ce_selfgen`, trained 08-25
before the trajectory file was rewritten -- provenance from timestamps and from its 312-step
count vs the rebuild's 258 at equal tokens) was re-scored at n=1319:

| adapter | math lane | free | R8 | R8 same-arm | R16 same-arm |
|---|---|---|---|---|---|
| base | -- | 87.8 | 78.8 | -- | -- |
| rebuild | StackMathQA (real) | 86.8 | 81.9 | +3.1 +/- 1.0 | +0.9 +/- 0.6 |
| **selfgen** | model-written, GSM8K-shaped | 88.2 | 84.1 | **+5.3 +/- 1.0** | +0.1 +/- 0.6 |

Head-to-head on the same 1319 questions: selfgen beats rebuild by +2.2 +/- 0.9 (z=2.39).
+5.3 is within noise of D12's published +6.0 (n=200). **The entire gap is the lane.**

Two signatures mark this as style-matching rather than residency robustness:
1. selfgen's FREE arm is 88.2, above base's 87.8. It improved the unconstrained model on
   GSM8K -- it taught the task. The rebuild's free arm sagged (86.8), which is what an
   adapter that only learns the constraint looks like.
2. selfgen's R16 delta is +0.1 vs the rebuild's +0.9. Genuine constraint robustness should
   help at every bound; a GSM8K-specialised adapter has nothing extra to give at R16.

This is the Orca-Math failure mode arriving by a different door. The lineage rule bans
"benchmark-family data in any form (test/train splits, synthetic derivatives)"; a model
generating GSM8K-format problems is a synthetic derivative in everything but provenance.
The 8-gram screen cannot catch it because the problems are novel text in the benchmark's
shape. The published +6.0 (gemma) and +6.5 (qwen, same pool) both rest on it.

**Supersedes the "prompt pool is lost" disposition.** The pool is not lost; its math lane
was regenerable all along (build_selfgen_lanes.py, selfgen_math_raw.pt). It was rejected
on principle, correctly. The reproducible honest number is the rebuild's; the published
number is reproducible but should not be reported as constraint robustness.

Next: selfgen's full surface (IFEval, MMLU, HumanEval, MBPP, WritingBench). If those are
flat while GSM8K is +5.3, the case is closed.

Mechanism, confirmed from `selfgen_math_raw.pt` (2,700 rows): stage A gave gemma seed
instructions ("Write a multi-step word problem needing at least three arithmetic operations
... Output ONLY the problem statement") and it authored the problems; stage B had it solve
its own problems, and the adapter trained on those solutions. Data-file trap for anyone
regenerating: `selfgen_math_2341.jsonl` holds the stage-A INSTRUCTIONS (120 distinct
templates, repeated), not the problems -- a pool spliced from it collapses to ~120 math
rows under dedup. The authored problems are `selfgen_math_prompts.jsonl` (2,671 distinct).
The qwen cross-model test uses those authored problems with qwen's own solutions.

### Style, not contamination (2026-08-27)

Two measurements separate the claims. 8-gram overlap with GSM8K test: selfgen 63 of 2,671
authored problems hit, StackMathQA 0, WildChat 0 -- but every one of the 17 distinct hit
grams is question-stem boilerplate ("how many hours will it take them to complete", "how much
money will she have left over after"), max 2 per row, no numbers or names. The gemma selfgen
pool was built through build_d7_prompts.py and so was screened; those rows never trained.
Style: selfgen prompts share a median 4.5% of their 4-grams with GSM8K test (p90 12.2%),
StackMathQA 0% (p90 3.4%). The lane reproduces GSM8K's phrasing and structure without
reproducing any item.

So the published +6.0 is not leaked test data. It is an adapter trained on problems in the
benchmark's exact format, which an n-gram screen cannot catch because the text is novel.
This is why the lineage rule bans synthetic derivatives by PROVENANCE rather than by
post-hoc filtering -- and why a self-generation pipeline that produces benchmark-format
items is a derivative even with a clean screen.

## Why the adapter fails on qwen: residency breaks ARITHMETIC, and the loss cannot see it (2026-08-27)

Producer: `analysis/residency/failure_filter.py` plus the equation checks in this section.
All from the committed n=1319 dumps (which carry full generated text), no GPU.

**The adapter repairs most damage on BOTH models; the difference is what it breaks.**

| | damage_fixed | adapter_broke | net |
|---|---|---|---|
| qwen | 103 | 92 | +11 (+0.8%) |
| gemma | 107 | 75 | +32 (+2.4%) |

Repair rate is ~70% of residency-damaged problems on each. On adapter_broke, the adapted
FREE arm is still right on 78/92 (qwen) and 63/75 (gemma): a reshuffle of which problems
survive the constraint, not a capability loss.

**Not truncation, loops, or extraction.** Cap-hit 0-9%, 8-gram repetition ~0, lengths normal.
A normalised extractor (strip $ , .00) adds ~+5 to EVERY arm uniformly and leaves the gap
untouched -- a scorer offset, not a residency effect.

**It is arithmetic.** Among wrong constrained generations (base-free-right subset), the
fraction containing a demonstrably false `a op b = c` equation:

| | base R8 | adapted R8 | noise on correct gens |
|---|---|---|---|
| qwen | 49% (73/150) | 41% (53/128) | 2-4% |
| gemma | 32% (46/142) | 28% (27/98) | 2-3% |

The slips are trivial: `5+4+2=8`, `7+8=30`, `30-27=7`, `160+330=794`, `6*20=60`. Mostly
two-term, mid-response (median position 0.55-0.67), qwen's on smaller operands (median 35
vs gemma's 80) -- 3.1% residency breaks easier arithmetic than 6.25% does. The plan and
setup are intact; the primitive fails. 37% (qwen) / 24% (gemma) of adapter_broke problems
are broken by a NEW false equation the base R8 did not make.

**Why CE cannot fix it.** In the qwen D7 trajectories, digit tokens are 6.3% of response
tokens and digits following '=' are 0.66%. The loss is >99% about tokens that do not fail.
The adapter therefore learns format and phrasing (which is why the free arm moves, and why
the selfgen lane 'works') and leaves the arithmetic primitive alone, while perturbing the
resident experts enough to introduce new slips.

**Proposed fix, at the source:** reweight the CE loss on digit tokens (`--digit-weight`),
so the gradient concentrates on the failing token class. Same data, same budget, one flag.

### Refined (same day): scorer artifacts are uniform; arithmetic is about half of real failures

Three extractors (as scored / last-number / bold-answer) give qwen R8 gaps of -9.2 / -8.8 /
-9.5 -- the extractor moves every arm together and leaves the gap alone. 18% of individual
constrained failures are scorer artifacts (`$21.00`, or "**$132** after 12 hours" where the
last-number rule takes 12), but the free arm has the same rate. Closed: the damage is real.

The first pass at this claimed 84% of real qwen failures (66% gemma) contained a false
equation. That number came from equation checks that were never committed and it does not
reproduce. The committed parser (`analysis/residency/slip_position.py`, LaTeX-aware: `\$`,
`\times`, `\text{}`, thousands commas) finds a false equation in about half of the
real-damage generations on both models: qwen 68/150 (25/47 unfixed, 43/103 fixed), gemma
63/142 (16/35, 47/107), or roughly 55% once the 18% scorer artifacts leave the denominator.
The parser is a lower bound (prose-form sums such as "300 + 200 + 500" written as "$800" are
not caught), so arithmetic is the largest failure class either way, but "84%" overstated it.

The cleaner statement is a rate rather than a share. Across all 1319 generations per arm,
the fraction of parsed equations that are false:

| | free | tight | adapted tight | adapted free |
|---|---|---|---|---|
| qwen (R8) | 0.68% | 5.03% | 3.77% | 0.53% |
| gemma (R8) | 0.81% | 4.82% | 2.75% | 0.75% |

The constraint multiplies the error rate by six or seven on both models. The qwen adapter
removes 29% of that excess; the gemma adapter removes 52%. That single number is the
qwen gap, and it does not depend on the scorer. It is the primary readout for the
digit-weight runs (standard error about 0.5 points at 2000-3500 equations per arm).

Where the slips land: the first false equation sits at median position 0.40-0.62 of the
generation and falls in the second half 29-61% of the time, against 55-64% for equations
in general. There is no late clustering on either model, so the slips are not accumulated
residency-state drift, and nothing here says prefix-independent CE cannot reach them. On
qwen the slips skew toward trivial operands (both sides at most 20: 39-64% of slips vs
27-43% of all equations); on gemma the reverse, on small counts.

### The mechanism is systematic across five independent gemma runs

| run | broke | of which NEW slip | fixed | of which slip gone |
|---|---|---|---|---|
| rebuild | 75 | 37 (49%) | 116 | 59 (51%) |
| seed1 | 100 | 45 (45%) | 112 | 59 (53%) |
| seed2 | 85 | 37 (44%) | 126 | 61 (48%) |
| seed3 | 97 | 42 (43%) | 123 | 68 (55%) |
| realmath | 81 | 31 (38%) | 114 | 55 (48%) |

Every run repairs ~120 problems and breaks ~90; in both directions about half the movement
is an arithmetic slip appearing or disappearing. The adapter RESHUFFLES which problems get
the slip. It does not lower the slip rate, because nothing in a full-trajectory CE loss
rewards the correct digit specifically. This is the quantity --digit-weight is meant to
change; the test is whether the "broke" column shrinks without the "fixed" column shrinking.

### Dose-response and the selfgen connection

Arithmetic-slip share of real constrained failures vs residency fraction (LaTeX-aware parse,
noise floor 10-17%):

| model | 3.1% | 6.25% | 12.5% |
|---|---|---|---|
| qwen base | 71% (107/150) | -- | 60% (64/107) |
| gemma base | -- | 63% (89/142) | **27% (9/33)** |

Monotone on both models; at 12.5% resident gemma's arithmetic slips nearly reach the noise
floor and there are only 33 failures left. Arithmetic breakage is the specific signature of
tight residency.

And it explains the selfgen lane's gain. Slips among gemma R8 failures: base 89, rebuild 57,
**selfgen 33**. Model-authored word problems are arithmetic-dense, so digit tokens were a
larger share of that lane's loss -- it upweighted arithmetic by accident, with
benchmark-format data as the vehicle. `--digit-weight` is the same lever applied on
purpose, on real prompts.
