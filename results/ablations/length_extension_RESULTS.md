# Does the blow-up mechanism generalize beyond IFEval? Code, knowledge, and long-form writing

The IFEval finding was that damage under rolling residency rides length: items
that blow up (hit the generation cap, or run more than twice their free-arm
counterpart) are where the constrained arm goes wrong. That claim rested on the
two surfaces that had per-item dumps. This note extends it to the three that did
not — HumanEval (code), MMLU (knowledge), WritingBench (long-form) — using the
trajectories recovered and regenerated in the dump-repair pass.

Definitions, unchanged from `plot_length_decomp.py`: **cap-hit** = generation
within 8 tokens of the cell's budget; **blow-up** = cap-hit in either arm OR one
arm more than twice the paired counterpart; all comparisons are paired on the
same document across the free and constrained arms of one model. WritingBench
has no binary correctness, so a **flip** there is operationalized as a per-item
critic-score change beyond ±1 SD of that subset's own delta distribution — a
choice made here, not inherited.

Producers: `analysis/residency/length_extension.py` (table + figure),
`mmlu_dual_lengths.py` (recovered think-off MMLU lengths),
`analysis/writingbench/wb_lengths.py`. Data: `length_extension.csv`,
`genbench_samples/`, `writingbench/`.

## Headline: yes, and it is strongest exactly where the model writes code

**Wrongness conditioned on blow-up, pooled over cells (constrained arm):**

| surface | wrong \| blow-up | wrong \| normal length | ratio | cells agreeing |
|---|---|---|---|---|
| HumanEval | **0.526** (195/371) | **0.047** (107/2253) | 11.1× | 13 of 16 |
| MMLU | **0.300** (223/744) | **0.148** (396/2676) | 2.0× | 10 of 15 |

- **Code is the extreme case.** A HumanEval item that blows up under the
  constraint is wrong more than half the time; one that keeps a normal length is
  wrong 6% of the time. The gap is an order of magnitude and it is not carried
  by one model: 9 of 10 cells show blown items wronger than normal ones.
- **Knowledge shows the same sign, half the strength.** MMLU's ratio is 1.9×.
  Multiple-choice has a floor — a truncated answer still has a 25% chance — which
  compresses the effect the length mechanism can produce.
- **The strongest single cells are the thinking-mode code cells.** gemma
  thinking-on at R=8 is 51 of 62 blown items wrong against 4 of 102 normal ones;
  gpt-oss-20b at high effort is 29 of 55 against 4 of 109. Both are the
  deliberation-heavy configurations, where a lost expert turns into a
  non-terminating chain of thought rather than a wrong token.

## Flip counts by direction

Flips are counted per cell between the free and constrained arm of the same
model, then pooled. All cells are **single runs**; at these n, individual cell
counts carry roughly ±sqrt(count) noise, so read the pooled ratios, not
individual cells.

| surface | cells | flips to wrong | of those, blow-up | rescues (to right) | of those, blow-up |
|---|---|---|---|---|---|
| HumanEval | 16 | **155** | 87 (56%) | 78 | 45 (58%) |
| MMLU | 15 | **287** | 111 (39%) | 207 | 73 (35%) |
| WritingBench | 12 | **273** | 78 (29%) | 152 | 36 (24%) |
| **total (new surfaces)** | **43** | **715** | 276 (39%) | 437 | 154 (35%) |

- **Damage is asymmetric everywhere**: 715 flips to wrong against 437 rescues,
  1.64:1 pooled. Direction holds on all three surfaces separately (2.0:1 code,
  1.4:1 knowledge, 1.8:1 writing).
- **The original IFEval-era headline was 174 flips-to-wrong against 85 rescues**
  (2.0:1) on the thinking/non-thinking generative cells. The new surfaces
  **reproduce the asymmetry on a 4.1× larger flip sample** at a slightly milder
  ratio (1.64:1); the direction is not in doubt, its magnitude softens as
  easier surfaces enter the pool.
- They do **not** reproduce the "blow-up wrongness direction 20 of 20" clean
  sweep, which becomes **23 of 31** here (13/16 HumanEval, 10/15 MMLU). The
  dissenters are concentrated in the configurations that barely deliberate and
  therefore barely blow up — gpt-oss at **low** effort and the gemma think-off
  cells, where a handful of blown items (as few as 5) decide the comparison.
  On the deliberation-heavy cells the direction is unanimous.
- **Blow-up is not the whole story.** It is involved in 41% of flips-to-wrong,
  and it is *equally* common among rescues (39%). Blow-up marks an item as
  high-variance under the constraint; conditional wrongness (the table above) is
  where the asymmetry lives, not in the flip counts alone.

## Cap traffic versus broad lengthening

Mean paired length change (constrained − free) and the part contributed by
cap-hit items:

- **HumanEval: +110 tokens per item on average, of which +28 is cap traffic.**
  Most of the lengthening is broad — the constrained model writes longer code and
  longer commentary on items that never approach the budget.
- **MMLU: +23 tokens, of which +9 is cap traffic.** Knowledge answers barely
  lengthen on average; what movement exists is concentrated in the few items that
  saturate.
- **WritingBench: −22 tokens on average**, i.e. no systematic lengthening at all
  on long-form writing, with individual cells split in both directions
  (gpt-oss-120b R4 −114, qwen R8 +52).
- **The two components are largely independent.** Cells with the largest total
  change are not the cells with the largest cap component — Qwen thinking-on
  HumanEval is +360 total against +94 cap, while LFM is +250 against +138.

## Measured at a fair budget: two thirds of qwen's MMLU damage was the budget

The cell above was re-measured at 8192, double its original budget, both arms,
everything else unchanged (single run per arm):

| qwen think-on MMLU | 4096 | 8192 |
|---|---|---|
| free | 0.8816 | **0.9386** |
| R = 8 | 0.8158 | **0.9167** |
| **damage (R8 − free)** | **−6.6 pts** | **−2.2 pts** |
| generations still at the cap | 63 free / 78 R8 | 2 free / 9 R8 |
| mean tokens | 2530 / 2614 | 2894 / 3363 |

- **Most of the measured damage was truncation, not residency.** At 4096, 27% of
  free-arm and 34% of constrained-arm generations were cut off mid-thought and
  scored as wrong. Give the model room to finish and the gap between the arms
  falls from 6.6 points to 2.2.
- **The −2.2 that remains is real** — and so is the mechanism behind the rest: the
  constrained arm still runs longer (3363 against 2894 mean tokens) and still hits
  the wall more often (9 against 2). Residency does push generations toward the
  budget. It just was not worth 6.6 points of accuracy; the budget was.
- This is the honest version of the §1 estimate in `TRUNCATION_RERUN_PLAN.md`,
  which put the shift at +7.4 points by *excluding* truncated items (landing at
  +0.8). Measuring at a budget the model can finish in lands at −2.2 instead of
  dropping the inconvenient items.

## Fair budget, second surface: IFEval reproduces the MMLU result

Every IFEval cell that was still budget-limited was re-measured at 16384 (double).
Metric is `prompt_level_strict_acc` (fraction of prompts satisfying every
instruction; 0-1, higher is better), n=200, single run per arm. "Trunc" is the
share of items finishing within 8 tokens of the cap — the thing being fixed.
Flips are paired on the same document; z is McNemar with continuity correction on
the discordant pairs, so **|z| > 2 is a real move, below that is rerun noise**.

| cell | trunc @8192 → @16384 | acc @8192 → @16384 | Δ | →right | →wrong | z |
|---|---|---|---|---|---|---|
| gpt-oss-20b high, free | 17.0% → 1.0% | 0.790 → 0.860 | +7.0 | 23 | 9 | 2.30 |
| gpt-oss-20b high, R=4 | 19.0% → 0.5% | 0.760 → 0.875 | +11.5 | 28 | 5 | 3.83 |
| qwen3.5-35B, R=8 | 10.0% → 2.0% | 0.700 → 0.800 | +10.0 | 33 | 13 | 2.80 |
| qwen3.5-35B, R=32 | 8.5% → 1.0% | 0.815 → 0.840 | +2.5 | 12 | 7 | 0.92 |
| gpt-oss-120b high, free | 6.0% → 0.5% | 0.845 → 0.885 | +4.0 | 17 | 9 | 1.37 |
| gpt-oss-120b high, R=4 | 6.0% → 0.5% | 0.840 → 0.890 | +5.0 | 18 | 8 | 1.77 |
| gpt-oss-120b high, R=16 | 9.0% → 0.5% | 0.835 → 0.895 | +6.0 | 21 | 9 | 2.01 |

**The gain is carried by exactly the items that were being cut off.** Splitting
each cell's net flips by whether the item was truncated in the 8192 run:

| cell | net gain from previously-truncated | from never-truncated |
|---|---|---|
| gpt-oss-20b high free / R4 | +15 / +21 | −1 / +2 |
| gpt-oss-120b high free / R4 / R16 | +7 / +7 / +12 | +1 / +3 / 0 |
| qwen3.5-35B R8 / R32 | +9 / +4 | **+11** / +1 |

For both gpt-oss models the never-truncated items are flat (−1 to +3 net across
five cells) while the truncated ones carry everything — the budget was the whole
story. **qwen R=8 is the exception and is flagged, not explained away**: half its
gain (+11 of +20) comes from items that never hit the cap, which the budget
cannot account for. That is consistent with the already-recorded finding that
qwen constrained-arm trajectories are not reproducible run-to-run (see "Flagged"
below) — so qwen's +10.0 is part fair-budget correction, part rerun draw, and
should not be read as a pure budget effect.

**Consequence for the damage numbers** (damage = constrained − free, negative =
residency hurts):

| cell | damage @8192 | damage @fair budget |
|---|---|---|
| gpt-oss-20b high, R=4 | −3.0 pts | **+1.5 pts** |
| gpt-oss-120b high, R=4 | −0.5 pts | **+0.5 pts** |
| gpt-oss-120b high, R=16 | −1.0 pts | **+1.0 pts** |
| qwen3.5-35B, R=8 | −16.5 pts | **−6.5 pts** |
| qwen3.5-35B, R=32 | −5.0 pts | **−2.5 pts** |

All three gpt-oss IFEval damages were already inside the noise floor and land on
the other side of zero at a fair budget: **there is no measurable residency damage
on gpt-oss IFEval, at either model size.** qwen's damage survives but shrinks by
roughly 60% — the same ratio, on a different surface, as the MMLU result above,
where two thirds of the damage was the budget. The pattern that generalizes is
not "residency costs N points"; it is that a budget-limited cell overstates
residency damage, because the constrained arm runs longer and therefore eats the
truncation penalty more often.

## Per-surface detail worth naming

- **Qwen thinking-on MMLU saturates its budget**: 63 of 228 free-arm items and 78
  of 228 constrained-arm items finish at the 4096-token cap. This cell is
  budget-limited, not knowledge-limited, and its scores should be read as such.
- **gemma thinking-on HumanEval** is the deliberation attractor seen elsewhere in
  this program: +191 mean tokens at R=8, 62 of 164 items blown, and the blow-ups
  carry essentially all of the damage.
- **WritingBench damage is small and mostly length-blind.** Mean critic-score
  changes are −0.07 to −0.31 on a 10-point scale, and the blow-up share of items
  is under 12% on every gemma and qwen cell. Where blow-up does concentrate
  (gpt-oss and LFM, 32–44% of items), it is because those models routinely write
  to the 4096-token cap in both arms.

## OLMoE-Instruct MMLU: relaxed versus strict

The grid carried OLMoE's MMLU as strict-extraction only. Both arms were
regenerated with dumps and dual-scored:

| arm | relaxed extraction | strict (flan `get-answer`) | prior strict grid row |
|---|---|---|---|
| free | **0.5526** | 0.5307 | 0.5000 |
| R=8 | **0.4561** | 0.3772 | 0.2851 |
| damage | **−9.6 pts** | −15.3 pts | −21.5 pts |

- **Relaxed free 0.5526, relaxed R=8 0.4561.** Against the strict grid cells the
  deltas are **+5.3 points on the free arm and +17.1 points on R=8**.
- **Strict extraction overstates residency damage on this model by a factor of
  two** (−21.5 points strict against −9.6 relaxed). The constrained arm loses the
  few-shot answer *format* faster than it loses the answer, and a format-strict
  filter charges that to knowledge.
- Both regenerated cells reproduce their grid rows within noise on the
  era-comparable metric (z = 0.7 free, z = 2.1 at R=8, binomial SE, n = 228).

## What was checked, flagged, and skipped

- **Consistency gate.** Every regenerated cell was compared against its grid row
  in binomial-SE units. Reproductions within gate: LFM HumanEval exact (0.8293 /
  0.6707), gemma think-off HumanEval (z ≤ 0.7), gemma think-on HumanEval
  (z ≤ 1.1), gpt-oss-20b HumanEval across three efforts (z ≤ 1.3), qwen MMLU
  strict (z ≤ 1.0), OLMoE (z ≤ 1.5).
- **Resolved, not variance: every gpt-oss MMLU cell moved because the sampling
  was fixed.** The old `mmlu_gptoss.py` hardcoded `temperature=1.0, top_p=1.0`
  into its `gen_kwargs`. The rewritten harness reads the model's shipped recipe,
  and gpt-oss ships **no** temperature/top_p — so the documented no-recipe
  fallback applies (`PROTOCOL_ERAS.md`: "no-recipe fallback 0.7/0.95"). The run
  logs confirm every new gpt-oss MMLU cell ran at temp 0.7 / top_p 0.95 against
  the old rows' ancestral 1.0/1.0. The old rows therefore violated the sampling
  protocol, which exists precisely because ancestral sampling was measured to
  depress scores (5–30 points across tasks on OLMoE). The new numbers are the
  protocol-correct ones and supersede the old; the largest move,
  `gptoss_20b_high` MMLU free **0.7763 → 0.8640**, is the expected direction and
  is largest on the noisiest model. No rerun is needed to explain these, and none
  was spent on them. Sampling is unchanged for every other model (they all ship
  recipes), so this explanation applies to gpt-oss MMLU only.
- **Flagged, both rows kept, old row not overwritten:**
  - `qwen35_instruct` HumanEval R8 **0.8902 vs 0.8110** and R32 **0.9634 vs
    0.8841**, while the free arm reproduced (0.9512 vs 0.9573). Adjudicated: the
    memory-fraction hypothesis is **refuted**. Rerunning at the protocol default
    (0.92, against the regeneration's 0.94) gives **R8 0.8902 again, to four
    decimals**, and R32 0.9512 against 0.9634 (a 2-item difference) — both arms
    far above the old rows, which sit 11–13 items lower. Yet **0 of 164
    generations are textually identical between the two R8 runs** and 26 items
    flip pass/fail. Constrained-arm trajectories are
    simply not reproducible run to run (batch-shape jitter rewrites every one);
    only the aggregate is. Everything else recoverable is identical across eras:
    the harness that wrote the old rows differs from today's only by dump code,
    the model is the same upstream revision (re-downloaded 2026-08-17, but
    upstream's latest commit predates both runs), and vLLM/torch/transformers
    were installed before either. With ~26 unstable items the run-to-run SD on
    pass@1 is about 1.5 points, so 0.8110 sits ~5 SD below two agreeing draws —
    too far for jitter, and **not further diagnosable, because that run saved no
    trajectories**. The new value is kept as reproduced-twice; the old is marked
    unverifiable rather than wrong.
  - **Protocol consequence:** per-item analysis on a constrained arm must use
    that run's own dump. A re-run reproduces the score but not the generations,
    so pairing items across runs is invalid.
- **Withdrawn flag: qwen thinking-on MMLU.** An earlier draft flagged this cell
  at z = 2.1 using `mmlu_flan_rescore.py`. That comparison was wrong, and the
  fault is in the rescore, not the run. lm_eval's filter takes the FIRST
  "answer is" in the text; in qwen's long thinking traces that is routinely a
  mid-reasoning aside, so the rescore extracts a stray letter or none at all. On
  the same 228 items the harness's own strict metric and the rescore disagree on
  **26 items in both directions** (strict finds an answer the rescore misses on
  the ones whose first "answer is" carries no letter; the rescore finds one
  strict misses on "the *correct* answer is (D)"). Two extractors, two failure
  modes — the gap measures the extractors, not the model. The like-for-like
  check is the harness's own `acc,strict-flan` against the old strict grid row:
  **0.8421 vs 0.8246 (z = 0.5), within gate**, and sampling is identical across
  eras here (qwen ships a recipe, both ran temp 1.0 / top_p 0.95 / top_k 20).
  `mmlu_flan_rescore.csv` remains useful for what it was built for — showing
  that stock extraction floors harmony-format answers (gpt-oss 0.05–0.11 against
  0.56–0.86 relaxed) — but its absolute values are not an era-comparable metric
  for think-in-text models and must not be cited as one.
- **An extraction trap worth recording.** The relaxed MMLU harness reports its
  own `acc,strict-flan` metric, whose regex requires the literal "the answer is".
  lm_eval's flan-CoT filter is `(?<=answer is )(.*)`, which also accepts "the
  **correct** answer is". Comparing the two across eras made gemma's regenerated
  MMLU look 17–26 points worse than its grid row; under the genuine filter the
  same generations reproduce it (free 0.8684 vs 0.8509). `mmlu_flan_rescore.py`
  now recomputes the era-comparable number from any dump, and
  `mmlu_flan_rescore.csv` carries it.
- **Skipped, with reason:** gpt-oss and LFM have no valid `mmlu_flan_cot_fewshot`
  comparison at all — stock extraction floors their answers (get-answer scores
  them 0.01–0.11 against 0.56–0.86 relaxed), which is why those cells are
  registered as never-live in `partition_eras.py`. Relaxed extraction is their
  only valid metric and no cross-era check is possible.
