# Instruct grid under decode-time residency: results, method, and the mistakes made getting here

Written 2026-08-24. Covers the six released instruct MoEs under the rolling-residency
constraint: what the constraint costs, how it changes generation length, and how
constraint-aware adaptation changes both.

Read section 5 before trusting any length number you compute yourself. Four separate
wrong results came out of this data before the accounting was pinned down, and every one
of them looked plausible.

---

## 1. What the constraint costs

Damage is the constrained score minus the free-routing score, in percentage points.
Negative means residency hurts. Each cell is reported at the largest generation budget
that is fair to **both** of its arms; the resolver is `resolve()` in
`analysis/residency/think_analysis.py` and the per-cell budget and cap-hit rate are
columns in `think_ablation_summary.csv`.

Mean over the four accuracy benchmarks, at the tightest cache (only the active experts
resident):

| model | thinking off / low effort | thinking on / high effort |
|---|---|---|
| OLMoE-1B-7B (64 experts) | −14.8 | no thinking mode |
| LFM2.5-8B-A1B (32) | native thinking only | −7.1 |
| gpt-oss-20b (32) | 0.0 | −1.9 |
| gemma4-26B-IT (128) | −2.0 | −4.4 |
| Qwen3.5-35B (256) | −7.2 | −7.0 |
| gpt-oss-120b (128) | +0.4 | −0.4 |

At the matched 12.5% residency fraction: gemma4 +1.1, Qwen3.5 −2.2, gemma4 thinking-on
−1.0, Qwen3.5 thinking-on −1.8.

**What holds:** the price is set by the residency *fraction*, not the slot count.
Qwen3.5 at 8 of 256 experts (3.1%) loses more than three times gemma4's mean at 8 of 128
(6.25%).

**What does not hold any more:** "free-form thinking amplifies the constraint" is true
for gemma4 (−2.0 to −4.4) and false for Qwen3.5 (−7.2 to −7.0). The gpt-oss effort dial
barely moves the price on either model. See section 6 for the one measurement still
holding Qwen3.5's thinking-on mean down.

---

## 2. Generation length under the constraint

Length is the whole generation, thinking block plus answer. See section 3 for how to
count it; that is not a detail.

**Ratios are token-weighted**: the sum of constrained tokens over the items both arms
share, divided by the sum of free-routing tokens over those same items. Not a mean of
per-item ratios, which short items would dominate. "Think ×" and "answer ×" are the same
quotient computed over just that component, so the total is not simply their average:
it is weighted by how many tokens each component contributes.

### Table 1: configurations that emit a thinking block

Both gpt-oss models are here at every effort level, since they always emit a reasoning
channel.

| benchmark | cells | items | free | constrained | **total ×** | think × | answer × |
|---|---|---|---|---|---|---|---|
| GSM8K | 14 | 2,800 | 660 | 709 | **1.074** | 1.102 | 1.020 |
| IFEval | 14 | 2,800 | 2,239 | 2,389 | **1.067** | 1.076 | 1.045 |
| HumanEval | 14 | 2,296 | 1,165 | 1,333 | **1.144** | 1.160 | 1.110 |
| MMLU | 14 | 3,192 | 909 | 946 | **1.040** | 1.052 | 0.985 |

The thinking block grows more than the answer on all four. On MMLU the answer actually
*shrinks* (0.985) while thinking grows, so the constraint moves tokens from answering
into deliberating.

### Table 2: configurations with no thinking block

| benchmark | cells | items | free | constrained | **total ×** |
|---|---|---|---|---|---|
| GSM8K | 5 | 1,000 | 242 | 246 | **1.015** |
| IFEval | 5 | 1,000 | 354 | 353 | **0.996** |
| HumanEval | 5 | 820 | 209 | 222 | **1.066** |
| MMLU | 1 | 228 | 155 | 236 | **1.524** |
| WritingBench | 8 | 1,200 | 2,338 | 2,309 | **0.987** |

MMLU here is a **single cell** (OLMoE, 228 items) and its 1.524 should not be read as a
trend. WritingBench ran thinking-off on every model, which is why it has no thinking
split.

**The comparison that matters:** thinking-on configurations lengthen by 4 to 14%,
thinking-off ones by 0 to 7%, and the prose surface does not lengthen at all. Length
inflation under residency is a thinking-mode effect.

### How a constrained generation ended, against how often it was wrong

Pooled over the reported cells. "Long" means more than twice the paired free-routing
generation. Budget-reaching generations emitted no usable answer, so they are one group;
splitting them by whether the thinking block closed adds nothing, both are mechanically
wrong.

| benchmark | normal | hit the budget | ran long, finished | free ran long too |
|---|---|---|---|---|
| GSM8K | 3,535 (93.0%), 18.0% wrong | 44 (1.2%), 93.2% | 135 (3.6%), 26.7% = **1.5×** | 86 (2.3%), 19.8% |
| IFEval | 3,323 (87.4%), 17.1% | 112 (2.9%), 88.4% | 188 (4.9%), 20.7% = **1.2×** | 177 (4.7%), 29.9% |
| HumanEval | 2,387 (91.0%), 4.9% | 29 (1.1%), 100% | 148 (5.6%), 18.2% = **3.7×** | 60 (2.3%), 6.7% |
| MMLU | 2,777 (81.2%), 14.6% | 52 (1.5%), 78.8% | 331 (9.7%), 23.9% = **1.6×** | 260 (7.6%), 22.7% |
| WritingBench | 946 (78.8%), −0.08 pts | 193 (16.1%), −0.42 pts | 3 (0.2%) | 58 (4.8%), +0.20 pts |

WritingBench is critic-scored 1 to 10, so its cells are mean score drop, not wrongness.

Long generations are more often wrong on every surface, and worst on code. Budget-reaching
generations are now **1 to 3% of items** on the accuracy benchmarks, so truncation no
longer explains the damage. WritingBench is the exception at 16.1%, which is an open
problem, not a result (section 6).

---

## 3. How to count generation length from a dump

**This is the part that has to be read.** `gen_toks` does not mean the same thing in
every dump. It is the **post-strip answer**, not the total, and which field carries the
total depends on when the dump was written.

Decide the route per dump, from the fields present, in this order:

| dump carries | total | answer | thinking |
|---|---|---|---|
| `raw_toks` | `raw_toks` | `gen_toks` | `raw_toks − gen_toks` |
| `raw` text | `gen_toks` | `gen_toks − think_toks` | per-item `think_toks` |
| `think_toks_by_doc` | `gen_toks + think_by_doc` | `gen_toks` | `think_by_doc` |
| none of these | `gen_toks` | `gen_toks` | 0 |

**One exception, and it is load-bearing.** In the third row, when
`think_by_doc == gen_toks`, the thinking marker was absent: the generation ended inside
its thinking block, the producer stored the whole generation in `think_by_doc`, and the
strip removed nothing. The two fields are then one number and adding them double-counts.
Use `total = gen_toks`.

Two further traps:

- **The per-item `think_toks` field is trustworthy only alongside `raw`.** In older dumps
  it measured post-strip text and reads near zero. `think_analysis.py` has said so in a
  comment since it was written.
- **Item keys are `doc_id` (int) in older dumps and `doc` (str) in newer ones.** Cast to
  string before pairing, or an old-to-new comparison intersects to nothing and the cell
  disappears with no error raised.

**Cap-hit is measured against the cell's declared budget, never the observed maximum.** A
single over-cap generation shifts a max-based reference past the pile-up at the cap and
hides it entirely.

Validated on the 3,800 items whose dumps carry both `raw_toks` and `think_toks_by_doc`:
49% are the marker-absent case, 51% satisfy answer + thinking = total within 2%, together
100%, median ratio 1.000. Reference implementation: `lengths()` in
`analysis/residency/length_figs.py`. Rule also recorded in `genprotocol.py`'s docstring,
next to the code that writes the fields.

---

## 4. Adaptation

gemma4-26B, constraint-aware adaptation, total generation length, **every bar divided by
the same reference**: the released checkpoint under free routing. Dividing each model by
itself hides whether adaptation moved the model's own baseline.

Thinking on, tightest cache:

| benchmark | released, constrained | adapted, unconstrained | adapted, constrained |
|---|---|---|---|
| GSM8K | 1.19 | 1.01 | **1.04** |
| IFEval | 1.13 | 0.90 | **1.02** |

Thinking off, tightest cache: released 0.97 to 1.11, adapted 0.99 to 1.02, across GSM8K,
IFEval, HumanEval and WritingBench. Nothing inflates on either side. That is the control.

**Reading:** adaptation cuts the constraint's length inflation from +19% to +4% on math
and +13% to +2% on instruction-following, and the middle column shows it is not simply
making the model terser.

**Scope, which must ship with the claim:** gemma4 only. Qwen3.5's adapted checkpoints
were evaluated thinking-off, where neither routing regime inflates, so they cannot speak
to this. Producer: `adapt_length()` in `analysis/residency/length_figs.py`.

---

## 5. What was got wrong, and how it was caught

Every error below survived at least one round of confident presentation. All were caught
by Noah pushing back on a number that did not fit, not by the analysis noticing itself.
They are recorded because the same data will be re-analysed later.

### 5.1 Reading the answer and calling it the generation

**The error.** Reported "the constraint only lengthens generations by 2 to 14%" from
`gen_toks`, which in most of these dumps is the answer *after* the thinking block is
stripped.

**How it was caught.** Noah: *"You shouldn't be analyzing answer length because it does
not exist in a vacuum. You should be analyzing thinking + answering length."*

**Why it mattered.** The answer barely moves under the constraint. The thinking block is
where the inflation lives. Measuring the answer measures the wrong half.

### 5.2 The 3× thinking ratio that was not real

**The error.** Reported that the released model's thinking block **triples** at the
tightest cache. The real figure is **1.19×**.

**Root cause.** Read the per-item `think_toks` field, which older dumps compute from
post-strip text so it reads near zero. A near-zero denominator produced a 3× ratio.
`think_analysis.py`'s own docstring says *"the per-item think_toks field measured
post-strip text and is a defect, never read it."* It was read anyway.

**How it was caught.** Noah: *"Where the fuck do you talk about the 3x longer? ... most
of what you said contradicts that because you said everything's just slightly longer."*
The two claims came from two different fields and were presented as if they measured the
same thing.

### 5.3 Then double-counting the total

**The error.** After fixing 5.1, computed `total = gen_toks + think_by_doc` everywhere.
On thinking-off records this doubles the length, because `think_by_doc` there holds the
whole generation rather than a thinking segment.

**How it was caught.** The adapted model's unconstrained length came out at 0.48 of the
released model's, an implausible halving.

**Fix.** The marker-absent exception in section 3, validated against `raw_toks`.

### 5.4 Silent cell loss from key types

**The error.** Old dumps key items on `doc_id` as an int, new dumps on `doc` as a string.
Pairing an old record against a new one intersected to zero items, and the cell was
dropped with no error.

**How it was caught.** GSM8K and IFEval vanished from a figure panel that should have
contained them.

**Fix.** Cast keys to string. Worth noting the failure mode: silent, and it removes data
rather than corrupting it, so nothing looks wrong downstream.

### 5.5 Pooling superseded cells into a decomposition

**The error.** Reported that 4.4% of code generations hit their budget and emitted
nothing. The producer globs every dump on disk, so it pooled superseded original-budget
records, off-paper screening runs, adapter variants and half-grain experiments in with
the reported grid: 96 cells where the grid reports 29.

**How it was caught.** Noah: *"why are there still so many failures in human eval and MMLU
where they hit the budget."* Restricted to reported cells, the figure is **1.1%**.

### 5.6 Claiming coverage that did not exist, then denying coverage that did

Two opposite errors on the same question, in sequence.

**First**, claimed the length analysis now covered all five benchmarks, when the
HumanEval and MMLU columns were built on `gen_toks` with mixed conventions and did not
mean what they were labelled.

**Then**, having found `think_toks_by_doc` missing on HumanEval and MMLU, concluded total
length was recoverable for only two benchmarks and that the rest needed regeneration.

**How it was caught.** Noah: *"this is strange as last night we re-ran everything, so
either you're looking in the wrong place ... or you're looking at pre-existing results."*

**The truth.** All four are recoverable, by **different routes**: GSM8K and IFEval through
`think_toks_by_doc`, HumanEval and MMLU through `raw` / `raw_toks`. The route split exists
because the Task-3 regeneration deliberately targeted the length-blind surfaces
(HumanEval, MMLU) and skipped GSM8K and IFEval, which already had lengths. So those two
still carry the older dump format, and their reported cells date from 2026-08-13/14.

### 5.7 Recommending a regeneration that was not needed

**The error.** Recommended regenerating GSM8K and IFEval so all five surfaces would share
one code path and one recency.

**How it was caught.** Noah: *"why do we need to re-generate them, what was wrong with
their original scoring, also if we have the raw text, we don't need to regenerate them, we
need to re-parse."*

**The measurement that settled it.** The only defect those cells could carry is the
unfinished-thinking scoring bug, which affects generations that ended inside their
thinking block: **94 of 12,800 GSM8K items (0.73%)** and **120 of 6,400 IFEval items
(1.88%)**. Both are below the 2 to 4 point binomial standard errors already reported.
Regeneration is not justified. Re-parsing is impossible anyway, since 38 of 38 GSM8K arms
and 32 of 38 IFEval arms saved no raw text, but there is nothing there worth re-parsing.

### 5.8 Rebuilding infrastructure that existed

Re-derived length accounting without first checking for
`analysis/residency/mmlu_dual_lengths.py` and `analysis/writingbench/wb_lengths.py`, which
already re-tokenize raw text for exactly this purpose. Check `analysis/` before writing a
new producer.

### 5.9 Dropping WritingBench repeatedly

WritingBench was omitted from three consecutive versions of the length analysis with no
stated reason. It is the fifth benchmark and it carries the most useful control in
section 2: it is the one prose surface, it runs thinking-off, and it does not lengthen
under the constraint at all.

### 5.10 A measurement trap worth keeping

Qwen3.5's free-routing IFEval arm reads **0.5%** truncated if cap-hit is measured against
the observed maximum token count, because one generation ran to 8,574. Against its
declared 8,192 budget it is **8.0%**: 15 items stacked at 8,189 to 8,193, with the next
longest natural generation at 1,878. This is very likely why the truncation sweep passed
over it.

---

## 6. Open items

- **Qwen3.5-35B, free routing, IFEval, at 16384.** The one arm the truncation sweep
  missed. Both constrained arms were rerun and are clean (2.0% and 1.0%), but without a
  matched free arm the whole cell falls back to 8,192, where the free arm is 8.0%
  truncated. This is the single number holding Qwen3.5's thinking-on mean at −7.0, so it
  decides whether thinking amplification holds for that model. One cell, base weights only.

- **WritingBench truncation, never swept.** At its 4,096 budget: gpt-oss-120b 30 to 36%,
  gpt-oss-20b 20 to 25%, LFM 21 to 27%, Qwen3.5 6 to 9%, gemma4 clean. The sweep covered
  HumanEval, MMLU and IFEval and never looked here. Section 6 of the paper leans on
  WritingBench for "prose is the robust surface", and budget-reaching essays lose 0.42
  critic points against 0.08 for normal-length ones. Both arms truncate at similar rates
  so the paired delta may survive, but it is untested.

- **Qwen3.5 thinking-on adaptation.** Needed to extend section 4 beyond gemma4. Gated on
  the adapter rebuild, since the pod deletion took both adapters and the prompt pool has no
  committed builder.

- **Two gemma4 thinking-on cells** sit at 6.5% (IFEval) and 6.1% (MMLU) cap-hit, left as a
  judgment call rather than chased.

---

## 7. Where things live

| what | where |
|---|---|
| authoritative grid rows | `results/ablations/instruct_genbench_vllm.csv` |
| fair-budget re-runs | `results/ablations/screening_genbench.csv` (`*_cap8k`, `*_cap16k`) |
| per-cell damage, budget, cap-hit | `results/ablations/think_ablation_summary.csv` |
| per-item generation dumps | `results/ablations/genbench_samples/` |
| WritingBench per-item scores and lengths | `results/ablations/writingbench/` |
| budget resolution across arms | `resolve()` in `analysis/residency/think_analysis.py` |
| length accounting | `lengths()` in `analysis/residency/length_figs.py`, rule in `genprotocol.py` |
| length figures | `analysis/residency/length_figs.py` |
| damage figures | `analysis/residency/plot_instruct.py` |
