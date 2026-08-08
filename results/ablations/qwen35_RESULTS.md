# Qwen3.5-35B-A3B-Base under rolling residency — test-time suite

Second model in the layer-freeing program, and the first at a different scale. Everything below is
test-time only: no training, no adaptation, so none of it depends on this checkpoint being base or
instruct. All 84 cells come from one model load replaying identical cached batches, so they are
matched on input by construction.

**Model.** 35B total / 3B active, 40 layers, 256 routed experts + 1 always-resident shared expert,
top-k 8. Attention is hybrid: 3 Gated DeltaNet layers per 1 full-attention layer.

**Data.** The audited held-out slice, decoded from its OLMoE token ids and re-tokenized with Qwen's
tokenizer, so both models score **byte-identical text**. The decode is verified exact by re-encoding.
Divisor 3.1642666 (ln2 x 4.5651 bytes/token) against OLMoE's 3.1089071. 16 sequences x 4096 tokens.

## Preflight — what licenses the rest

| check | result |
|---|---|
| residency disabled vs stock router | max abs delta-logit **0.000e+00** |
| R = E (resident set is everything) vs stock | max abs delta-logit **0.000e+00** |
| resident experts per token | min 8, max 8 |

The hook is bitwise inert when it should be. Every difference below is therefore the constraint.

## 1. The constraint is nearly free at this scale

| cell | resident (routed) | BPB | damage |
|---|---|---|---|
| free baseline | 100% | **0.625133** | — |
| all 40 layers constrained, R=8 | 3.12% | 0.679937 | **+0.054804** |
| all 40 layers constrained, R=32 | 12.50% | 0.655061 | **+0.029928** |
| free {0,1,38,39}, R=8 | 3.12% | 0.660238 | +0.035105 |

The comparison that matters is against OLMoE at the **same resident fraction**. OLMoE runs R=k=8 of
64 = 12.5% and pays **2.078 BPB** (0.6727 -> 2.7507). Qwen3.5 at 12.5% resident pays **0.0299**.

> **A 69x smaller price for the same constraint, on a model 5x larger.** At matched absolute R=8 it
> is 38x. Residency gets dramatically *cheaper* with scale, which is the opposite of the pessimistic
> prior that more experts means more to lose.

Two mechanisms could produce this and they imply different things: Qwen's always-resident shared
expert gives every token an intact path regardless of the resident set, which OLMoE has no analogue
for; or 256 experts at top-8 simply leaves more substitutable capacity than 64 at top-8. The first
makes the result architecture-specific, the second makes it general.
`qwen_shared_ablation.py` separates them and has not yet been run.

## 2. The damage profile does NOT transfer — and the recipe is half wrong here

Per-layer damage, constraining exactly one layer and leaving the other 39 free (R=8, x1e-5):

| layers 0-9 | 78 | 114 | 77 | 32 | 85 | 61 | 73 | 67 | 15 | 42 |
|---|---|---|---|---|---|---|---|---|---|---|
| **10-19** | 74 | 103 | 56 | 50 | 44 | 23 | 24 | 52 | 31 | 23 |
| **20-29** | 15 | 28 | 48 | 68 | 37 | 22 | 3 | 43 | 62 | 68 |
| **30-39** | 72 | 82 | **246** | **315** | **335** | 123 | **183** | **201** | **328** | **970** |

OLMoE's profile is U-shaped: both ends elevated relative to the middle. Qwen3.5's is **also**
U-shaped -- but the weight has moved to the tail. Measured against each model's central half:

| | middle-half mean | first 2 layers | last 2 layers | front/back |
|---|---|---|---|---|
| OLMoE (16 layers) | 0.08971 | **2.66x** | 1.47x | 1.81 — front-heavy |
| Qwen3.5 (40 layers) | 0.00044 | **2.21x** | **14.87x** | 0.15 — back-heavy |

So the *direction* transfers and the *balance* inverts. Freeing the ends is right for both models;
on OLMoE most of the value is at the input end, on Qwen3.5 it is overwhelmingly at the output end,
where layer 39 alone costs 970e-5 against a middle-half mean of 44e-5.

> **Correction.** An earlier version of this section claimed Qwen's first layers were
> "indistinguishable from the middle third" and that the `{0,1,L-2,L-1}` recipe therefore "spends
> half its budget on layers that cost nothing here". That was wrong, and wrong for a specific
> reason: the middle window used was layers 8-31, which includes the tail's rise from layer 32
> onward and so inflated the baseline the front was compared against. Against a clean central half
> the first two layers are 2.21x the middle -- close to OLMoE's 2.66x. The recipe transfers; it is
> simply unbalanced at this scale, not half-wasted.

Figure: [`results/phase0/figures/residency_profile_transfer.png`](../phase0/figures/residency_profile_transfer.png),
both profiles normalised by their own mean against relative depth, since the absolute damages differ
by ~70x and the question is shape.

The caution from the OLMoE work applies unchanged and is why none of this is stated as a
prescription: **solo damage does not predict joint value** (section 3 of the layer-freeing results
records three independent contradictions, one of them controlled).

## 3. Damage is superadditive, mildly

Sum of the 40 solo damages vs constraining all 40 at once: +0.04374 vs +0.05480 at R=8 (**1.25x**),
+0.02263 vs +0.02993 at R=32 (**1.32x**). Layers interact, but weakly — most of the cost is already
visible one layer at a time, unlike the free-set interactions seen on OLMoE.

Producer: `analysis/ple/qwen_sweep.py`. Data: `qwen35_residency_suite.csv` (84 cells).


## 4. Adaptation was attempted and abandoned — throughput, not a result

Three matched training arms were set up (constrained; ends-freed + attention LoRA; unconstrained
null), on 67.8M Qwen-tokenized training tokens disjoint from the eval slice, with 461M LoRA
parameters and 8-bit Adam fitting comfortably in 73.4 GB. The arms ran, and then were stopped.

**Measured throughput: 176 s/step, or 93 tok/s.** The same model evaluates at 4700 tok/s. At that

> **This figure was later misused and the misuse is withdrawn.** 93 tok/s is correct for what it
> measures: Qwen3.5 (40 layers x 256 experts) carrying 461M of EXPERT LoRA through
> `_experts_forward_lora`'s Python loop at micro-batch 1. `bench_train_fused.py` then cited it as the
> "stock" baseline for a Qwen3-30B benchmark using attention-only LoRA and the stock expert forward,
> yielding a 22.7x and a 65x that do not exist. Stock in the configuration actually trained ran at
> **6,274 tok/s** (the 50M Qwen3-30B arm, 49,987,584 tokens in 132.8 min). Do not use this number as
> a baseline for any other configuration. See `analysis/ple/results/ablations/crossmodel_RESULTS.md S9` §1.
rate the 30M-token budget needs 90 hours, and the 100-minute per-arm cap would have bought 0.6M
tokens -- far too few to move BPB detectably, let alone to separate three arms.

The cause is structural rather than a misconfiguration. `Qwen3_5MoeExperts.forward` iterates in
Python over every hit expert; with 40 layers and 256 experts that is up to 10240 small GEMMs per
forward. The per-expert LoRA branch adds two more linears inside that loop, and gradient
checkpointing recomputes the whole thing on the backward pass. OLMoE, with 64 experts over 16
layers, trains at 11300 tok/s -- a 120x gap that the 3x difference in active parameters does not
come close to explaining.

So the honest position is: **this document contains no Qwen adaptation result**, and the OLMoE
recovery numbers have no Qwen counterpart yet. Making one needs a batched expert kernel (grouped GEMM
over the hit set rather than a Python loop), which is a real piece of engineering and not something
to attempt against a deadline. What the night bought instead is the test-time characterisation above
and in section 5, which is what the scaling question actually asked and which does not depend on this
checkpoint being base or instruct.


## 5. The price of residency, the right free set, and where the advantage comes from

One model load, 30 cells, identical cached batches. Free baseline on this slice is 0.648087 BPB
(24 sequences; section 1 used 16, hence the different absolute anchor -- damages within each table
are internally matched).

### A. Price as a function of resident budget

| R | resident (routed) | BPB | damage |
|---|---|---|---|
| 8 | 3.12% | 0.703758 | +0.055671 |
| 16 | 6.25% | 0.686977 | +0.038890 |
| 32 | 12.50% | 0.678965 | +0.030878 |
| 64 | 25.00% | 0.670946 | +0.022859 |
| 128 | 50.00% | 0.663396 | +0.015309 |

Damage falls roughly as R^-0.59 -- a 32x increase in resident budget buys only a 7.6x reduction in
damage, and even holding **half** the experts resident still costs 0.0153. The knee is early: R=8 is
already within 3.6x of R=128 while holding a sixteenth as much. For serving, that is the useful
shape -- most of the recoverable loss is recovered by the first few resident slots.

### B. Which layers to free, measured jointly at matched budget

| free set | layers freed | damage (R=8) | of the constraint recovered |
|---|---|---|---|
| none | 0 | +0.055671 | — |
| first 2 | 2 | +0.052592 | 5.5% |
| **last 2** | **2** | **+0.038181** | **31.4%** |
| first2 + last2 (OLMoE recipe) | 4 | +0.035294 | 36.6% |
| **last 4** | **4** | **+0.032781** | **41.1%** |
| first2 + last4 | 6 | +0.030022 | 46.1% |
| last 8 | 8 | +0.019903 | 64.2% |

> **At matched budget the tail-only set beats the recipe inherited from OLMoE.** Four freed layers:
> last-4 recovers 41.1% against the {0,1,38,39} recipe's 36.6%. Two freed layers: last-2 recovers
> 31.4% against first-2's 5.5% -- nearly six times as much for the same serving cost. Freeing the
> input end, which is where most of OLMoE's value sat, is close to worthless here.

The same ordering holds at R=32 (last-4 +0.016147 vs first2+last2 +0.017180), so it is not an artefact
of the harsher budget.

This is worth flagging as a methodological exception. On OLMoE, solo damage **did not** predict joint
value -- three contradictions, one controlled (section 3 of the layer-freeing results). Here it does:
the per-layer profile said back-heavy, and the jointly-measured free sets agree. Solo damage is not
reliable in general; it happened to be right on this model, and the joint measurement is what
establishes that, not the profile.

### C. Where the 70x advantage comes from — mostly not the shared expert

| shared expert | free | constrained (R=8) | damage |
|---|---|---|---|
| live | 0.648087 | 0.703758 | +0.055671 |
| zeroed | 1.161288 | 1.344755 | +0.183467 |

**Ratio 3.30x.** Removing Qwen's always-resident shared expert makes residency 3.3x more damaging, so
it genuinely does absorb part of the constraint -- but only part. With the shared expert gone
entirely, Qwen still pays **0.183** where OLMoE pays **2.078**, and it pays it at 3.12% resident
against OLMoE's 12.5%. That is still an **11x** advantage at a 4x harsher budget.

So the answer is mixed, and the mixture is the interesting part: roughly a factor of 3 of Qwen's
robustness is architecture-specific and would not transfer to a model without a shared expert, while
the remaining order of magnitude is not, and is most plausibly expert redundancy -- 256 experts at
top-8 leaves far more substitutable capacity than 64 at top-8. The prediction that follows is that
residency keeps getting cheaper as expert count grows, with or without a shared expert. Kimi K3's
896 experts would be the test.

Producer: `analysis/ple/qwen_cost_curve.py`. Data: `qwen35_cost_curve.csv` (30 cells).


## 6. The free-set claim, with error bars

Section 5B rests on a 0.0025 BPB gap out of a 0.056 constraint, which is small enough that it needed
an error bar before anyone acted on it. Each set was rescored on **three disjoint 32-sequence blocks**,
with damage measured against a per-block free baseline so block difficulty cancels.

| free set | mean damage | sd across blocks |
|---|---|---|
| none | +0.057606 | 0.001743 |
| first 2 | +0.054157 | 0.001513 |
| last 2 | +0.040167 | 0.001647 |
| first2 + last2 | +0.036733 | 0.001502 |
| last 4 | +0.034443 | 0.001805 |
| last 8 | +0.021150 | 0.001933 |

Block-to-block spread is ~0.0017, which on its own is comparable to the effect being claimed. The
paired per-block difference is the sensitive test, because difficulty is common to both arms:

| comparison (same budget) | per-block differences | mean +- sd | sign consistent |
|---|---|---|---|
| last-4 minus first2+last2 | -0.002370, -0.002564, -0.001936 | **-0.002290 +- 0.000321** | 3/3 |
| last-2 minus first-2 | -0.014169, -0.013962, -0.013839 | **-0.013990 +- 0.000166** | 3/3 |

Pairing tightens the standard deviation about 5x relative to the raw damages. Both effects are large
against their own spread -- roughly 7 sigma and 84 sigma -- and both hold in the same direction on
every block. **The tail-only free set is genuinely better than the recipe inherited from OLMoE at
matched budget, and it is not close for the two-layer case.**

Producer: `analysis/ple/qwen_freeset_precision.py`. Data: `qwen35_freeset_precision.csv`.


## 7. Qwen3-30B-A3B-Base: the redundancy control settles the mechanism

Qwen3.5 pays ~70x less than OLMoE for the same residency rule, and zeroing its always-resident shared
expert accounted for only 3.3x of that. Qwen3-30B-A3B-Base decides the rest: **128 experts, top-8
(identical), no shared expert at all, standard attention**. Preflight passed on this family
independently -- residency-off and R=E both 0.000e+00 against stock, exactly 8 resident per token.

### Damage at matched resident fraction (12.5%)

| model | experts | shared expert | damage |
|---|---|---|---|
| OLMoE 1B-7B | 64 | no | **+2.078** |
| Qwen3-30B-A3B | 128 | **no** | **+0.046964** |
| Qwen3.5-35B-A3B | 256 | yes | **+0.029928** |

> **The shared expert is not the explanation.** Qwen3-30B has none and is still ~44x cheaper than
> OLMoE. Damage falls monotonically as expert count doubles -- 64 -> 128 -> 256 gives 2.078 -> 0.047
> -> 0.030 -- which is what expert redundancy predicts and what an architecture-specific buffer does
> not. The prediction that residency keeps getting cheaper with expert count now has two points
> supporting it and one confound removed.

The caveat that keeps this honest: OLMoE and the Qwen models differ in far more than expert count --
pretraining data, depth, total parameters, overall quality. The clean part of the comparison is that
a model with **no** shared expert reproduces most of the advantage, which is what the ablation in
section 5C could not establish on its own.

### Cost curve, and an internal check

| R | resident | damage |
|---|---|---|
| 8 | 6.25% | +0.104836 |
| 16 | 12.50% | +0.046964 |
| 32 | 25.00% | +0.019537 |
| 64 | 50.00% | +0.005374 |
| 128 | 100.00% | **+0.000000** |

R=128 is every expert resident, and damage is exactly zero to six decimals. That is the R=E no-op
verified at full measurement scale rather than only in the preflight's single forward.

### The free-set recipe does NOT transfer between the two Qwen models

| free set (4-layer budget) | Qwen3-30B | Qwen3.5-35B |
|---|---|---|
| first2 + last2 | **45.9%** | 36.6% |
| last 4 | 41.8% | **41.1%** |

**Section 5B's conclusion is model-specific.** On Qwen3.5 the tail-only set wins; on Qwen3-30B the
OLMoE-style both-ends set wins. What survives across all three models is weaker and more robust: the
last layers are worth far more than the first (last2 vs first2 is 36.0% vs 8.3% here, 31.4% vs 5.5%
on Qwen3.5). What does not survive is the stronger claim that extending the tail beats adding the
head.

This run used 8 sequences, cut down to fit a deadline, where the Qwen3.5 free-set ordering was
verified on three disjoint 32-sequence blocks with paired differences. The flip is therefore
suggestive and not yet established to the same standard; it needs the same treatment before anyone
picks a free set from it.

Producer: `analysis/ple/qwen_cost_curve.py --family qwen3`. Data: `qwen3_30b_cost_curve.csv`.


## 8. Residency is free at inference -- and the slowness was never residency

Every throughput number reported earlier in this program was measured with the hook installed at
batch 1, with no stock reference, so none of them could distinguish HuggingFace's MoE implementation
from our machinery. Measured properly on Qwen3-30B (seq 512, bf16, one H100, one model load):

| variant | bs=8 | bs=16 | bs=32 | bs=64 |
|---|---|---|---|---|
| stock (HF as shipped) | 13,369 | 17,770 | 21,495 | **23,540 tok/s** |
| hook, all layers free | 0.99x | 1.01x | 1.00x | **1.00x** |
| hook, residency R=8 | **1.06x** | 1.04x | 1.01x | **1.00x** |

> **Rolling residency costs nothing in throughput.** The machinery is 1.00x when inert, and running
> the constraint is 1.00-1.06x -- at smaller batches it is *faster*, because with 8 of 128 experts
> resident the union of experts touched across a batch is smaller and the expert loop runs fewer
> iterations. The technique buys serving memory at no speed cost, which is a stronger claim than the
> memory result alone.

What was actually slow, in order of magnitude:

| cause | cost |
|---|---|
| running at batch 1-4 instead of 32-64 | **8.3x** (2,847 -> 23,540 tok/s) |
| an "optimisation" of ours that was a regression | **3x** (see below) |
| loading weights over network storage rather than RAM | 11 min -> 54 s per load |

### A documented negative result

`_experts_forward_fast` hoists the expert hit-list to host once, on the reasoning that
`for expert_idx in expert_hit:` over a CUDA tensor costs a device-to-host copy per expert per layer
-- roughly 6,000 stalls per forward at 128 experts over 48 layers. It benchmarks at **0.35x stock**
at batch 1-4 and 0.51x at batch 16: three times *slower*. The `.tolist()` is a pipeline barrier,
since nothing can be queued until one_hot/sum/nonzero have completed, so the CPU cannot run ahead;
stock's per-iteration syncs happen after kernels are already in flight and cost far less than
serialising the launch stream. It is kept in the tree, defaulted off and labelled, because it is the
obvious fix to an obvious-looking bottleneck and the next reader will have the same idea.

The general lesson is the one this section exists to record: the hypothesis was formed by reading
code, the fix was written and deployed into running jobs, and it was never measured against a
baseline that had never been established. Two rounds of tuning were spent before the first
measurement.

Producer: `analysis/ple/bench_inference.py`. Data: `qwen_inference_bench.csv`.


## 9. Expert kernels: no win available, and two self-inflicted measurement errors

transformers ships an `ExpertsInterface` registry (`grouped_mm`, `batched_mm`, `deepgemm`,
`sonicmoe`) chosen by `config._experts_implementation`, which Qwen leaves unset. That looked like a
free 14x: at batch 64 the expert path reaches ~7% of the H100's bf16 peak. It is not.

| implementation | tok/s | vs stock | max abs delta-logit | BPB delta | top-1 agreement |
|---|---|---|---|---|---|
| stock (untouched) | 18,011 | 1.000 | — | — | — |
| grouped_mm | 18,086 | **1.004** | 3.391 | **-0.000493** | **93.16%** |
| batched_mm | — | — | — | — | tried to allocate 384 GiB |
| deepgemm / sonicmoe | — | — | — | — | packages not installed |

**`grouped_mm` is rejected twice over.** It is 1.004x -- no speedup at all -- and it changes what the
model computes: the top-1 token differs at 7% of positions and BPB moves by 4.93e-04. That last
figure is the reason this matters rather than being a footnote. The aux-loss correction in section 4
was 4.85e-04 and free-set differences in section 5B are ~2.5e-03, so this kernel's error is the same
size as the effects the program measures. Adopting it for speed would have silently corrupted every
subsequent number.

The harness was validated before the candidate was judged: stock against stock gives max abs
delta-logit exactly 0.000e+00 and identical BPB to six decimals, so the disagreement is the kernel.

### Two measurement errors, both the same mistake

**A 3x regression.** `_experts_forward_fast` hoists the expert hit-list to host to remove ~6k
per-forward device-to-host stalls. It measures 0.35x stock: the `.tolist()` is a pipeline barrier, so
the CPU cannot run ahead, whereas stock's syncs land after kernels are already in flight.

**A phantom 2.69x speedup.** Setting `_experts_implementation = None` explicitly -- intended as "use
eager" -- drops throughput from 18,011 to 6,770 tok/s, because `ExpertsInterface.get_interface`
warns on every expert module call, 48 times per forward. Measured against that crippled baseline,
`grouped_mm` appeared to be 2.69x. Against stock it is 1.004x.

Both errors have the same shape: numbers compared across two harnesses instead of variants measured
side by side in one process. The fix is procedural rather than technical, and both are recorded here
because the speedups were reported before they were checked.

**What actually made things faster** was unglamorous: batch size (2,847 -> 23,540 tok/s from batch 1
to 64, 8.3x) and staging weights in RAM rather than network storage (11 min -> 54 s per load). No
kernel-level win is available in this stack without installing deepgemm/sonicmoe or writing a correct
grouped GEMM.

Producer: `analysis/ple/bench_experts.py`, `analysis/ple/check_grouped_mm.py`.
Data: `qwen_expert_kernels.csv`.


## 10. Naive imposition survives on a 128-expert model -- the deployability result

BPB is a likelihood and can look healthy while a model has lost the ability to answer anything. The
question that decides whether residency is usable is what happens to tasks when the constraint is
switched on with **no adaptation at all**. On OLMoE the answer is catastrophic, which is why the
whole adaptation programme exists. On Qwen3-30B it is not.

Ten 0-shot tasks, full sets (78,459 scored continuations per arm), same harness and metric convention
as the era table `olmoe_adapt_downstream.csv` (now archived: `../archive/olmoe_wrong_renorm/`).
Retention is imposed/free within each model, so it controls for
Qwen simply being the stronger model.

| task | metric | OLMoE 64e @12.5% resident | Qwen3-30B 128e @6.25% resident |
|---|---|---|---|
| arc_easy | acc | 0.7715 -> 0.2799 (**36.3%**) | 0.7980 -> 0.7088 (**88.8%**) |
| arc_challenge | acc | 0.4701 -> 0.2150 (45.7%) | 0.5444 -> 0.4411 (81.0%) |
| hellaswag | acc_norm | 0.7822 -> 0.2657 (34.0%) | 0.8131 -> 0.7508 (92.3%) |
| piqa | acc | 0.7873 -> 0.5180 (65.8%) | 0.8145 -> 0.7388 (90.7%) |
| winogrande | acc | 0.6922 -> 0.4909 (70.9%) | 0.7238 -> 0.6140 (84.8%) |
| boolq | acc | 0.7018 -> 0.4037 (57.5%) | 0.8141 -> 0.6590 (80.9%) |
| sciq | acc | 0.9370 -> 0.2930 (31.3%) | 0.9630 -> 0.9210 (**95.6%**) |
| openbookqa | acc | 0.3220 -> 0.1460 (45.3%) | 0.3400 -> 0.2680 (78.8%) |
| lambada_openai | acc | 0.7056 -> **0.0000** (0.0%) | 0.7475 -> 0.5979 (80.0%) |
| copa | acc | 0.8500 -> 0.5600 (65.9%) | 0.9000 -> 0.8100 (90.0%) |
| **mean over 16 metrics** | | **45.8%** | **87.0%** |

> **A 128-expert model keeps 87% of its zero-shot capability with 6.25% of experts resident and no
> training whatsoever, where a 64-expert model keeps 45.8% at twice that residency budget.** OLMoE's
> `lambada_openai` goes to exactly zero -- it cannot predict a final word at all -- while Qwen3-30B
> retains 80% of it.

This is the claim that changes what can be deployed. The adaptation programme exists because
imposing residency on OLMoE destroys the model; on a model with more experts, most of the serving
win is available with no training run at all. Adaptation would then be an optimisation rather than a
prerequisite.

BPB moved 0.582025 -> 0.686861 (+0.1048) on the same arms, so the likelihood and the task numbers
agree in direction and were computed from the same forward configuration.

**What this does not establish.** OLMoE and Qwen3-30B differ in pretraining data, depth, parameter
count and quality as well as expert count, so "more experts causes the survival" is supported by the
cost-curve evidence in section 7 but not isolated by this table alone. The honest statement is that
the survival is a property of this model, and that expert count is the variable most strongly
associated with it across the three models measured.

Producer: `analysis/ple/qwen_downstream.py`. Data: `qwen3_30b_downstream_naive.csv`.


## 11. Correction: R < k is not a valid operating point

Sections 5A and 7 reported an R=4 row in each cost curve, labelled "1.56% resident" and "3.12%
resident". Those rows are withdrawn. Both models use top-k=8, and the router selects k experts from
the resident set: with R=4 only four experts carry non-zero probability, while `topk` still returns
eight indices and the model dispatches eight. Verified directly -- at E=128, k=8, R=4 the top-k
weights come back as `[0.4796, 0.2155, 0.1774, 0.1275, 0, 0, 0, 0]`.

So that configuration measures **degraded top-4 routing with four slots of wasted compute**, not
residency at a smaller budget. Reporting it as "R/E resident" implied a serving trade-off that was
not what ran.

**R = k is the tightest meaningful constraint** -- the setting the OLMoE runs used throughout, and
the one this program should have carried over. Every other row in both curves (R = 8, 16, 32, 64,
128) is at or above k and stands unchanged; the withdrawn rows were the extreme point of each curve,
so the R^-0.59 fit in section 5A is refitted on the valid points only.

`residency_qwen.assert_valid_R` now raises on R < k, and `qwen_cost_curve.py` skips those points, so
the configuration cannot be measured again by accident. The rows are annotated in the CSVs rather
than deleted, since a silently shorter table invites the same experiment a second time.


## 12. Three-model profile: a stable head, a wildly variable tail

With Qwen3-30B added, the per-layer damage profiles can be compared across 64, 128 and 256 experts.
Each is normalised by its own central-half mean, since absolute damages differ by ~70x.

| model | experts | first 2 / middle | last 2 / middle |
|---|---|---|---|
| OLMoE 1B-7B | 64 | **2.69x** | 1.49x |
| Qwen3-30B-A3B | 128 | **2.60x** | **28.08x** |
| Qwen3.5-35B-A3B | 256 | **2.89x** | 19.46x |

The input end is remarkably stable: 2.6-2.9x the middle on all three models, across a 4x range of
expert count and 16-to-48 layers. The output end is not stable at all -- 1.5x, 28x, 19x. So the
earlier framing of "both ends are elevated" understates the asymmetry. What generalises is a modest,
consistent premium on the first two layers; what varies by model is whether the tail dominates.

This also does **not** explain the free-set flip in section 7. Qwen3-30B has the most extreme tail of
the three (28x) and yet `{first2, last2}` beats `{last4}` there, while Qwen3.5 with a milder tail
(19x) prefers tail-only. A profile that says "the tail is 28x the middle" would predict the opposite.
That is the third independent instance in this program of solo per-layer damage failing to predict
joint free-set value, after the three recorded in the OLMoE work.

Figure: [`results/phase0/figures/residency_profile_transfer.png`](../phase0/figures/residency_profile_transfer.png).
Producer: `analysis/ple/plot_profile_transfer.py`.


## 13. CORRECTION — the cross-model comparison is confounded; ratios were ~10x too large

An independent audit of this codebase found a defect that invalidates the load-bearing comparison in
sections 1, 7, 10 and 12. It is recorded here in full because those sections were committed and
quoted before the check was made.

### The defect

Residency is imposed by masking non-resident experts to `-inf` before the router's softmax. Whether
that is a *clean* intervention depends on a per-model config flag:

| model | `norm_topk_prob` | what masking does |
|---|---|---|
| Qwen3-30B / Qwen3.5 | **True** | top-k weights are renormalised to sum to 1 in **both** arms, so masking only restricts *which* experts are eligible. Clean. |
| OLMoE-1B-7B | **False** | gate weights are the **raw** softmax-over-64 probabilities, which sum to ~0.40 for the top-8. After masking, the softmax is over the 8 residents and sums to **1.0**. |

So on OLMoE the constraint does not only change which experts serve -- it multiplies every MoE
block's output by roughly 2.5x, compounded across 16 layers. That is an activation-scale blow-up, not
a routing constraint, and Qwen structurally cannot suffer it.

### Magnitude, measured on BPB (OLMoE, R=8 of 64, no adaptation)

| arm | BPB | damage vs free |
|---|---|---|
| free | 0.7875 | — |
| residency **as implemented** (mask -> softmax over residents) | 2.8318 | **+2.0443** |
| residency, same resident sets, **stock gate values** | 0.9786 | **+0.1910** |

The as-implemented arm reproduces the published +2.078 to within 2%, confirming it is the same
intervention this program has been running. **About 91% of OLMoE's residency damage is the gate-mass
artifact, not residency.**

### What this changes

| claim as published | corrected |
|---|---|
| Qwen3.5 is **69x** cheaper than OLMoE at matched 12.5% resident | ~**4x** (0.0299 vs 0.191) |
| Qwen3-30B is **44x** cheaper at matched 12.5% resident | ~**4x** (0.0470 vs 0.191) |
| OLMoE retains ~0% of above-chance skill under naive imposition | **withdrawn** -- that collapse is the signature of an activation blow-up, and has not been re-measured with gate mass preserved |
| "expert count is the mechanism" (section 7) | **weakened, not refuted.** A 4x gap across 64 -> 128 -> 256 experts is still monotone and still favours more experts, but it is an ordinary effect rather than the order-of-magnitude one claimed |

**Qwen's own numbers are unaffected** -- its intervention is clean, so the cost curves, per-layer
profiles, free-set orderings and the 78.9% retention under naive imposition all stand. What is wrong
is every *cross-model ratio*, because the OLMoE comparator is inflated ~10x.

### Two further findings from the same audit

**Throughput (section 8) does not measure residency.** All experts remain in HBM in every arm, so the
benchmark measures the masking machinery, not the swap traffic that is the entire point -- at a
measured swap rate of 1.0 expert/token/layer that is ~450 MB/token of expert weight movement on
Qwen3-30B, which the benchmark never pays. Separately, the benchmark feeds uniform-random token ids,
and residency *reduces* the number of distinct experts hit per forward (64.0 -> 57.3 on real text,
56.1 -> 45.2 on random ids), which is why the constrained arm appeared 1.06x *faster*. The "0%
throughput cost" claim is withdrawn.

**Downstream is measured where the constraint is weakest.** The resident set cold-fills at position 0
with the exact top-R and holds no state across forward calls, so damage grows with position: on
OLMoE, +1.256 BPB at positions 0-8 rising to +2.118 at 512-1023. Zero-shot contexts here are ~20-150
tokens, so the task numbers run at roughly 55-75% of the long-context constraint strength. This
inflates retention for *both* models, so it affects the absolute 78.9% more than the comparison.
A corollary worth recording: with per-call state, **incremental decoding would impose no constraint
at all**, since every single-token step re-cold-fills to the exact top-R.

### The fix

Residency should change *which* experts serve, not how much they contribute. For a
`norm_topk_prob=False` model the mask must select the resident top-k while the gate values are taken
from the **unmasked** softmax. Until that is implemented and OLMoE re-measured, no cross-model ratio
from this program should be quoted.


## 14. Re-measurement under the fix: the artifact was 91.6%, and OLMoE's profile INVERTS

`olmoe_remeasure.py`, one model load, gate mass preserved. The `renorm` arm reproduces the published
+2.078 to within 4%, confirming it is the same intervention the program has been running.

### The artifact

| arm | BPB | damage |
|---|---|---|
| free | 0.670290 | — |
| constrained R=8, **as published** (`renorm`) | 2.671729 | **+2.001439** |
| constrained R=8, **gate mass preserved** | 0.839290 | **+0.169000** |

**91.6% of OLMoE's measured residency damage was the gate-mass artifact.**

### Corrected cross-model comparison, matched 12.5% resident

| model | experts | damage | published claim | corrected |
|---|---|---|---|---|
| OLMoE 1B-7B | 64 | +0.169000 | — | — |
| Qwen3-30B-A3B | 128 | +0.046964 | 44x cheaper | **3.6x** |
| Qwen3.5-35B-A3B | 256 | +0.029928 | 69x cheaper | **5.6x** |

The direction survives and is still monotone in expert count (0.169 -> 0.047 -> 0.030 across
64 -> 128 -> 256), so "residency gets cheaper with more experts" stands. Its magnitude was overstated
by roughly an order of magnitude.

OLMoE also now has a proper cost curve rather than one point: R=8 (12.5%) +0.169000, R=16 (25%)
+0.098104, R=32 (50%) +0.042913, R=64 (100%) +0.000000. The last row is the R=E no-op verified at
full measurement scale.

### The per-layer profile inverts

Section 2's U-shape was produced under the artifact. Re-run with gate mass preserved (damage x1e-3):

| layer | 0 | 1 | 2 | 5 | 9 | 10 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| published | 217.8 | **258.8** | 140.8 | 98.6 | 82.2 | 83.7 | 72.9 | 74.8 | 122.5 | 140.8 |
| corrected | 6.4 | 5.9 | 4.8 | 3.8 | 8.0 | 9.6 | 9.3 | 8.2 | 11.4 | **22.3** |

| | published | corrected |
|---|---|---|
| first 2 / middle | **2.66x** | **1.00x** |
| last 2 / middle | 1.47x | **2.71x** |
| most damaging | **L1, L0, L2**, L15, L14 | **L15, L14, L10, L12, L13** |

> **The profile flips from front-heavy to back-heavy.** Corrected, layers 0 and 1 sit exactly at the
> middle (1.04x and 0.96x) -- no premium at all -- while damage rises monotonically from L5 to L15,
> which alone is 3.6x the middle.

This matters beyond bookkeeping. **`{0,1,14,15}` was chosen partly because L0 and L1 measured as the
two most damaging layers.** On corrected data they are unremarkable, and a tail-weighted set --
`{12,13,14,15}` or `{14,15}` -- is what the profile now supports.

It also reconciles OLMoE with the Qwen models for the first time. Section 12 reported a "stable head
premium" of 2.6-2.9x across all three; that was 2.66x for OLMoE **only under the artifact**, and is
1.00x corrected. All three models are back-heavy; OLMoE simply has no head premium, where Qwen has a
modest one. The earlier framing was an artifact masquerading as a cross-model regularity.

Producer: `analysis/ple/olmoe_remeasure.py`. Data: `olmoe_gatemass_remeasure.csv`.
