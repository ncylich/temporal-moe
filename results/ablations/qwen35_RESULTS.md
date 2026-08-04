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
| 4 | 1.56% | 0.764537 | +0.116450 |
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
| 4 | 3.12% | +0.265620 |
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
