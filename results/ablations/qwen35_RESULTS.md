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
