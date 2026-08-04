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

OLMoE's profile is **U-shaped**: layers 0-2 and 14-15 both elevated, layer 1 the single worst at 1.99x
uniform, middle third at 0.54-0.64x. Qwen3.5's is not U-shaped. It is **late-heavy**: layers 0-2
(78, 114, 77) are indistinguishable from the middle third (mean 45), while layers 32-39 rise steeply
and layer 39 alone is **970** — 12x the first layer and 21x the middle.

| | first 2 + last 2 | middle third | ratio |
|---|---|---|---|
| R=8 | +0.00373 | +0.00045 | 8.25x |
| R=32 | +0.00243 | +0.00016 | 14.75x |

That ratio is carried almost entirely by the last two layers. **The `{0,1,L-2,L-1}` recipe inherited
from OLMoE spends half its budget on layers that cost nothing here** — freeing L0 and L1 buys
0.00078 + 0.00114 while L38 and L39 buy 0.00328 + 0.00970. A Qwen-shaped recipe would free the tail,
not both ends.

The caution from the OLMoE work applies unchanged and is the reason this is not stated as a
prescription: **solo damage does not predict joint value** (section 3 of the layer-freeing results
records three independent contradictions, one of them controlled). What transfers is the *finding
that the profile is not flat*; the specific shape does not.

## 3. Damage is superadditive, mildly

Sum of the 40 solo damages vs constraining all 40 at once: +0.04374 vs +0.05480 at R=8 (**1.25x**),
+0.02263 vs +0.02993 at R=32 (**1.32x**). Layers interact, but weakly — most of the cost is already
visible one layer at a time, unlike the free-set interactions seen on OLMoE.

Producer: `analysis/ple/qwen_sweep.py`. Data: `qwen35_residency_suite.csv` (84 cells).
