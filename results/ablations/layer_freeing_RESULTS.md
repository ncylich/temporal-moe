# Per-layer residency relaxation — results

Numbers: `layer_freeing_results.csv` (tidy, sliceable by its `group` column). Code:
[`analysis/ple/`](../../analysis/ple/README.md) — `layer_ablation.py`, `joint_free.py`, and
`train_ple.py --free-set`.

**This is a separate line of inquiry from per-layer embeddings** ([`ple_RESULTS.md`](ple_RESULTS.md)).
PLE *adds* a token-indexed lookup while leaving the residency constraint intact. Layer freeing
*removes* the constraint from chosen layers and adds nothing. They share a base model, an eval slice
and a set of published references, and little else.

**Metric.** BPB = cross-entropy nats ÷ 3.10891 on the Stage-1 audited held-out slice, divisor
byte-derived. **Lower is better.** `recovery = 1 − (BPB − 0.6727)/(2.7507 − 0.6727)`: 0% is rolling
residency `R=k=8` of 64 imposed untrained, 100% is the base model with free routing.
**2σ = 0.012 BPB.**

**What "freeing" costs.** A freed layer must keep all 64 of its experts resident instead of 8. FLOPs
are unchanged — both regimes activate exactly top-8 of 64 per token, and residency only restricts
*which* eight are eligible. So this trades **serving memory only**, which is precisely the currency
the temporal-MoE thesis claims to save.

| free set | resident expert slots | vs full residency |
|---|---|---|
| none | 128 | — |
| 1 layer | 184 | +43.8% |
| 2 layers | 240 | +87.5% |
| 3 layers | 296 | +131.2% |
| 4 layers | 352 | +175.0% |

---

## 1. Headline

**Freeing MoE layers 0, 1 and 15 beats the full finetune.** The CE recipe (router + norm gains +
LoRA r32, no PLE) at 50M tokens reaches **0.797810 (93.98%)**, beating F′ = 0.8106 — a finetune of
all 6.92B parameters — by 1.07σ. It is the strongest quality in the adaptation line. The price is
**+131% resident expert memory**.

It does not reach 95% recovery (BPB 0.7766), falling 1.77σ short.

| cell | BPB | recovery | free set | memory |
|---|---|---|---|---|
| **ce_free_0_1_15** | **0.797810** | **93.98%** | 0, 1, 15 | +131.2% |
| ce_free2 | 0.814440 | 93.18% | 0, 1 | +87.5% |
| *CE@50M, full residency* | *0.826900* | *92.58%* | — | baseline |
| *F′ full finetune 6.92B* | *0.810600* | *93.36%* | — | baseline |

Both cells beat the full-residency CE comparator, `ce_free2` marginally (1.04σ) and
`ce_free_0_1_15` clearly (2.42σ).

## 2. Per-layer damage

Constraining exactly one MoE layer at a time, base model, no training. Both anchors reproduce
published values to six decimals (all-free 0.672736 vs 0.6727; all-constrained 2.750704 vs 2.7507),
which is what licenses the sixteen numbers between them.

Damage is **U-shaped**, not decaying with depth: layers 0–2 and 14–15 sit at or above the uniform
share of 0.1299, the middle third (7–13) at 0.54–0.64× it. Layer **1 is the single most damaging** at
1.99× uniform, nearly double layer 15.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| damage | 0.2178 | **0.2588** | 0.1408 | 0.1153 | 0.1168 | 0.0986 | 0.1064 | 0.0792 |

| layer | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|
| damage | 0.0810 | 0.0822 | 0.0837 | 0.0698 | 0.0729 | 0.0748 | 0.1225 | 0.1408 |

Figure: [`results/phase0/figures/ple_layer_damage.png`](../phase0/figures/ple_layer_damage.png).

The shape is cleaner than "early layers matter": the constraint hurts most where routing is closest
to the token — at the input, and again at the output where the next-token decision is formed.

## 3. Solo damage does not predict joint value

Freeing a *subset*, measured directly against the additive prediction from the solo profile:

| free set | damage | additive pred. | interaction | memory | recovered |
|---|---|---|---|---|---|
| {0} | 1.931151 | 1.860135 | +0.071 | +43.8% | 0.147 |
| {1} | 1.913915 | 1.819152 | +0.095 | +43.8% | 0.164 |
| {0,1} | 1.702781 | 1.601319 | +0.101 | +87.5% | 0.375 |
| **{0,1,2}** | **1.504779** | 1.460475 | +0.044 | +131.2% | **0.573** |
| {0,1,15} | 1.669080 | 1.460562 | +0.209 | +131.2% | 0.409 |
| {0,1,14,15} | 1.582700 | 1.338077 | +0.245 | +175.0% | 0.495 |

**Layers 2 and 15 have near-identical solo damage (0.14084, 0.14076) and therefore near-identical
additive predictions (1.460475, 1.460562) — yet freeing them alongside {0,1} recovers 0.573 versus
0.409.** The third layer is worth **0.198 if it is layer 2 and 0.034 if it is layer 15**, a factor of
5.8 at identical memory cost. Layer 15's damage largely *overlaps* what freeing 0–1 already repairs;
layer 2's is comparatively independent. No single-layer profile can show this.

{0,1,2} also dominates {0,1,14,15} outright — more recovery for less memory. A contiguous early
block beats splitting across both ends.

Across the whole network the constraint is mildly **super-additive** (singles sum 1.861 against a
full damage of 2.078, ratio 0.896), but within any freed subset every interaction term is
**positive**: freeing a set always recovers *less* than the profile predicts. Network-level
super-additivity does not imply it within a subset.

## 4. Training-free ordering did not survive adaptation

The one case tested came out the wrong way round. Layer 15 was predicted to add 0.034 of recovery on
top of {0,1}; trained, it added **0.0166 BPB — more than layers 0–1 contributed together (0.0125)**.

| step | trained gain | training-free recovery |
|---|---|---|
| full residency → {0,1} | 0.0125 | 0.375 |
| {0,1} → {0,1,15} | **0.0166** | **0.034** |

So adaptation compresses the first two layers' large training-free advantage almost to nothing, then
*amplifies* the third layer's small one. I have no explanation for the asymmetry and will not
construct one from a single cell.

**Unresolved.** The controlled `{0,1,2}` cell — identical recipe, budget and memory, differing only
in whether the third freed layer is 2 or 15 — was cancelled before completing. Without it I cannot
say whether the damage profile is a useful design tool or a misleading one. On present evidence it
does **not** predict trained outcomes, and configurations should not be chosen from solo damage
alone.

## 5. Diminishing returns

`ce_free_0_1_15` per 10M: −0.0114, −0.0035, −0.0032, −0.0014. `ce_free2`: −0.0139, −0.0063, −0.0029,
−0.0018. Both essentially flat by 40M, so more tokens at either configuration will not close the
remaining gap to 95%.

## 6. Reading

This is a genuine quality/memory frontier point and the strongest quality the adaptation line has
produced — F′-level from a 235M-parameter LoRA recipe at 50M tokens. But it concedes the paper's
central claim that serving memory tracks active parameters, and by a large factor.

**If a 95% target is real, R is the likelier lever than layer count.** The residency-dose curve is
smooth in R, whereas free-versus-constrained is an all-or-nothing jump to R=64 on a layer — the most
expensive point on that curve. R=16 on layers 0–1 would cost roughly +9% resident memory against
+87.5%; whether it captures a useful share of the gain is untested.
