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

**Superseded by §7 (2026-08-03): `{0,1,14,15}` at 250M reaches 0.781199 / 94.78%, and the best downstream score is `{0,1,14,15}`+attention. This section records the state as of the first round.**

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

Figure: [`results/phase0/figures/layer_freeing_damage.png`](../phase0/figures/layer_freeing_damage.png).

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

**Resolved, 2026-08-03: the damage profile is not a design tool.** The controlled `{0,1,2}` cell was
run — identical recipe, budget and memory to `{0,1,15}`, differing only in whether the third freed
layer is 2 or 15. Training-free, layer 2 was worth 0.198 of recovery against layer 15's 0.034, a
factor of 5.8 at identical cost. Trained, it is **worse**:

| third freed layer | training-free recovery | trained BPB | mean downstream acc |
|---|---|---|---|
| 15 (`ce_free_0_1_15`) | 0.034 | **0.797810** | **0.5959** |
| 2 (`ce_free_0_1_2`) | 0.198 | 0.808615 | 0.5855 |

The profile predicted a large win for layer 2 and delivered a 0.0108 BPB loss and the worst
downstream score of any free-set cell. Together with the `{0,1}`→`{0,1,15}` reversal above, that is
two independent contradictions, one of them controlled. **Do not choose free sets from solo damage.**

A third arrangement was also run: `{0,1,14,15}`, freeing both ends. Training-free it was *dominated*
by `{0,1,2}` — less recovery (0.495 vs 0.573) at more memory (+175.0% vs +131.2%) — and the write-up
above says `{0,1,2}` "dominates it outright". Trained, `{0,1,14,15}` is the best cell in the program
by a wide margin (§7). That is the third contradiction.

> **Caveat on this comparison, 2026-08-04 — the defect is fixed, these cells predate the fix.**
> Freed layers used a different aux formula from constrained ones, `E·Σ(P²)` against `E·Σ(f·P)`,
> which at the uniform optimum are 1 and *k*. Since the returned aux is the mean over all layers,
> freed layers diluted it in proportion to free-set size: 33.86 with none freed, 30.41 for `{0,1,2}`,
> 29.43 for `{0,1,15}`, 27.46 for `{0,1,14,15}`. Monotone along the axis the ladder varies, with the
> best-BPB rung least regularised, so free-set size and regularisation strength were confounded.
> `{0,1,2}` and `{0,1,15}` free the same *number* of layers and still differed by 2.9 points.
>
> `aux_z_from_router_logits` now uses one formula everywhere, matching this repo's own convention:
> mask the distribution, never the loss. Dilution is gone (+1.1% to +1.3% across the ladder), a freed
> layer is exactly HF's `load_balancing_loss_func` to 1.9e-06, and at R=k the change is provably a
> no-op (`checks.py auxparity`, difference 0.00e+00), so nothing *already* trained moves for that
> reason.
>
> **Tested 2026-08-04, and the conclusions survive.** Two 50M cells were run under the unified aux.
> `ce_auxfix_50M` (full residency, no freed layers, so the aux is provably unchanged) reproduces the
> published CE curve to 8e-4 at all five checkpoints, ending 0.827549 against 0.8269 — the refactor
> left the untouched path untouched. `ce_auxfix_free_attn_50M` is the identical configuration to
> `ce_free_0_1_14_15_attn`, differing only in the aux formula, and lands **0.784717 against 0.785201,
> a shift of 4.85e-4**. The freed layers' regularisation rose ~15x and moved the cell by less than
> five ten-thousandths of a BPB, against a ladder spread of 0.028165 — **58x larger**. The two
> data-seed twins in this document bound seed noise at 8.8e-4, so the shift sits below that floor.
>
> So the dilution was real, monotone in free-set size, and immaterial to every conclusion drawn from
> these cells. They remain trained under the old formula and are not bit-comparable to newer ones,
> but the 0.0108 BPB gap this section rests on is not in doubt from this cause.

## 5. Diminishing returns

`ce_free_0_1_15` per 10M: −0.0114, −0.0035, −0.0032, −0.0014. `ce_free2`: −0.0139, −0.0063, −0.0029,
−0.0018. Both essentially flat by 40M, so more tokens at either configuration will not close the
remaining gap to 95%.

## 6. Reading

This is a genuine quality/memory frontier point and the strongest quality the adaptation line has
produced — F′-level from a 235M-parameter LoRA recipe at 50M tokens. But it concedes the paper's
central claim that serving memory tracks active parameters, and by a large factor.

## 7. 2026-08-03: nine more cells, and the first downstream evaluation

Attention LoRA (`attn`) is on the projections q/k/v/o and is the one mechanism no previous arm of
this program had ever trained; see §8. The table is generated — its definitions are in the caption.

<!-- SUMMARY:BEGIN -->

*Generated by `analysis/ple/summary_table.py` from the per-cell JSONs and `layer_freeing_downstream.csv`. Do not edit by hand — run the script. 12 cells, 11 scored downstream.*

BPB lower is better; recovery is the share of the constraint's BPB damage undone (0% = residency imposed untrained at 2.7507, 100% = base model with free routing at 0.6727). Mean acc is over ten 0-shot tasks, higher better (base free routing 0.6823, imposed untrained 0.3164); `—` means not scored. Memory is extra resident expert slots against full residency. `attn` is the LoRA rank on the attention projections.

| cell | free set | memory | tokens | attn | seed | BPB | recovery | mean acc |
|---|---|---|---|---|---|---|---|---|
| `ce_free_0_1_14_15_250M` | 0,1,14,15 | +175.0% | 250M | — | 0 | 0.781199 | 94.78% | 0.6073 |
| `ce_free_0_1_14_15_attn_250M` | 0,1,14,15 | +175.0% | 250M | 32 | 0 | 0.783079 | 94.69% | 0.6097 |
| `ce_free_0_1_14_15_attn_ds1` | 0,1,14,15 | +175.0% | 50M | 32 | 1 | 0.784325 | 94.63% | 0.6081 |
| `ce_free_0_1_14_15_attn` | 0,1,14,15 | +175.0% | 50M | 32 | 0 | 0.785201 | 94.59% | 0.6081 |
| `ce_free_0_1_14_15` | 0,1,14,15 | +175.0% | 50M | — | 0 | 0.786275 | 94.53% | 0.5958 |
| `ce_free_0_1_15_250M` | 0,1,15 | +131.2% | 250M | — | 0 | 0.790846 | 94.31% | 0.6013 |
| `ce_free_0_1_15_200M` | 0,1,15 | +131.2% | 200M | — | 0 | 0.791767 | 94.27% | 0.6008 |
| `ce_free_0_1_15_attn` | 0,1,15 | +131.2% | 50M | 32 | 0 | 0.796195 | 94.06% | 0.5982 |
| `ce_free_0_1_15` | 0,1,15 | +131.2% | 50M | — | 0 | 0.797810 | 93.98% | 0.5959 |
| `ce_free_0_1_15_ds1` | 0,1,15 | +131.2% | 50M | — | 1 | 0.797814 | 93.98% | — |
| `ce_free_0_1_2` | 0,1,2 | +131.2% | 50M | — | 0 | 0.808615 | 93.46% | 0.5855 |
| `ce_free2` | 0,1 | +87.5% | 50M | — | 0 | 0.814440 | 93.18% | 0.5883 |
| *F' full 6.92B finetune* | — | *baseline* | *250M* | — | — | *0.810600* | *93.36%* | — |
| *CE-adapted, full residency* | — | *baseline* | *250M* | — | — | *0.814700* | *93.17%* | *0.5888* |

<!-- SUMMARY:END -->

**95% is not reachable this way.** `{0,1,14,15}` at 250M reaches 94.78%, and its curve is flat from
120M — the last 130M tokens bought 0.0015. The 95% bar (0.7766) is 0.0046 further. Freeing layers
and adding tokens does not get there; §6's suggestion that R, not layer count, is the remaining
lever is untested and now the obvious next thing.

**BPB overstates recovery.** The best model is 94.8% recovered on BPB and **79.5%** on downstream
accuracy. Any "we recovered N% of the constraint's cost" claim in this line is a BPB-space claim and
does not survive translation to task accuracy.

**BPB ranks coarse differences and not fine ones.** Across the nine scored cells, Pearson r between
BPB and mean accuracy is **−0.90** and Spearman ρ is **+0.83**, so BPB is a sound coarse instrument.
Every inversion, however, sits below ~0.012 BPB: the 0.0115 gap between `{0,1,14,15}` and `{0,1,15}`
at 50M produces a 0.0001 accuracy difference, and `ce_free_0_1_2` beats `ce_free2` by 0.0058 BPB
while losing to it on accuracy. That threshold is, ironically, this program's own pre-registered
2σ = 0.012 bar, which is far too wide as a statement about measurement precision and about right as
a statement about what transfers.

**Replicate spread is not one number.** Two same-configuration pairs on different corpus draws:
`ce_free_0_1_15` vs `_ds1` agree to **0.000004** BPB, `ce_free_0_1_14_15_attn` vs `_ds1` to
**0.000876**. Anything quoted as a noise floor should use the larger. On mean downstream accuracy the
second pair agrees to **0.000030**, because both models are scored on identical items.

## 8. Attention adaptation, and what it exposes about the metric

Every arm of this program adapts the router, the RMSNorm gains and the expert MLPs, and freezes
attention — including every cell above. F′ unfroze all 6.92B parameters, so attention was trainable
there, but no efficient arm had ever asked whether a small attention adapter contributes. Rank-32
LoRA on q/k/v/o is 8.4M parameters and costs **zero** resident expert memory, the currency this whole
line spends.

| free set | control | + attention | ΔBPB | Δacc |
|---|---|---|---|---|
| 0,1,14,15 | 0.786275 / 0.595840 | 0.785201 / 0.608130 | +0.0011 | **+0.01229** |
| 0,1,14,15 (seed 1) | 0.786275 / 0.595840 | 0.784325 / 0.608100 | +0.0020 | **+0.01226** |
| 0,1,15 | 0.797810 / 0.595880 | 0.796195 / 0.598240 | +0.0016 | +0.00236 |

**The `{0,1,14,15}` result replicates to 3e-5 on a different corpus draw.** Two independently trained
models agreeing that closely on a 0.0123 effect is not a fluctuation.

**And the intervention is nearly invisible to BPB.** Its BPB movement, 0.0011, is *inside* the
0.000876 replicate spread of that very cell — not an established effect — while its accuracy movement
is 400× its own spread. The metric this line selects on cannot see the intervention that most
improves what the line is for.

The benefit is free-set-dependent — 5× larger on `{0,1,14,15}` than on `{0,1,15}` — which is not
explained here and should not be explained from two free sets.

At 250M the picture changes again: attention ends 0.0019 **worse** on BPB (0.783079 vs 0.781199) and
0.0024 better on accuracy, inside noise. Attention front-loads: `{0,1,14,15}`+attention at **50M**
scores 0.6081, statistically level with the same configuration without attention at **250M**
(0.6073). Read as a token-efficiency result it is a 5× saving; read as a quality result at matched
budget it is a tie.

**Not done.** Attention with the expert LoRA switched off — 8.4M parameters against 235M — did not
fit before the session deadline. It asks whether the expensive adapter is the one doing the work.

**If a 95% target is real, R is the likelier lever than layer count.** The residency-dose curve is
smooth in R, whereas free-versus-constrained is an all-or-nothing jump to R=64 on a layer — the most
expensive point on that curve. R=16 on layers 0–1 would cost roughly +9% resident memory against
+87.5%; whether it captures a useful share of the gain is untested.


## Effective expert count: what residency does to routing, and what adaptation gives back

`eff_load = 1 / sum_e p_e^2` over the dispatch distribution, per layer, range 1..64. It counts how
many experts actually carry the corpus: 1 is total collapse onto one expert, 64 is perfect balance.
It is the quantity the auxiliary load-balancing loss exists to hold up, so it is the honest place to
look for damage when the aux changes. All four rows below score the **same** audited held-out slice
(4 x 4096 = 16384 tokens), so they differ only in the model state, not in the input.

| model state | median | min | max |
|---|---|---|---|
| stock OLMoE, no residency — the ceiling | 57.1 | 50.2 | 62.0 |
| stock OLMoE, residency R=k imposed, untrained | 20.5 | 13.6 | 52.5 |
| adapted 50M, full residency (`ce_auxfix_50M`) | 48.6 | 39.4 | 55.4 |
| adapted 50M, free {0,1,14,15} + attention LoRA | 50.9 | 45.6 | 61.8 |

Read top to bottom, this is the mechanism the BPB numbers only imply. Imposing residency on an
untrained model **collapses routing**: the median layer falls from 57 experts to 20, and the worst
layer to 13.6, because the resident set is chosen by a scan the router was never trained to satisfy.
Adaptation recovers most of it — 20.5 back to 48.6, about 78% of the way to the free ceiling — which
is what 50M tokens of training the router under the constraint buys. Freeing four layers recovers a
further 2.3 and lifts the worst layer from 39.4 to 45.6, consistent with those layers being the ones
the constraint hurt most, and with §4's finding that *which* layers you free matters more than how
many.

No layer in any adapted state is anywhere near collapse, so the unified aux is doing its job; this
was the check that could have invalidated the run and did not.

**The first version of this measurement was wrong and is withdrawn.** It scored `torch.randint`
token ids rather than the audited slice. Effective expert count is a property of the *routing*, and
a router fed uniform-random ids routes nothing like one fed text: that version read a free-regime
median of 16.3 against the 57.1 above, a 3.5x error, and would have made adaptation look like it
*tripled* balance rather than restoring it. The producer now loads `bpb_slice_ids.pt`, the same
tokens the training-time log scores.


## The unconstrained null: 20% of the apparent gap was the recipe, not the constraint

Every recovery percentage in this document is computed against 0.6727, the untrained base model,
which silently assumes that continual training on this corpus is *neutral* -- that a model trained on
these tokens with this adapter and no residency constraint would stay where it started. That had
never been tested. `ce_freeall_50M` tests it: every one of the 16 MoE layers free, so the residency
machinery is provably inert (`swap=0.0000`, `eff_load` median 58.5 against the stock 57.1), with the
tokens, LoRA rank, optimiser, schedule and data order identical to the constrained arms.

It does not stay where it started. It gets monotonically worse:

| tokens | 10M | 20M | 30M | 40M | 50M |
|---|---|---|---|---|---|
| BPB | 0.683090 | 0.687481 | 0.690524 | 0.692876 | **0.695064** |

So 50M tokens of this corpus, with no constraint at all, costs the base model **0.0224 BPB**. The
ceiling actually reachable under this recipe is 0.695064, not 0.6727.

That changes the arithmetic in the direction that *favours* the technique:

| reference ceiling | BPB | best cell's recovery |
|---|---|---|
| stock OLMoE (as published) | 0.672700 | 94.61% |
| null arm, same tokens and adapter, no constraint | 0.695064 | **95.64%** |

The best cell's gap to stock is 0.112017; its gap to the achievable ceiling is 0.089653. **Twenty
percent of the gap we have been attributing to the residency constraint is caused by the adaptation
recipe itself** and would be paid by an unconstrained model too.

Two cautions. This does not rescale the *differences between* free sets -- every cell paid the same
recipe cost, so the ladder's internal ordering is untouched. And the degradation is monotone in
tokens, so a longer run would widen it; the 250M cells are measured against a ceiling that has moved
further than this table shows, and their recovery percentages are correspondingly more understated.
