# Layer Lexicality, Round 2: the C3 correction and what to run next

**Status: a correction to a conclusion already committed in
[`LAYER_LEXICALITY.md`](LAYER_LEXICALITY.md), plus the next round of tests.** Round 1 measured H1 at
full depth, ran C3 in both directions, and recorded H2 as falsified because the per-layer cost
profile came out U-shaped rather than monotone. That reading is incomplete: the U decomposes into an
**endpoint effect** and an **interior gradient**, and only the second is about lexicality. H2 is
falsified as literally pre-registered and supported over the interior of the network.

Everything below reproduces from committed CSVs. The decomposition is
[`analysis/probes/swap_shape.py`](../../../analysis/probes/swap_shape.py) ->
`results/ablations/swap_shape.csv`.

## 1. The correction

C3 at 1e18 coarse (6 of 64 experts, 9 layers so MoE layers 2–9). Cost is the test-CE penalty of
applying the residency constraint to exactly one layer (`impose_one`, on the unconstrained
checkpoint, native CE 3.9185) or removing it from exactly one layer (`unmask_one`, on the temporal
checkpoint, native CE 3.9095). Contextual share is that layer's median context-minus-token AUC from
the locus probe, held-out documents, w=k — higher = more contextual, less lexical.

| MoE layer | l/L | impose cost | unmask cost | ctx−tok (temporal) | ctx−tok (unconstrained) |
|---|---|---|---|---|---|
| 2 | 0.22 | **+0.4115** | +0.0425 | +0.067 | −0.285 |
| 3 | 0.33 | +0.2731 | +0.0463 | +0.104 | −0.254 |
| 4 | 0.44 | +0.2526 | +0.0350 | +0.116 | −0.256 |
| 5 | 0.56 | +0.2191 | +0.0337 | +0.111 | −0.224 |
| 6 | 0.67 | +0.2409 | +0.0326 | +0.096 | −0.208 |
| 7 | 0.78 | +0.1840 | +0.0300 | +0.094 | −0.198 |
| 8 | 0.89 | +0.2151 | +0.0374 | +0.090 | −0.203 |
| 9 | 1.00 | **+0.4665** | +0.0579 | +0.098 | −0.165 |

Rank correlations, with permutation p-values (n is 6–8, so a permutation null rather than a
t-approximation):

| arm | span | cost vs depth | cost vs contextual share | endpoints ÷ interior |
|---|---|---|---|---|
| impose (unconstrained ckpt) | full L2–9 | ρ = −0.286, p = 0.50 | ρ = −0.286, p = 0.50 | **1.90×** |
| impose | **interior L3–8** | **ρ = −0.886, p = 0.035** | **ρ = −0.886, p = 0.033** | 1.90× |
| unmask (temporal ckpt) | full L2–9 | ρ = −0.071, p = 0.89 | ρ = +0.000, p = 1.00 | 1.40× |
| unmask | interior L3–8 | ρ = −0.429, p = 0.42 | ρ = +0.143, p = 0.80 | 1.40× |

**What this changes.**

1. **H2's direction survives across the interior**, significantly, in the direction with the ~7×
   larger effect (mean impose cost +0.278 vs mean unmask cost +0.039). Reporting the profile as a
   single U hid this: including the two endpoint layers drives the correlation from −0.886 to −0.286
   and the p-value from 0.035 to 0.50.
2. **The endpoint spikes are almost certainly not a lexical effect.** Layer 9 has the **highest**
   contextual share of any layer in the unconstrained model (−0.165, ranking 8 of 8 on
   least-lexical) and near the top in the temporal one — and it is the single most expensive layer
   to constrain. A lexical account predicts the opposite. The obvious alternative is architectural
   position: the first MoE layer sits directly after the dense block and the last writes into the
   final norm and unembedding, so perturbing either reaches the loss through a short path regardless
   of what it routes on.
3. **The output lens corroborates the endpoint reading.** Median effective vocabulary at layer 9 is
   anomalous in **all four** 1e18 arms (18,068 / 14,810 / 20,158 / 19,584 against interior values of
   7,600–15,800). An effect present in every regime and granularity is a property of the position,
   not of the constraint.
4. **The unmask arm is too weak to confirm any of this on its own** — no correlation reaches
   significance, and its endpoint ratio is 1.40× against the impose arm's 1.90×. Round 1 reported
   the two directions as agreeing on "the same U with the same vertex"; they agree on the endpoint
   spike and disagree on the interior.

## 2. Why 1e18 cannot settle it

Over the interior of this model the unconstrained arm's contextual share is monotone in depth, so
`cost ~ depth` and `cost ~ contextual share` are **the same measurement** — which is why both
correlations come out at exactly −0.886. Nothing in this cell can distinguish "cost falls with
depth" from "cost falls with contextual share", and the second is the one H2 is about.

Distinguishing them needs a model whose contextual share is **non-monotone** in depth. The 14-layer
1e19 coarse temporal arm is exactly that: its contextual share rises to +0.107 at layer 8 and then
declines to +0.083 by layer 14, while depth keeps increasing. That is the dissociation.

## 3. H2, restated as two separable claims

**H2a — interior gradient.** Among MoE layers not adjacent to the embedding or the unembedding, the
cost of imposing rolling residency falls as the layer's contextual share rises.

*Supported* at 1e18 in the impose direction (ρ = −0.886, p = 0.035, n = 6). Not yet separable from a
pure depth effect. Falsified by a flat or positive interior correlation once depth and contextual
share are dissociated.

**H2b — endpoint effect.** The first and last MoE layers cost more to constrain than the interior
gradient predicts, for reasons of architectural position rather than routing locus.

*Consistent with* the 1e18 data (1.90× / 1.40×, and the last layer is the least lexical while being
the most expensive). Falsified if the spike tracks depth-relative position rather than
first/last-layer adjacency — i.e. if in a 14-layer model it moves to ~⅔ depth instead of staying at
the first and last MoE layers.

## 4. The next round, in order

### N1 — C3 at 1e19, both directions. Eval-only. The decision experiment.

Run [`scripts/phase0/constraint_swap_sweep.sh`](../../../scripts/phase0/constraint_swap_sweep.sh)
over `g1_tmoe_coarse_1e19` (unmask) and `moe_coarse_1e19` (impose), layers 2–14. Two
pre-registered readings:

**Test A — endpoint position (works in both directions, strong signal).** Where do the spikes land
in a 14-layer model?

- Spikes at layers 2 and 14 -> **H2b confirmed**, the effect is adjacency to the embedding and the
  unembedding, and the "U" should be reported as two outliers plus a trend, not a curve.
- Spikes at ~⅔ depth (layers 9–10) -> H2b rejected; the shape is intrinsic and Round 1's vertex
  reading was right.

**Test B — dissociation (unmask arm only; this is the only arm where the two explanations come
apart).** Between layers 8 and 14 the temporal contextual share *falls* while depth rises. Does
interior cost:

- rise again after layer 8, tracking the contextual share -> **H2a confirmed as a lexical effect**;
- keep falling with depth -> H2a is a depth effect, and H2's mechanism is not about lexicality at
  all.

Note the limitation up front: the unmask arm carries the weaker signal (interior spread ~0.016 CE at
1e18 versus ~0.09 for impose), and the impose arm cannot run Test B because the *unconstrained*
model's contextual share stays monotone through layer 14. If Test B is inconclusive, say so rather
than leaning on Test A.

### N1b — C3 on the fine granularity at 1e18. Eval-only, cheap.

C3 has only ever run on the coarse arm. Running `flame38m_g3_temporal` / `flame38m_g3_moe` tests
whether the endpoint spike replicates across granularity before any of it is built on.

### N2 — re-frame C3 in [`LAYER_LEXICALITY.md`](LAYER_LEXICALITY.md).

§3's H2 status and the two-line summary currently report the U without the decomposition, and state
that both directions agree on the same vertex. Replace with H2a/H2b above, and record that the
interior correlation and the depth correlation are collinear within one model.

### N3 — C8, causal token and context substitution.

Still the strongest untried non-training evidence for H1, and more valuable now: if the endpoint
spikes are positional, C8's per-layer sensitivity ratio is the clean way to show the routing story
is an interior-layers story. Hold the context fixed and substitute the current token, measure how
far the selected expert set moves; then hold the token fixed and shuffle the context. Forward passes
only.

### N4 — correct §4 of [`delexicalization.md`](delexicalization.md).

The published output-lens claim — temporal experts barely distinguishable from no signal, baseline
markedly narrower — rests on rows that the layer-keying bug attributed one layer too shallow, and it
does not replicate at 1e18. Median effective vocabulary (data-weighted, 50k vocabulary; **lower =
narrower, more lexical**):

| MoE layer | coarse full MoE | coarse temporal | fine full MoE | fine temporal |
|---|---|---|---|---|
| 2 | 8,824 | 9,624 | 11,937 | 18,231 |
| 3 | 8,690 | 9,065 | 13,233 | 13,633 |
| 4 | 8,896 | 14,821 | 13,450 | 12,953 |
| 5 | 10,414 | 12,356 | 9,724 | 12,226 |
| 6 | 10,364 | 7,501 | 12,499 | 10,599 |
| 7 | 7,640 | **720** | 12,163 | 8,625 |
| 8 | 9,640 | 2,477 | 13,549 | 15,759 |
| 9 | 18,068 | 14,810 | 20,158 | 19,584 |

The coarse pair shows a crossover — temporal broader through layer 5, sharper from layer 6 — and the
fine pair shows no consistent direction at all (temporal broader at layers 2, 3, 5, 8; sharper at 4,
6, 7, 9). Round 1 summarised this as "temporal experts write *sharper* vocabularies", quoting layer
7 of the coarse arm, which is the single most extreme cell in the table. **The defensible statement
is that the output-lens regime difference does not replicate at 1e18**, not that it reverses. §4
needs restating or retracting, and its concluding sentence — that input side, output side and
structure agree — cannot stand as written.

### N5 — training. Held, and the design changes if N1 Test A confirms H2b.

Round 1 proposed redesigning T2 to contrast {2,3,8,9} against {4,5,6,7}, which is right for a
symmetric U. If the spikes are positional, the schedule worth testing is narrower and much cheaper:
**free only the first and last MoE layer, constrain the rest.** Resident expert-slot counts at 1e18
coarse (8 MoE layers, E = 64, k = 6):

| schedule | slots | vs baseline |
|---|---|---|
| unconstrained baseline | 512 | 1.00× |
| free {2,3,8,9}, constrain {4,5,6,7} | 280 | 0.55× |
| **free {2,9}, constrain {3..8}** | **164** | **0.32×** |
| uniform R = k | 48 | 0.09× |

The matched-memory control for the 164-slot schedule is uniform R = 20 (160 slots), which is where
the existing uniform-R dose curve already provides the reference. Do not start any of this before
N1.

## 5. What the advisor's prior does and does not survive

The original suggestion was to keep the first and last layers fully resident and constrain the
middle, on the grounds that first and last layer experts are bound to token identity.

**The recommendation survives** — the first and last MoE layers are by some margin the most
expensive to constrain, so exempting them is the right engineering call, and it is cheaper than any
schedule we proposed.

**The stated reason does not.** Layer 9 is the *least* lexical layer in the stack by our own probe
and the most expensive to constrain. Whatever makes the end layers expensive, it is not that they
route on token identity. Worth communicating both halves, since the reason determines whether the
exemption should be two layers or four, and whether it generalises to deeper models — which is
exactly what N1 measures.

## 6. Reproduction

```bash
python3 analysis/probes/swap_shape.py                    # -> results/ablations/swap_shape.csv
python3 analysis/plots/plot_locus_by_layer.py            # H1 curves + slope/shape statistics
```
