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

## 4. The next round, in priority order

Every run here is **eval-only or a single forward pass — no training**. Training is deferred to §5
and should be started only when these are done and there is spare capacity.

Timing estimates are derived from the FLOP budgets, anchored on the one wall-clock fact the repo
records — the 1e18 seed replicates ran overnight on an A6000
([`seed_replicates.csv`](../../../results/ablations/seed_replicates.csv), `box` column) — scaled to a
single H100 at roughly 3–4x effective throughput. Individual evals are startup-dominated (Megatron
init plus checkpoint load) rather than compute-dominated, so arm counts matter more than model size.
Treat every estimate as ±2x.

| # | run | cost | answers |
|---|---|---|---|
| **N1** | Sham-perturbation control at 1e18 | ~18 evals, 1.5 h | Is the endpoint spike positional? Directly, rather than by inference |
| **N2** | Multi-layer schedule and additivity check | ~6 evals, 40 min | Do single-layer costs predict a schedule at all? |
| **N3** | C3 on a second seed at 1e18 | ~18 evals, 1.5 h | Is the U a property of the regime or of one trained model? |
| **N4** | C3 on the fine granularity at 1e18 | ~18 evals, 1.5 h | Does the endpoint spike replicate across granularity? |
| **N5** | C3 at 1e19, both directions, layers 2–14 | ~30 evals, 3–5 h | H2a: does cost track contextual share or depth? The dissociation |
| **N6** | C8 — causal token and context substitution | 2–4 h | H1, causally rather than by probe. Still never run |
| **N7** | Per-layer cost versus per-layer hit rate | free, CPU | Rules out (or in) cacheability as a third explanation |
| **N8** | Captures: re-take 3 x 1e19, add 1e16/1e17 arms | ~7 passes, 2 h | Makes the 1e19 output lens valid; unfreezes the low-budget end |
| **N9** | Document corrections | no compute | Two claims still stand next to their own refutations |

## 4a. Results (N1, N2, N5, N6 measured)

| # | status | headline |
|---|---|---|
| **N1** | **done, both directions** | The sham **reproduces** the U in the impose direction (r = 0.78; L2 1.39× vs real 1.45×). H2b supported: the endpoints are positional. The unmask direction disagrees (r = 0.38) and should be discounted — there is no valid sham for "un-apply a perturbation the model never had", and its baseline is an off-distribution model at CE 4.895 vs native 3.909. |
| **N2** | **done, both directions** | Endpoints add independently, interior does not: {2,9} ratio **1.02** imposing and **0.98** unmasking; {3–8} **0.81**. So single-layer costs predict the endpoint schedule and overstate an interior one by 23%. |
| **N5** | **unmask arm done** | **Cost tracks depth, not lexicality**: r = **+0.770** with depth, **+0.152** with contextual share. Between L8 and L14 the contextual share falls while cost more than doubles. Test A: quadratic R² 0.92 vs linear 0.59, vertex L5.8, **L14 at 2.71× mean** — spikes at layers 2 and 14, i.e. the architectural boundaries. Impose arm running. |
| **N6 (C8)** | **done, both regimes** | De-lexicalization shown causally. Ratio (context shift ÷ token shift): temporal **1.34–1.66**, unconstrained **0.30–0.73**, opposite sides of 1 at every layer. Token sensitivity falls 43% *and* context sensitivity nearly triples — a decomposition no AUC difference can separate. |

**What this settles.** The constraint's effect on *routing* is large and causal (N6). The per-layer
*cost* profile is architectural — endpoint sensitivity plus an interior depth trend — and lexicality
explains almost none of it (r = 0.15). Those are different quantities; H2 conflated them.

**A Round-1 reading this corrects.** The vertex at roughly two thirds depth was an artifact of the
9-layer model, where "⅔ of the way down" and "just before the last layer" are not separable. At 14
layers they separate and the endpoint reading wins.

**Do N1–N4 first.** They fit in one pod day, they are all cheap, and N1 can answer H2b outright — in
which case N5 narrows to the H2a dissociation question only.

### N1 — sham-perturbation control. The cheapest decisive test for H2b.

H2b currently rests on an *inference*: the first and last MoE layers are expensive to constrain, the
last layer is the least lexical layer in the stack, therefore the spike is architectural. Test it
instead.

Sweep the same layers with a perturbation of matched magnitude that carries **no lexical
information**: keep k experts active but choose the resident set uniformly at random, or add noise to
the router logits calibrated to produce a comparable mean CE penalty. Then compare profiles.

- Sham reproduces the same U -> **H2b confirmed outright.** The endpoints are positional sensitivity
  and have nothing to do with routing; report C3 as an interior trend plus two structural outliers.
- Sham is flat while the constraint's profile is U-shaped -> the endpoints are specific to residency,
  and H2b needs the depth-scaling test in N5.

This is more decisive for H2b than N5 and costs a third as much.

### N2 — multi-layer schedule and additivity. De-risks six training runs for six evals.

Every C3 number is a **single**-layer perturbation. The thing we would ship is a **multi**-layer
schedule, and nothing has checked that the two relate. Run:

- impose on {3..8} together, on the unconstrained checkpoint;
- impose on {2,9} together;
- unmask {2,9} together, on the temporal checkpoint;
- compare each against the sum of the corresponding single-layer costs.

If the effects are not additive, single-layer C3 does not predict schedule performance and T2's
design rests on an unchecked assumption. This also produces the first direct eval-time estimate of
the proposed "free the endpoints" schedule, which is the thing §5 would spend six training runs on.

### N3 — C3 on a second seed at 1e18.

`flame38m_g3_temporal_s2`, `flame38m_g3_temporal_s3` and `flame38m_g1_temporal_s3` all have preserved
checkpoints. The entire U-shape currently rests on one trained model per direction. C3 is
deterministic given a checkpoint, so this is not a noise estimate — it tests whether the shape is a
property of the regime or of that particular run.

### N4 — C3 on the fine granularity at 1e18.

C3 has only ever run on the coarse arm. `flame38m_g3_temporal` / `flame38m_g3_moe` test whether the
endpoint spike survives a change of granularity.

### N5 — C3 at 1e19, both directions, layers 2–14. The H2a dissociation.

Run [`scripts/phase0/constraint_swap_sweep.sh`](../../../scripts/phase0/constraint_swap_sweep.sh)
over `g1_tmoe_coarse_1e19` (unmask) and `moe_coarse_1e19` (impose). Two pre-registered readings:

**Test A — endpoint position (both directions, strong signal).** Where do the spikes land in a
14-layer model? At layers 2 and 14 -> adjacency to the dense block and the unembedding, H2b
confirmed. At ~⅔ depth (layers 9–10) -> intrinsic, and Round 1's vertex reading was right.

**Test B — dissociation (unmask arm only).** Between layers 8 and 14 the temporal contextual share
*falls* while depth rises. Does interior cost rise again after layer 8, tracking the contextual share
(**H2a is a lexical effect**), or keep falling with depth (**H2a is a depth effect and the mechanism
is not about lexicality**)?

Two limitations, stated rather than discovered: the unmask arm carries the weaker signal (interior
spread ~0.016 CE at 1e18 against ~0.09 for impose), and the impose arm cannot run Test B because the
unconstrained model's contextual share stays monotone through layer 14. This model also gives 11
interior points against 1e18's 6, which is the real power gain for H2a — the current
ρ = −0.886 rests on six.

### N6 — C8, causal token and context substitution.

Hold the context fixed and substitute the current token, measuring how far the selected expert set
moves; then hold the token fixed and shuffle the context. Forward passes only. Still the strongest
untried evidence for H1, and more valuable now: if the endpoints are positional, C8's per-layer
sensitivity ratio is the clean way to show the routing story is an interior-layers story.

### N7 — per-layer cost versus per-layer hit rate. Free.

Cacheability is a third candidate explanation for the interior gradient and nobody has ruled it out.
Per-layer hit rate now exists for 22 runs (`e6_per_layer_ranking.csv`). At 1e18 it is collinear with
both depth and contextual share over the interior, so it needs the same dissociation treatment as
H2a — fold it into N5's analysis rather than running anything.

### N8 — the capture gaps.

Re-take the three 1e19 captures with the fixed layer keying, so C5 (output lens) is valid there and
not only at 1e18. Then capture 1e16/1e17 sibling arms: the low-budget end is frozen at layers 2–4 and
2–6 because the runs behind the published rows are absent from `MANIFEST.csv`, but same-shape,
same-budget checkpoints do exist, so *new* low-budget arms are reachable even though those exact
cells never will be.

### N9 — document corrections. No compute.

1. §4 of [`delexicalization.md`](delexicalization.md) now places its correction *above* the original
   claim, so the retracted paragraph — ending "input side, output side, and structure now agree" —
   still sits at the bottom of the section as though it stands. Delete it or mark it explicitly.
2. The same section says the fine pair "shows the same crossover from layer 5 on". It does not
   cleanly: at layers 5 and 8 the unconstrained arm is the sharper one. "Does not replicate" is the
   defensible claim.
3. §3's H2 status in [`LAYER_LEXICALITY.md`](LAYER_LEXICALITY.md) reports the U without the
   endpoint/interior decomposition, and states that both directions agree on the same vertex — they
   agree on the endpoint spike and disagree on the interior. Replace with H2a/H2b from §3 above.

## 5. Deferred: training runs, for when there is spare capacity

**Nothing here should start before §4 is done.** N1 and N5 determine which schedule is worth
training, and N2 determines whether single-layer costs predict a schedule at all. Running T2 against
the wrong exemption set would burn six 1e18 runs on a spurious null, which is exactly what Round 1's
original T2 design would have done.

| # | run | count | per run | total |
|---|---|---|---|---|
| T1 | Single-layer constraint sweep at s0/1e16 | 3 | ~10 min | under 1 h |
| T2 | Matched-memory schedule contrast at 1e18 | 6 | 5–6 h | ~1.5 days |
| T3 | Full per-layer resolution plus schedule-versus-uniform at 1e18 | ~10 | 5–6 h | ~2.5 days |
| T4 | Dense-final-block variant at 1e18 | 1–2 | 5–6 h | ~10 h |

**T1 is nearly free and worth running regardless** — three runs at ten minutes each is cheaper than
arguing about whether inference-time C3 generalises to a co-adapted model, which is the one thing C3
structurally cannot tell us.

**T2's exemption set depends on N1 and N5.** Resident expert-slot counts at 1e18 coarse (8 MoE
layers, E = 64, k = 6):

| schedule | slots | vs baseline |
|---|---|---|
| unconstrained baseline | 512 | 1.00x |
| free {2,3,8,9}, constrain {4,5,6,7} | 280 | 0.55x |
| **free {2,9}, constrain {3..8}** | **164** | **0.32x** |
| uniform R = k | 48 | 0.09x |

If the endpoints are positional, the two-layer exemption is the right schedule and the four-layer one
wastes 116 slots. The matched-memory control for the 164-slot schedule is uniform R = 20 (160 slots),
where the existing uniform-R dose curve already provides the reference.

**T4 is the last resort for H2b**, and only if N1, N3, N4 and N5 leave it ambiguous. If the endpoint
spike is adjacency to the unembedding, then training a model whose final block is dense should move
the spike to the new last MoE layer. Every existing config uses `--moe-layer-freq "[0]*1+[1]*(L-1)"`,
so this needs new training rather than a new analysis.

For the full T1–T3 rationale and the power calculation that set their testbeds, see §5 of
[`LAYER_LEXICALITY.md`](LAYER_LEXICALITY.md).

## 6. What the advisor's prior does and does not survive

The original suggestion was to keep the first and last layers fully resident and constrain the
middle, on the grounds that first and last layer experts are bound to token identity.

**The recommendation survives** — the first and last MoE layers are by some margin the most
expensive to constrain, so exempting them is the right engineering call, and it is cheaper than any
schedule we proposed.

**The stated reason does not.** Layer 9 is the *least* lexical layer in the stack by our own probe
and the most expensive to constrain. Whatever makes the end layers expensive, it is not that they
route on token identity. Worth communicating both halves, since the reason determines whether the
exemption should be two layers or four, and whether it generalises to deeper models — which is
what N1 and N5 measure.

## 7. Reproduction

```bash
python3 analysis/probes/swap_shape.py                    # -> results/ablations/swap_shape.csv
python3 analysis/plots/plot_locus_by_layer.py            # H1 curves + slope/shape statistics
```
