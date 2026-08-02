# Findings without a write-up elsewhere

Five results that were produced during the multi-pod experiment program, committed as CSVs, and
reported only inside the pod control-channel transcripts. Each is reconstructed here from its
committed CSV, not from the transcript, so every number below is reproducible from the repo.

**What is already documented, and is not repeated here:**

| topic | lives in |
|---|---|
| Phase-0 isoflop sweeps, dense floor, temporal MoE, fine-graining at G1/G3 | [`results/ablations/FINDINGS.md`](../../results/ablations/FINDINGS.md) |
| OLMoE residency adaptation, Stage 2 LR sweep and the eight-arm bake-off | [`results/ablations/olmoe_adapt_RESULTS.md`](../../results/ablations/olmoe_adapt_RESULTS.md) |
| De-lexicalization, narrative write-up (see the corrections doc first) | [`mechanism/delexicalization.md`](mechanism/delexicalization.md) |
| What rolling residency does to routing — findings, by claim | [`mechanism/01-findings.md`](mechanism/01-findings.md) |
| Corrections to the published de-lexicalization write-up | [`mechanism/02-corrections.md`](mechanism/02-corrections.md) |
| Selection-shaping program: anticipatory loss, bursty loss, Karen, momentum variants | [`ablations/alignment-program.md`](ablations/alignment-program.md) |
| MinFlow / O-series offline scheduling, and the calibration and init-axis screens (Cal-0, Cal-2) | [`olmoe-adaptation-plan.md`](olmoe-adaptation-plan.md) close-out, and the `olmoe_minflow_*` / `olmoe_cal*` rows of [`results/ablations/README.md`](../../results/ablations/README.md) |
| Block-local routing, serving floor | [`ablations/local-global-program.md`](ablations/local-global-program.md), [`background/batch1-offload-feasibility.md`](background/batch1-offload-feasibility.md) |
| Per-CSV index for everything | [`results/ablations/README.md`](../../results/ablations/README.md) |

**Metric conventions.** `BPB = CE_nats / (ln2 · bytes_per_token)`, bits per byte, lower is better.
Divisors are byte-derived per corpus and never inherited: **2.9780** for the pythia-50k corpus,
**2.7600** for the G3-era 16k corpus, **3.1089** for the OLMoE audited held-out slice. `R` is the
rolling-residency size, `k` the top-k; `R=k` with at most one expert swap per token is the
deployable serving constraint. Downstream accuracy is 0-shot, higher is better.

---

## 1. Fine-graining hurts free MoE more than it hurts temporal residency

Extending the panel to GRAIN=5 (320 experts, top-30) at the 1e18 budget, both cells single-variable
against the standard panel and trained for 2121 iterations at seed 1234.
Source: [`flame1e18_k30.csv`](../../results/ablations/flame1e18_k30.csv).

| grain | experts / top-k | temporal BPB | free-MoE BPB | temporal − MoE | basis |
|---|---|---|---|---|---|
| G1 | 64 / 6 | 1.3123 | 1.3175 | −0.0052 | seed means (5 and 3 seeds) |
| G3 | 192 / 18 | 1.3338 | 1.3478 | −0.0140 | seed means (3 and 3 seeds) |
| G5 | 320 / 30 | **1.3491** | **1.3625** | −0.0134 | single seed |

Seed-mean values are computed from [`flame38m_1e18_cells.csv`](../../results/ablations/flame38m_1e18_cells.csv);
observed seed ranges are 0.0027–0.0040 BPB, so the G3 and G5 gaps sit above seed noise and the G1
gap sits near it.

Fine-graining degrades both paradigms, but not equally: G1→G5 costs the temporal cell +0.037 BPB
and the free-MoE cell +0.045. The temporal advantage therefore widens from −0.005 at G1 to roughly
−0.014 at G3 and G5. Both K=30 cells still beat the dense floor, the temporal cell by 0.042 and
the MoE cell by 0.029. The floor here is `flame38m_dense_local` (1.3911), the same-corpus local
dense run, not the cross-data `flame38m_dense` (1.3893).

The plausible mechanism is that rolling residency damps the routing churn that over-fine-graining
introduces: with 320 experts and a ≤1-swap budget the served set changes slowly, so the model
cannot chase a noisier top-30 the way free routing does.

**Caveats.** The G5 cells are single-seed while G1 and G3 are seed means, so the G5 delta carries
no error bar of its own; it is quoted because it falls close to the G3 delta, not because it was
replicated. Against *temporal-coarse* (G1, 1.3123) rather than the dense floor, K=30 is worse by
~0.037 — fine-graining costs absolute quality even where it improves the temporal-vs-MoE gap. The
payload is the robustness gap, not a pass against any single bar.
Downstream at K=30 separates nothing (g5_temporal 0.348, g5_moe 0.361, both near chance at 38M),
consistent with the rest of the 1e18 hygiene table.

---

## 2. Early-router overlap: promoted on a screen, falsified at depth

Two architecture variants aimed at widening the expert-prefetch window were implemented behind
flags and screened at 1e17 before promotion to 1e18. **V1 early-router** routes on the
pre-attention normalized state `LN1(x)` instead of the expert input; **V2 parallel-FFN** is the
PaLM/GPT-J restructure, `y = x + Attn(LN1(x)) + MoE(LN2(x))`.

Flag-off parity was established before any cell ran: the edited-off vs original-off difference
(1.2e-4) was smaller than the run-to-run non-determinism floor measured from two identical
edited-off runs (9.7e-4), while flags-on moved the loss 10–100× that floor.

**Screen at 1e17, shape s2 (L=6, 5 MoE layers), divisor 2.7600**
([`flame1e17_overlap.csv`](../../results/ablations/flame1e17_overlap.csv)):

| variant | cell | BPB | Δ vs standard | bar | gate |
|---|---|---|---|---|---|
| V1 early-router | temporal | 1.2909 | +0.0036 | 0.005 | PASS |
| V1 early-router | MoE | 1.2712 | +0.0004 | 0.010 | PASS |
| V2 parallel-FFN | temporal | 1.3137 | +0.0264 | 0.005 | FAIL |
| V2 parallel-FFN | MoE | 1.2901 | +0.0193 | 0.010 | FAIL |

**Promotion to 1e18, flame38m (L=9, 8 MoE layers), divisor 2.9780**
([`flame1e18_overlap.csv`](../../results/ablations/flame1e18_overlap.csv)):

| leg | V1 BPB | standard BPB | Δ | screen predicted | miss |
|---|---|---|---|---|---|
| temporal | 1.3798 | 1.3354 | +0.0444 | +0.0036 | 12× |
| MoE | 1.4964 | 1.3461 | +0.1503 | +0.0004 | 375× |

V1 fails on both legs. The MoE leg lands *worse than the dense floor* (1.4964 vs 1.3911, a higher
BPB), meaning early-routing erases the entire MoE advantage at this depth. Both runs were clean: monotonic loss
decay, zero NaN iterations, flags confirmed in the argument dumps.

**The protocol result, which matters more than the architecture result.** The damage is worse on
the free-MoE leg, so it is not temporal-specific. Routing on the pre-attention state degrades
compounds per MoE layer, and the screen had 5 MoE layers where the promotion cell had 8. A
shallower screen therefore systematically under-predicts. **Screen-to-promotion gates must screen
at the deployment depth, or carry an explicit depth-transfer check.** On this evidence V1 should
not have promoted.

Worth keeping from the wreckage: the temporal leg is about 3.4× more robust to the insult than free
MoE (+0.044 vs +0.150), which is the same residency-damps-routing-error effect that shows up in §1.

**Caveats.** One seed per cell. Depth is confounded with shape, since s2 and flame38m differ in
more than layer count; the claim is that the screen did not transfer, and depth is the leading
hypothesis rather than an isolated variable. V1 routes on `LN1(x)`; the `LN2(x)` alternative, which
would isolate earliness from the choice of norm, was never built.

---

## 3. OLMoE Stage 3: adaptation recovers three quarters of the downstream loss and reproduces the de-lexicalization signature

The Stage-2 bake-off (documented separately) measured recovery in BPB. Stage 3 asks whether that
transfers to downstream capability. Three cells, 10 tasks, 0-shot: the base model with free
routing, the base model with the R=8 residency mask imposed and no training, and the CE-adapted
model (router + RMSNorm gains + LoRA r32, 0.25B tokens) evaluated at R=8.
Source: [`olmoe_adapt_downstream.csv`](../../results/ablations/olmoe_adapt_downstream.csv).

Mean accuracy over the 10 tasks at each task's primary metric (acc_norm for hellaswag and
openbookqa, acc for the other eight): **base-free 0.715, impose-R8 0.329, CE-adapted-R8 0.617**,
giving a downstream recovery of **74.7%** of the accuracy the mask destroys. The recovery fraction
is stable across metric selection (74.4% acc-only to 74.7% primary-metric), so quote it as roughly
three quarters rather than to a tenth of a point. The absolute means move by about 0.03 with the
convention (acc-only gives base 0.682, adapted 0.589), while the recovery fraction does not, which
is why the fraction is the number to quote.

That is meaningfully below the 93.2% BPB recovery from the same adapted model. Downstream is the
harder test, and the residual constraint cost concentrates in the hardest tasks: near-full recovery
on knowledge and commonsense (sciq recovers to within 0.001 of base after the mask had driven it to
near-random; piqa, boolq strong), partial on multi-step reasoning (arc_challenge, hellaswag), and
lambada_openai goes from 0.706 to **0.000** under the untrained mask before recovering to 0.570.

**The dense-1B bracket.** Against era-matched released dense checkpoints, the CE-adapted model at
R=8 scores 0.617 mean against OLMo-1B-0724's 0.630, a gap of **−0.012**. It wins on ARC
(arc_easy +0.061, arc_challenge +0.036) and ties on sciq and openbookqa. So a residency-constrained
MoE serving from roughly a fifth of the parameters (1.3B active of 7B total; serving VRAM itself was
not benchmarked, since Stage 4 was cancelled) lands within about one point of the unconstrained
dense-1B peer, and 0.091 below the OLMo-7B anchor.

**Routing forensics: adaptation reproduces the de-lexicalization signature, weakly.** Running the
Section-4 locus probe on the adapted model asks whether adaptation reshapes routing the way
training the constraint in from scratch does. It does, in direction: under the untrained R=8 mask
routing is marginally token-driven (context-minus-token AUC −0.0041), and after CE adaptation the
sign flips to context-driven (+0.0493), with context AUC rising 0.603→0.673 while token AUC barely
moves (0.607→0.624). The generalist fraction drops to 0, so experts become more selective, not
less. Source: [`olmoe_adapt_forensics.csv`](../../results/ablations/olmoe_adapt_forensics.csv).

The effect is real but small and must be sized as such. The underlying AUCs sit at 0.60–0.67
against a 0.5 chance floor, so the clean signal is the paired context-minus-token difference, not
the absolute discriminability, and the flip is far milder than the from-scratch 1e19 battery where
context-dominated experts go from 0% to 88–96%. Demand-prediction AUC is ~0.96 in both cells, a
non-discriminator here: a ≤1-swap residency schedule is history-forecastable by construction, so
the mechanism change appears in the locus rather than the forecastability.

**Caveats.** Era-matched, not data-matched (OLMo-1B trained on ~3T Dolma tokens against OLMoE's
~5.1T). This compares an adapted-under-constraint model against a free dense model, so it is a
memory-class comparison and not iso-training. Stderr for every downstream cell is in the CSV. The
forensics is a single probe run over layers 2–6, 16 packs, so its magnitudes are indicative rather
than replicated; the sign of the locus flip is the load-bearing claim.

---

## 4. Adapting under the constraint is far cheaper than training into it

Evaluating the released from-scratch OLMoE-0924 pretraining checkpoints with free routing on the
same audited held-out slice gives a quality-versus-tokens ladder, which locates where the adapted
models sit. Source: [`olmoe_scratch_ladder.csv`](../../results/ablations/olmoe_scratch_ladder.csv).

| from-scratch checkpoint | tokens | BPB (free routing) |
|---|---|---|
| step5000 | 20B | 0.8700 |
| step10000 | 41B | 0.8182 |
| step25000 | 104B | 0.7772 |
| step55000 | 230B | 0.7565 |
| step125000 | 524B | 0.7431 |

Interpolating in token space, the CE-adapted model (0.8147 BPB, reached with **0.25B** tokens of
adaptation under a hard R=8 constraint) matches from-scratch *unconstrained* quality at about
**46B** tokens. The cheap norms-only arm (0.8505) matches it at about **28B**. That is roughly
**110–180×** fewer tokens to reach equivalent held-out quality, while running under the residency
constraint rather than without it.

The harness was independently validated before this claim: the base model's CE on c4 en-validation
measured 2.4730 nats/token through our perplexity path, against 2.4807 from the published 0924
wandb run, in the expected direction since 0125 is the improved checkpoint.

**Caveats.** This is equivalence on one held-out slice (dolmino dclm), not across capabilities;
§3's downstream numbers are the harder comparison and are less favourable. The from-scratch ladder
runs free, the adapted models run constrained, so this measures cost-to-reach-quality and not an
iso-training comparison. Non-membership of the audited slice rests on dataset identity plus a
high-index heuristic, not per-shard proof.

---

## 5. Temporal routing yields more Gaussian weights, and quantizes better

A five-part stability appendix over the 1e18 38M cells and the 1e19 cells found a consistent
signature. Sources: [`stability_weights.csv`](../../results/ablations/stability_weights.csv),
[`stability_activations.csv`](../../results/ablations/stability_activations.csv),
[`stability_gradnorms.csv`](../../results/ablations/stability_gradnorms.csv),
[`stability_fakequant.csv`](../../results/ablations/stability_fakequant.csv).

**Weights and activations are less heavy-tailed under temporal routing.** Routed-expert excess
kurtosis (max over experts) is lower for every matched pair: g1_temporal 0.66 vs g1_moe 2.63,
g3_temporal 2.00 vs g3_moe 6.01, temporal_coarse 0.57 vs moe_coarse 0.97. FFN intermediate
activations show the same direction (median excess kurtosis g1_temporal 0.21 vs g1_moe 1.02,
g3_temporal 0.38 vs g3_moe 1.04), and it holds at 1e19 (temporal_coarse 0.305 vs moe_coarse 0.457).

**No expert is pinned.** Maximum residency is 15–25% across temporal cells, and router-row L2 norm
does not predict residency (correlation −0.27 to +0.03), so weight magnitude is not what decides
which experts stay resident.

**The consequence: better low-bit quantization.** Per-group symmetric round-to-nearest on routed
expert weights only, test CE delta against the 16-bit baseline:

| run | 8-bit | 4-bit | 3-bit |
|---|---|---|---|
| g1_moe | +0.0001 | +0.0180 | +0.1077 |
| g1_temporal | +0.0000 | **+0.0144** | **+0.0866** |
| g3_moe | +0.0000 | +0.0126 | +0.0722 |
| g3_temporal | +0.0001 | **+0.0106** | **+0.0618** |
| moe_coarse_1e19 | +0.0000 | +0.0076 | +0.0452 |
| temporal_coarse_1e19 | +0.0001 | **+0.0066** | **+0.0383** |
| temporal_fine_1e19 | +0.0001 | +0.0059 | +0.0340 |

8-bit is lossless everywhere. At both 4-bit and 3-bit the temporal cell degrades less than its
matched free-MoE pair, in all three pairs; that within-pair advantage is the clean result. The
1e19 models are also more quantization-robust than the 38M models at 3-bit, by roughly 2× on
matched pairs (1.8× fine, 2.3× coarse temporal, 2.4× coarse MoE; up to ~3.2× on the most
favourable cross-pairing), and fine-grained cells are more robust than coarse ones.

This composes with the memory story: the same constraint that cuts resident memory also produces
weights that survive low-bit quantization slightly better, so the two savings stack rather than
trade off.

**Caveats.** `temporal_fine_1e19` has no matched free-MoE pair in this table, so its column is
descriptive only. The quantization sweep covers routed expert weights, not attention or shared
experts. Gradient norms were clean on all 38M cells (pre-clip max ≤2.2); two 1e19 cells showed larger
pre-clip transients that clipping absorbed with no divergence (dense_1e19 max 7.71 at iter 2280 of
4310; temporal_coarse_1e19 max 12.47 at iter 830). These are mid-run rather than warmup: the dense
excursion sits near the middle of training, coinciding with the mid-run loss bump rather than the
opening iterations, and both runs recovered.

---

## Methodological notes worth carrying forward

Three process findings from the same program that generalise beyond any single result.

**Single-seed quality wins in this family do not survive replication.** Two headline claims were
retracted after a second seed: a momentum-on-plain-substrate BPB win (−0.0036 vs baseline) turned
out to sit inside a seed spread of 0.006–0.009 and became a tie; the Karen cell's "quality
positive" framing (−0.0031) reversed sign on seed 2 and became parity. Plain-temporal's own
two-seed spread is 0.0004, so the momentum family is roughly 10–20× noisier than the baseline it
was being compared against. Effects below ~0.01 BPB in this family need two seeds before they are
claims.

**Windowing a scheduling evaluation inflates it.** Captured-mass and BPB comparisons run with a
256-token window and a cold fill per window roughly doubled the estimated captured mass and
understated the greedy policy's accumulated suboptimality, because frequent free cold-fills reset
the policy. The full-sequence measurement moved the greedy-versus-optimal gap from 0.007 to
0.023 ± 0.008 BPB. Any residency evaluation should state its cold-fill regime.

**Divisors must be byte-derived per corpus, never inherited.** Three divisors are live in this
repo (2.7568, 2.7600, 2.9780) and they correspond to genuinely different corpora rather than to an
error. One early CSV was written with the 16k divisor for 50k-vocab runs. The convention is to
re-derive `ln2 · bytes_per_token` from the exact evaluation and record it in the CSV header.
