# Temporal-to-free curriculum for full-MoE pretraining: plan (2026-09-04)

Question. Rolling-residency (temporal) training gives weights that behave better in every way we
have measured except unconstrained BPB: experts that are more substitutable, less alike in output,
lower-kurtosis, quantisation-friendlier, with a router that swaps at most one expert per token. Can
some of that be used to make an ordinary full MoE, evaluated with every expert resident and no
constraint, reach a lower test BPB at the same FLOPs?

## Setting and baselines (no re-runs; seed variance is below the effects we chase)

Shape s2 at 1e17: 6 layers, H 256, global batch 256, cosine lr 3e-3 to 3e-4, 3,861 iterations
(grain 3, 192 experts, k 18) or 3,917 (grain 1, 64 experts, k 6), the pythia-50k tokenizer and
the DCLM corpus of the 1e18 and 1e19 runs. Correction (08:10): the recorded `g*_moe_s2_1e17` and
`g*_tmoe_s2_1e17` cells were trained with the 16k tokenizer on its own corpus (test CE 3.5074 /
3.5530 on grain 3, 3.4930 / 3.5486 on grain 1, BPB divisor 2.7568, so 1.272 / 1.289 BPB), which
makes their cross-entropies incomparable with runs on the 50k corpus. The curriculum stays on the
50k corpus, the one the promotion targets use, and the reference for every 1e17 comparison is
the C0 control: the full MoE trained through the same router path (R = E) on the same corpus,
one run per grain. Every arm keeps C0's batch, iteration count and lr schedule; residency changes
no FLOPs, so every arm is compute-matched by construction. The 1e19 fine-run speed recipe is on
(`MOE_TORCH_GMM=1 MOE_PERMUTE_FUSION=1 MOE_NO_LAYER_LOG=1 CE_FUSION=1`, legacy grouped GEMM path,
rope fusion off), bit-identical to the reference path, 1.17 s per iteration at this shape.

Noise bar. The 1e18 seed triplets (`flame38m_g3_moe`, `_s2`, `_s3`: 4.0087, 4.0170, 4.0152; the
temporal triplet 3.9768, 3.9724, 3.9675) put the seed standard deviation near 0.005 CE. A recipe
counts as a win at 0.010 CE (0.0034 BPB) below C0, as promising within 0.005 of it, and is
dropped otherwise. Test CE is the final 20-iteration test-split eval in `train.log`, the same
quantity the baseline row carries, and the final eval of every curriculum run is unconstrained.

Cost. One 1e17 run is about 3,900 iterations at 1.5 s (1.6 h, less with the speed recipe). One
1e18 run (`flame38m` shape, 2,121 iterations at 12.7 s) is about 7.5 h. One 1e19 fine run is 5.3
days. So 1e17 admits a dozen arms in a day and the promotion is a one-arm decision at each scale.

## Mechanism knobs (all in `temporal/temporal_router.py`, none change FLOPs)

Existing: `TEMPORAL_RESIDENCY_R` (R = k is the shipped constraint, R = E is a full MoE),
`TEMPORAL_R_SCHEDULE` (per-layer R), `TEMPORAL_COHERENCE_LAMBDA` (BCE pulling raw router logits
toward the resident set), the swap trigger threshold and the swaps-per-token count.

To add, default off so nothing recorded changes:
- `TEMPORAL_ITER_SCHEDULE`: `<iter>:<R>,...`, R read from Megatron's `curr_iteration`. One run,
  one process, the switch or ramp happens inside the training loop; no exit-and-resume, so the lr
  schedule and data order are exactly the baseline's.
- `TEMPORAL_FREE_FRAC_SCHEDULE`: `<iter>:<fraction>,...`, the fraction of sequences in each
  micro-batch that see no constraint (rows of the mask set to all-True, chosen by a generator
  seeded on the iteration and layer), a heterogeneous batch with an annealable mix.
- `TEMPORAL_SHADOW=1`: compute the resident set at R but do not apply it; only the coherence loss
  sees it. The soft constraint: a free model regularised toward temporally coherent routing.

## Round 1 (grain 3, the setting where the 1e19 gap is largest)

| arm | recipe | what it tests |
|---|---|---|
| C0 | R = E throughout, through the router path | the reference: full MoE on this corpus, same code path |
| C1 | R = k for the first half, then R = E | the user's proposal, early switch |
| C2 | R = k for the first two thirds, then R = E | late switch: more temporal, less recovery |
| C3 | R ramp k, 2k, 4k, 8k, E at 0, 20, 40, 60, 75% of training | annealed constraint, no shock |
| C4 | constrained fraction of sequences 1 until 40%, linear to 0 at 80% | heterogeneous batches |
| C5 | free model, shadow resident set at R = k, coherence lambda 0.01 | the soft constraint alone |

Each run logs the test loss every tenth of training, so the unmask shock and the recovery after a
switch are visible (`analysis/parse_run.py`). Decision rules after each result:

- C1 or C2 wins: refine the switch point (0.4, 0.8) and try a WSD schedule whose decay starts at
  the switch, so the free phase gets the full annealing; then run the best on grain 1 for
  transfer, then promote.
- C1 and C2 both lose and the loss trajectory shows the shock never fully recovers: C3 and C4
  are the shock-free versions; if one of them wins, refine its schedule.
- Everything ends at the baseline within 0.005: the constraint early in training is neither
  helping nor hurting the final basin; run C5 and C6 (below) before concluding.
- Everything loses by more than the temporal-at-R=k gap explains: the constrained phase costs
  optimisation progress that the free phase cannot recoup at this budget; report and stop at 1e17.

Driver arm names: C1 = SW0p5, C2 = SW0p667, C3 = RAMP0p75, C4 = HET0p4-0p8, C5 = SHD0p01, C6 = SAND
(`tmoe_curriculum_1e17.sh`); round 2 refinements are chosen by `curriculum_decide.py`.

## Round 2 candidates (chosen by round 1)

- C6 sandwich: free, then R = k for a middle quarter, then free. The constraint as a mid-training
  perturbation that breaks expert co-adaptation, in the spirit of the substitution result.
- C7 router reset at the switch: re-initialise the router (or raise its temperature) when the
  constraint lifts, if the shock is a router artefact rather than an expert one.
- C8 depth-wise curriculum with `TEMPORAL_R_SCHEDULE`: lift the constraint layer by layer, early
  layers first, since substitution tolerance varies with depth.
- C9 swap-budget anneal: keep R = k and raise the swaps per token from 1 toward k, a different
  path from the constraint to freedom than growing R.
- C10 alternating batches: every other iteration constrained, a batch-level heterogeneous mix.

## Promotion

The best 1e17 recipe on grain 3 is run once at grain 1 (transfer) and, if it wins there too or is
within noise, once at 1e18 (`flame38m_g3` config, against the seed triplet mean 4.0136, bar 0.010)
and then at 1e19 (`moe_fine_g3_1e19` config, against 3.1578). Every run is a single seed; the
recorded baselines are never re-run.

## Records

Runs land in `results/phase0/runs/<name>` with `run.meta` and `train.log`; the summary table
`results/ablations/curriculum_1e17.csv` (arm, recipe, test CE, BPB, delta vs C0, per-tenth
validation losses) is produced by `analysis/residency/curriculum_csv.py`. The log of what was run and
why goes in `REBUILD_RESULTS.md` as it happens.

## Round W: the weak constraint (2026-09-05, user's proposal)

The shipped constraint keeps k - 1 of the previous token's residents (one swap per token). A
weaker policy keeps only k/2 (k/2 swaps per token): the same inductive bias, less of it, and
less to give up when the constraint lifts. The scan already implements a swap budget
(`swaps`, bit-exact Triton kernel checked against the reference at startup); it is now exposed
as `TEMPORAL_SWAPS` with `TEMPORAL_SWAPS_SCHEDULE` for annealing. Grain 1 (k = 6), 1e17, the
50k corpus; controls C0 3.7661 and C0b 3.7421 (both with router blow-ups), shadow arms
3.7297 and 3.7337. Whole-run constrained arms are re-scored unconstrained in place after
training (sweep eval, tag `cross`), and their own constrained quality is the `native` row.

| arm | recipe | question |
|---|---|---|
| WK3 | 3 swaps per token (k/2), whole run | does the weak bias help the free model? what does it cost under its own constraint? |
| WK1 | 1 swap per token, whole run (the shipped temporal model on this corpus) | the anchor: strong bias, both evals |
| WK2 | 2 swaps per token | the dose in between |
| WK3SW0p5 | 3 swaps until half, then free | the curriculum with the weak bias |

Then, by the results: `WKA3-<f>` anneals the budget from 3 to k (free) at fraction f; the
best is replicated once on grain 3 if it wins on grain 1.
