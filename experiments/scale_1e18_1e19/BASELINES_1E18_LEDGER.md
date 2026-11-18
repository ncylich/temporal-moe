# Published locality losses as pretraining baselines at 1e18 (fine grain, 18 of 192)

Two trained baselines against the 1e18 rows of the paper's seed table: CoSMoEs' block-wise
expert selection loss (BlES, Huber et al., arXiv:2503.00245) and sticky routing (Kayyam,
arXiv:2607.08780). One run each, seed 1234, on a 4x H200 pod (RunPod `k425im1s3drxei`,
secure tier, 2026-11-18 UTC). Entries are in the order things happened. Rules are written
before the runs they govern and are not edited afterwards.

Metrics. BPB is test bits per byte, lower is better (CE in nats / 2.9780 for the pythia 50k
tokenizer at 1e18, / 2.7568 for the 16k tokenizer at 1e16). BPB tax is the relative increase
over the standard MoE on the same split. Loads per token-layer is router demand on one fixed
batch: the experts that must load if only the previous token's top-k set stays resident
(`analysis/probes/cosmoes_metrics.py`, column `xfers_tl`), between 0 and k = 18, lower is
better. Effective experts is exp(entropy) of the marginal top-k load, between 1 and 192,
higher means the pool is in use.

## B0. Which launcher and data produced the 1e18 reference rows

- Launcher: `experiments/scale_1e18_1e19/flame38m_run.sh` with `GRAIN=3 MICRO_BATCH=32`,
  as sequenced by `flame38m_1e18_seq.sh`. Standard MoE adds `MOE_FULL=1`
  (`pretrain_gpt.py`); temporal is the default mode with `TEMPORAL_EVICT=min_logit`.
  Shape 256 hidden, 9 layers, 192 experts, top-18, expert width 58, shared 352, global
  batch 1024, 2,121 iterations, lr 3e-4 to 3e-5 WSD, pythia-12b tokenizer, seed 1234.
- Checked against the published checkpoints' own logs (pulled with `scripts/artifacts.py`):
  `flame38m_g3_moe` ends at test CE 4.008667 = 1.3461 BPB and `flame38m_g3_temporal` at
  3.976762 = 1.3354 BPB. These are the seed-1 cells of the seed table (means 1.3478 and
  1.3339 over three seeds).
- Data: both logs list 12 shards of `data/dclm_tokenized` in the order
  11 04 06 10 08 02 03 07 09 05 01 00. The published corpus now has 22 shards and `find`
  returns them in a different order on this pod, which would move the 90/5/5 split. The
  launchers gained a `SHARDS` variable and `baselines_1e18.sh` pins that list and order.
- Environment: torch 2.4.1+cu124, Python 3.11.10, TransformerEngine built from the pinned
  submodule `fc03478`, apex from source, as in `docs/ENVIRONMENT.md`.

## B1. Weight rules, fixed before any 1e18 result

**CoSMoEs: weight 0.03.** Rule: the lightest weight in the paper's 1e16 table
(`tab:cosmoes`, fine grain) that is worse than our model on both axes, BPB tax and loads.
Ours at 1e16 is +1.2% BPB, served at 1.00 loads per token-layer (raw demand 11.39). Weight
0.01 gives 12.85 loads at +0.8%, which is not worse on BPB. Weight 0.03 gives 11.93 loads
at +3.2%, worse on both. So 0.03. Recorded 2026-11-18 05:40 UTC, before the run started.

**Sticky routing: chosen by the same rule from a 1e16 sweep (B4).** The lightest weight
whose 1e16 cell is worse than ours on both BPB tax and loads per token-layer, with "ours"
measured on the same 1e16 split as the sweep. If no weight changes loads at all, or every
weight beats ours on BPB, that is reported and the weight with the largest load reduction
is used instead. The weight is not changed after its 1e18 result is seen.

## B2. Setup checks (05:18 to 06:40 UTC)

- Smoke, standard MoE at the 1e18 config, one card: starts and logs loss, 11.4 s per
  iteration (the reference H100 log shows 12.5).
- Smoke, two cards, BlES 0.03 through the patched router at R=E: 6.0 s per iteration, and
  the log prints `cosmoes loss active on rank 0` and `on rank 1`. Its loss tracks the
  one-card standard-MoE smoke (10.1150 vs 10.1155 at iteration 10).
- Re-evaluating the published `flame38m_g3_moe` checkpoint here with frozen weights
  (`EVALONLY=1`, one card, pinned shards) gives test CE 4.008724 against the recorded
  4.008667, a difference of 0.00002 BPB. Its validation number, 4.011249, matches the
  reference's in-training validation eval (4.011211) and not its end-of-run "on validation
  set" line (4.004456), because that second eval continues the same iterator and so covers
  the next 50 batches. The test iterator is fresh in both cases, so test numbers compare
  directly.

## B3. Two-card training, validated at 1e16

Same standard MoE cell (shape s0, 18 of 192, tok16k, seed 1234, 2,232 iterations) trained
once on one card and once on two with the same global batch.

| run | test CE | test BPB | loads / token-layer | effective experts |
|---|---|---|---|---|
| one card (`v_g3_moe_1c`) | 4.005924 | 1.4531 | 14.15 | 191.2 |
| two cards (`v_g3_moe_2c`) | 4.019447 | 1.4580 | 14.06 | 191.2 |
| recorded (`g3_ref_moe_1e16`, 24-shard corpus) | 4.033053 | 1.4629 | 14.13 | 190.9 |

- One card against two: 0.0049 BPB apart, inside the 0.005 bound but only just, and above
  the 0.003 quoted as seed noise. `seed_replicates.csv` puts same-seed cross-box pairs at
  0.0004 to 0.0089, so this is ordinary run-to-run spread, not evidence of a two-card fault.
- Against the recorded value: the two-card run is 0.0049 away and the one-card run 0.0098
  away. This part of the check is not clean and cannot be made clean here: the recorded
  sweep trained on 24 tok16k shards and only parts 00 to 11 are published, so these cells
  use a different train and test split. All 1e16 comparisons below are therefore made
  against the same-split references trained tonight, not against the paper's table.
- Probe on the two-card checkpoint: 14.06 loads against 14.15 (one card) and 14.13
  (recorded); effective experts 191.2, 191.2, 190.9.
- Auxiliary loss on every rank: confirmed in the smoke (B2) and again in both 1e18 logs.
- Stronger evidence at 1e18 itself: both two-card runs start from the reference's seed and
  data, and their first logged losses match the one-card reference log. Reference
  `flame38m_g3_moe`: 10.81400 at iteration 5, 10.44734 at iteration 10. BlES run: 10.81355
  and 10.44825. Sticky run: 10.81393 at iteration 5.

Verdict: two-card training accepted. A dedicated two-card BlES cell at 1e16 was planned
and dropped for time (see B6).

## B4. Sticky routing sweep at 1e16 (fine grain, one card each, seed 1234)

Loss as in the paper (arXiv:2607.08780, Eq. 9 and 10): mean over consecutive token pairs of
the squared L2 distance between the two tokens' full-softmax gate vectors, averaged over
the MoE layers, added to the standard MoE with its usual load-balancing loss. No anchor
variant, no temperature, no stop-gradient. Unit test:
`temporal/tests/test_ablation_mechanisms.py::test_sticky_routing_loss_matches_eq9_by_hand`.
BPB tax is against the one-card standard MoE of B3 (1.4531), same split.

| cell | test BPB | BPB tax | loads / token-layer | effective experts |
|---|---|---|---|---|
| standard MoE | 1.4531 | -- | 14.15 | 191.2 |
| temporal (ours), raw demand | 1.4713 | +1.25% | 11.27 | 185.3 |
| temporal (ours), served | 1.4713 | +1.25% | 1.00 | 185.3 |
| sticky 0.05 | 1.4546 | +0.10% | 14.01 | 191.2 |
| sticky 0.15 | 1.4586 | +0.38% | 13.62 | 191.3 |
| sticky 0.5 | 1.4607 | +0.52% | 12.69 | 191.3 |
| sticky 5 | 1.4755 | +1.54% | 10.25 | 190.5 |
| sticky 50 | 1.4931 | +2.75% | 8.44 | 166.4 |
| sticky 500 | 1.5428 | +6.17% | 10.06 | 76.3 |

- The paper's range (0.05 to 0.5) is nearly inert here: with 192 experts the gates are
  diffuse and the loss term is about 0.003 to 0.009 out of a possible 2, so the sweep was
  widened to 500 from the start. Taxes below about 0.4% are inside the one-card against
  two-card spread of B3 and should be read as zero.
- Loads fall to 8.44 at weight 50 and rise again at 500, where the pool collapses to 76
  effective experts. No weight comes near a bounded set.
- Every weight is worse than ours on loads (ours serves at 1.00). On BPB, 0.05, 0.15 and
  0.5 are better than ours, and 5 is the lightest that is worse (+1.54% against +1.25%).

**Sticky weight for 1e18: 5**, by the rule in B1. Fixed 2026-11-18 07:28 UTC, when the
1e18 sticky run was launched. Caveat recorded with the choice: the margin over ours on BPB
at 1e16 is 0.004 BPB, the size of run-to-run spread, so "worse on BPB" at this weight is
not a robust 1e16 statement. The rule was applied as written.

## B5. The 1e18 runs

- CoSMoEs BlES 0.03: `METHOD=cosmoes LAMBDA=0.03 GPUS=2,3 baselines_1e18.sh`, started
  06:58 UTC on two cards, before the one-card half of B3 had finished (it shared its cards
  with the last 1e16 cells for about 20 minutes, which only slowed it).
- Sticky 5: `METHOD=sticky LAMBDA=5 GPUS=0,1 baselines_1e18.sh`, started 07:28 UTC.
- Both: seed 1234, the reference's 12 shards and order, 2,121 iterations, global batch
  1024, micro-batch 32 per rank.
