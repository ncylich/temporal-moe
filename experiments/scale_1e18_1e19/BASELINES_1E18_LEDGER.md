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
and dropped for time (see B7).

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

## B6. Results at 1e18 (fine grain, 18 of 192, seed 1234, same split)

Test BPB is each run's end-of-run test eval (50 batches of 1,024 sequences), the same eval
as the reference rows. BPB tax is against the standard MoE of the same seed and split
(1.3461). Loads and effective experts come from one probe (`router_probe.py` on the first
training micro-batch, then `cosmoes_metrics.py`), run tonight on all four checkpoints.

| model | test BPB | BPB tax over standard MoE | loads / token-layer | effective experts (of 192) |
|---|---|---|---|---|
| temporal (ours) | 1.3354 | -0.80% | 1.00 served (raw demand 11.35) | 183.1 |
| standard MoE | 1.3461 | -- | 13.85 | 190.7 |
| CoSMoEs BlES, weight 0.03 | 1.4089 | +4.67% | 9.35 | 165.1 |
| sticky routing, weight 5 | 1.3612 | +1.13% | 10.83 | 189.5 |

Test CE in nats: temporal 3.976762, standard MoE 4.008667, CoSMoEs 4.195695, sticky
4.053793. The paper's seed table reports three-seed means for the first two rows (1.3339
and 1.3478); against the mean MoE the taxes are -1.03%, +4.53% and +1.00%.

- CoSMoEs is worse than ours on both axes. It lands 0.0735 BPB above ours and 0.0628 above the
  standard MoE, about twenty times the 0.003 seed spread, and above the dense floor
  (1.3911). It needs 9.35 loads per token-layer against our 1.00, and its effective pool
  falls to 165 of 192 with 13 experts neglected per layer on average. Its tax grew from
  +3.2% at 1e16 to +4.67% here at the same weight.
- Sticky routing is also worse than ours on both axes. It lands 0.0258 BPB above ours
  (about eight times the seed spread) and 0.0151 above the standard MoE, and it needs 10.83 loads per
  token-layer against our 1.00. The pool stays in use (189.5 effective experts, none
  neglected). The 1e16 margin over ours was noise-sized (B4); at 1e18 it is not, because
  our model moves below the standard MoE at this budget while sticky stays above it.
- Neither loss bounds the demanded set. The lowest demand reached is 9.35 experts per
  token-layer, and our own model's raw demand (11.35) is in the same range as both
  baselines. The 1.00 in our row comes from the residency rule, which the two losses do
  not have.

Checks on these numbers:

- One-card re-evaluation with frozen weights (`EVALONLY=1`) of each two-card checkpoint:
  CoSMoEs test CE 4.195698 against the run's own 4.195695, sticky 4.053793 against
  4.053793. The two-card runs' test numbers are not an artifact of two-rank evaluation.
- The same re-evaluation on the published references: standard MoE 4.008724 against the
  recorded 4.008667, temporal 3.976817 against 3.976762.
- First logged losses against the one-card reference log (10.81400, 10.44734): CoSMoEs
  10.81355, 10.44825; sticky 10.81393, 10.44745.
- No NaN or skipped iterations in either run. Both logs print the loss as active on rank 0
  and rank 1.

Raw material: `results/ablations/baselines_1e18.csv`,
`experiments/scale_1e18_1e19/baselines_1e18_logs/` (filtered train logs, run metadata, eval
and probe lines, per-run probe metrics, the pod's `pip freeze --local`).

## B7. What went wrong and how it was caught

- Run time at 1e16 was misjudged. The plan assumed a 1e16 run takes about 10 minutes.
  A fine-grain cell takes 36 minutes alone on one card (0.97 s per iteration, the same as
  the recorded sweep's own logs). I noticed when the first status check showed 2.0 s per
  iteration with three jobs on a card. I first read that as a sharing penalty, killed the
  round and relaunched one job per card, then saw the same 0.95 s per iteration alone and
  checked the recorded logs. Sharing a card gives about 1.4 times the throughput of running
  serially, so the remaining jobs were added back alongside. Net cost about 8 minutes of
  restarts, and the 1e16 round took 75 minutes instead of the 20 planned.
- Four jobs died with their launcher. Killing the launcher script to change the queue
  also killed its four running children. The next status check showed it (their iteration
  counters had stopped at 110) and relaunched detached. About 5 minutes of four cards lost.
- The two-card BlES cell at 1e16 was dropped. It was meant to check the auxiliary loss
  on two ranks against the recorded 1e16 BlES number. With runs at 36 minutes it would have
  delayed the 1e18 starts, and the recorded number is on a different split anyway. The
  per-rank log line and the 1e18 loss parity stand in for it.
- The recorded 1e16 split cannot be reproduced, because the recorded sweep used 24 tok16k
  shards and 12 are published. I caught this by reading the `data_path` line of the recorded
  logs. The 1e16 table in B4 carries its own references for that reason, and the CoSMoEs
  weight rests on the older table, which was not re-run on tonight's split.
- `find` returned the 1e18 shards in a different order and count than the reference logs.
  This was caught the same way, before any run, and fixed with `SHARDS`.
- One 1e16 process aborted at exit. `g3_sticky0.5_1e16` received SIGABRT during
  teardown, after its test eval and checkpoint save, so the cell script stopped before the
  probe. The missing metrics line gave it away, and I ran the probe by hand on the saved
  checkpoint. No other run aborted this way.
- The first TransformerEngine build failed in under a minute because cmake
  could not find `cudnn.h`. Rebuilt with `CUDNN_PATH` pointing at the pip `nvidia/cudnn`
  directory; 25 minutes.
- Four H200s were only available on the secure tier at $21.16 per hour
  (community tier, $14.36, had no stock). The pod ran from 05:18 to 11:29 UTC, about $131.
  The pod's SSH daemon was not running at boot and had to be started through the gateway.

## B8. Not done or not verified

- One seed per baseline. The 0.003 seed spread quoted is from the reference rows.
- The baselines trained on two cards and the references on one. The evidence that this
  does not matter is in B3 and B6; a one-card repeat of either baseline was not run.
- The baselines went through the patched router at R=E and the standard MoE reference
  through stock `pretrain_gpt.py`. A no-loss run through the patched path at 1e18 was not
  made. At 1e16 the earlier sweep treated the two as the same model, and the first-step
  losses here agree with the stock reference to three decimals.
- Sticky routing was implemented from the paper's HTML text (Eq. 9 to 11). Batch
  aggregation is not stated there and is taken as a mean. The paper's experiments use 4
  experts with top-2; nothing here reproduces its numbers in its own setting.
- The CoSMoEs weight was chosen from the paper's 1e16 table, measured on a different split
  and TransformerEngine build than tonight's cells.
- Loads are router demand on one fixed batch of 32 sequences, not a measurement in a
  serving engine.
- The weights were chosen so that each baseline is worse than ours at 1e16. Matched-quality
  points (BlES 0.01, sticky 0.5) and heavier points (BlES 0.1, sticky 50) were not run at
  1e18.

## B9. Pod state

Pod `k425im1s3drxei` (`h200x4-baselines-1e18`) was stopped at 11:29 UTC and shows EXITED.
It was not terminated. Its 250 GB volume at `/workspace` holds the repository checkout at
this branch with the built venv (TransformerEngine and apex), both corpora
(`data/dclm_tokenized`, `data/tok16k_full`), and under `results/phase0/runs/` the two new
1e18 checkpoints (`flame38m_g3_cosmoes0.03_1e18`, `flame38m_g3_sticky5_1e18`, 1.6 GB each
with their router logs), the two reference checkpoints, and every 1e16 cell of B3 and B4.
A rerun needs only `METHOD=... LAMBDA=... GPUS=... bash
experiments/scale_1e18_1e19/baselines_1e18.sh` after the pod is started; the SSH daemon may
need `service ssh start` through the gateway first.
