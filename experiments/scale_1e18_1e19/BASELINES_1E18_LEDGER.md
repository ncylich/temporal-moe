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
