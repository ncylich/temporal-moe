# ARCHIVE — OLMoE renorm-era runs. ALL RESULTS WRONG. DO NOT USE.

Every adaptation result in this directory was produced by code that applied the **wrong
gate convention** to OLMoE-1B-7B: after top-k selection (and after the residency mask),
expert weights were **renormalized to sum to 1** over the selected set. OLMoE is a
`norm_topk_prob=False` model — its correct convention keeps the raw softmax mass of the
selected experts (~0.40 of total on average) with **no** post-top-k normalization.

The error's measured effect (`results/ablations/olmoe_gatemass_remeasure.csv`, producer
`analysis/ple/olmoe_remeasure.py`): renormalization raises top-k gate mass from ~0.40 to
1.0, scaling every MoE block's output by ~2.5x compounded over 16 layers. The identical
untrained R=8 cell measures **BPB 2.6717 under renorm vs 0.8393 under the correct
convention** — the intervention these files study is not rolling residency on OLMoE, but
rolling residency on a differently-scaled model that does not exist.

Consequently:

- **Every BPB, CE, recovery %, downstream number, and conclusion in the
  `olmoe_adapt_*` files is wrong.** That includes the published Stage-2 results:
  impose 2.7507, the ~1.28 router-only arms, the CE winner 0.8149, the full-finetune
  0.8106, and all "91-93% recovery" claims.
- **Every checkpoint in `adapt_ckpts/` was trained under the wrong convention** and
  must not be loaded as an adaptation surface for correct-convention work.
- **No number here may be compared against current-era results in any form**,
  absolute or relative.

These files are kept solely as records of *ideas explored* (arm designs: router-only,
norm-gain calibration, LoRA stacking, anneal curricula, self-distillation; the LR sweep
shape; telemetry/forensics methodology), in case any idea is revisited — under the
correct convention, from scratch.

`scripts/` holds the era's producer code (train_bakeoff, train_cal2, the O-series minflow
scripts, lmeval harnesses, dense_bracket, forensics). It is the code that implemented the
wrong convention and writes absolute paths from its original home; keep it for reading,
never for producing results.

Era distinction within this directory: the `olmoe_minflow_*` capture studies scored
candidate resident sets by **base softmax mass and never renormalized** (see their
headers), so they are not convention-poisoned — they are archived as superseded
old-era exploration, not as wrong results. Everything else listed above is wrong.

The correct-convention record lives in `results/ablations/` (see
`sweep_RESULTS.md`, `olmoe_gatemass_remeasure.csv`, `layer_freeing_RESULTS.md`):
untrained impose 0.839 at R=8, best adapted (T=1 distillation, 100M tokens) 0.7779.
