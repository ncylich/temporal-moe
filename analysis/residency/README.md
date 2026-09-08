# Per-Layer Embeddings (PLE) and per-layer residency relaxation

Implementation for [`PLE_PLAN.md`](../../PLE_PLAN.md). **Results and findings live in
[`results/ablations/ple_RESULTS.md`](../../results/ablations/ple_RESULTS.md); every number is in
`results/ablations/ple_results.csv`.** This file documents only the code.

## Files

| file | what it is |
|---|---|
| `residency.py` | the rolling-residency constraint, plus `set_free_layers()` for per-layer relaxation |
| `ple.py` | the factored PLE table and the post-MoE install |
| `train_ple.py` | every training cell: PLE ranks, LoRA surface, free-layer sets, calibrated init, resume |
| `olmoe_paths.py` | path contract for the base checkpoint and corpus, with manifest verification |
| `calibrate.py` | closed-form Δ capture, shrinkage at an estimated λ\*, precision-weighted SVD |
| `cal_stack.py` | Cal-0 norm calibration, and the norm↔PLE stack in either order |

| `eval_table.py` | score a table with NO training (§9), and bucket recovery by frequency (§8.3) |
| `locus.py` | §8.1 token-vs-context locus probe |
| `row_norms.py` | §2 diagnostic: row norm bucketed by occurrence count |
| `accounting.py` | parameters, flash fetch, training memory, corpus coverage |
| `memory_probe.py` | measured memory decomposition per rank and micro-batch |
| `heldout.py` | builds the held-out token set the zero-property check uses |
| `checks.py` | all correctness checks: `init`, `placement`, `grad`, `zero`, `bitwise` |
| `report.py` | §5 ladder gates, and the layer-damage figure |
| `consolidate.py` | folds every intermediate into the two committed CSVs |

Per-layer residency **relaxation** is a separate line of inquiry sharing this code:

| file | what it is |
|---|---|
| `layer_ablation.py` | per-layer damage: constrain one MoE layer at a time |
| `joint_free.py` | joint free-set damage vs the additive prediction from the solo profile |
| `train_ple.py --free-set` | training cells with chosen layers unconstrained |

Its results are in `results/ablations/layer_freeing_results.csv` and
[`layer_freeing_RESULTS.md`](../../results/ablations/layer_freeing_RESULTS.md), kept apart from the
PLE tables on purpose: PLE adds a lookup and leaves the constraint intact, layer freeing removes the
constraint and adds nothing.

## Artifacts and paths

`residency.py` is the adaptation program's `olmoe_residency.py` with two hardcoded paths replaced by
resolution through `analysis/paths.py`, plus `load_c_adapted()` and `set_free_layers()`. The
unmodified original is archived at [`scripts/adaptation/`](../../scripts/adaptation/README.md) and
the two are **not** interchangeable: the archive is the record of what produced the published
numbers.

The base checkpoint (27 GB) and corpus (4.4 GB) resolve via `$TMOE_OLMOE_MODEL` / `$TMOE_OLMOE_DATA`,
then `$TMOE_OLMOE_HOME`, then `<repo parent>/olmoe-adapt`. The corpus files are in
`results/MANIFEST.csv` with sha256; `olmoe_paths.py --full` verifies a local copy, which is what
licenses using local disk instead of re-downloading.

Each script writes a small intermediate CSV into `results/ablations/`. Those are **gitignored**;
`consolidate.py` folds them into the two tracked files, `ple_results.csv` and
`layer_freeing_results.csv`. Re-run any script, then `consolidate.py`, and the committed artifacts
are rebuilt.

## Design decisions that differ from the plan's literal text

**Gate initialisation.** §2 asks for both a zero-init table and a gate "initialised so the branch
starts inert". Both at once is a permanent fixed point — `dL/dU = dL/dV = dL/dg = 0`, so the branch
can never leave zero and a cell would train nothing while logging a healthy loss curve. Verified by
`checks.py init`. The gate starts at 1.0; the zero table alone provides inertness, bit-identical
parity, and the rare-row zero property.

**Weight decay is 0 on every rung**, making rank the only regulariser so a null at low rank is
attributable. §9's λ\* is *not* a source for it — λ is count-space pseudo-observations, decay is a
loss coefficient, and the functional forms differ. The cost is acknowledged: with Adam and no decay a
row seen once takes nearly the step of a row seen ten thousand times, so full rank is exposed to
memorising noise in rare rows. `row_norms.py` is the diagnostic; measured norms rise monotonically
with frequency, so the risk did not materialise.

**Gradient clipping is per-surface.** Clipping the C parameters and the PLE tensors jointly would
make the C-surface updates depend on the PLE gradient norm, confounding PLE's contribution with a
changed C trajectory.

**Free layers relax the constraint.** A cell using `--free-set` or `--free-layers` is not comparable
to a full-residency number without stating the cost: a freed layer keeps all 64 experts resident
instead of 8. FLOPs are unchanged — both regimes activate exactly top-8 of 64, and residency only
restricts which eight are eligible.

## Placement

Post-MoE only, added to the layer output: `out = h + MoE(LN2(h)) + g_l * PLE[tok, l]`. There is no
pre-MoE flag and no code path that could produce one (§13). `checks.py placement` verifies the
guarantee that follows: layer 0's router logits are bitwise identical with and without an active
table. Deeper layers' routing does move, which is inherent to writing into the residual stream.

## Reproducing

```bash
export TMOE_ROOT=$(git rev-parse --show-toplevel)
PY=<adaptation venv python>        # torch 2.4.1+cu124, transformers 5.12.1, bitsandbytes

$PY analysis/residency/olmoe_paths.py --full          # verify the corpus against MANIFEST.csv
$PY analysis/residency/accounting.py --accounting     # params / bandwidth / training memory   (CPU)
$PY analysis/residency/accounting.py --coverage --model-for-loss C                            # (GPU)
$PY analysis/residency/checks.py init                 # zero-init property, gradient reachability (CPU)
$PY analysis/residency/checks.py placement            # post-MoE guarantee                     (GPU)
$PY analysis/residency/checks.py grad                 # gradient survives checkpointing        (GPU)
$PY analysis/residency/heldout.py                     # build the zero-property held-out set   (GPU)
$PY analysis/residency/layer_ablation.py              # per-layer damage                       (GPU)

# a training cell, e.g. the rank ladder and the best free-set configuration
$PY analysis/ple/train_ple.py --tag ladder_r512 --rank 512 --tokens 50000000 \
      --mb 16 --eval-every 10000000 --table-wd 0.0 --adam8bit --heldout
$PY analysis/ple/train_ple.py --tag ce_free_0_1_15 --rank off --tokens 50000000 \
      --mb 16 --eval-every 10000000 --lora 32 --free-set 0,1,15

# after any cell
$PY analysis/residency/checks.py zero --trained <ple_table_TAG.pt> --train-tokens 50000000
$PY analysis/residency/consolidate.py                 # rebuild the committed CSV
```

`checks.py bitwise` needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` and, for exact gradient comparison,
`--no-flash`. Flash attention is **on** for every training cell per §10; flash-off is only for
correctness checks. `report.py figure` needs matplotlib, absent from the adaptation venv.
