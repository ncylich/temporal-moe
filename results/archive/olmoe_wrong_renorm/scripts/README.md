# OLMoE adaptation scripts — verbatim archive

Every number in the OLMoE residency-adaptation results table was produced by the code in this
directory. Until this commit it existed in exactly one place, `/workspace/olmoe-adapt/scripts/` on
a rented pod, and nothing in the repository referenced it by name. `results/MANIFEST.csv` indexes
the checkpoints, logs and captured tensors that code produced, but never the code itself.

**These files are byte-identical copies of what ran.** Paths are hardcoded, nothing is tidied, and
the `analysis/paths.py` convention is deliberately not applied. A cleaned-up copy could no longer
testify to what actually executed. Any adaptation to make them run from a fresh clone belongs in a
later commit, so the diff shows exactly what changed between "what ran" and "what runs now".

## What produced which published number

| number | BPB | script |
|---|---|---|
| base, free routing | 0.6727 | `impose_bpb.py`, restated by `impose_restate.py` |
| impose R=8, untrained | 2.7507 | `impose_bpb.py` |
| A: router only | 1.2825 | `train_router.py` via `run_sweep.sh` |
| C: router + norm gains | 0.8505 | `train_bakeoff.py` arm C via `run_bakeoff.sh` |
| E: router + LoRA r32 | 0.8507 | `train_bakeoff.py` arm E |
| CE: router + norms + LoRA | 0.8149 | `train_bakeoff.py` arm CE; merged by `merge_ce.py` |
| F′: full finetune, 6.92B | 0.8106 | `train_fprime.py` via `run_fprime.sh` |

Supporting infrastructure, which matters as much as the trainers:

| concern | script |
|---|---|
| the constraint itself | `olmoe_residency.py` |
| audited held-out slice and its divisor **D = 3.1089** | `build_bpb_slice.py` |
| 1B-token adaptation corpus | `build_finetune_corpus.py` |
| eval-noise σ | `eval_noise_sigma.py` |
| downstream lm-eval | `lmeval_downstream.py`, `lmeval_impose.py` |
| from-scratch ladder / harness validation | `eval_scratch_ladder.py`, `eval_c4val.py`, `dense_bracket.py` |
| de-lexicalization and locus forensics | `hf_delex.py` |
| O-series residency scheduling | `oseries_*.py` |
| calibration probes (both rejected) | `cal0.py`, `train_cal2.py` |
| mask correctness checks | `verify1_identity.py`, `verify23_scan.py` |

## External dependencies these scripts assume

They are verbatim, so they still reference absolute paths that will not exist in a fresh clone:

- `olmoe_residency.py:19` inserts `/workspace/FLAME-MoE` on `sys.path` and imports
  `temporal.temporal_router`. That module **is** versioned in this repo, so the residency scan
  itself was never at risk — only the code around it.
- `olmoe_residency.py:237` defaults the base checkpoint to `/workspace/olmoe-adapt/model`
  (`allenai/OLMoE-1B-7B-0125`, re-downloadable).
- `train_bakeoff.py:21` writes to `/workspace/olmoe-adapt/data` and
  `/workspace/FLAME-MoE/results/ablations/adapt_ckpts`.
- The corpus tensors `finetune_ids.pt` and `bpb_slice_ids.pt` are in `results/MANIFEST.csv` with
  sha256 and are on Hugging Face.

## Optimizer settings, for the record

All four trainers set weight decay explicitly; none relies on a framework default:

```
train_bakeoff.py:51  torch.optim.AdamW(masters,   lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
train_router.py:31   torch.optim.AdamW(masters,   lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
train_cal2.py:86     torch.optim.AdamW(masters,   lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
train_fprime.py:85   Adam8bit(ft_params,          lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
```

Gradient clipping is 1.0 in all of them.

## Completeness

37 files, all verified byte-identical to the source directory by sha256 at archive time. Nothing was
reconstructed or reimplemented. All local imports resolve within this directory (`olmoe_residency`),
and every sibling script referenced by the four shell runners is present. `__pycache__` was excluded
as build output.
