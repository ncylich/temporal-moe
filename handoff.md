# Handoff: PLE adaptation program

You are picking up an experiment program on a repository with a long history. This document is the
context you need that `PLE_PLAN.md` does not carry: what has already been established, where the
artifacts live, and which mistakes this program has already paid for.

## What this repo is

Temporal MoE: a rolling-residency constraint on mixture-of-experts routing. Only `R` experts per
layer are resident, gates are a softmax over resident logits only, and at most one expert may be
swapped in per token per layer. With `R = k` the resident set is the active set, so serving memory
scales with active rather than total parameters. Two lines of evidence exist: FLAME-MoE models
trained with the constraint from scratch (1e16 → 1e19 FLOPs), and a released checkpoint
(`allenai/OLMoE-1B-7B-0125`) adapted to the constraint after the fact.

Your program extends the second line.

## What is already established — do not re-derive these

Adaptation of OLMoE to `R = k = 8` of 64, all on the audited held-out slice, divisor 3.1089:

| result | number | where |
|---|---|---|
| Zero-shot impose gap | base 0.6727 → 2.7507 BPB (+2.078) | `results/ablations/olmoe_adapt_impose.csv` |
| Router-only adaptation | 1.2825, 70.7% recovery | `olmoe_adapt_bakeoff.csv` |
| Router + norm gains (133K params) | 0.8505, 91.4% | same |
| Router + LoRA r32 (235M params) | 0.8507, 91.4%, tying the 133K version | same |
| Router + norms + LoRA | 0.8149, 93.2%, bake-off winner | same |
| Full finetune, all 6.92B params | 0.8106, 93.4%, the "constraint price" | same |
| Downstream recovery | 74.7% of the accuracy the mask destroys | `olmoe_adapt_downstream.csv` |
| Locus flip under adaptation | ctx−tok AUC −0.0041 → +0.0493 | `olmoe_adapt_forensics.csv` |
| Token-efficiency crossing | 0.25B adapted ≈ 28–46B from scratch | `olmoe_scratch_ladder.csv` |

Dead ends, closed with evidence. **Do not propose these again**: constraint annealing (both on the
routing surface and the capacity surface), self-distillation from the free-routing teacher (both
surfaces), closed-form moment-matching calibration, calibrated initialization, LoRA rank above 32,
and offline MinFlow scheduling (the live greedy scan beats a forced hindsight-optimal schedule
because the model's own logits drift).

Narrative write-ups: `docs/research/FINDINGS.md`,
`results/ablations/olmoe_adapt_RESULTS.md`, `docs/research/olmoe-adaptation-plan.md` (its close-out
section carries the corrections), `docs/research/mechanism/delexicalization.md`. The per-CSV index
is `results/ablations/README.md`.

## Where the artifacts are

**The pod's disk no longer matters.** Every checkpoint, log, and captured tensor lives in four
public Hugging Face repos, indexed by `results/MANIFEST.csv` (local path, HF repo, HF path, size,
sha256):

- `ncylich/temporal-moe-ckpts`: model checkpoints and run logs
- `ncylich/temporal-moe-extras`: probe captures and router logs
- `ncylich/temporal-moe-router-adapt`: adaptation checkpoints (router-only are ~4 MB and tracked
  in-repo under `results/ablations/adapt_ckpts/`; LoRA-bearing ones are on HF)
- `ncylich/temporal-moe-corpus`: the 1B-token adaptation corpus and the audited held-out slice

Fetch what you need from the manifest; do not assume a path exists on local disk.

## Tooling

- `analysis/paths.py` is the canonical root resolver: `$TMOE_ROOT`, then git toplevel, then file
  location. Probes import `from paths import ROOT`. Do not hardcode absolute paths; that was
  cleaned up repo-wide and should stay clean.
- `analysis/probes/` holds the probe suite: locus and de-lexicalization (`delex_*.py`), stability
  and quantization (`stability_*.py`, `fakequant_eval.py`), lm-eval plumbing (`run_lmeval.py`,
  `lmeval_*_to_csv.py`, and `lmeval_task_dataset_id.patch`, which repoints task datasets to their
  canonical Hugging Face IDs for recent `huggingface_hub`).
- `run_lmeval.py` installs the residency router when `TEMPORAL=1`, so temporal checkpoints are
  scored in their native masked regime. Forgetting this silently costs 0.05–0.07 accuracy.
- `temporal/temporal_router.py` is the constraint itself. `compute_resident_mask` and its
  accelerated variant are framework-independent; only `install()` is Megatron-specific.

## Mistakes this program has already paid for

Each of these cost real time or produced a wrong number that had to be retracted.

1. **Evaluating a schedule in windows inflates it.** A 256-token window with a cold fill per window
   roughly doubled estimated captured mass and hid a policy's accumulated suboptimality. State the
   cold-fill regime in any residency evaluation.
2. **A screen at the wrong depth does not transfer.** An architecture variant passed a 5-MoE-layer
   screen at +0.0036 BPB and cost +0.0444 at 8 layers, a 12× miss and 375× on its MoE leg.
   Screen where you deploy.
3. **Single-seed wins under ~0.01 BPB evaporate.** Two headline claims were retracted after a
   second seed. σ here is 0.006; treat 2σ as the floor for a claim.
4. **Divisor confusion.** Three divisors are live in this repo (2.7568, 2.7600, 2.9780) plus 3.1089
   for the OLMoE slice, and they are genuinely different corpora, not an error. Re-derive
   `ln2 × bytes_per_token` from the actual evaluation and record it in the CSV header.
5. **Reporting drift.** An executor once ran a cancelled arm and a deprioritized one for ~6.5 GPU
   hours because it never polled its instruction channel; another quoted a stale ETA for hours
   after the job had already finished. Read the log's terminal line before quoting status, and
   check for new instructions at every task boundary.

## What is expected of you

Work the phases in `PLE_PLAN.md` in order. Report each deliverable as its own message with its CSV
committed **and pushed**, then verify the file is actually on the branch — a commit that did not
reach the remote has happened here before.

Stop and report, rather than deciding, when: a gate is ambiguous, a result contradicts a
pre-registered expectation, a run fails, or the plan does not cover the situation. Do not start
work beyond the plan without sign-off. Flagging a problem with the plan is welcome and has improved
several of these programs; silently working around it is not.

Numbers you report will be recomputed from your CSVs before they are believed. That is not
distrust. It is the standard the whole program has been held to, including its orchestrator, whose
own comparator error was caught this way.
