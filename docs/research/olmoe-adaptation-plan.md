# OLMoE residency-adaptation plan (FINAL_TOUCHES row 10)

**Status: PLAN OF RECORD (2026-07-19). Nothing launched.** Target locked after the verified
candidate survey: `allenai/OLMoE-1B-7B-0125`, with DeepSeek-V2-Lite as fallback. Survey basis:
best measured local-routing consistency among small open MoEs (arXiv:2505.16056), no shared
expert, MoE on every layer, softmax top-k router matching our gating math, open pretraining
data and intermediate checkpoints, Apache 2.0, native llama.cpp `olmoe` architecture.

## Question and deliverables

Can a released, free-routed MoE checkpoint be adapted to the hard residency constraint by
router-only finetuning, and how much of the imposed quality gap does adaptation recover?

Paper deliverables, in order of value:
1. **Recovery number**: fraction of the zero-shot impose gap closed on held-out BPB.
2. **Impose-at-7B point**: extends the Section 4 impose column's scale trend (+0.43 BPB at
   $10^{19}$, ~200M active) to 7B total / 1.3B active, measured before any training.
3. **Real-weights serving row**: the fork's deploy measurement on a usable model instead of
   random weights.
4. **Routing forensics**: does adaptation reproduce the de-lexicalization signature we see when
   the constraint is trained in from scratch (selectivity drop, context-dominated experts)?

## Target facts and cell geometry

| | OLMoE-1B-7B-0125 |
|---|---|
| Params | 7B total, 1.3B active |
| Layers / hidden | 16 (MoE on all) / 2048 |
| Experts / top-k | 64 routed, top-8, no shared expert |
| Router | softmax top-k, aux loss 0.01, z-loss 0.001 |
| Context / tokenizer | 4096 / OLMoE (GPT-NeoX-style) |
| Hub format | safetensors stored fp32 (~28GB download), cast to bf16 at load |
| Training data | OLMoE-mix-0924 (~5T, ~one pass: treat all of it as seen), dolmino-mix-1124 anneal |

**Constraint cell: R = k = 8 of 64 per layer.** The paper's regime (the resident set is the
active set, so serving memory scales with active parameters), and it keeps active FLOPs identical
to the base model, so every quality delta attributes to the constraint rather than to reduced
capacity. Reuse `temporal/temporal_router.py::compute_resident_mask` / `_accel` verbatim (both
are Megatron-free; only `install()` is Megatron-specific). Residency state does a t=0 cold fill
at the start of each packed 4096 sequence, matching FLAME training; document boundaries inside a
pack are measured (EOD-churn probe) rather than special-cased.

Out of scope, by standing decisions: the Instruct variant (optional artifact stage after the
base-model result, on `-0125-Instruct` whose SFT data is the open Tulu 3 mix), R > k cells
(breaks the active-only memory story), pinned residents, and systems/bandwidth accounting.

## Stage 0 — mask port and zero-shot impose (~half a day; gate for everything else)

1. **HF-side patch.** Wrap `OlmoeTopKRouter.forward` (module swap on a loaded model, no
   transformers fork): compute full router logits, run the resident scan, `masked_fill(~mask,
   -inf)`, then hand off to the existing softmax/top-k unchanged.
2. **Verify** before any numbers leave the machine:
   - R = E reproduces base-model outputs exactly (the mask is then all-ones).
   - Triton scan matches the torch reference on random logits (existing parity test).
   - Measured swaps per layer per token never exceed 1.
3. **Impose eval, no training**: held-out BPB (Stage 1 slice) plus 3 quick lm-eval tasks, base
   vs base+mask.

Gate: any finite, coherent gap proceeds to Stage 2. Incoherent output means a port bug until
proven otherwise. The gap number itself is deliverable 2 regardless of what follows.

## Stage 1 — disjoint in-distribution data (1–2 days, parallel with Stage 0)

The finetune corpus must match the training distribution while sharing no tokens with it
(gradients on memorized text are shaped by memorization, and a reviewer can ask whether the
adapted router leans on it).

1. **Unconsumed-tail check first**: AllenAI publishes exact training configs and data order. If
   the -0125 run consumed fewer tokens than the tokenized mix contains, the unconsumed tail is
   identifiable by index — in-distribution and unseen by construction. One config-reading task.
2. **Else parent-pool sampling**: every OLMoE-mix component subsamples a larger parent
   (DCLM-baseline pool; Dolma's peS2o, StarCoder, arXiv, OpenWebMath, Wikipedia). Sample fresh
   shards the mix's public file manifest excludes, reweight to the published proportions. Same
   manifest-exclusion logic for a ~30% fraction of unseen dolmino-mix-1124 shards (the anneal
   used a subsample, so unseen shards exist).
3. **Slices**: finetune corpus up to 5B tokens (only 1B needed to start); held-out BPB slice
   ~100M tokens; n-gram dedup of the finetune corpus against the BPB slice and the lm-eval task
   data. Commit the exclusion-audit output next to the corpus recipe.
4. Tokenize with the OLMoE tokenizer, pack at 4096.
5. **Derive the BPB divisor for this tokenizer and slice from actual byte counts and record it
   in the CSV header.** House rule after the 2.7568/2.9780 incidents: never inherit a divisor.

Kickoff side-task (10 minutes, before locking the token budget): read the ReMoE
(arXiv:2605.27081) and MELINOE (arXiv:2602.11192) methods sections for router-finetuning token
budgets — the only literature priors on this number, still unread.

## Stage 2 — router-only finetune (2–4 days including iteration)

- **Trainable**: the per-layer router linear only, 16 × (2048×64) ≈ 2.1M params. Everything
  else `requires_grad=False`. Forward and backward still traverse the full active network
  (router gradients need it), so wall-clock is ~4·N_active·D, not free.
- **Losses**: LM loss through the resident-masked softmax, plus OLMoE's own aux (0.01) and
  z-loss (0.001) computed on the masked distribution. Knob to A/B cheaply at 0.5B tokens if
  usage drifts: aux on full logits instead.
- **Constraint on from step 0.** Trained-in from the start, matching the paper. No curriculum.
- **Schedule**: LR sweep {3e-5, 1e-4, 3e-4} × 0.5B tokens, pick by held-out BPB. Run 1B at the
  winner, checkpoint + eval every 0.5B, extend toward 5B only while the recovery curve still
  descends.
- **Hardware**: fits the 48GB A6000 (14GB bf16 weights + checkpointed activations); an H100
  roughly halves wall-clock. Compute envelope $10–60 at the 1–5B range.
- **Durability**: push checkpoints off-pod at every eval point (the a6000 wipe precedent).
- **Scan cost**: ~1ms per layer call at our benchmarked shapes, 1–2% of step time.

Gate at 1B tokens: adapted BPB strictly below impose BPB, and still descending or recovery
already ≥ ~25% of the gap. A flat curve below that bar stops the run and opens the stage-2b
decision (add LoRA on expert down-projections) rather than silently spending more.

## Stage 2b — escalation bake-off (protocol locked 2026-07-20, runs after the 1B gate run)

The zero-shot gap (+2.08 BPB on the audited slice) is large enough that the escalation question
gets answered by measurement, not by default. One bake-off: equal-footing arms at 0.25B tokens
each, all from the base router init at the sweep's winning LR, same data order, evals every 50M
tokens always at R = 8 (the serving condition), usage-entropy and swap-rate telemetry per arm.

| Arm | Axis | Recipe |
|---|---|---|
| A (reference) | — | Router-only, cold. Reuses the incumbent 1B run's first 0.25B, no new compute |
| B | Schedule | Annealed R: 64→8 stepwise over the first ~150M tokens, then hold. The regime-annealing lesson from AR→diffusion adaptation (DiffuLLaMA-style attention-mask annealing); every intermediate R is a well-defined mask, and the Stage-0 R-sweep showed the intermediate regimes are mild |
| C | Calibration | Router + layernorm gains, conditional on OLMoE's norms having learnable parameters (skip and note if non-parametric) |
| D | Objective | Router + self-distillation: 0.5 data CE + 0.5 KL toward the frozen base's free-routing logits. ~2x step cost, the standard recovery tool for surgically modified models |
| E | Capacity | Router + LoRA r=32 on expert up/down projections (MELINOE's recipe) |
| F (optional) | Ceiling | Low-LR full finetune (8-bit Adam). Not a candidate: calibrates the recoverable ceiling so recovery fractions are quoted honestly |

Selection rule, in order: (1) disqualify arms that improve BPB while collapsing usage entropy or
gaming swap rate (mechanisms Goodhart; the alignment-program lesson); (2) rank by audited-slice
BPB at 0.25B against a measured eval-noise sigma (re-evals of the base model on a few subsample
seeds); (3) an arm within sigma of the leader but with clearly steeper end-slope may win on
headroom; (4) ties break toward the simpler claim: router-only, then +LN, then anneal, then
+LoRA, then distillation combos; (5) a 3-task lm-eval on the provisional winner's 0.25B
checkpoint confirms BPB gains translate downstream before the 5B run. If two different-axis
mechanisms both clear the bar, one combo arm at 0.25B tests stacking. The winning recipe runs
Phase C: 5B tokens, full eval curve, telemetry, milestone checkpoints, then Stage 3 forensics.

## Stage 3 — evaluation (1–2 days)

1. **BPB triplet** base / impose / adapted on the held-out slice, plus one external corpus
   (a Paloma subset, or a Marin nemotron-cc slice, both largely disjoint from OLMoE's mix) as a
   robustness check. Headline: recovery = 1 − (adapted − base)/(impose − base), reported with
   the token-budget curve, not as a single point.
2. **lm-eval**: the widened 10-task set with the per-task stderr protocol and pinned venv, base
   vs adapted-under-constraint, identical harness both sides.
3. **Routing forensics** on the adapted model: swap rate per layer, residency dwell times,
   usage balance, and the Section 4 probes (selectivity, context-dominance). Reproducing the
   de-lexicalization signature via adaptation alone would be a second headline.
4. Optional: EOD-boundary churn analog for the appendix.

## Stage 4 — serving (3–5 days, starts as soon as any adapted checkpoint exists)

1. **GGUF**: stock conversion (weights are architecturally unchanged; the constraint is
   runtime-only), fp32 → bf16, then Q4_K_M and bf16 variants.
2. **Fork port**: repeat the qwen3moe integration pattern for the `olmoe` graph. `temporal.cu`,
   the `mul_mat_id` hook, and R-slot registration are architecture-agnostic; the work is olmoe
   graph wiring and tensor naming. No shared-expert path needed.
3. **Bench on the a6000**: ceiling / kernel / deploy rows analog to Table 2, VRAM, decode and
   prefill; perplexity spot-check of the fork against the HF adapted model as sanity.

Gate: the deploy row shows active-scaled VRAM at a decode ratio consistent with Table 2's story,
on real weights.

## Extension stage — sub-expert division for deeper sparsity (after Stage 3, decision-gated)

Native OLMoE geometry caps residency at 8 of 64: a 1/8 expert-sparsity ratio, ~5x end-to-end
VRAM, which matches but does not extend the paper's existing deploy story. No small open MoE
does better natively (the survey's candidates all sit between 1/6 and 1/11); frontier-style
ratios (1/18 to 1/32) exist only in models far over our budget. The route to deeper sparsity on
a released checkpoint is dividing the experts we have, which is also the paper's own
fine-grained prediction (fine geometry cushions residency) tested post hoc.

1. **Split** each expert's intermediate dimension in half: E = 64 → 128 sub-experts of width
   512. Function-preserving at init: duplicate each router row for its two halves and double
   each half's down-projection (softmax mass splits evenly across the duplicated logits, so the
   2x restores the parent's output). Gate: the split model at free top-16 must match the base
   model's outputs to numerical tolerance before any constraint is applied.
2. **Ladder** of adaptation cells at R = k in {16, 12, 8} of 128:
   - 16 of 128 = 1/8, the parity control. Function-preserved at init, so its recovery should
     closely match Stage 2's. A mismatch means the split itself is the problem.
   - 12 of 128 = 1/10.7, the paper's trained-from-scratch geometry, now on a real checkpoint.
   - 8 of 128 = 1/16, the doubled-sparsity target. Active params halve here, so quality
     attribution needs the parity control alongside, and router-only will plausibly stall:
     sub-expert specialization was never trained, so the stage-2b LoRA decision is expected,
     not exceptional, in this cell.
3. **Stretch, only if the ladder is clean**: an m = 4 split (256 sub-experts of 128) at
   R = k = 8 gives 1/32, the Methods inequality's frontier ratio (FINAL_TOUCHES row 12) reached
   by division and adaptation instead of new from-scratch training cells.

Each cell reuses Stage 0's patch, Stage 1's corpus, Stage 3's harness, and Stage 4's fork port
unchanged (GGUF and the R-slot ring are agnostic to expert count and width). Marginal cost is
one split script plus per-cell finetunes at the Stage 2 envelope.

## Paper integration

Main text: one short subsection, one table (base / impose / adapted × held-out BPB, downstream
average, VRAM, decode tok/s). Appendix: recovery curve, forensics, hyperparameters, data recipe.
CSVs land in `results/ablations/` with provenance rows in its README, as usual.

## Risks

| Risk | Handling |
|---|---|
| Router-only recovery stalls | Stage-2b LoRA-on-experts is a decision gate, not automatic. A clean partial-recovery number is still reportable; the paper stands on from-scratch evidence either way |
| Unconsumed tail doesn't exist | Parent-pool path is fully specified and always available |
| transformers/harness version drift | Pinned venv, same workaround as the stderr program |
| Pod wipe | Off-pod checkpoint pushes at every eval point |
| Timeline | ~1.5–2 weeks total against a ~9-week runway; Stage 4 parallelizes with Stages 2–3 after the first checkpoint |
