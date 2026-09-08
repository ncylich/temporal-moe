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

## Stage 2b — escalation bake-off (protocol locked 2026-07-20; resequenced same day: runs directly after the LR sweep, and the 1B hard gate migrates to Phase C's eval schedule, applied to the selected recipe)

The zero-shot gap (+2.08 BPB on the audited slice) is large enough that the escalation question
gets answered by measurement, not by default. One bake-off: equal-footing arms at 0.25B tokens
each, all from the base router init at the sweep's winning LR, same data order, evals every 50M
tokens always at R = 8 (the serving condition), usage-entropy and swap-rate telemetry per arm.

| Arm | Axis | Recipe |
|---|---|---|
| A (reference) | — | Router-only, cold. Reuses the incumbent 1B run's first 0.25B, no new compute |
| B | Schedule | Annealed R: 64→8 one expert at a time (56 rungs, equal-token, ~2.7M tokens each) across the first ~150M tokens, then hold. The regime-annealing lesson from AR→diffusion adaptation (DiffuLLaMA-style attention-mask annealing); every rung runs the exact deployed mechanism at the current R, and the Stage-0 R-sweep showed the intermediate regimes are mild. Reserved refinement if the final rungs spike: tail-weighted dwell (difficulty concentrates below R≈16). Rejected: soft-mask λ-ramps (break the residency invariant mid-anneal; soft pressure Goodharts) |
| C | Calibration | Router + layernorm gains, conditional on OLMoE's norms having learnable parameters (skip and note if non-parametric) |
| D | Objective | Router + self-distillation: 0.5 data CE + 0.5 KL toward the frozen base's free-routing logits. ~2x step cost, the standard recovery tool for surgically modified models |
| E | Capacity | Router + LoRA r=32 on expert up/down projections (MELINOE's recipe) |
| F′ (after G) | Ceiling | Staged probe, supersedes from-scratch F: merge the best adapter of {E, G} into the expert weights (exact linear merge, phase 2 starts at phase-1 quality), then full finetune at LR 1e-5 (8-bit Adam) for ~200M tokens, evals every 25M. Breaks the ~91.5% plateau = the floor was adapter capacity and Phase C adopts the staged recipe; plateaus at the same level with every parameter free = the residual is the constraint's price, matching the from-scratch story |

Selection rule, in order: (1) disqualify arms that improve BPB while collapsing usage entropy or
gaming swap rate (mechanisms Goodhart; the alignment-program lesson); (2) rank by audited-slice
BPB at 0.25B against a measured eval-noise sigma (re-evals of the base model on a few subsample
seeds); (3) an arm within sigma of the leader but with clearly steeper end-slope may win on
headroom; (4) ties break toward the simpler claim: router-only, then +LN, then anneal, then
+LoRA, then distillation combos; (5) a 3-task lm-eval on the provisional winner's 0.25B
checkpoint confirms BPB gains translate downstream before the 5B run. If two different-axis
mechanisms both clear the bar, one combo arm at 0.25B tests stacking. The winning recipe runs
Phase C with the full eval curve, telemetry, milestone checkpoints, then Stage 3 forensics.

**Screening policy (adopted 2026-07-20, prospective).** Fast-adapting arms at standard LR capture
95–99% of their gain by 50M tokens (arm A 95.6%, arm E 99.3%), so new mechanism arms screen at
50M with evals every ~10M. Promotion rule: an arm within 2 sigma of the leader at 50M, or still
visibly descending, extends to 250M by checkpoint-resume; far behind and flat stops at 50M.
Structural exceptions run full length by design: schedule/ramp arms, low-LR floor probes (F′),
and Phase C. Already-committed arms D and G keep their 250M spec: G is the recipe contender so
its floor is the decision input, and D doubles as the slow-start test of this policy (if D's 50M
position mispredicts its 250M position, the promotion rule earns its keep).

**Arm CE (combo screen, inserted 2026-07-20 after C's live 50M signal of 90.1% recovery with
133K norm-gain params).** Router + norm gains + LoRA r32 together, under the 50M screening
policy. The disambiguator: CE ≈ E at 50M means norms and LoRA fix the same thing (calibration)
and the cheap surface carries the recipe; CE above E by >2 sigma means they stack. G's exact
definition (LoRA+distill vs norms+distill) is decided after CE and C's final land.

**Optional-idle block** (only if the GPU would otherwise sit before Phase C selection, in this
order): rank screens r ∈ {8, 64} under the 50M policy (bracketing r=32: does 91% survive a 4x
smaller adapter, does 2x larger buy anything), then arm H (LoRA + zone-anneal R 24→8, a ramp arm
so full-length by the exception rule).

**Phase C sizing note.** The chosen recipe converges within ~100M tokens and tokens do not move
floors, so Phase C defaults to 1B with cosine decay (harvesting the decay dividend E's flat-LR
plateau leaves on the table), extending only while the decayed curve still moves.

## O-series — MinFlow oracle scheduling program (approved 2026-07-21, queues after F′)

The residual gap decomposes into weight-side terms (measured by the bake-off) and a
scheduling-side term this program prices. Reward field: the frozen base's plain softmax mass
over all 64 experts per token per layer (one free forward; sparsified to top-24 after verifying
≥99.5% cumulative mass — never renormalized after truncation). Captured mass M(S,t) is the
first-order surrogate: masked gating preserves resident ratios, so 1−M bounds the per-token
deviation from free behavior. The hindsight schedule is then an EXACT min-cost flow per
(layer, sequence): 8 slot-units through the experts×time graph, node capacity 1 per (e,t),
stay arcs collect reward, switch arcs route through a capacity-m admission hub per token
(m=1 is the paper constraint), cold fill = 8 free admissions at t=0. Milliseconds per 4096
sequence; solvable in the dataloader.

- **O-0 calibration** (GPU minutes + CPU): free forward over ~10M tokens; rank-mass histogram
  (truncation check); captured-mass ladder replayed over stored logits with no model evals:
  static-best-8 < greedy scan < MinFlow < per-token top-8 bound. Kill the program here if
  MinFlow ≈ greedy in reward.
- **O-1 m-sweep** (CPU): ladder at m ∈ {1, 2, 4}. Separates better-choices from more-budget;
  m>1 is analysis and a serving bandwidth knob, not a method change.
- **O-2 reward→BPB transfer** (~$3): frozen base's actual BPB under 4-5 ladder schedules. Fits
  the surrogate's validity and yields the headline schedulability number (MinFlow zero-shot vs
  greedy 2.7507). Tests whether better scheduling also shrinks the calibration term. Kill if
  MinFlow ≈ greedy in BPB.
- **O-3 learned causal MinFlow router** (~$5, 50M screen): behavior-clone the oracle's
  admissions into the EXISTING router surface (64-way admit classification, teacher-forced
  residency) so serving needs zero new machinery. Evaluate frozen-weights BPB vs greedy and vs
  arm A (2σ bars). Expressible target, exogenous and frozen (Goodhart channel closed);
  demand-prediction AUC 0.93-0.98 says the oracle's choices are largely context-predictable.
- **O-4 restack** (~$15, conditional on O-3): norms+LoRA on top of the learned scheduler; does
  it beat 93.2%.
- **Deployment note**: prefill sees the whole prompt, so serving can use the exact flow
  schedule in prefill and the learned causal router in decode.

Sequencing: O-0's forward runs right after F′; O-0 analysis + O-1 (CPU) and Stage 4 port dev
interleave with Phase C's training; O-2 onward after Phase C completes, gated orch-side.

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

## Program close-out and corrections (2026-07-22)

The program completed with the evidence record in `results/ablations/` (see the README's
olmoe_* rows and `olmoe_adapt_RESULTS.md` for the full table). Corrections and lessons the
plan's original text got wrong, kept here so the doc reads honestly against the record:

- **Windowed evaluation understates accumulation.** TW=256 replay cold-fills masked most of the
  greedy-vs-oracle gap (0.007 windowed vs 0.023 full-sequence). Schedule comparisons must run
  at full sequence length.
- **Replay scheduling is ~0.40 BPB more optimistic than live serving.** Schedules computed on
  frozen free-routing logits misalign under the model's own residency-induced logit drift
  (free-vs-drift 0.4034 +/- 0.0178). The decisive test on the adapted winner came out NEGATIVE:
  the live scan beats a forced free-logit MinFlow optimum by 0.0113 +/- 0.0021 — online
  adaptivity beats offline optimality under drift, and offline residency scheduling (O-3) is
  dead on the winner, not merely unnecessary.
- **The exact flow solver costs ~36 s per (layer, 4096-sequence) with C=32 pruning**, not the
  milliseconds the O-series spec estimated; supervision at training volumes is intractable,
  evaluation ladders are fine.
- **Both the path axis and the init axis are closed.** Five path variations (anneal x2,
  distill x2, surface identity C=E=F') and the calibrated-init screen (Cal-2:
  undone-to-single-basin, detour cost 0.047) all converge to the same endpoint. Closed-form
  moment matching is neither a solution (Cal-0: 31.5% clipped, cos~0.01 to the learned gains)
  nor a useful starting point.
- **m>1 swap budgets are diagnostic-only** (each extra swap is another expert fetch of
  slow-storage bandwidth per layer per token); m=1 is the only deployment budget.
- Stage 4 (serving artifact) was cancelled by decision 2026-07-22: the program's deliverable is
  the scientific answer. Phase C (1B cosine run) remains optional and unlaunched.
