# Rolling residency across three MoE models, adapted at 50M tokens

What this is: rolling residency (R of E experts held resident per layer, top-k chosen among the
residents, at most one swap per token) imposed on **every** MoE layer at R = k = 8, then adapted for
50M tokens with cross-entropy plus attention LoRA rank 32. No expert LoRA, no freed layers, bf16, and
the same 300,658,454 bytes of source text for all three models, re-tokenized per tokenizer.

R = k = 8 is the tightest **valid** setting. R < k cannot fill the top-k -- the router returns
weights like `[0.48, 0.22, 0.18, 0.13, 0, 0, 0, 0]` -- so rows at R < k were withdrawn from earlier
work and are not measured here.

BPB is bits per byte, `CE_nats / (ln2 * bytes_per_token)`, lower is better. It is used instead of
per-token cross-entropy because the three tokenizers differ and BPB is tokenizer-invariant.

Data: [`crossmodel_nofree_50M.csv`](crossmodel_nofree_50M.csv).
Throughput work: [`../../analysis/residency/TRAINING_OPTIM_PLAN.md`](../../analysis/residency/TRAINING_OPTIM_PLAN.md).
Producers: `analysis/ple/train_ple.py`, `analysis/residency/train_qwen.py`, `analysis/residency/olmoe_ref_256.py`,
`analysis/residency/score_corpus_candidates.py`, `analysis/residency/fetch_corpus_candidates.py`.


## 0. WITHDRAWN: every recovery percentage in this document

The BPB measurements below are real. **The recovery percentages derived from them are not, and are
withdrawn.** Three independent reasons, any one of which is sufficient:

1. **No expert LoRA.** All three runs used `--lora 0`, so ~90% of each model -- the experts, the
   component residency reroutes tokens between -- was frozen. Every published OLMoE cell carries
   `lora=32`, including `ce_freeall_50M`, the null these figures are scored against. OLMoE's 26.0%
   therefore subtracts an expert-LoRA null from an attention-LoRA arm.
2. **Qwen routers were frozen.** `train_ple.py` trains the router; `train_qwen.py` did not. Rolling
   residency is a constraint on routing, so both Qwen arms adapted around it rather than to it.
3. **Qwen aux was 10x too strong** (0.01 against the shipped 0.001) and the **corpus was half-unread**,
   making the 50M runs 1.49 epochs of repeated data.

See S9 below. What survives is S1's training-free result, which uses no
adaptation at all.


## 1. The headline

| model | experts | resident | untrained free | untrained R=8 | trained | constraint cost | ~~recovery~~ |
|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 64 | 12.5% | 0.672723 | 0.842848 | 0.804438 | **+0.170125** | ~~22.6%~~ |
| Qwen3-30B-A3B-Base | 128 | 6.25% | 0.582025 | 0.686861 | 0.642922 | **+0.104836** | ~~41.9%~~ |
| Qwen3.5-35B-A3B-Base | 256 | 3.125% | 0.586121 | 0.637653 | 0.641656 | **+0.051532** | ~~-7.8%~~ |

The `untrained free`, `untrained R=8` and `constraint cost` columns are training-free and stand. The
`trained` column is a real measurement of a recipe we would not now run. The recovery column is
struck: see S0.

`recovery = (untrained_constrained - trained) / (untrained_constrained - untrained_free)`, the share
of the constraint's damage that adaptation removed. Negative means training ended worse than not
training at all.

**The result that holds without any adaptation: the constraint gets cheaper as expert count grows,
monotonically -- +0.170 -> +0.105 -> +0.052 BPB across 64 -> 128 -> 256 experts -- while the resident
FRACTION simultaneously falls 12.5% -> 6.25% -> 3.125%.** Both quantities move the right way at once.
This is the strongest claim in this table because it needs no training and no ceiling assumption.

**The recovery column does not follow that pattern, and Qwen3.5 is negative.** Sections 3 and 4 are
the two measurements that explain it.


## 2. Three caveats that change how rows should be read

**Evaluation subsets differ between models, by design.** Each model's untrained references are
measured on the subset its own trained cell is scored on -- OLMoE on the 256 sequences
`train_ple.py:204` selects via `linspace(0, n-1, 256)`, Qwen on the first 8 that `train_qwen.py`
takes. Recovery is valid **within** a row. Absolute BPB **across** rows is approximate: the free
baseline of the OLMoE slice alone varies 0.6359..0.6822 across disjoint blocks, so cross-row BPB
differences under ~0.05 carry no meaning.

This bit an earlier version of this table. OLMoE's recovery was first computed by differencing a
256-sequence trained number against 16-sequence references, giving 20.6%. `olmoe_ref_256.py`
re-measured both references on the correct subset; the corrected figure is **22.6%**, and the
corrected untrained baseline (0.672723) reproduces the programme's long-published 0.6727.

**Every recovery figure is a LOWER BOUND.** They are measured against the untrained unconstrained
model, which assumes continual training on this corpus is neutral. It is not -- see §3.

**The recipe is not uniform across models.** Attention LoRA reaches 16 of 16 layers on OLMoE and 48
of 48 on Qwen3-30B, but only **10 of Qwen3.5's 40**: its other 30 layers are Gated DeltaNet and have
no q/k/v/o to adapt. Qwen3.5 receives a structurally thinner intervention.


## 3. The corpus was the wrong one, and it is measured

Qwen3's pretraining corpus is proprietary (36T tokens, web plus PDF-extracted documents plus
synthetic textbooks from Qwen2.5-Math/Coder, annotated for educational value -- arXiv 2505.09388), so
"representative" has to be measured rather than matched. Scored under Qwen3-30B, residency off, 8 MB
per corpus at matched byte counts:

| corpus | Qwen3-30B BPB | delta | Qwen3.5 BPB | delta | bytes/token (Qwen3) |
|---|---|---|---|---|---|
| **FineWeb-Edu** | **0.632977** | **-0.103202** | **0.633418** | **-0.105280** | 4.6011 |
| Dolma (incumbent -- what all three runs used) | 0.736179 | -- | 0.738698 | -- | 4.4906 |
| DCLM-baseline | 0.773312 | +0.037134 | 0.779714 | +0.041016 | 4.3139 |

**FineWeb-Edu fits Qwen ~14% better than the corpus these runs used, and it replicates.** Two models
with different tokenizers, different expert counts (128 vs 256) and different attention architectures
(full vs 30/40 Gated DeltaNet) agree on the ordering and on the magnitudes to within 0.002 BPB.

A third, model-free measure agrees as well: the BPB ranking matches the tokenizer-fertility ranking
exactly (4.6011 > 4.4906 > 4.3139 bytes/token), and fertility is computed without running the model.

The incumbent was chosen to match OLMoE, and kept for comparability once OLMoE had already trained on
it. That was a defensible call for this batch and the wrong corpus for Qwen. Nemotron-CC -- the prior
pick, since it ensembles the FineWeb-Edu and DCLM classifiers and adds synthetic rephrasing -- is
gated on the Hub and could not be tested.


## 4. The unconstrained null, and what it does to Qwen3.5's row

The null trains the identical recipe with `--free-set all`, making the residency machinery provably
inert (`swap = 0.0000` at every checkpoint). It measures what continual training costs by itself.

| | 10M | 20M | 30M | 40M | 50M |
|---|---|---|---|---|---|
| OLMoE null (`ce_freeall_50M`) | 0.683090 | 0.687481 | 0.690524 | 0.692876 | 0.695064 |
| Qwen3.5 null (`null_attn_50M`) | 0.588538 | 0.588852 | 0.588513 | 0.588468 | **0.588630** |

**These behave completely differently.** OLMoE's null degrades monotonically and does not stop,
losing 0.0224 BPB over 50M tokens. Qwen3.5's rises ~0.0025 BPB and then goes flat (sd 0.00019 across
checkpoints). So this corpus is roughly 9x less hostile to Qwen3.5 than to OLMoE.

Completed, the null lands at 0.588630, so the corpus costs Qwen3.5 **+0.002509 BPB** over 50M tokens
and stops. That gives the honest arithmetic:

| Qwen3.5 | BPB | constraint cost |
|---|---|---|
| untrained, unconstrained | 0.586121 | -- |
| trained 50M, unconstrained (null) | 0.588630 | -- |
| untrained, R=8 of 256 | 0.637653 | **+0.051532** |
| trained 50M, R=8 of 256 | 0.641656 | **+0.053026** |

**Recovery against the achievable ceiling is -8.2%**, against -7.8% versus the untrained model. An
earlier reading of this document said the corpus "accounts for most of Qwen3.5's degradation". That
is true of the RAW rise (0.0025 of 0.0040) but it does NOT explain the missing recovery: with the
corpus effect removed the figure is still negative. The constraint costs 0.051532 BPB before
adaptation and 0.053026 after -- **statistically unchanged**, the difference sitting inside one
standard deviation of checkpoint noise (sd 0.001635).

So the finding is not that adaptation was masked by a falling ceiling. On Qwen3.5 this recipe is
**inert**: it does not move the constraint's cost in either direction.

Why nothing? Qwen3.5 starts with the least damage to repair (0.0515 BPB, half Qwen3-30B's) and the
thinnest adapter reach (10 of 40 layers). There may simply be very little for this recipe to do.

**Qwen3-30B has no null.** Its 41.9% is therefore a lower bound and the true figure is higher.


## 5. Adaptation saturates early

Both models that recovered anything did so almost entirely in the first 10M tokens.

| tokens | 10M | 20M | 30M | 40M | 50M |
|---|---|---|---|---|---|
| OLMoE | 0.809402 | 0.806927 | 0.805392 | 0.804825 | 0.804438 |
| Qwen3-30B | 0.643013 | 0.643471 | 0.643298 | 0.642064 | 0.642922 |

OLMoE's last 10M tokens bought 0.0004 BPB against the first 10M's 0.0025. Qwen3-30B is flat from 10M
onward, oscillating within 0.0015. **A 10M-token budget would have produced nearly the same result at
a fifth of the GPU time**, which matters for any cost argument built on these runs.


## 6. Throughput: the residency scan's cost scales with expert count

The null and the constrained arm are identical but for the scan, on the same hardware, so the
difference is the scan.

| model | experts | constrained | unconstrained | scan cost |
|---|---|---|---|---|
| Qwen3-30B | 128 | -- | -- | 1.6-7.6% (measured separately, fused path) |
| Qwen3.5-35B | 256 | 216.5 min | ~156 min | **~28%** |

The scan is O(sequence x experts) per layer, so this direction is expected; the magnitude was not.
**More experts make the constraint cheaper in quality and more expensive in wall-clock.** Earlier
write-ups quoted only the first half.


## 7. The hyperparameters were inherited from a mis-specified run

None of these settings was tuned for Qwen. All were inherited, and their provenance is worse than
"untuned": `train_ple.py:32` labels the learning rate 3e-4 the "bake-off winner LR, arm C", selected
on OLMoE **while the gate-mass artifact was active**. In that regime the constrained arm's damage read
+2.001439 BPB; corrected it is +0.169000. The LR was chosen to recover a hole **11.8x larger than the
real one** and never re-tuned after the fix.

| issue | measured |
|---|---|
| LR provenance | 3e-4 selected against 2.0014 BPB of damage; the real figure is 0.1690 |
| Half the corpus is never read | packed at 4096, `--seq 2048` slices `[:, :2048]` -> 33.5M usable, so 50M tokens is **1.49 epochs** |
| Adapter capacity differs 3.9x between the Qwen runs | 26.7M trainable (Qwen3-30B, 192 projections) vs 6.9M (Qwen3.5, 40) |
| Aux coefficient inherited | `AUX_C=0.01` x aux 20..55 ~ 0.2..0.55 against lm 2.0..2.7, i.e. 10-20% of the gradient |
| No sweep of any kind | one LR, one rank, one aux weight, both models |

**This confounds the Qwen3.5 conclusion.** Its null moved +0.002509 BPB over 50M tokens where OLMoE's
moved +0.0224 -- 9x less. The null measures how much the recipe perturbs the model at all, with no
constraint involved, so Qwen3.5 was barely trained. "Adaptation is inert on Qwen3.5" cannot be
separated from "we barely trained it" on this evidence. The defensible claim is narrower: *under
settings inherited from a mis-specified OLMoE run, Qwen3.5 showed no recovery.*

**What is unaffected:** §1's monotone fall in constraint cost with expert count is training-free. No
hyperparameter enters it.

**The sweep this calls for is cheap.** §5 shows adaptation saturates by 10M tokens, so points cost
~26 min on Qwen3-30B and ~43 min on Qwen3.5 rather than a full run. A 3x3 over LR {1e-4, 3e-4, 1e-3}
and rank {32, 128} on Qwen3-30B at 10M, scored against the null, would establish whether 3e-4 is
anywhere near right before any further conclusions are drawn from the recovery column.


## 8. What is still missing

- **Qwen3-30B unconstrained null.** Queued behind the Qwen3.5 null and skipped if the clock did not
  allow it. Without it Qwen3-30B's 41.9% stays a lower bound.
- **Downstream task accuracy for both Qwen models.** `train_qwen.py` originally saved no adapter, so
  the trained model was discarded. Fixed mid-run: Qwen3.5 has a saved adapter, Qwen3-30B does not, so
  a matched downstream comparison needs Qwen3-30B re-run.
- **Any run on FineWeb-Edu.** §3 says the corpus is wrong and nothing here has been re-run on the
  better one.


## 9. Withdrawn numbers and the mistakes that produced them

Every number retracted this session, with the specific error and the check that would have
caught it. Kept so the same class of mistake is visible rather than rediscovered.


Numbers that were reported and are now retracted, each with the specific error and the check that
would have caught it. Kept so the same class of mistake is visible rather than rediscovered.


### 1. The 22.7x and 65x training speedups -- WITHDRAWN

**Claimed:** the fused MoE library gave 22.7x at fixed micro-batch and 65x end-to-end (93 -> 6,046
tok/s) for Qwen adaptation.

**What was wrong:** the baseline and the measurement were DIFFERENT MODELS. (An earlier version of
this entry also said the adapter differed. It did not -- `bench_train_fused.py` targets
`gate_proj/up_proj/down_proj` as well as attention, so both sides carried expert LoRA. The adapter was
matched; the model was not.)

| | model | adapter | expert path | micro-batch | tok/s |
|---|---|---|---|---|---|
| "stock" baseline (qwen35_RESULTS.md:100) | **Qwen3.5**, 40L x 256E | expert + attn LoRA | `_experts_forward_lora` Python loop | 1 | 93 |
| fused measurement (bench_train_fused.py) | **Qwen3-30B**, 48L x 128E | expert + attn LoRA | fused grouped GEMM | 1-8 | 2,109-6,046 |

The 93 tok/s figure is CORRECT for what it measured. It was cited as the baseline for a benchmark of
a different model, on a card where the two differ ~2x in cost per token before any kernel is involved.

**Do not over-correct this into "fused is no faster than stock".** The honest comparison is:

| config | model | adapter | tok/s |
|---|---|---|---|
| stock | Qwen3-30B | attention only | 6,274 |
| fused | Qwen3-30B | **attention + expert** | 6,046 |

The fused path adapts the experts -- ~90% of the parameters, and the thing residency actually reroutes
-- at essentially the throughput stock manages on attention alone. That is a real and useful result.
What is withdrawn is the *number* 22.7x, not the utility of the library. A matched
stock-vs-fused-at-identical-adapter measurement has never been run; `verify_fused_claim.sh` is queued
to produce it.

**Ground truth:** stock, in the configuration actually trained, ran the 50M Qwen3-30B arm at
49,987,584 tokens in 132.8 min = **6,274 tok/s** -- 67x the claimed baseline, and above the fused
benchmark's own best of 6,046 tok/s (at seq 1024, so not strictly like-for-like).

**The check that would have caught it, in one line:** compare the claimed baseline against a real
run's throughput. 93 vs 6,274 is not a subtle discrepancy.

**Consolation:** the fused path was never wired into `train_qwen.py`, so no run was slowed by the
error. That was luck, not design -- the code path was simply never connected, which is failure 2.

**Third occurrence of this class.** The programme already recorded a phantom 2.69x from crippling a
baseline with `_experts_implementation = None` (qwen35_RESULTS.md:353), and a 2x inference claim that
measured 1.15x. The pattern is always the same: a speedup measured against a baseline nobody runs.

**Standing rule:** a speedup is only reportable against the configuration actually used in production
runs. Any baseline that is not literally a config we run must be labelled as a microbenchmark and
excluded from headline numbers.


### 2. The fused path was validated and never deployed

`residency_fused.py` was written, `check_fused_kernels.py` passed it (BPB shift 9.6e-06 against a
same-kernel bf16 noise floor of 6.26e-05), and then every training run was launched against
`train_qwen.py`, which imports `residency_qwen` -- the stock path. The two were never connected.

Cost: nothing, because stock turned out to be at or above fused speed (failure 1). But the failure
mode -- build a thing, validate it, then run something else -- is independent of that luck.

**Check:** read the command line you are about to launch and confirm it uses the component you just
validated.


### 3. OLMoE recovery computed across mismatched eval subsets -- CORRECTED

**Claimed:** 20.6% recovery, and "training removed 35% of the constraint's cost".

**Wrong because:** `train_ple.py:204` scores trained cells on 256 sequences spread across the slice
(`linspace(0, n-1, 256)`); `olmoe_remeasure.py` measured the untrained references on the FIRST 16
(`ids[:16]`). Differencing them is not a recovery measurement -- the free baseline alone ranges
0.6359..0.6822 across disjoint blocks of that slice, ~40x the effect being claimed.

**Corrected for the subset error:** references re-measured on the 256-subset by `olmoe_ref_256.py`,
giving 22.6% against the untrained model and 26.0% against the null. The corrected untrained baseline
(0.672723) reproduces the programme's long-published 0.6727, confirming that diagnosis.

**But those figures are ALSO invalid, for a different reason -- see S9.** The 26.0% differences an
attention-only-LoRA arm against an expert-LoRA null. Fixing the eval subset did not fix the surface
mismatch, because I had not yet noticed it.


### 4. "The corpus explains most of Qwen3.5's degradation" -- CORRECTED

Said after seeing only the raw BPB rise. The null shows the corpus costs Qwen3.5 +0.002509 BPB of the
+0.004003 total, so it explains the raw drift but NOT the missing recovery: measured against the
achievable ceiling the figure is still -8.2%. The constraint costs 0.051532 BPB before adaptation and
0.053026 after -- statistically unchanged. Adaptation was inert, not masked.


### 5. "Monotone degradation" called from two points -- CORRECTED

Qwen3.5's constrained arm read 0.640819 then 0.644467 and was described as monotonically degrading.
The third point (0.642330) contradicted it; the series is oscillation around ~0.642, sd 0.001635.


### 6. "The aux fights residency by pushing resident sets to vary" -- WITHDRAWN

Varying resident sets is the objective, not a cost. A static resident set is a pruned model; the
constraint is on swap RATE (<=1/token), not on variety, and effective expert count rising 20.5 ->
49.5 under adaptation is the success signal. Correction supplied by the user, not by me.


### 7. Setups diverged between models without being noticed

Three separate instances, all found only when specifically looked for:

- **Qwen routers were never trained.** `train_ple.py:135` calls `freeze_all_but_router`; `train_qwen.py`
  froze every base parameter. Rolling residency is a constraint ON ROUTING, so both Qwen arms adapted
  around the constraint rather than to it. Fixed by `--train-router`.
- **RMSNorm gains.** `train_ple.py:137` trains them; `train_qwen.py` did not. Fixed by `--train-norms`.
- **Aux coefficient.** Both Qwen models ship `router_aux_loss_coef = 0.001`; every run used OLMoE's
  0.01, i.e. 10x. Fixed by defaulting to the model's own config.

**Consequence:** the completed cross-model table is not comparing like with like. OLMoE's 26.0% came
from a strictly richer adaptation surface than Qwen3-30B's 43.9% or Qwen3.5's -8.2%. Both Qwen figures
are lower bounds until re-run.


### 8. Corpus half-unread, forcing repetition -- FIXED

`train_qwen.py` sliced `corpus[ptr:ptr+mb, :seq]` from rows packed at 4096, so `--seq 2048` never read
the second half of any row: 33.5M usable of 66.9M, making the 50M runs **1.49 epochs** while unseen
tokens sat on disk. Repeating data at this scale is a bug, not a tuning choice. The loader now
reshapes `(n,4096) -> (2n,2048)` and prints the epoch count, warning above 1.0.


### 9. Expert LoRA was silently dropped from every run -- THE LARGEST ERROR

**What happened:** `--lora 0` was set on all three 50M runs and both queued sweeps, so no expert LoRA
was trained anywhere. Experts are ~90% of each model's parameters and are the component rolling
residency actually reroutes tokens between. The adaptation surface was router + RMSNorm gains +
attention LoRA, and the experts were frozen.

**Why:** expert LoRA runs through `_experts_forward_lora`, a Python loop over experts with two extra
linears each -- the configuration measured at 93 tok/s. I removed the requirement to avoid the cost
instead of using the fused library, which was already built, already validated, and ships
`LoraMoeFusedLinear` for exactly this. The cut was never surfaced as a decision.

**It also breaks the OLMoE comparison outright.** Every published OLMoE cell carries expert LoRA:

| cell | expert LoRA | attn LoRA | final BPB |
|---|---|---|---|
| `ce_auxfix_free_attn_50M` (best published) | 32 | 32 | 0.784717 |
| `ce_auxfix_50M` | 32 | 0 | 0.827549 |
| `ce_freeall_50M` (**the null used as the ceiling**) | 32 | 0 | 0.695064 |
| `ce_attn_nofree_50M` (**this session**) | **0** | 32 | 0.804438 |

So the 26.0% recovery subtracts an expert-LoRA null from an attention-LoRA arm. Different surfaces on
each side of the subtraction; the number is meaningless, as are the Qwen equivalents.

**Every recovery figure produced this session is withdrawn.** The BPB values are real measurements;
the recovery percentages derived from them are not.
