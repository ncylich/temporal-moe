# Published baselines: what to run, and what each method actually does

Written 2026-08-24. Scope: the external comparisons the paper is missing, the venue each
one belongs in, and a plan per run.

## Why this document exists

The paper compares rolling residency against a dense floor, a full-MoE ceiling, and a
constructed offload floor. All three are ours. A grep of this branch across `analysis/`,
`scripts/`, `temporal/`, `experiments/` and `results/`, excluding archives, finds **no
published competitor implemented or run anywhere**:

```
skliar -> NONE      cosmoe -> NONE       promoe -> NONE
pregated -> NONE    oracle-moe -> NONE   eliseev -> NONE
blockffn -> NONE    remoe/melinoe -> results/ablations/README.md only
```

ReMoE and MELINOE appear once, as router-finetuning budget priors in the archived OLMoE
corpus audit. Everything else lives in the bibliography and the reading list.

The gap is no longer infrastructure. Both venues exist and work: the isoFLOP sweep
(`experiments/isoflop_1e16_1e17/drive.sh`) and the gemma4 D-ladder
(`analysis/residency/train_gemma_ce.py`). What is missing is that D12 beats only our own
variants, and Appendix E argues against CoSMoEs by citing their reported numbers rather
than running them.

## What to run

| # | Baseline (method from) | Venue | What we run | Cost |
|---|---|---|---|---|
| 1 | **CoSMoEs**, Huber et al., arXiv:2503.00245 | FLAME isoFLOP sweep | Their block-selection loss on a vanilla MoE, no residency mask. Weight sweep at 1e16, both granularities; winning weight repeated at 1e17. Convert each model's achieved swap rate into the R it needs, then plot as one point on the existing memory-quality frontier. | ~10 small runs |
| 2 | **ReMoE**, Zhu et al., arXiv:2605.27081 | gemma4-26B-A4B-IT | Faithful remake: router-only finetune, recency-bias reuse objective, residency constraint OFF during training. Everything else repeats the gemma4-26B-IT policy. Evaluate free and R8. | 1 D-ladder arm |
| 2b | **ReMoE**, fair-surface variant | gemma4-26B-A4B-IT | Same objective on D12's own surface. Disambiguates objective from capacity. See below. | 1 D-ladder arm |
| 3 | **Cache-conditional experts**, Skliar et al., arXiv:2412.00099 | gemma4-26B-A4B-IT | Additive bonus for currently-resident experts inside the swap-trigger scoring. No training. Sweep bonus strength, evaluate at R8. | Eval only |
| 4 | **Offloading with LRU and prefetch**, Eliseev and Mazur, arXiv:2312.17238 | llama.cpp fork | One serving-table row at the same VRAM as deploy, replacing free-routing-with-no-cache as the floor. | 1 row |
| 5 | **Oracle-MoE**, Zhou et al., ICML 2025 | FLAME isoFLOP sweep | Their attention-derived routing space, one budget, one granularity, same swap-rate-to-R conversion as #1. | Days of implementation, then 1-2 runs |

Corpus for #2, #2b and #3: repeat the policy used for gemma4-26B-IT. The corpus itself is
being reformulated separately; this document does not specify it.

Priority if the queue is contended: 1, then 2, then 3. #4 is presentation more than
science, since `e5_eviction_policy_headroom.csv` and `e1_victim_cache_hitrate.csv` already
answer the policy question. #5 answers the same objection as #1 at roughly ten times the
cost and competes for the GPU with A3 in `RECOVER_DATA_PLAN.md`, which is worth more.

## What each method actually does

**CoSMoEs.** Cut the text into fixed-length blocks. Add a penalty that grows with how many
different experts get used inside a block. The model trains from scratch with that penalty,
so it learns to stay with a smaller set of experts within each block. Nothing is forbidden;
switching is just made expensive. Their own paper reports the trade: fewer expert changes,
and a measurable drop in quality.

**ReMoE.** Give the router a bonus for experts it picked on recent tokens, so it tends to
pick them again. Take an already-trained model and adjust only the router to lean into that
bonus. Nothing is forbidden and the router can still choose any expert, so the set held in
memory is not bounded. The effect is that experts stay in use for longer stretches, so a
cache misses less often. Reported result is 26% better expert reuse on DeepSeek and Qwen
models, at roughly 33M training tokens.

**Cache-conditional experts.** At serving time, add a fixed bonus to any expert already
sitting in memory. The router still scores everything normally, but close calls break
toward what is already loaded. No training in the basic version.

**Offloading with LRU and prefetch.** Keep more experts in fast memory than a single token
needs. When something new is required, evict whichever resident expert went unused the
longest. In parallel, guess which experts the next token will want and start loading them
early. Purely a serving system; the model is untouched.

**Oracle-MoE.** Instead of routing on the current word's representation, build a compact
summary of the surrounding passage from attention and route on that. Because the summary
changes slowly from token to token, neighbouring tokens land on similar experts. Requires
training from scratch with the new router.

**How all five differ from ours.** Each of them makes expert reuse more likely without ever
making it guaranteed, so each still has to provision memory for the experts it might miss.
That is why none of them can serve at R = k by construction, and we can.

## Plans

### 1. CoSMoEs on the FLAME sweep

- **Implement.** One function in `temporal/ablation_mechanisms.py`, a sibling of the
  existing `bursty_window_loss` and `coherence_bce_loss`. Same signature, same env-var
  wiring through `pretrain_temporal.py`.
- **Train.** Vanilla MoE, no residency mask, plus their loss. Four weights by two
  granularities at 1e16, then the winning weight at 1e17. Same shape as the coherence-loss
  weight scan already in Appendix E.
- **Compare.** Their loss reduces switching but never bounds the resident set, so "CoSMoEs
  at R = k" is not something their method can satisfy. Measure the trained model's achieved
  swaps per layer per token, find the smallest R that serves it inside the one-swap budget,
  and plot (R, BPB) against the existing frontier in `residency_dose_curve`. One point on a
  curve we already have is the whole comparison, and it states the claim directly: their
  locality buys a smaller R, ours buys R = k.
- **Why this one first.** Appendix E currently rebuts CoSMoEs by citing their table. That is
  the most attackable move in the paper, and it costs about ten small runs to fix.

### 2. ReMoE remake on gemma4-26B-A4B-IT

- **Implement.** A `--surface router` option in `train_gemma_ce.py`, freezing everything
  except the 30 router linears. A recency-bias term on router scores, which the
  `ablation_mechanisms.py` momentum helpers already cover in shape.
- **Train.** Router-only. Recency-bias reuse objective. **Residency constraint OFF during
  training**, because that is what ReMoE does: it shapes a free router toward locality
  rather than training under a bound. Learning rate swept over three points, since
  router-only tolerates much higher rates than D12's 3e-5. No KL anchor, since with a frozen
  backbone there is nothing to anchor to. Otherwise repeat the gemma4-26B-IT policy
  throughout: same trajectory generation, same lineage rules, same budget discipline.
- **Evaluate.** GSM8K, IFEval, HumanEval, MMLU at free and R8, on the 200-item instrument,
  with the same budget resolver as the rest of the grid. Report streamed expert diversity
  alongside, so the union-collapse question stays visible.
- **Read against.** D12 at R8: GSM8K 0.0, IFEval -1.0, HumanEval -1.2, MMLU -1.8, versus
  base unadapted -6.0, 0.0, -6.1, -0.2.

### 2b. The fair-surface variant

ReMoE is router-only, so the faithful remake is router-only, and that is what a reviewer
asking "did you reproduce their method" wants to see. But it creates a confound: D12 trains
expert-LoRA r16 plus attention-LoRA r32, so if the faithful arm loses, we cannot tell
whether the objective is weaker or the surface is simply smaller.

The fix is one extra cell holding the surface fixed and varying only the objective:

| arm | objective | surface | constraint during training |
|---|---|---|---|
| D12 (incumbent) | constraint-aware CE + KL 0.05 | expert-LoRA r16 + attn-LoRA r32 | ON, R8 |
| 2 (faithful) | ReMoE recency bias | router only | OFF |
| 2b (fair) | ReMoE recency bias | expert-LoRA r16 + attn-LoRA r32 | OFF |

Run 2 first. Add 2b only if 2 underperforms D12, since its only job is to separate "their
objective is weaker" from "their surface is smaller." If 2 already matches D12, 2b answers
nothing.

This decomposition is worth stating in the paper regardless. The D-ladder already sweeps the
**surface** axis (C norms, E LoRA, F' full finetune). It has never swept the **objective**
axis. ReMoE is the published objective, and 2b is the controlled version of that comparison.

### 3. Cache-conditional bias on gemma4-26B-A4B-IT

- **Implement.** An additive bonus on resident experts in `compute_resident_mask`'s scoring,
  applied before the swap trigger. A few lines beside the existing `evict` knob.
- **Train.** Nothing. Zero-shot on base gemma4 at R8, sweeping bonus strength. Add one light
  finetuned variant only if the zero-shot arm lands close to D12.
- **Why.** Cheapest arm here, and it closes the "you did not need to adapt at all" objection.

### 4. LRU and prefetch serving row

- **Implement.** The fork already has the slot machinery. Add an LRU policy with a
  one-step-ahead prefetch, driven by real routing traces rather than a modelled miss rate.
- **Run.** One row at the same VRAM as the deploy row.
- **Why.** The serving table's floor is currently free routing with no cache at all, and
  Appendix C says so plainly. A reader who reaches the appendix will discount the 3-4x
  headline. An LRU-plus-prefetch row is the honest floor, and we probably still win.

### 5. Oracle-MoE

Deferred until the others land. It changes the routing architecture rather than adding a
loss, so it costs days of implementation before the first run, and what it buys is a second
answer to the objection #1 already answers.

## Not running, and why

- **MELINOE**, arXiv:2602.11192. Its objective is to shrink the number of distinct experts
  used per sequence, and its payoff is throughput. Throughput is not our target, and the
  objective is the opposite of our design: Section 5 reports temporal models touching 83-98%
  of the pool over 2048 tokens with no expert resident more than 40% of the time. We also
  already measured this objective as harmful when trained in, since Appendix E's anticipatory
  loss collapsed the expert union from 158 to 50 of 192 and worsened BPB. Differentiate in
  related work using `e2_streamed_diversity.csv`; do not run it.
- **Pre-gated MoE**, arXiv:2308.12066. We do not need to run this one, but the serving
  table's `router-early` row is their mechanism and has to cite them there. That is an
  attribution fix, and leaving it is the kind of thing a reviewer flags as undisclosed reuse.
- **Sticky Routing**, arXiv:2607.08780. Concurrent for a September deadline. Cite and
  distinguish.
- **BlockFFN**, arXiv:2507.08771. Chunk-level activation sparsity is mechanically close to
  our own `bursty_window_loss`, whose negative result is already in Appendix E.
- **ProMoE and MoE-Infinity.** Prefetch systems. #4 covers the "you ignored caching"
  objection, and the Belady and oracle replays in Appendix E already bound what better policy
  can buy.

## Naming caution

There is an unrelated and better known **ReMoE** (ReLU-routed MoE, arXiv:2412.14711).
Disambiguate in the bibliography or a reviewer will assume the wrong paper was cited.

## Provenance

- ReMoE and MELINOE method summaries: abstracts read directly. Training budgets, learning
  rates and the MELINOE surface detail come from the reading notes recorded in
  `results/archive/olmoe_wrong_renorm/olmoe_adapt_corpus_audit.md`, not re-verified here
  against the methods sections.
- CoSMoEs, Pre-gated, Oracle-MoE, Skliar and Eliseev summaries: abstracts and the paper's own
  related-work section. Confirm each protocol against its paper before slotting a run into a
  venue, since putting one in the wrong regime is the error this document is meant to prevent.
- The OLMoE 70.7% recovery figure that circulated earlier is void. Those artifacts are
  archived under `results/archive/olmoe_wrong_renorm/`.

## Protocol correction (2026-08-28): compare each method at ITS OWN setting, on three axes

The measured comparisons above (ReMoE and the Skliar deadband, both at our R=8 on GSM8K)
are not a fair comparison and should not be presented as one. Two problems, both raised by
Noah on 2026-08-28:

1. Forcing a competitor into our R=8 bound tests it where it was never designed to operate.
   Each method must be run at the configuration its paper used (ReMoE and CoSMoEs with no
   resident bound; Skliar at their cache sizes, typically half the experts resident, which
   is about 4x our RAM at R=8 on gemma). They may well win on quality there; the point is to
   show what that quality costs in memory and in speed, not to show them losing on our turf.
2. GSM8K alone is not a surface. Every method gets the same five benchmarks we report for
   ourselves (GSM8K, IFEval, HumanEval, MMLU, MBPP; WritingBench where the producer exists).

Every cell is therefore a triple, and the paper's comparison is a plot on these axes:

- **resident memory**: fraction of expert weights resident (ours 6.25% gemma / 3.1% qwen at
  R8; Skliar ~50%; ReMoE and CoSMoEs 100% unless paired with a cache);
- **expert swaps per layer per token**, MEASURED on the eval generations
  (`TEMPORAL_COUNT_SWAPS=1`, `swap_stats()`), not simulated. Above 1 swap per layer per
  token the system becomes very slow, so this is the speed axis. The floor of the memory
  axis is the streaming-only MoE (nothing resident, every selected expert loaded on demand):
  same or better memory than ours and no quality drop, at k swaps per layer per token
  (8 on both models), which is why it is not a usable system. Ours runs at about 1.0 at R8
  (0.9987 measured on gemma GSM8K), i.e. at that threshold by construction of the
  one-swap-per-token rule;
- **quality** on the full surface, same-arm against the unadapted base at the same setting.

What this means for the existing rows: keep the R=8 ReMoE and deadband numbers as an
ablation of "what does bounding do to a bandwidth method" if at all, and build the real
table from runs at each method's own configuration. Not yet run.

**The eviction deadband (ours, Skliar-inspired) against the fair baseline.** The deadband is training-free, so its baseline is the
untrained base at the same R, and on that comparison it is not a null: +1.2 +/- 1.0 at rho
0.5 (positive at rho 0.25, 0.5 and 1.25, each inside noise on its own) and up to ~36% fewer
swaps at flat quality on all five benchmarks. Ours is +3.1 for 3.4M training tokens with
swaps unchanged at 1.0 per layer per token. The two stack (ours + rho 0.5: +3.5). The paper
should present them as complementary at equal memory: theirs moves the speed axis, ours the
quality axis, and the combination is the best cell measured.

**Correction (2026-08-28): the "Skliar" rows are not Skliar.** Every `gemma4_skliar_rho*`
record ran our own residency rule at E=128, k=8, R=8 with one line changed:
`do_swap = nom_val > worst_val + RHO` (`temporal/temporal_router.py:182`; RHO=0 is our
published rule, bit-identical). That is a hysteresis ablation of OUR eviction rule, inspired
by cache-conditional experts, inside OUR bound. It is not their routing, not their cache
sizes, not their memory, and at R=k there is no slack for a deadband to exploit, which is
why quality is flat until the swap rate falls too far. Refer to those rows as "eviction
deadband (ours, inspired by Skliar)" and never as a Skliar baseline. Skliar's method at its
own configuration has not been run.

## Skliar (cache-conditional experts) at their own setting: implemented (2026-08-30)

`analysis/residency/cache_bias.py`, selected with `TEMPORAL_WALKER=cache_bias`. Per MoE layer and decode token: an LRU cache of `C` experts (`TEMPORAL_CB_C`; half the pool, gemma 64/128, qwen 128/256, the paper's 50% regime), ranking logits `z' = z + lambda * delta_avg[layer] * cached` with `delta_avg` the online running mean of `max(z) - min(z)` per layer, top-k by `z'` with the top-J experts of the original `z` guaranteed (`TEMPORAL_CB_J`, 1), gate weights from the original `z` (non-selected experts are masked to -inf, so vLLM's own top-k and softmax reproduce the paper's step 3), and every selected non-cached expert counted as a load (the swap axis) before the LRU update. Prefill is observed unbiased and warms the cache (LRU state after a sequential pass = the C most recently used experts); prefill loads are not counted, matching our own metric. Unit tests: lambda 0 equals plain top-k; the top-1 guarantee holds; exactly k experts per token; the cache never exceeds C; on random logits (E=32, k=4, C=16) loads per token fall 1.99 -> 0.70 -> 0.50 -> 0.48 for lambda 0 / 0.25 / 0.5 / 1.0. Batch-1 semantics are kept per request even though the eval batches requests.

Plan (`tmoe_skliar.sh <model>`): lambda in {0.25, 0.5, 0.75, 1.0} on GSM8K n=1319 with loads/token, then the full surface at lambda 0.5 and 1.0. Three axes per cell: resident fraction 50% (vs ours 6.25% gemma / 3.1% qwen), loads per layer per token, quality vs the unadapted base at the same setting.

## Status of the fair-setting runs (2026-08-30 11:20)

Queued behind the gemma full-pool run, in this order, each on both models, each reported as (resident fraction, loads or swaps per layer per token measured on the eval generations, quality on the full surface vs the unadapted base at the same setting):

1. **Skliar** at their setting (`tmoe_skliar.sh`): LRU cache of half the experts (gemma 64/128, qwen 128/256), lambda in {0, 0.05, 0.1, 0.2, 0.4} on GSM8K n=1319 with loads/token (lambda 0 is the plain-LRU reference, i.e. what a cache of that size costs with no method at all), then the full surface at two operating points chosen from the sweep: the lambda whose loads/token is closest to ours (1.0, equal speed) and the best-GSM8K lambda. Smoke inside vLLM passed (gemma, 8 prompts: same generation lengths as the free arm; at lambda 0.5 the bias is so strong that only the top-J guarantee forces loads, 0.047 per token-layer, which is why the sweep starts at 0.05).
2. **ReMoE** at their setting (`tmoe_remoe_fair.sh`): router-only finetune, recency-reuse objective (lambda 1.0, gamma 0.9), residency OFF during training, our d7 trajectories at 3.4M CE tokens, lr in {1e-4, 3e-4, 1e-3}; per lr GSM8K n=1319 on the free arm (their operating point, 100% resident) and R8 (the bounded ablation); the best free-arm lr gets the full surface on the free arm and its loads/token under a plain LRU cache of half the experts (the only way to put a swap number on an unbounded method).
3. **CoSMoEs**: NOT runnable on this pod tonight. It is a pretraining-time loss and needs the FLAME isoFLOP sweep in the Megatron stack, whose submodules are empty here (`import megatron` fails; TransformerEngine and apex must be built from source, then a sweep of dozens of 1e16 runs). This is a day of environment work plus the sweep, not an overnight item. Until then the paper compares against CoSMoEs' reported numbers with that stated, as Appendix E already does.

## Skliar at their setting, gemma sweep (2026-08-30 15:17)

Cache-conditional experts with an LRU cache of 64 of 128 experts per layer (50% resident), top-J=1, k=8, no training; GSM8K n=1319 with loads per token-layer measured on the generations. For scale, ours at R8: 6.25% resident, 1.0 swaps per token-layer, GSM8K 84.0 (adapted) / 78.8 (unadapted); the free arm is 87.8.

| lambda | loads / token-layer | GSM8K R8-trigger |
|---|---|---|
| 0 (plain LRU) | 0.307 | 87.9 |
| 0.05 | 0.165 | 87.7 |
| 0.1 | 0.104 | 87.2 |
| 0.2 | 0.057 | 87.5 |
| 0.4 | 0.032 | 87.9 |

Reading: at half the experts resident, a plain LRU cache already needs only 0.31 loads per token-layer at base quality, and the bias takes that down tenfold (0.032 at lambda 0.4) with GSM8K flat within noise (87.2-87.9 vs free 87.8). The operating point our speed axis would call "equal" (1.0 swaps per token-layer) does not exist for them: they are faster than us at every lambda, at 8x our resident memory. Full surfaces run at lambda 0 (plain LRU reference) and lambda 0.4 (best GSM8K, fewest loads); qwen follows (C=128 of 256).

### Skliar, gemma, full surfaces on the three axes (2026-08-30 16:26)

Quality vs the free model (100% resident), the right reference for a 50%-memory method; ours is shown against the same reference.

| run | resident memory | GSM8K | IFEval | MMLU | HumanEval | MBPP | swaps or loads / token-layer |
|---|---|---|---|---|---|---|---|
| base, free | 100% | 87.8 | 88.7 | 93.0 | 99.4 | 91.2 | 0 |
| base at R8, unadapted | 6.25% | 78.8 | 86.9 | 92.5 | 94.5 | 78.0 | 1.00 |
| ours, R8 adapted (KL T=2, d7 pool) | 6.25% | 84.0 (-3.8) | 86.7 (-2.0) | 94.3 (+1.3) | 96.3 (-3.0) | 82.2 (-9.0) | 1.00 |
| Skliar, plain LRU (lambda 0) | 50% | 87.9 (+0.1) | 88.7 (0.0) | 92.5 (-0.4) | 98.8 (-0.6) | 91.2 (0.0) | 0.31 |
| Skliar, lambda 0.4 | 50% | 87.9 (+0.2) | 89.6 (+0.9) | 93.0 (0.0) | 99.4 (0.0) | 90.4 (-0.8) | 0.03 |

At half the experts resident their method is lossless on every cell and needs 0.03-0.31 loads per token-layer: better quality and fewer loads than ours, at eight times our resident memory. The two methods occupy different corners of the memory/speed/quality space; ours is the one defined at 6.25%. Qwen (C=128 of 256): plain LRU needs 0.84 loads per token-layer at free-level GSM8K (86.0); sweep in progress.

### Skliar, qwen sweep (2026-08-30 17:19)

LRU cache of 128 of 256 experts (50% resident), top-J=1, k=8, no training; GSM8K n=1319 (free arm 85.9; ours at R8: 3.1% resident, 1.0 swaps per token-layer, 83.5 adapted / 76.6 unadapted).

| lambda | loads / token-layer | GSM8K R8-trigger |
|---|---|---|
| 0 (plain LRU) | 0.837 | 86.0 |
| 0.05 | 0.294 | 86.4 |
| 0.1 | 0.160 | 86.1 |
| 0.2 | 0.087 | 85.6 |
| 0.4 | 0.069 | 85.8 |

Quality flat at the free level for every lambda while loads fall twelvefold. With 256 experts a plain LRU at 50% needs 0.84 loads per token-layer, close to our 1.0, so on qwen the lambda-0 point is nearly an equal-speed comparison: same swap traffic, sixteen times our resident memory, free-level quality against our 83.5. Surfaces at lambda 0 and 0.4 (the lm-eval code stages needed HF_ALLOW_CODE_EVAL and are re-run in a follow-up).

### Skliar, qwen, full surfaces on the three axes (2026-08-30 18:28)

| run | resident memory | GSM8K | IFEval | MMLU | HumanEval | MBPP | loads or swaps / token-layer (GSM8K, IFEval, code) |
|---|---|---|---|---|---|---|---|
| base, free | 100% | 85.9 | 86.5 | 93.4 | 92.7 | 79.4 | 0 |
| base at R8, unadapted | 3.1% | 76.6 | 82.6 | 92.1 | 90.9 | 75.2 | 1.00 |
| ours, R8 adapted (full pool 1.0x) | 3.1% | 83.5 (-2.4) | 82.6 (-3.9) | 91.7 (-1.8) | 92.7 (0.0) | 76.4 (-3.0) | 1.00 |
| Skliar, plain LRU (lambda 0) | 50% | 86.0 (+0.1) | 86.9 (+0.4) | 92.1 (-1.3) | 95.7 (+3.0) | 80.6 (+1.2) | 0.84, 0.53, 0.76 |
| Skliar, lambda 0.4 | 50% | 85.8 (-0.1) | 85.6 (-0.9) | 93.0 (-0.4) | 95.7 (+3.0) | 78.6 (-0.8) | 0.07, 0.05, 0.06 |

Lossless within noise at half the experts resident, at 0.05-0.84 loads per token-layer. The plain-LRU point is close to our swap rate (0.84 vs 1.00) at sixteen times our resident memory and free-level quality: on qwen the comparison reduces to the memory axis. (The lm-eval code tasks need HF_ALLOW_CODE_EVAL=1; the Skliar script now exports it.)

### ReMoE at its setting, gemma (2026-08-30 20:09)

Router-only finetune with the recency-bias reuse objective, residency constraint OFF during
training (their setting: the free model is the product), lr swept on 3.4M tokens of the d7
pool; GSM8K n=1319 free and R8-trigger per lr. Base free 87.8 / R8 78.8.

| lr | GSM8K free | GSM8K R8 |
|---|---|---|
| 1e-4 | 87.6 | 78.1 |
| 3e-4 (pick, by free) | 88.9 | 80.4 |
| 1e-3 | 79.3 | 65.2 |

Full surface of the pick on its own (free) arm, full budgets:

| run | resident memory | GSM8K | IFEval | MMLU | HumanEval | MBPP | loads or swaps / token-layer |
|---|---|---|---|---|---|---|---|
| base, free | 100% | 87.8 | 88.7 | 93.0 | 99.4 | 91.2 | 0 |
| ReMoE lr 3e-4, free | 100% | 88.8 (+1.0) | 89.6 (+0.9) | 93.4 (+0.4) | 98.8 (-0.6) | 89.8 (-1.4) | 0 |
| ReMoE lr 3e-4, LRU C=64 | 50% | 88.8 (+1.0) | | | | | 0.29 |
| ours, R8 adapted (KL T=2, d7 pool) | 6.25% | 84.0 | 86.7 | 94.3 | 96.3 | 82.2 | 1.00 |

Reading: at its own setting ReMoE is quality-neutral (5-cell mean -0.1, noise floor ~1/cell)
and its router regularizer does not reduce cache traffic below the untrained base under the
same LRU (0.29 vs 0.31 at C=64), though it stays lossless there (GSM8K 88.8 = its free arm).
Its R8 arm (80.4, +1.6 over base) trails our R8-adapted 84.0 by 3.6: training never sees the
residency bound. Qwen ReMoE runs the same chain next.

### ReMoE at its setting, qwen (2026-08-30 21:47)

Same chain as gemma (router-only, recency-bias reuse objective, residency OFF in training,
3.4M tokens); GSM8K n=1319 per lr. Base free 85.9 / R8 76.6.

| lr | GSM8K free | GSM8K R8 |
|---|---|---|
| 1e-4 (pick, by free) | 86.1 | 63.1 |
| 3e-4 | 85.4 | 65.4 |
| 1e-3 | 49.0 | 34.0 |

On qwen the objective actively damages the residency-bounded arm (R8 63-65 vs base 76.6)
while the free model stays at base level. Full surface of the pick on its own (free) arm:

| run | resident memory | GSM8K | IFEval | MMLU | HumanEval | MBPP | loads or swaps / token-layer |
|---|---|---|---|---|---|---|---|
| base, free | 100% | 85.9 | 86.5 | 93.4 | 92.7 | 79.4 | 0 |
| ReMoE lr 1e-4, free | 100% | 86.1 (+0.2) | 85.0 (-1.5) | 93.4 (0.0) | 92.7 (0.0) | 79.4 (0.0) | 0 |
| ReMoE lr 1e-4, LRU C=128 | 50% | 86.9 (+1.0) | | | | | 0.79 |
| ours, R8 adapted (full pool 1.0x) | 3.1% | 83.5 | 82.6 | 91.7 | 92.7 | 76.4 | 1.00 |

Reading: quality-neutral at its own setting (5-cell mean -0.3), lossless under the 50% LRU,
and, as on gemma, the regularizer does not cut cache traffic below the untrained base under
the same cache (0.79 vs 0.84). ReMoE on both models: a free-model method; it neither
survives the residency bound (R8 63-80) nor reduces loads; its axis of merit is that the
free model costs nothing to keep.

## Speed axis restated hardware-independently (2026-08-31)

Wall-clock on our H100 harness is not a claim (hooks simulate residency with all experts in
HBM, and H100 link bandwidth says nothing about a deployment target). The speed axis the
paper reports is BYTES MOVED PER TOKEN, measured swap/load rate x expert size (bf16; gemma
expert 11.9MB x 30 layers, qwen 6.3MB x 40 layers). The reader divides by their own link.

| method | gemma MB/token (resident) | qwen MB/token (resident) |
|---|---|---|
| ours, R8-trigger | 357 (6.25%) | 252 (3.1%) |
| Skliar plain LRU C=E/2 | 110 (50%) | 212 (50%) |
| Skliar lambda 0.4 | 11 (50%) | 18 (50%) |
| ReMoE pick under the same LRU | 102 (50%) | 200 (50%) |

ReMoE 2b (fair-surface variant) launched per this doc's own rule: the faithful arm
underperformed at R8, so one cell holds D12's LoRA surface and lr fixed and varies only the
objective (recency reuse, constraint off). Gemma first.

### ReMoE 2b result, gemma (2026-08-31 17:45)

GSM8K n=1319: free 86.9 / R8 77.3, vs the faithful router-only arm 88.9 / 80.4, base
87.8 / 78.8, ours 84.0 at R8 on the identical surface, lr and data. The fair-surface cell
answers the confound: with D12's own LoRA surface the recency objective does WORSE under
the bound than router-only, so the faithful arm's R8 gap is the objective, not the surface.
Training cost 14 min (plain CE + recency term, 4.1k tok/s). Qwen 2b runs the same chain.

### ReMoE 2b result, qwen (2026-08-31 18:22)

GSM8K n=1319: free 87.2 / R8 67.6 (base 85.9 / 76.6, faithful router-only 86.1 / 63.1,
ours 83.5 at R8). Same verdict as gemma on the identical surface, lr and data: the recency
objective leaves the free model healthy and damages the bounded arm on either surface.
The fair-surface cell closes the ReMoE comparison on both models.

## Skliar at OUR memory budget, C=8 (2026-08-31 20:15) — the cross-setting cells

GSM8K n=1319, loads/token-layer measured. Gemma C=8 of 128 (6.25%, a 16x reduction),
qwen C=8 of 256 (3.1%, 32x). Ours at the same memory: gemma adapted 84.0, qwen 83.5, at a
HARD cap of 1.0 swap/token-layer.

| cell | quality | loads/token-layer (mean, unbounded per token) |
|---|---|---|
| gemma C8 plain LRU (lam 0) | 87.8 | 4.67 |
| gemma C8 lam 0.4 | 85.8 | 1.33 |
| qwen C8 plain LRU (lam 0) | 86.0 | 5.63 |
| qwen C8 lam 0.4 | 74.5 | 1.01 |

Reading: their cache stays lossless at aggressive memory only by paying ~5x our traffic
(the LRU thrashes: 8 active experts vs 8 slots). Biasing traffic down toward our budget is
where the methods separate: on qwen at matched memory AND traffic they lose 9 points to us
(74.5 vs 83.5); on gemma the bias holds up better (85.8 at 1.33x our traffic, above our
84.0), so at a 16x reduction the two methods genuinely compete and ours wins on the traffic
guarantee (hard cap vs mean), while at a 32x reduction ours is clearly ahead. Framing per
the user: these methods work well at ~2x memory reductions; ours is the frontier for
aggressive 5-30x reductions, with the gap widening as the budget tightens.

### Skliar C=8 pushed to matched traffic, gemma (2026-08-31 20:55)

Full traffic-quality curve at our memory (C=8 of 128, GSM8K n=1319):

| lambda | loads/token-layer | GSM8K |
|---|---|---|
| 0 (plain LRU) | 4.67 | 87.8 |
| 0.4 | 1.33 | 85.8 |
| 0.5 | 0.94 | 80.7 |
| 0.6 | 0.70 | 69.5 |
| 0.8 | 0.52 | 50.2 |
| 1.2 | 0.49 | 45.1 |

At matched traffic (0.94 vs our hard 1.00) their best point is 80.7 against our adapted
84.0 (paired on shared items: n=1319 ours_fixes=114 skliar_fixes=67 z=+3.49), and below our budget they fall off a cliff (69.5 at
0.70, 45.1 at 0.49). Combined with qwen (74.5 at 1.01 vs our 83.5), the cross-setting
claim is uniform: at 5-30x memory reductions, no measured Skliar operating point matches
our quality at or under our traffic budget on either model, and only the burst-unbounded
1.33x point on gemma comes within two points.

## CoSMoEs BlES on the isoFLOP venue (2026-09-01, first pass): quality axis

Faithful BlES loss (Eq 4-7 of 2503.00245, unit-tested), vanilla routing at R=E; refs
retrained on the rebuilt corpus (bit-identical tokenizer; the fresh g3 vanilla ref matches
the original pre-wipe checkpoint's final CE to 0.19%, validating the pipeline). s0 @ 1e16,
locked phase0 HPs, rope fusion off everywhere (TE 2.16). Test BPB:

| cell | grain 1 | grain 3 |
|---|---|---|
| vanilla MoE (reference) | 1.4486 | 1.4629 |
| temporal MoE (ours, hard 1-swap cap) | 1.4625 (+0.96%) | 1.4807 (+1.2%) |
| BlES lambda 0.1 | 1.5379 (+6.2%) | 1.5373 (+5.1%) |
| BlES lambda 1 | 1.6001 (+10.5%) | 1.5976 (+9.2%) |
| BlES lambda 10 | 1.7757 (+22.6%) | 1.7581 (+20.2%) |
| BlES lambda 100 | 1.9225 (+32.7%) | 1.8103 (+23.7%) |

At every weight measured, BlES costs 5-25x our constraint tax before any switching benefit
is priced in. Their paper states no lambda; a low tail (0.01, 0.03) runs next to trace the
knee fairly, then the router-probe pass adds the other axes: achieved switch rate (their
loads), effective experts, over-use share, neglected-expert count (user note: fewer
switches via collapse is not locality).

Low-lambda knee (pass 2, 2026-09-01 15:21) — complete BlES quality curve:

| BlES lambda | g1 BPB (delta vs vanilla) | g3 BPB (delta) |
|---|---|---|
| 0.01 | 1.4619 (+0.9%) | 1.4753 (+0.8%) |
| 0.03 | 1.5002 (+3.6%) | 1.5100 (+3.2%) |

At lambda 0.01 BlES matches OUR tax (+0.9% vs our +1.0/1.2%); everything above it is
strictly worse. The probe pass decides the verdict: if lambda 0.01 barely moves the switch
rate off vanilla, their method at matched quality buys no locality, while ours delivers
R=k at the same price.

### CoSMoEs verdict: the three axes together (2026-09-01 15:42, probe pass complete)

Fair-usage probe (one fixed batch per trained cell, raw router logits; vanilla anchors:
g1 0.799 switches/token-layer with 63.7/64 effective experts, g3 0.785 with 190.9/192):

| cell | switches/tok-layer (g1 / g3) | effective experts | neglected (<10% uniform) | BPB tax |
|---|---|---|---|---|
| temporal (ours) | 0.62 / 0.63 | 61.7/64, 186/192 | 0 / 0 | +1.0% / +1.2% |
| BlES 0.01 | 0.74 / 0.71 | 59.4, 157.8 | 0.3 / 14.7 | +0.9% / +0.8% |
| BlES 0.03 | 0.62 / 0.66 | 43.1, 120.0 | 12.3 / 41.3 | +3.6% / +3.2% |
| BlES 0.1 | 0.52 / 0.57 | 20.5, 68.4 | 32.7 / 84.7 | +6.2% / +5.1% |
| BlES 1 | 0.55 / 0.58 | 19.8, 64.6 | 34.7 / 88.7 | +10.5% / +9.2% |
| BlES 10 | 0.41 / 0.48 | 12.5, 43.4 | 47.3 / 129.0 | +22.6% / +20.2% |
| BlES 100 | 0.32 / 0.41 | 10.2, 36.1 | 51.0 / 142.3 | +32.7% / +23.7% |

Readings:
1. At matched quality (lambda 0.01, the knee), BlES removes only 8-9% of the switching and
   already starts neglecting experts; it buys essentially no locality.
2. To merely MATCH our raw-demand reuse (0.62), BlES needs lambda 0.03 and pays 3x our
   quality tax while abandoning 19-22% of the expert pool -- the "fewer switches by
   over-dependence" failure mode, measured (user prediction confirmed).
3. Every further switching gain is bought by collapse (up to 74% of experts neglected),
   the demanded set never becomes bounded (mean 0.32+ switches with unbounded per-token
   misses), so no BlES point can serve at R=k; ours serves at R=k (9.4% resident) by
   construction with full balance and a +1% tax.
Appendix E's argument against CoSMoEs is now a measurement, not a citation.

Runtime-cost column added (user note): xfers/token-layer = per-slot churn x k, the experts
that must LOAD if only the previous token's set is resident -- the closest demand-side
proxy for real transfer cost. Ours serves at a hard 1.0 regardless of demand.

| cell | g1 xfers/tok-layer (k=6) | g3 xfers/tok-layer (k=18) | BPB tax (g1/g3) |
|---|---|---|---|
| vanilla anchor | 4.79 | 14.13 | -- |
| temporal (ours), raw demand | 3.73 | 11.39 | +1.0% / +1.2% |
| temporal (ours), SERVED | 1.00 (hard cap) | 1.00 (hard cap) | (same models) |
| BlES 0.01 | 4.42 | 12.85 | +0.9% / +0.8% |
| BlES 0.03 | 3.70 | 11.93 | +3.6% / +3.2% |
| BlES 0.1 | 3.14 | 10.19 | +6.2% / +5.1% |
| BlES 1 | 3.28 | 10.44 | +10.5% / +9.2% |
| BlES 10 | 2.48 | 8.58 | +22.6% / +20.2% |
| BlES 100 | 1.91 | 7.33 | +32.7% / +23.7% |

In transfer units the gap is stark: even at maximum collapse (74% of experts abandoned,
+24-33% BPB) BlES still demands 1.9-7.3 loads per token-layer, while our models SERVE at
1.0 by construction at a +1% tax. No BlES operating point reaches our runtime cost at any
quality. (cosmoes_metrics.csv regenerated with k and xfers_tl; v1 kept alongside.)

Consolidated CoSMoEs table (all axes in one view; supersedes the two partial tables above):

| cell | xfers/tok-layer g1 / g3 | effective experts (of 64 / 192) | neglected g1 / g3 | BPB tax g1 / g3 |
|---|---|---|---|---|
| vanilla anchor | 4.79 / 14.13 | 63.7 / 190.9 | 0 / 0 | -- |
| temporal (ours), raw demand | 3.73 / 11.39 | 61.7 / 186.0 | 0 / 0 | +1.0% / +1.2% |
| temporal (ours), SERVED | 1.00 / 1.00 (hard cap) | 61.7 / 186.0 | 0 / 0 | (same models) |
| BlES 0.01 | 4.42 / 12.85 | 59.4 / 157.8 | 0.3 / 14.7 | +0.9% / +0.8% |
| BlES 0.03 | 3.70 / 11.93 | 43.1 / 120.0 | 12.3 / 41.3 | +3.6% / +3.2% |
| BlES 0.1 | 3.14 / 10.19 | 20.5 / 68.4 | 32.7 / 84.7 | +6.2% / +5.1% |
| BlES 1 | 3.28 / 10.44 | 19.8 / 64.6 | 34.7 / 88.7 | +10.5% / +9.2% |
| BlES 10 | 2.48 / 8.58 | 12.5 / 43.4 | 47.3 / 129.0 | +22.6% / +20.2% |
| BlES 100 | 1.91 / 7.33 | 10.2 / 36.1 | 51.0 / 142.3 | +32.7% / +23.7% |
