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
