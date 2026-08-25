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
