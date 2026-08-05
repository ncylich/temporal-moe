# Hyperparameter sweep — residency adaptation

**Goal.** Find a learning rate and LoRA rank that actually adapt the model to rolling residency.
Throughput is a separate document ([`TRAINING_OPTIM_PLAN.md`](TRAINING_OPTIM_PLAN.md)); this one is
about what to train and at what settings.

**Everything measured before this plan is void.** The completed 50M runs froze the experts
(`--lora 0`) and, on Qwen, the router too — so ~90% of each model never adapted, and the recovery
percentages were computed against a null with a different trainable surface. Details in
`results/ablations/crossmodel_RESULTS.md` §0 and §9. Nothing from them carries forward except the
training-free constraint costs.

## Fixed, not swept

    trainable  expert LoRA + attn LoRA + router gates + RMSNorm gains — identical across models
    residency  R = k = 8 on every MoE layer (R < k cannot fill top-k; degenerate)
    data       corpus reshaped so every token is reachable; keep epochs < 1
    budget     15M tokens, evals at 5M/10M/15M — the 50M runs saturated by 10M
    batch      16,384 tokens/optimiser-step, matched across models so LR transfers as a model
               property and not a batch artefact

**Aux loss is NOT a hyperparameter.** Each model ships its own `router_aux_loss_coef` — 0.001 for both
Qwen models, 0.01 for OLMoE — and the trainers now read it from the config. Every previous run used
0.01 everywhere, i.e. 10× the intended pressure on Qwen. Use the vendor's number; do not search it.

**Aux scope: micro-batch, deliberately.** Qwen3 pretrains with *global-batch* load balancing, on the
argument that micro-batch balancing suppresses expert specialisation over trillions of tokens. We are
adapting an already-specialised model for ~15M tokens and cannot un-specialise it, so the reason for
global-batch does not apply. Micro-batch is simpler and matches what the model sees.

## What to sweep

| axis | grid | note |
|---|---|---|
| **learning rate** | 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 | 3e-4 is the inherited value, fitted for a *different* intervention on a model under the gate-mass artifact. Never validated here |
| **LoRA rank** | 16, 32 | rank does not transfer between models — the same label buys ~4× different capacity depending on depth and head geometry. On Qwen3.5 it may be a memory decision rather than a capacity one |

Nothing else. Selection: lowest held-out BPB at 15M; prune runs that diverge or that sit within noise
(~0.003) of untrained; tie-break on the 5M checkpoint, preferring whichever got there soonest.

Recovery percentages need a null at the **same** LR — a higher LR damages the null too, so scoring
against a null from another LR silently changes the reference. Run nulls only at the finalists.

## Order, and why it is deferred

Qwen3-30B first (most headroom), then Qwen3.5. OLMoE only if a cheap reference is wanted.

**Do not start this until the Unsloth work lands.** Unsloth ships a tuned MoE fine-tuning setup —
their own LoRA defaults, gradient checkpointing, and kernels — which is likely a better starting
point than sweeping our hand-rolled path. Sweeping now would tune a configuration we intend to
replace, and rank and LR both interact with whatever adapter implementation ends up underneath.
