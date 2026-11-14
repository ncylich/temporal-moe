# Hyperparameter sweep — residency adaptation

**Goal.** Find the learning rate that adapts each model to rolling residency. Throughput is
[`TRAINING_OPTIM_PLAN.md`](TRAINING_OPTIM_PLAN.md); this is what to train and at what settings.

**Everything measured before this plan is void.** The 50M runs froze the experts (and, on Qwen,
the router), so ~90% of each model never adapted — `results/ablations/crossmodel_RESULTS.md` §0/§9.
Only the training-free constraint costs carry forward.

## Fixed, not swept

    trainable  expert LoRA r32 + attn LoRA r32 + router gates + RMSNorm gains, all models
    residency  R = k = 8 on every MoE layer (R < k cannot fill top-k; degenerate)
    data       corpus reshaped so every token is reachable; epochs < 1
    budget     15M tokens, evals at 5M/10M/15M — the 50M runs saturated by 10M
    batch      16,384 tokens/optimiser-step, matched across models so LR transfers as a
               model property and not a batch artefact
    harness    OLMoE: stock path (train_ple.py, sweep_olmoe.sh). Qwen: Unsloth path
               (train_unsloth.py, sweep_qwen_unsloth.sh). Implementations carry O(1e-03)
               BPB offsets under the constraint (unsloth_parity.md), so every number that
               enters a comparison — run, null, baseline — comes from one path per model

Optimizer: OLMoE keeps its published cells' fp32-master AdamW; Qwen uses AdamW8bit +
cut_cross_entropy, the only configuration that fits r32 on one H100. One optimizer per model,
and runs with different optimizers are never differenced.

**Aux loss is NOT a hyperparameter.** Each model ships `router_aux_loss_coef` — 0.001 both
Qwen, 0.01 OLMoE — and the trainers read it from the config; previous runs used 0.01
everywhere, 10× the intended pressure on Qwen. Scope is micro-batch, deliberately: global-batch
exists to protect specialisation over trillions of pretraining tokens, and adapting an
already-specialised model for 15M tokens is not that regime.

## What to sweep

| axis | grid | note |
|---|---|---|
| **learning rate** | 1e-5, 3e-5, 1e-4, 3e-4, 1e-3 | 3e-4 is inherited from a different intervention under the gate-mass artifact; never validated here |
| **LoRA rank** | OLMoE only: 32, 128 | Qwen is fixed at r32 — r128 is ~4× r32's 1.9B trainable params, physically impossible on one H100 |
| **null arm** | `--free-set all`, at each finalist only | the achievable ceiling; a null at another LR silently changes the reference |

Selection (pre-registered, `summarize_sweep.py`): lowest held-out BPB at 15M; prune diverged
runs and runs within noise (~0.003) of untrained; tie-break on the 5M checkpoint.

## Order

OLMoE first (cheapest, largest constraint damage, matched null on disk), then Qwen3-30B
(~3.4 h per grid), then Qwen3.5 (~6.3 h) — both unblocked by the accepted Unsloth path.
LR should transfer reasonably (all three have hidden_size 2048, so adapters are identically
shaped) but is verified per model, not assumed.

## Final deliverable — downstream eval on the winning config

Per model: the ten-task downstream suite for best config vs matched null vs untrained baseline.
Table 1: every run — downstream evals, BPB increase over min(null, baseline), % recovery from
the untrained constrained model. Table 2: per model, best vs null vs baseline — BPB increase,
% recovery, avg raw performance, and avg raw / avg baseline (performance retained).
