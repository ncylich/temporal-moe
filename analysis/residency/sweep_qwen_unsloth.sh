#!/usr/bin/env bash
# Stage 2: Qwen LR sweep on the Unsloth path (SWEEP_PLAN.md), one family per invocation.
#
#   sweep_qwen_unsloth.sh qwen3        # Qwen3-30B first: 2.2x the throughput of Qwen3.5
#   sweep_qwen_unsloth.sh qwen3_5
#
# Configuration is the accepted one (results/ablations/unsloth_parity.md): r32, AdamW8bit +
# CCE, bf16 adapters, residency_unsloth, aux from the model's shipped config, micro-batch
# aux scope. 15M tokens, evals at 5/10/15M, 16,384 tok/step matched across models.
#
# RANK: fixed at r32 for Qwen (SWEEP_PLAN.md) -- r128 is ~4x r32's 1.9B trainable params,
# physically impossible on one H100. The rank ablation lives on OLMoE alone.
#
# Nulls (--free-set all) are NOT run here: SWEEP_PLAN.md runs nulls at the finalists only,
# after summarize_sweep.py picks them.
#
# Env, per unsloth_parity.md: autotune off everywhere (autotuners bench at peak memory and
# OOM a full card -- warm the fla cache first for qwen3_5); compile off for qwen3_5 only
# (its CUDA-graph pools hold ~7.4 GB of VRAM the model needs); expandable segments always.
set -u
cd "$(dirname "$0")"
FAM="${1:?usage: sweep_qwen_unsloth.sh qwen3|qwen3_5}"
PY=/workspace/venv_fla/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNSLOTH_MOE_DISABLE_AUTOTUNE=1
[ "$FAM" = "qwen3_5" ] && export UNSLOTH_COMPILE_DISABLE=1

for LR in 1e-5 3e-5 1e-4 3e-4 1e-3; do
  echo "=== [$(date -u +%H:%M)] $FAM lr=$LR ==="
  $PY train_unsloth.py --tag "sweep_lr${LR}" --family "$FAM" --lr "$LR" --lora 32 \
      --tokens 15000000 --eval-every 5000000 > "/tmp/sweep_${FAM}_lr${LR}.log" 2>&1
  echo "=== [$(date -u +%H:%M)] lr=$LR exit=$? ==="
done
echo "=== ${FAM} LR SWEEP COMPLETE ==="
