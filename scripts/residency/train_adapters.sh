#!/bin/bash
# RECOVER_DATA_PLAN sections 1.3 / 1.4 -- retrain the two CE adapters.
#
#     train_adapters.sh gemma    # GPU 1
#     train_adapters.sh qwen     # GPU 2
#
# Four stages per model, gated in order: smoke -> KL precompute -> train -> mirror.
# Each stage prints its own DONE marker so a monitor can chain on it.
#
# QWEN USES train_gemma_ce.py --family qwen35 --no-unsloth, not train_qwen_ce.py.
# RECOVER_DATA_PLAN 1.4 names train_qwen_ce.py, but that script builds on unsloth's
# FastModel, and gemma_adapt_RESULTS records the opposite for the committed r2 config:
# "HF stack (unsloth's batched constrained path drifts 4.9% on qwen where plain HF shows
# 0.0-0.3%)", with TODO section 6 calling it "the --no-unsloth/HF+peft path".
# train_gemma_ce.py carries a --family qwen35 router patch and is layout-generic
# otherwise, so it is the trainer that matches the documented recipe.
#
# Settings differ from the published run in three ways, all deliberate:
#   expert-LoRA r32 (gemma) / r16 (qwen), not r16/r8. Published ranks were a capacity
#     MATCH, not a memory limit -- gemma r16 = 0.48B params, qwen r8 = 0.46B. Equal rank
#     would NOT preserve that: qwen has 2x the experts and 1.33x the layers, so r32 both
#     sides means qwen 1.85B against gemma 0.95B. r32/r16 doubles both while holding the
#     match to within 3% (0.95B vs 0.92B).
#   no gradient accumulation: --accum 16 --micro-batch 16 gives accum_batches=1, the same
#     16 rows per optimizer step as the published --accum 16 --micro-batch 2, so the
#     optimizer step is unchanged and only the schedule is faster.
#   full AdamW at micro-batch 16 on BOTH models. The published qwen run used paged
#     8-bit Adam at micro-batch 2 because 70GB of weights left about 10GB of headroom on
#     an 80GB card; on a 143GB H200 that headroom is about 73GB, so the accommodation is
#     unnecessary and quantised optimiser states only add noise. Same reasoning retires
#     gradient accumulation on the qwen side.
#   gemma runs --no-unsloth. The published gemma run used unsloth, but on this pod
#     unsloth's gemma4 patches are NON-DETERMINISTIC: six identical forwards of the
#     same probe, with no residency patch, no expert-LoRA and no constraint, differ
#     pairwise by 5.75 to 10.75 max|dlogit|. Divergence appears at the first
#     transformer block (layer 0 identical, layer 1 = 0.25) and compounds to 31.6 by
#     layer 30. Plain HF is bit-exact (0.000000) on the same weights and machine, and
#     the grouped-path parity gate then reads 0.000. The project already prefers HF
#     where unsloth drifts -- gemma_adapt_RESULTS says exactly that for qwen.
#   trajectories cut to seq 4096 whole-row (cut_trajectories.py), from 8192-cap
#     generation rather than the published 2048/3072 caps.
# Budget stays 3.4M response tokens: the ladder shows 10M collapses the KL recipe.
set -euo pipefail

ROOT=/workspace/temporal-moe
export TMOE_ROOT=$ROOT
export PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $ROOT

PY=/workspace/venv_fla/bin/python
T=analysis/residency/train_gemma_ce.py
DATA=/workspace/olmoe-adapt/data
LOG=${LOG_DIR:-/workspace/rerun-logs}
mkdir -p $DATA $LOG

# GPU is overridable: this box is SHARED. A third-party job (run_acts.py) claimed GPU 0
# and GPU 1 mid-run on 2026-08-25 and OOM'd gemma inside loss.backward() by taking 61 GiB
# out from under it. Pin a lane to a device that is actually free rather than assuming.
case "${1:?usage: train_adapters.sh gemma|qwen}" in
  gemma)
    DEV=${GPU:-2}; MODEL=/dev/shm/gemma4-26b-it; TRAJ=gemma4_d7_seq4096
    FAM=(--family gemma4 --no-unsloth); RANK=32; KLW=0.05; MB=16; OPT=(--opt adamw)
    OUT=$DATA/gemma_ce_realmath_adapter.pt ;;
  qwen)
    DEV=${GPU:-2}; MODEL=/dev/shm/qwen35-35b-a3b; TRAJ=qwen35_d7_seq4096
    FAM=(--family qwen35 --no-unsloth); RANK=16; KLW=0.1; MB=16; OPT=(--opt adamw)
    OUT=$DATA/qwen_ce_realmath_adapter.pt ;;
  *) echo "unknown: $1" >&2; exit 2 ;;
esac
KL=/workspace/instruct-traj/${TRAJ}_klref.pt
export CUDA_VISIBLE_DEVICES=$DEV
# --out belongs in COMMON, not just on the train stage: the trainer refuses any
# --expert-lora-r run whose --out is still the default, to stop an expert-LoRA
# adapter clobbering the attention-only one. Smoke and KL never write it.
COMMON=(--model $MODEL --traj $TRAJ --max-seq 4096 --expert-lora-r $RANK \
        --out $OUT "${FAM[@]}" "${OPT[@]}")

echo "### $1 STAGE 1/4 smoke $(date -u +%H:%M)"
$PY -u $T "${COMMON[@]}" --micro-batch $MB --smoke 2>&1 | tee $LOG/adapt_${1}_smoke.log
echo "### $1 SMOKE DONE $(date -u +%H:%M)"

if [ ! -s "$KL" ]; then
  echo "### $1 STAGE 2/4 KL precompute (base model, forward only) $(date -u +%H:%M)"
  $PY -u $T "${COMMON[@]}" --micro-batch $MB --precompute-kl $KL 2>&1 | tee $LOG/adapt_${1}_kl.log
else
  echo "### $1 STAGE 2/4 KL anchor already present, skipping"
fi
echo "### $1 KL DONE $(date -u +%H:%M)"

echo "### $1 STAGE 3/4 train, 3.4M response tokens $(date -u +%H:%M)"
$PY -u $T "${COMMON[@]}" --micro-batch $MB --accum 16 --lr 3e-5 \
    --tokens 3400000 --kl-anchor $KL --kl-weight $KLW \
    2>&1 | tee $LOG/adapt_${1}_train.log
echo "### $1 TRAIN DONE $(date -u +%H:%M)"

echo "### $1 STAGE 4/4 mirror to Hugging Face $(date -u +%H:%M)"
$PY $ROOT/scripts/residency/mirror_artifact.py --path $OUT --kind adapter
echo "### $1 ALL DONE $(date -u +%H:%M)"
