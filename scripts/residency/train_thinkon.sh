#!/bin/bash
# B2 (RECOVER_DATA_PLAN Part 0 group B): think-on CE adaptation.
#
# Section 7's length result rests on gemma alone, because qwen's adapted checkpoints were
# only ever evaluated think-off -- where neither routing regime lengthens, so the
# comparison is empty. This trains the think-on adapter that makes it non-empty.
#
# Sequence is 8192, not 4096: think-on responses are far longer (qwen median 3363, p90
# 7446, 56.5% over 3072), which is why gemma_adapt_RESULTS says a valid think-on run needs
# a >=6k envelope. That forces gradient accumulation back on -- micro-batch 16 at seq 8192
# is 4x the activation memory of the think-off run's 126GB and will not fit. Effective
# batch stays 16 rows per optimizer step, so the optimizer step is unchanged.
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

case "${1:?usage: train_thinkon.sh gemma|qwen}" in
  gemma) MODEL=/dev/shm/gemma4-26b-it;  TRAJ=gemma4_d7think_seq8192
         FAM=(--family gemma4 --no-unsloth); RANK=32; KLW=0.05
         OUT=$DATA/gemma_ce_thinkon_adapter.pt ;;
  qwen)  MODEL=/dev/shm/qwen35-35b-a3b; TRAJ=qwen35_d7think_seq8192
         FAM=(--family qwen35 --no-unsloth); RANK=16; KLW=0.1
         OUT=$DATA/qwen_ce_thinkon_adapter.pt ;;
  *) echo "unknown: $1" >&2; exit 2 ;;
esac
KL=/workspace/instruct-traj/${TRAJ}_klref.pt
export CUDA_VISIBLE_DEVICES=${GPU:-0}
COMMON=(--model $MODEL --traj $TRAJ --max-seq 8192 --expert-lora-r $RANK --out $OUT
        "${FAM[@]}" --opt adamw --micro-batch ${MB:-4})

echo "### thinkon-$1 STAGE 1/4 smoke $(date -u +%H:%M)"
$PY -u $T "${COMMON[@]}" --smoke 2>&1 | tee $LOG/thinkon_${1}_smoke.log
echo "### thinkon-$1 SMOKE DONE $(date -u +%H:%M)"
if [ ! -s "$KL" ]; then
  echo "### thinkon-$1 STAGE 2/4 KL precompute $(date -u +%H:%M)"
  $PY -u $T "${COMMON[@]}" --precompute-kl $KL 2>&1 | tee $LOG/thinkon_${1}_kl.log
fi
echo "### thinkon-$1 KL DONE $(date -u +%H:%M)"
echo "### thinkon-$1 STAGE 3/4 train $(date -u +%H:%M)"
$PY -u $T "${COMMON[@]}" --accum 16 --lr 3e-5 --tokens 3400000 \
    --kl-anchor $KL --kl-weight $KLW 2>&1 | tee $LOG/thinkon_${1}_train.log
echo "### thinkon-$1 TRAIN DONE $(date -u +%H:%M)"
$PY $ROOT/scripts/residency/mirror_artifact.py --path $OUT --kind adapter
echo "### thinkon-$1 ALL DONE $(date -u +%H:%M)"
