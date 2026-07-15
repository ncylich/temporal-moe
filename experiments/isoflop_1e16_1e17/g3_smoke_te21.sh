#!/bin/bash
# Validate the TE 2.1 upgrade + --moe-permute-fusion. 3 stages on shape s0, GRAIN=3, mb=64, 50 iters,
# seed 1234. A (MoE no-PF) and B (MoE +PF) must give ~identical loss (PF is numerically equivalent).
# C (temporal +PF) must descend + scan==reference + no NaN.
set -uo pipefail
cd /workspace/FLAME-MoE
ROOT=$(pwd)
export TOKENIZER_MODEL=$ROOT/data/tok16k DATA_DIR=$ROOT/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7600 GRAIN=3 HF_TOKEN= MICRO_BATCH=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_AT_END=1
export EXTRA_ARGS="--train-iters 50 --lr-decay-iters 50 --lr-warmup-iters 5 --eval-iters 1"

run_one() {  # $1=name $2=extra_env_desc ; remaining env vars set by caller
  local name=$1
  rm -rf results/phase0/runs/$name
  RUN_NAME=$name SHAPE=s0 TARGET_FLOPS=1e16 RDZV_PORT=$((29600 + RANDOM % 100)) \
    bash scripts/phase0/run.sh > results/phase0/${name}.out 2>&1 || true
  local d=results/phase0/runs/$name
  local loss=$(grep "lm loss:" $d/train.log 2>/dev/null | tail -1 | grep -oE "lm loss: [0-9.E+]+" | grep -oE "[0-9.E+]+$")
  local nan=$(grep -oE "nan iterations:   [0-9]+" $d/train.log 2>/dev/null | tail -1)
  local oom=$(grep -ciE "out of memory|OutOfMemory" $d/train.log 2>/dev/null)
  local done=$(grep -c "after training is done" $d/train.log 2>/dev/null)
  echo "RESULT $name: done=$done loss@last=$loss $nan oom=$oom"
  pkill -9 -f pretrain_gpt 2>/dev/null; pkill -9 -f pretrain_temporal 2>/dev/null; sleep 3
}

echo "=== A: MoE, TE2.1, NO permute-fusion $(date) ==="
MOE_PERMUTE_FUSION= run_one g3sm_A_moe_nopf

echo "=== B: MoE, TE2.1, +permute-fusion $(date) ==="
MOE_PERMUTE_FUSION=1 run_one g3sm_B_moe_pf

echo "=== C: temporal, TE2.1, +permute-fusion $(date) ==="
MOE_PERMUTE_FUSION=1 TEMPORAL=1 TEMPORAL_EVICT=min_logit run_one g3sm_C_tmoe_pf
grep -i "rolling-residency router installed\|scan path" results/phase0/runs/g3sm_C_tmoe_pf/train.log 2>/dev/null | head -2

echo "=== SMOKE DONE $(date) ==="
