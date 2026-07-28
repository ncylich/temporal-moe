#!/bin/bash
# Overlap-arch 1e17-g3-s2 screens (orch 0150 step 3). Reproduces the standard g3-s2-1e17 config
# (16k vocab, cosine, gb256/mb64/lr3e-3/3861it/seed1234, GRAIN=3 -> 192 exp/top-18/moe_ffn58/shared352,
# CE_FUSION, EVAL_AT_END) and adds one overlap flag via EXTRA_ARGS. 4 cells run sequentially on 1 H100:
#   v1 early-router x {temporal, moe} ; v2 parallel-ffn x {temporal, moe}.
# Standard baselines (16k divisor 2.7600): temporal 1.2873, moe 1.2708. Gate: v-temporal<=1.2923 AND v-moe<=1.2808.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
cd "$ROOT"
export TOKENIZER_MODEL=$ROOT/data/tok16k
export DATA_DIR=$ROOT/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7600 GRAIN=3 EVAL_AT_END=1 MICRO_BATCH=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SHAPE=s2 TARGET_FLOPS=1e17 PEAK_LR=3e-3 WARMUP_FRAC=0.05 GLOBAL_BATCH=256 SEED=1234
export AUX_COEFF=0.01 LR_DECAY_STYLE=cosine

screen() {  # $1=run_name $2=port $3=extra_args $4=temporal(0/1)
  echo "[screens] $(date -u +%FT%TZ) START $1 (temporal=$4, $3)"
  rm -rf results/phase0/runs/"$1"
  if [ "$4" = 1 ]; then
    env TEMPORAL=1 TEMPORAL_EVICT=min_logit EXTRA_ARGS="$3" RUN_NAME="$1" RDZV_PORT="$2" bash experiments/run.sh
  else
    env EXTRA_ARGS="$3" RUN_NAME="$1" RDZV_PORT="$2" bash experiments/run.sh
  fi
  echo "[screens] $(date -u +%FT%TZ) DONE $1 rc=$?"
}

screen g3_tmoe_s2_1e17_ovlEarly  29601 "--overlap-early-router"  1
screen g3_moe_s2_1e17_ovlEarly   29602 "--overlap-early-router"  0
screen g3_tmoe_s2_1e17_ovlParFFN 29603 "--overlap-parallel-ffn"  1
screen g3_moe_s2_1e17_ovlParFFN  29604 "--overlap-parallel-ffn"  0
echo "[screens] ALL 4 OVERLAP SCREENS DONE"
