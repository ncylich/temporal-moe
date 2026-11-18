#!/bin/bash
# Published locality losses as pretraining baselines at 1e18, fine grain (18 of 192).
# Same shape, data, tokenizer, schedule and iteration count as the 1e18 standard-MoE reference
# (flame38m_run.sh, GRAIN=3, mb 32, seed 1234); the auxiliary loss is the only change. The run
# goes through the patched router at R=E (mask all-true, i.e. standard routing) because that is
# where the loss hooks live; stock pretrain_gpt.py would train with no auxiliary loss at all.
#
#   METHOD=cosmoes LAMBDA=0.03 GPUS=0,1 bash experiments/scale_1e18_1e19/baselines_1e18.sh
#   METHOD=sticky  LAMBDA=...  GPUS=2,3 bash experiments/scale_1e18_1e19/baselines_1e18.sh
#   PROBE=1 ... same line: router probe on the trained checkpoint (one card).
#   METHOD=none: no loss, patched router at R=E (control for the path itself).
set -euo pipefail
METHOD=${METHOD:?cosmoes | sticky | none}
LAMBDA=${LAMBDA:-0}
GPUS=${GPUS:-0}
case $METHOD in
  cosmoes) export COSMOES_LAMBDA=$LAMBDA;;
  sticky)  export STICKY_LAMBDA=$LAMBDA;;
  none)    ;;
  *) echo "bad METHOD $METHOD"; exit 1;;
esac
export CUDA_VISIBLE_DEVICES=$GPUS
export NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
export GRAIN=3 MICRO_BATCH=${MICRO_BATCH:-32} TEMPORAL_EVICT=min_logit TEMPORAL_RESIDENCY_R=192
export RUN_NAME=${RUN_NAME:-flame38m_g3_${METHOD}${LAMBDA}_1e18}
export RDZV_PORT=${RDZV_PORT:-$((29560 + ${GPUS%%,*}))}
exec bash "$(dirname "${BASH_SOURCE[0]}")/flame38m_run.sh"
