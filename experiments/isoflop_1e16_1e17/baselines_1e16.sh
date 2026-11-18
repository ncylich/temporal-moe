#!/bin/bash
# One 1e16 cell (shape s0, fine grain 18 of 192) in the venue of the CoSMoEs sweep
# (scripts/residency/orchestration/tmoe_cosmoes_sweep.sh): tok16k corpus, locked phase0 HPs
# (lr 3e-3, warmup 0.05, gb 256, seed 1234), rope fusion off, eval at end. Trains, then runs the
# router probe and the load metrics (analysis/probes/cosmoes_metrics.py) on the checkpoint.
#
#   NAME=g3_sticky0.1_1e16 METHOD=sticky LAMBDA=0.1 GPUS=2 bash .../baselines_1e16.sh
#   METHOD=moe: stock standard MoE (pretrain_gpt.py). METHOD=none: patched router at R=E, no loss.
#   GPUS=0,1 trains data-parallel on two cards (same global batch and iteration count).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
cd "$ROOT"
NAME=${NAME:?run name}; METHOD=${METHOD:?moe | none | cosmoes | sticky}; LAMBDA=${LAMBDA:-0}
GPUS=${GPUS:-0}
export TOKENIZER_MODEL=$ROOT/data/tok16k DATA_DIR=$ROOT/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7568 EVAL_AT_END=1 EXTRA_ARGS="--no-rope-fusion"
export GRAIN=3 SHAPE=s0 TARGET_FLOPS=1e16 PEAK_LR=3e-3 WARMUP_FRAC=0.05 GLOBAL_BATCH=256 SEED=1234
export RUN_NAME=$NAME RDZV_PORT=${RDZV_PORT:-$((29510 + ${GPUS%%,*}))}
export TEMPORAL=1 TEMPORAL_EVICT=min_logit TEMPORAL_RESIDENCY_R=192
case $METHOD in
  moe)     export TEMPORAL=0; unset TEMPORAL_RESIDENCY_R;;
  none)    ;;
  cosmoes) export COSMOES_LAMBDA=$LAMBDA;;
  sticky)  export STICKY_LAMBDA=$LAMBDA;;
  *) echo "bad METHOD $METHOD"; exit 1;;
esac
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ') CUDA_VISIBLE_DEVICES=$GPUS bash experiments/run.sh
"$PY" analysis/parse_run.py "$CKPT_ROOT/$NAME" | grep '^SUMMARY' || echo "[cell] $NAME parse failed"
# Probe on one card. Raw router logits only, so TEMPORAL=0; --save to a throwaway dir.
TEMPORAL=0 PROBE=1 CUDA_VISIBLE_DEVICES=${GPUS%%,*} RDZV_PORT=$((RDZV_PORT + 20)) \
  EXTRA_ARGS="--no-rope-fusion --save $CKPT_ROOT/$NAME/probe_junk_ckpt" bash experiments/run.sh > /dev/null
"$PY" analysis/probes/cosmoes_metrics.py "$NAME"
echo "### cell $NAME DONE"
