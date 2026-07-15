#!/bin/bash
# Probe max micro-batch per shape (3-iter runs; report FIT/OOM). One GPU, serial.
# expandable_segments reduces allocator fragmentation (numerically safe). Waits for GPU to free
# between probes so a prior OOM's teardown can't poison the next probe.
set -uo pipefail
cd /workspace/FLAME-MoE
ROOT=$(pwd)
export TOKENIZER_MODEL=$ROOT/data/tok16k DATA_DIR=$ROOT/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7600 GRAIN=3 HF_TOKEN=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVAL_AT_END=1 EXTRA_ARGS="--train-iters 3 --lr-decay-iters 3 --lr-warmup-iters 1 --eval-iters 1"
PORT=29570
for spec in "$@"; do
  read shape flops mb <<< "$spec"
  d=results/phase0/runs/g3_probe
  rm -rf "$d"
  # wait for GPU to be free
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 2; done
  PORT=$((PORT+1))
  echo ">>> PROBE shape=$shape flops=$flops mb=$mb $(date)"
  MICRO_BATCH=$mb SHAPE=$shape TARGET_FLOPS=$flops RUN_NAME=g3_probe RDZV_PORT=$PORT \
    bash experiments/run.sh > results/phase0/g3_probe_${shape}_${flops}_mb${mb}.out 2>&1 || true
  if grep -q "after training is done" "$d/train.log" 2>/dev/null; then
    echo "RESULT: FIT  $shape@$flops mb=$mb"
  elif grep -qiE "out of memory|OutOfMemory" "$d/train.log" 2>/dev/null; then
    echo "RESULT: OOM  $shape@$flops mb=$mb"
  else
    echo "RESULT: UNKNOWN $shape@$flops mb=$mb"; tail -2 "$d/train.log" 2>/dev/null
  fi
  pkill -9 -f pretrain_gpt 2>/dev/null; sleep 4
done
rm -rf results/phase0/runs/g3_probe
echo "=== PROBE DONE $(date) ==="
