#!/bin/bash
# Time candidate configs (40-iter runs) to pick final mb + permute-fusion. Reports steady per-iter
# (median of last 5) + FIT/OOM. Args: "shape flops mb pf(0/1)" ...
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
cd "$ROOT"
export TOKENIZER_MODEL=$ROOT/data/tok16k DATA_DIR=$ROOT/data/tok16k_full
export CE_FUSION=1 BPB_DIVISOR=2.7600 GRAIN=3 HF_TOKEN=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_AT_END=1
export EXTRA_ARGS="--train-iters 40 --lr-decay-iters 40 --lr-warmup-iters 4 --eval-iters 1"
PORT=29700
for spec in "$@"; do
  read shape flops mb pf <<< "$spec"
  d=results/phase0/runs/g3_time
  rm -rf "$d"
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 2; done
  PORT=$((PORT+1))
  MOE_PERMUTE_FUSION=$([ "$pf" = "1" ] && echo 1 || echo "") \
  MICRO_BATCH=$mb SHAPE=$shape TARGET_FLOPS=$flops RUN_NAME=g3_time RDZV_PORT=$PORT \
    bash experiments/run.sh > results/phase0/g3_time_${shape}_${mb}_pf${pf}.out 2>&1 || true
  if grep -qiE "out of memory|OutOfMemory" "$d/train.log" 2>/dev/null; then
    echo "TIME $shape@$flops mb=$mb pf=$pf : OOM"
  else
    med=$(grep -oE "elapsed time per iteration \(ms\): [0-9.]+" "$d/train.log" 2>/dev/null | grep -oE "[0-9.]+$" | tail -5 | sort -n | head -3 | tail -1)
    echo "TIME $shape@$flops mb=$mb pf=$pf : ${med} ms/iter (FIT)"
  fi
  pkill -9 -f pretrain_gpt 2>/dev/null; sleep 3
done
echo "=== TIME PROBE DONE ==="
