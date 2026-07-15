#!/bin/bash
set -uo pipefail
cd /workspace/FLAME-MoE
for mb in "$@"; do
  d=results/phase0/runs/flame38m_probe
  rm -rf "$d"
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 2; done
  echo ">>> PROBE G3-temporal mb=$mb $(date)"
  GRAIN=3 MICRO_BATCH=$mb TRAIN_ITERS=4 EVAL_ITERS=1 RUN_NAME=flame38m_probe RDZV_PORT=$((29530+mb)) \
    bash scripts/phase0/flame38m_run.sh > results/phase0/flame38m_probe_mb${mb}.out 2>&1 || true
  if grep -q "after training is done" "$d/train.log" 2>/dev/null; then
    ms=$(grep -oE "elapsed time per iteration \(ms\): [0-9.]+" "$d/train.log" 2>/dev/null | grep -oE "[0-9.]+$" | tail -1)
    peak=$(grep -oE "max allocated: [0-9.]+" "$d/train.log" 2>/dev/null | tail -1)
    echo "RESULT: FIT mb=$mb  ~${ms}ms/iter  $peak MiB"
  elif grep -qiE "out of memory|OutOfMemory" "$d/train.log" 2>/dev/null; then
    echo "RESULT: OOM mb=$mb"
  else echo "RESULT: UNKNOWN mb=$mb"; tail -3 "$d/train.log" 2>/dev/null; fi
  pkill -9 -f pretrain_temporal 2>/dev/null; pkill -9 -f pretrain_gpt 2>/dev/null; sleep 3
done
rm -rf results/phase0/runs/flame38m_probe
echo "=== MBPROBE DONE ==="
