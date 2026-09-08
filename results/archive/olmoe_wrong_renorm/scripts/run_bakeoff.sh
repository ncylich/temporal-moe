#!/bin/bash
# Stage 2b bake-off runner. Usage: run_bakeoff.sh <winner_lr> <arm> [arm ...]
# Runs eval-noise sigma ONCE (if not already done), then each arm's 0.25B base-router-init run
# sequentially at the winner LR, evals every 50M at R=8 with telemetry. Arm A = sweep winner (no rerun).
set -u
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MB=16 HF_HUB_OFFLINE=0
cd /workspace/olmoe-adapt
LR=$1; shift
if [ ! -f data/eval_sigma.json ]; then
  echo "=== eval-noise sigma $(date -u +%H:%M) ==="
  venv/bin/python -u scripts/eval_noise_sigma.py > sigma.log 2>&1
  echo "  sigma: $(grep -a '\[sigma\] R8' sigma.log | tail -1)"
fi
for arm in "$@"; do
  tag="bake_${arm}"
  echo "=== BAKE-OFF arm $arm (0.25B, lr=$LR, eval every 50M) $(date -u +%H:%M) ==="
  venv/bin/python -u scripts/train_bakeoff.py "$arm" "$LR" 250000000 "$tag" 50000000 > "bake_${tag}.log" 2>&1
  echo "  $arm done: $(grep -aE '\[DONE|\[ABORT' bake_${tag}.log | tail -1)"
done
echo "ALL BAKE-OFF ARMS DONE"
