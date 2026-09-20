#!/bin/bash
set -u
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MB=16 HF_HUB_OFFLINE=0
cd /workspace/olmoe-adapt
rm -f data/live_sweep.txt
for lr in 3e-5 1e-4 3e-4; do
  tag="sweep_lr${lr}"
  echo "=== SWEEP $tag (0.25B, eval every 50M) $(date -u +%H:%M) ==="
  venv/bin/python -u scripts/train_router.py "$lr" 250000000 "$tag" 50000000 > "sweep_${tag}.log" 2>&1
  echo "  $tag done: $(grep -aE '\[DONE|\[ABORT' sweep_${tag}.log | tail -1)"
done
echo "ALL SWEEP DONE"
