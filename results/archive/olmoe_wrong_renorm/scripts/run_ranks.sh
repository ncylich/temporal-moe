#!/bin/bash
# Optional-idle rank screens (per orch 0074): E-recipe LoRA at r=8 then r=64, 50M/eval-10M screening.
# Tests adapter minimality vs E(r32)@50M=0.8642. Promote to 250M only if beats E@50M by >2sigma or
# still descending. Fills idle GPU before Phase-C selection; both remain OPTIONAL.
set -u
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MB=16 HF_HUB_OFFLINE=0
cd /workspace/olmoe-adapt
for arm in Er8 Er64; do
  echo "=== BAKE-OFF rank screen $arm (50M, lr=3e-4, eval 10M) $(date -u +%H:%M) ==="
  venv/bin/python -u scripts/train_bakeoff.py "$arm" 3e-4 50000000 "bake_${arm}" 10000000 > "bake_bake_${arm}.log" 2>&1
  echo "  $arm done: $(grep -aE '\[DONE|\[ABORT' bake_bake_${arm}.log | tail -1)"
done
echo "RANK SCREENS DONE (Er8, Er64). H remains optional."
