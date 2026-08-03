#!/bin/bash
# Arm H (optional-idle, per 0072/0074): E-recipe (router+LoRA r32/a64) + ZONE-CONFINED anneal — R starts
# at 24 (inside B's transfer zone), anneals one expert/rung over first 50M (16 rungs), holds R=8 for the
# remaining 200M. Tests whether EXPERT-representation (LoRA) benefits from a constraint curriculum where
# routing (arm B) did not. base-init, lr 3e-4, 250M, eval 50M at R=8. Bar: beat E 0.8507 by >2sigma.
set -u
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MB=16 HF_HUB_OFFLINE=0
cd /workspace/olmoe-adapt
echo "=== BAKE-OFF arm H (E-recipe + zone anneal 24->8, 250M, eval 50M) $(date -u +%H:%M) ==="
venv/bin/python -u scripts/train_bakeoff.py H 3e-4 250000000 bake_H 50000000 > bake_bake_H.log 2>&1
echo "  H done: $(grep -aE '\[DONE|\[ABORT' bake_bake_H.log | tail -1)"
echo "ARM H DONE. Optional-idle set complete."
