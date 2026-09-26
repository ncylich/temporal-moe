#!/bin/bash
# Post-C queue per orch 0078/0079: CE (router+norms+LoRA) FULL 250M — promotion confirmed (additive,
# still descending: 30M=0.8340 beats E@50M by ~5sigma). Evals every 10M (screen grid 10-50M + beyond),
# full checkpoint/resume every eval. Then D (self-distill router-only, 250M, MB=4). G + F-prime after.
set -u
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MB=16 HF_HUB_OFFLINE=0
cd /workspace/olmoe-adapt
echo "=== BAKE-OFF arm CE (250M, lr=3e-4, eval every 10M, checkpointed) $(date -u +%H:%M) ==="
venv/bin/python -u scripts/train_bakeoff.py CE 3e-4 250000000 bake_CE 10000000 > bake_bake_CE.log 2>&1
echo "  CE done: $(grep -aE '\[DONE|\[ABORT' bake_bake_CE.log | tail -1)"
echo "=== BAKE-OFF arm D re-run (250M, lr=3e-4, eval every 50M, MB=4 self-distill) $(date -u +%H:%M) ==="
venv/bin/python -u scripts/train_bakeoff.py D 3e-4 250000000 bake_D 50000000 > bake_bake_D.log 2>&1
echo "  D done: $(grep -aE '\[DONE|\[ABORT' bake_bake_D.log | tail -1)"
echo "POST-C QUEUE DONE (CE-250M, D). G + F-prime next."
