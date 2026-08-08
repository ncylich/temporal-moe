#!/bin/bash
# Arm F-prime (per orch 0074/0079): full-FT floor probe. Warm-start = CE's full checkpoint
# (router+norms plain-load + exact LoRA->expert merge), identity-check reproduces parent BPB, then
# unfreeze everything, 8-bit Adam lr 1e-5, ~200M tokens, evals every 25M at R=8. Self-aborts if the
# merge isn't identity (>0.003 BPB). Reads: breaks ~93% plateau=capacity floor; same=constraint price.
set -u
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MB=4 HF_HUB_OFFLINE=0
cd /workspace/olmoe-adapt
echo "=== BAKE-OFF arm F-prime (full-FT, 200M, 8-bit Adam lr1e-5, eval 25M) $(date -u +%H:%M) ==="
venv/bin/python -u scripts/train_fprime.py data/ckpt_bake_CE.pt 200000000 fprime 25000000 > bake_bake_Fprime.log 2>&1
echo "  F-prime done: $(grep -aE '\[DONE|\[ABORT' bake_bake_Fprime.log | tail -1)"
echo "F-PRIME DONE. Bake-off complete (all 8 arms)."
