#!/bin/bash
# Arm G (per orch 0078/0079 additive branch): router + norm gains + LoRA r32/a64 + self-distillation
# (0.5 data_CE + 0.5 soft_CE from frozen BASE free-routing teacher = base router + base norms + LoRA-off).
# base-init, lr 3e-4, 250M, evals every 50M at R=8, MB=4 (2 fwd), checkpointed. Answers: does a richer
# training signal push the ~91.5% calibration+capacity plateau higher?
set -u
NV=/usr/local/lib/python3.11/dist-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:/usr/local/cuda/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=0
cd /workspace/olmoe-adapt
echo "=== BAKE-OFF arm G (150M, lr=3e-4, eval every 50M, MB=4 norms+LoRA+distill) $(date -u +%H:%M) ==="
venv/bin/python -u scripts/train_bakeoff.py G 3e-4 150000000 bake_G 50000000 > bake_bake_G.log 2>&1
echo "  G done: $(grep -aE '\[DONE|\[ABORT' bake_bake_G.log | tail -1)"
echo "ARM G DONE. F-prime next (warm-start = CE final deltas)."
