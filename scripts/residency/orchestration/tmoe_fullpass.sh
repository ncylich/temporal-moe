#!/bin/bash
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
echo "### fullpass TRAIN 7356394 tokens (one complete pass over the pool) $(date -u +%H:%M)"
/workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py \
  --model /dev/shm/gemma4-26b-it --family gemma4 --no-unsloth --traj gemma4_d7_seq4096 \
  --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16 --accum 16 --lr 3e-5 \
  --tokens 7356394 --kl-anchor /workspace/instruct-traj/gemma4_d7_seq4096_klref.pt \
  --kl-weight 0.05 --out /workspace/olmoe-adapt/data/gemma_ce_fullpass_adapter.pt
echo "### fullpass TRAIN DONE $(date -u +%H:%M)"
