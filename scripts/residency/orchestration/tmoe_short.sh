#!/bin/bash
# $1 = token budget, $2 = tag suffix. Two arms separate LENGTH from REPETITION:
#   835k  -> one pass over the short subset (no repetition, small budget)
#   3.4M  -> published budget over the same subset (~4 epochs)
# If only the 3.4M arm helps, repetition matters; if both help, length is the mechanism;
# if neither, the hypothesis is dead.
set -euo pipefail
cd /workspace/temporal-moe
TOK=$1; SUF=$2
G=$(NEED_GB=100 TIMEOUT=7200 scripts/residency/wait_for_gpu.sh) || exit 1
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G
KL=/workspace/instruct-traj/gemma4_d7_seq4096_short640_klref.pt
echo "### short-$SUF on GPU $G, ${TOK} tokens $(date -u +%H:%M)"
COMMON="--model /dev/shm/gemma4-26b-it --family gemma4 --no-unsloth
        --traj gemma4_d7_seq4096_short640 --max-seq 4096 --expert-lora-r 32
        --opt adamw --micro-batch 16
        --out /workspace/olmoe-adapt/data/gemma_ce_short${SUF}_adapter.pt"
if [ ! -s "$KL" ]; then
  echo "### short-$SUF KL precompute $(date -u +%H:%M)"
  /workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON --precompute-kl $KL
fi
/workspace/venv_fla/bin/python -u analysis/residency/train_gemma_ce.py $COMMON \
  --accum 16 --lr 3e-5 --tokens $TOK --kl-anchor $KL --kl-weight 0.05
echo "### short-$SUF TRAIN DONE $(date -u +%H:%M)"
/workspace/venv_fla/bin/python scripts/residency/mirror_artifact.py \
  --path /workspace/olmoe-adapt/data/gemma_ce_short${SUF}_adapter.pt --kind adapter
echo "### short-$SUF ALL DONE $(date -u +%H:%M)"
