#!/bin/bash
# Merge + GSM8K for the short640 arm trained at the PUBLISHED budget (3.4M tokens,
# ~4 epochs over the 2,299 short rows). Pairs against short640-1pass (0.6M, -4.5) to
# separate "short responses help" from "one pass over short rows helps".
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=$(NEED_GB=100 TIMEOUT=7200 scripts/residency/wait_for_gpu.sh) || exit 1
export CUDA_VISIBLE_DEVICES=$G
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-short3p4m-merged
A=/workspace/olmoe-adapt/data/gemma_ce_short3p4m_adapter.pt
scripts/residency/disk_budget.sh || exit 3
[ -d $M ] || { /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
  --model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096_short640 \
  --max-seq 4096 --expert-lora-r 32 --opt adamw --micro-batch 16 --out $A --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true; }
/workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
export PATH=/workspace/venv_vllm312/bin:$PATH HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
/workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_short3p4m \
  --tasks "gsm8k_cot_zeroshot=200" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.94
echo "### short3p4m GSM8K DONE $(date -u +%H:%M)"
