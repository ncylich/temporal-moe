#!/bin/bash
set -euo pipefail
SUF=$1
cd /workspace/temporal-moe
G=$(NEED_GB=100 TIMEOUT=7200 scripts/residency/wait_for_gpu.sh) || exit 1
export TMOE_ROOT=/workspace/temporal-moe HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PATH=/workspace/venv_fla/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/dev/shm/gemma4-26b-it; M=/root/models/gemma4-short${SUF}-merged
A=/workspace/olmoe-adapt/data/gemma_ce_short${SUF}_adapter.pt
scripts/residency/disk_budget.sh || exit 3
if [ ! -d $M ]; then
  echo "### short-$SUF MERGE on GPU $G $(date -u +%H:%M)"
  CUDA_VISIBLE_DEVICES=$G /workspace/venv_fla/bin/python analysis/residency/train_gemma_ce.py \
    --model $B --family gemma4 --no-unsloth --traj gemma4_d7_seq4096_short640 --max-seq 4096 \
    --expert-lora-r 32 --out $A --merge-out $M
  cp $B/processor_config.json $M/ 2>/dev/null || true
fi
/workspace/venv_fla/bin/python analysis/residency/verify_merge.py --base $B --merged $M
echo "### short-$SUF VERIFY DONE $(date -u +%H:%M)"
export PATH=/workspace/venv_vllm312/bin:$PATH HF_ALLOW_CODE_EVAL=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
CUDA_VISIBLE_DEVICES=$G /workspace/venv_vllm312/bin/python -u \
  analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct --path $M \
  --arms free,R8,R16 --record-as gemma4_ce_short${SUF} \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.94
echo "### short-$SUF GSM8K DONE $(date -u +%H:%M)"
