#!/bin/bash
# Confirm the headline operating points at n=1319. The frontier was mapped with n=200
# screening runs (+/-3-4 points each), which is fine for locating a cliff but not for
# quoting an operating point in a paper.
set -euo pipefail
cd /workspace/temporal-moe
ARM=$1; RHO=$2; PREF=$3
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEMPORAL_COUNT_SWAPS=1
echo "### confirm $PREF $ARM RHO=$RHO $(date -u +%H:%M)"
TEMPORAL_RHO=$RHO /workspace/venv_vllm312/bin/python -u \
  analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms $ARM \
  --record-as ${PREF}_n1319 --tasks "gsm8k_cot_zeroshot=0" \
  --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### confirm $PREF DONE $(date -u +%H:%M)"
