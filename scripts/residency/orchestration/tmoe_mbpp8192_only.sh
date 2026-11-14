#!/bin/bash
# MBPP only at 8192, for an arm whose HumanEval@8192 already exists.
set -euo pipefail
cd /workspace/temporal-moe
TAG=$1; M=$2
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /workspace/venv_vllm312/bin/python -u analysis/residency/mbpp_gemma.py \
  --path "$M" --arms free,R8,R16 --tag "${TAG}_m8192" \
  --max-tokens 8192 --max-model-len 9216 --gpu-mem 0.90
