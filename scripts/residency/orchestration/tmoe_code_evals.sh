#!/bin/bash
# Both code surfaces for one merged arm. Does the "adapter does not fix code" null hold
# across training runs, or was it specific to the rebuild run?
#   tmoe_code_evals.sh <tag> <merged-dir>
set -euo pipefail
cd /workspace/temporal-moe
TAG=$1; M=$2
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "### $TAG MBPP $(date -u +%H:%M)"
/workspace/venv_vllm312/bin/python -u analysis/residency/mbpp_gemma.py \
  --path "$M" --arms free,R8,R16 --tag "$TAG" --max-model-len 4096 --gpu-mem 0.90
echo "### $TAG HumanEval $(date -u +%H:%M)"
/workspace/venv_vllm312/bin/python -u analysis/residency/humaneval_gemma.py \
  --path "$M" --arms free,R8,R16 --tag "$TAG"
echo "### $TAG CODE DONE $(date -u +%H:%M)"
