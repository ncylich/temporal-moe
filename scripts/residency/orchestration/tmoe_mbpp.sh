#!/bin/bash
# MBPP through the channel-aware producer. Assumes the slot wrapper set CUDA_VISIBLE_DEVICES.
#   tmoe_mbpp.sh <tag> <model-path> [limit]
set -euo pipefail
cd /workspace/temporal-moe
TAG=$1; MPATH=$2; LIM=${3:-}
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /workspace/venv_vllm312/bin/python -u analysis/residency/mbpp_gemma.py \
  --path "$MPATH" --arms free,R8,R16 --tag "$TAG" \
  --max-tokens ${MAXTOK:-1536} --max-model-len ${MML:-4096} --gpu-mem 0.90 ${LIM:+--limit $LIM}
