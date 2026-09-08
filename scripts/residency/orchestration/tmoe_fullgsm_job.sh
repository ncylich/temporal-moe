#!/bin/bash
# Full GSM8K test split (1319 problems) for one record. Assumes CUDA_VISIBLE_DEVICES set.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe HF_ALLOW_CODE_EVAL=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ninja lives in the vllm venv; torch builds inline CUDA extensions at engine
# init and dies with FileNotFoundError if it is not on PATH (2026-08-26).
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
exec /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path "$2" --arms free,R8,R16 --record-as "$1"_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
