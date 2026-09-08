#!/usr/bin/env bash
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null)}
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
exec /workspace/venv_vllm312/bin/python -u analysis/residency/mmlu_gptoss.py \
  --model qwen35_instruct --path /root/models/qwen35-rebuild-merged --arms free,R8,R32 \
  --think off --record-as qwen35_ce_rebuild_n_dual --gpu-mem 0.90
