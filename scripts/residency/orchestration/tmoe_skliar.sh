#!/bin/bash
# Baseline #3 sweep: cache-conditional experts at R8, full GSM8K split.
# RHO is read once at import, so each value needs its own process.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for RHO in 0.25 0.5 1.0; do
  echo "### skliar RHO=$RHO $(date -u +%H:%M)"
  TEMPORAL_RHO=$RHO /workspace/venv_vllm312/bin/python -u \
    analysis/residency/instruct_genbench_vllm.py \
    --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms R8 \
    --record-as gemma4_skliar_rho${RHO/./p}_n1319 \
    --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
done
echo "### skliar SWEEP DONE $(date -u +%H:%M)"
