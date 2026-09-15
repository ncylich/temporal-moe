#!/usr/bin/env bash
# Carve the full qwen Skliar curve at OUR memory (C=8 of 256, 3.1%): lambda points between
# the known ends (0: 5.63 loads/86.0, 0.4: 1.01/74.5) and below the budget. GSM8K n=1319.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 TMOE_PRIO=4
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
for LAM in 0.1 0.2 0.3 0.5 0.7; do
  ( export TEMPORAL_WALKER=cache_bias TEMPORAL_CB_J=1 TEMPORAL_CB_K=8 TEMPORAL_COUNT_SWAPS=1 TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_CB_C=8
    echo "### skliar C8 qwen lam$LAM $(date -u +%H:%M)"
    $L $PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct --path /root/models/qwen35-35b-a3b --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 --arms R8 --record-as qwen35_skliar_C8_lam${LAM/./p}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90 )
done
echo "### skliar C8 qwen sweep DONE $(date -u +%H:%M)"
