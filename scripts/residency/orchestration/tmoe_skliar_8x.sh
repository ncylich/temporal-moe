#!/usr/bin/env bash
# Within-model reduction test, feasible direction: Skliar's cache at the 8x points we already
# hold ours at (gemma R16 -> C=16 of 128; qwen R32 -> C=32 of 256), lambda curves with loads.
# Pairs with the existing 16x/32x C=8 curves: does our lead widen from 8x to 16x/32x within
# the same model? GSM8K n=1319.
set -uo pipefail; cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 TMOE_PRIO=4
L=scripts/residency/gpu_lease.sh; PY=/workspace/venv_vllm312/bin/python
CB="TEMPORAL_WALKER=cache_bias TEMPORAL_CB_J=1 TEMPORAL_CB_K=8 TEMPORAL_COUNT_SWAPS=1"
for LAM in 0.2 0.4 0.6; do
  echo "### skliar C16 gemma lam$LAM $(date -u +%H:%M)"
  ( export $CB TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_CB_C=16; $L $PY -u analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms R8 --record-as gemma4_skliar_C16_lam${LAM/./p}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90 )
  echo "### skliar C32 qwen lam$LAM $(date -u +%H:%M)"
  ( export $CB TEMPORAL_CB_LAMBDA=$LAM TEMPORAL_CB_C=32; $L $PY -u analysis/residency/instruct_genbench_vllm.py --model qwen35_instruct --path /root/models/qwen35-35b-a3b --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 --arms R8 --record-as qwen35_skliar_C32_lam${LAM/./p}_n1319 --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90 )
done
echo "### skliar 8x DONE $(date -u +%H:%M)"
