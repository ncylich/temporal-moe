#!/bin/bash
cd /workspace/temporal-moe
G=$(NEED_GB=100 TIMEOUT=7200 scripts/residency/wait_for_gpu.sh) || exit 1
echo "### kl003 gsm8k on GPU $G $(date -u +%H:%M)"
export TMOE_ROOT=/workspace/temporal-moe PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /root/.cache/huggingface/token) HF_ALLOW_CODE_EVAL=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=$G /workspace/venv_vllm312/bin/python -u \
  analysis/residency/instruct_genbench_vllm.py --model gemma4_instruct \
  --path /root/models/gemma4-kl003-merged --arms free,R8,R16 --record-as gemma4_ce_kl003 \
  --tasks "gsm8k_cot_zeroshot=200" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.94
echo "### kl003 gsm8k DONE $(date -u +%H:%M)"
