#!/bin/bash
# Adapted think-on arm at the SAME 16384 budget as the base reference just measured, full
# split. The committed think-on cells were n=200 adapted vs n=200 base at 4096 -- never a
# matched pair.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path /root/models/gemma4-thinkon-merged --arms free,R8,R16 \
  --think on --record-as gemma4_ce_thinkon_16k_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 16384 --max-model-len 18432 --gpu-mem 0.90
