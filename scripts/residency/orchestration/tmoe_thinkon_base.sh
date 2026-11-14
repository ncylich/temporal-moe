#!/bin/bash
# Matched base reference for the gemma think-on cells. The adapted arm (gemma4_ce_thinkon)
# ran at max_gen_toks=16384 but the only base think-on arm ran at 4096, so the adapted
# arm's +0.0/+3.5 were absolute scores with no valid reference -- a think-on model given
# 4x the budget is a different experiment. Same budget, full split.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms free,R8,R16 \
  --think on --record-as gemma4_think_on_16k_n1319 \
  --tasks "gsm8k_cot_zeroshot=0" --gen-cap 16384 --max-model-len 18432 --gpu-mem 0.90
