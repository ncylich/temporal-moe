#!/bin/bash
# Does the stock mbpp_instruct task score gemma4 at all? HumanEval floored at 0.000 for
# this model because gemma emits <channel|> reasoning spans that the stock filter mistakes
# for the answer, which is why humaneval_gemma.py exists. Check before spending a full run.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms free \
  --record-as gemma4_mbpp_smoke --tasks "mbpp_instruct=25" \
  --gen-cap 1536 --max-model-len 4096 --gpu-mem 0.90
