#!/bin/bash
# qwen R128 = 50% resident, E/R=2 -- pairs with gemma R64 at the same E/R on a model with
# twice the experts. Two independent E/R=2 points and two independent E/R=8 points is what
# separates "E/R sets the cliff" from "each model has its own curve".
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEMPORAL_COUNT_SWAPS=1
for RHO in 0 3.0 3.5 4.0; do
  echo "### frontier qwenR128 RHO=$RHO $(date -u +%H:%M)"
  TEMPORAL_RHO=$RHO /workspace/venv_vllm312/bin/python -u \
    analysis/residency/instruct_genbench_vllm.py \
    --model qwen35_instruct --path /root/models/qwen35-35b-a3b --arms R128 \
    --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
    --record-as qwenr128_swapmeasure_rho${RHO/./p} --csv-name screening_genbench.csv \
    --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
done
echo "### frontier qwenR128 DONE $(date -u +%H:%M)"
