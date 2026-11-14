#!/bin/bash
# gemma at R4 = 3.1% resident -- the SAME resident fraction as qwen R8, on a different
# model. If the cliff is set by fraction rather than model, these two must agree. This is
# the cleanest test of the frontier law available without training anything.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEMPORAL_COUNT_SWAPS=1
for RHO in 0 1.0 1.5 2.0; do
  echo "### frontier R4 RHO=$RHO $(date -u +%H:%M)"
  TEMPORAL_RHO=$RHO /workspace/venv_vllm312/bin/python -u \
    analysis/residency/instruct_genbench_vllm.py \
    --model gemma4_instruct --path /dev/shm/gemma4-26b-it --arms R4 \
    --record-as gemma4r4far_swapmeasure_rho${RHO/./p} --csv-name screening_genbench.csv \
    --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
done
echo "### frontier R4 DONE $(date -u +%H:%M)"
