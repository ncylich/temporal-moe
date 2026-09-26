#!/bin/bash
# Complete qwen's R-vs-critical-rate curve. qwen R8 (E/R=32) breaks at 0.83 and R32
# (E/R=8) at ~0.78 -- nearly flat where gemma falls 0.53 -> 0.25 over the same span. R16
# and R64 fill the middle and the low end: if qwen stays ~0.8 throughout, the
# memory-for-bandwidth substitution simply does not exist on this model.
set -euo pipefail
cd /workspace/temporal-moe
ARM=$1; RHOS=$2
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEMPORAL_COUNT_SWAPS=1
for RHO in ${RHOS//,/ }; do
  echo "### qwencurve $ARM RHO=$RHO $(date -u +%H:%M)"
  TEMPORAL_RHO=$RHO /workspace/venv_vllm312/bin/python -u \
    analysis/residency/instruct_genbench_vllm.py \
    --model qwen35_instruct --path /root/models/qwen35-35b-a3b --arms $ARM \
    --think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5 \
    --record-as qwen${ARM}c_swapmeasure_rho${RHO/./p} --csv-name screening_genbench.csv \
    --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
done
echo "### qwencurve $ARM DONE $(date -u +%H:%M)"
