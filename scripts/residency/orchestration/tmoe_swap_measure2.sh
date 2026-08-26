#!/bin/bash
# Real swap rates at the cliff. The simulated axis reported 0.000 for BOTH 1.75 and 2.0,
# yet those differ by 5.3 quality points -- so the simulation's logit scale does not match
# the model's and cannot explain the cliff. These are the two points that decide whether
# the collapse coincides with swaps actually reaching zero.
set -euo pipefail
cd /workspace/temporal-moe
KEY=${1:-gemma4_instruct}; MPATH=${2:-/dev/shm/gemma4-26b-it}; ARM=${3:-R8}; PREF=${4:-gemma4}; RHOS=${5:-1.75,2.0}
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TEMPORAL_COUNT_SWAPS=1
EXTRA=()
[ "$KEY" = "qwen35_instruct" ] && EXTRA=(--think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5)
for RHO in ${RHOS//,/ }; do
  echo "### swapmeasure2 $PREF RHO=$RHO $(date -u +%H:%M)"
  TEMPORAL_RHO=$RHO /workspace/venv_vllm312/bin/python -u \
    analysis/residency/instruct_genbench_vllm.py \
    --model $KEY --path "$MPATH" --arms $ARM "${EXTRA[@]}" \
    --record-as ${PREF}_swapmeasure_rho${RHO/./p} --csv-name screening_genbench.csv \
    --tasks "gsm8k_cot_zeroshot=200" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
done
echo "### swapmeasure2 $PREF DONE $(date -u +%H:%M)"
