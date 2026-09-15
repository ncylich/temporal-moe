#!/bin/bash
# Generalise the swap-rate finding: does quality stay flat as swaps collapse, on other
# models / arms / adapted checkpoints? At RHO=0 this is exactly the published rule.
#   tmoe_skliar_gen.sh <model-key> <path> <arms> <record-prefix> <rho-list>
set -euo pipefail
cd /workspace/temporal-moe
KEY=$1; MPATH=$2; ARMS=$3; PREF=$4; RHOS=$5
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXTRA=()
[ "$KEY" = "qwen35_instruct" ] && EXTRA=(--think off --temperature 0.7 --top-p 0.8 --presence-penalty 1.5)
for RHO in ${RHOS//,/ }; do
  echo "### $PREF RHO=$RHO $(date -u +%H:%M)"
  TEMPORAL_RHO=$RHO /workspace/venv_vllm312/bin/python -u \
    analysis/residency/instruct_genbench_vllm.py \
    --model $KEY --path "$MPATH" --arms $ARMS "${EXTRA[@]}" \
    --record-as ${PREF}_rho${RHO/./p}_n1319 \
    --tasks "gsm8k_cot_zeroshot=0" --gen-cap 2048 --max-model-len 5632 --gpu-mem 0.90
done
echo "### $PREF SWEEP DONE $(date -u +%H:%M)"
