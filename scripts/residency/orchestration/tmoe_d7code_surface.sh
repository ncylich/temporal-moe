#!/bin/bash
# Complete the deliverable surface for the chosen recipe (d7code = D7 + 26.7% code).
# Canonical checkpoint is the MEDIAN of the three seeds by GSM8K recovery
# (+1.7 / +4.2 / +2.4 -> seed 2), which avoids both cherry-picking the best run
# and being stuck with an unlucky default.
# GSM8K, MBPP and HumanEval already exist; this fills IFEval, MMLU and WritingBench so the
# arm is reportable on all five cells against matched base references.
set -euo pipefail
cd /workspace/temporal-moe
WHAT=$1
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/models/gemma4-d7code_s2-merged
case "$WHAT" in
  ifeval)
    echo "### d7code IFEval full 541 $(date -u +%H:%M)"
    exec /workspace/venv_vllm312/bin/python -u analysis/residency/instruct_genbench_vllm.py \
      --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_d7code_s2_full \
      --tasks "ifeval=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90 ;;
  mmlu)
    echo "### d7code MMLU dual $(date -u +%H:%M)"
    exec /workspace/venv_vllm312/bin/python -u analysis/residency/mmlu_gptoss.py \
      --model gemma4_instruct --path $M --arms free,R8,R16 \
      --record-as gemma4_ce_d7code_s2_full_dual --gpu-mem 0.90 ;;
  wb)
    echo "### d7code WritingBench $(date -u +%H:%M)"
    export GPU=${CUDA_VISIBLE_DEVICES:?}
    exec scripts/residency/wb_arm.sh $M gemma4_d7code_s2 R8,R16 ;;
esac
