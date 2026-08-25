#!/bin/bash
# HumanEval for a merged adapter, through the CHANNEL-AWARE producer its architecture needs.
#
#     GPU=3 remeasure_humaneval.sh gemma
#     GPU=2 remeasure_humaneval.sh qwen
#
# HumanEval is NOT servable from instruct_genbench_vllm.py's stock `humaneval` task. These
# models emit channel/think markers, so the stock extractor finds no code and scores every
# arm 0.000 -- which looks like a catastrophic result rather than a broken instrument. The
# published rows all come from bespoke producers with pass@1,channel-aware:
#     gemma4        -> humaneval_gemma.py   -> task humaneval_gemma_fixed
#     qwen35 / LFM  -> humaneval_think.py   -> task humaneval_think
#     gpt-oss       -> humaneval_gptoss.py  -> task humaneval_gptoss
# Same lesson as MMLU (mmlu_gptoss.py, acc,relaxed-extract): check the producer the paper
# reads, not the task name the harness happens to offer.
set -euo pipefail
ROOT=/workspace/temporal-moe
export TMOE_ROOT=$ROOT
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd $ROOT
PY=/workspace/venv_vllm312/bin/python
LOG=${LOG_DIR:-/workspace/rerun-logs}

case "${1:?usage: remeasure_humaneval.sh gemma|qwen}" in
  gemma)
    export CUDA_VISIBLE_DEVICES=${GPU:-3}
    echo "### gemma HUMANEVAL (channel-aware) $(date -u +%H:%M)"
    $PY -u analysis/residency/humaneval_gemma.py \
        --path /dev/shm/gemma4-rebuild-merged --arms free,R8,R16 \
        --tag gemma4_ce_rebuild --think off 2>&1 | tee $LOG/he_gemma.log ;;
  qwen)
    export CUDA_VISIBLE_DEVICES=${GPU:-2}
    for arm in free R8 R16; do
      echo "### qwen HUMANEVAL $arm (channel-aware) $(date -u +%H:%M)"
      $PY -u analysis/residency/humaneval_think.py --model qwen35_instruct \
          --path /dev/shm/qwen35-rebuild-merged --arm $arm \
          --tag qwen35_ce_rebuild --think off 2>&1 | tee -a $LOG/he_qwen.log
    done ;;
  *) echo "unknown: $1" >&2; exit 2 ;;
esac
echo "### $1 HUMANEVAL DONE $(date -u +%H:%M)"
