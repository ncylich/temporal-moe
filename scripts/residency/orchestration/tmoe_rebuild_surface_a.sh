#!/bin/bash
# Lane A of the rebuild arm's remaining surface: IFEval then MMLU.
# GSM8K is already done at the full split; HumanEval and WritingBench are lane B.
# Settings match grid_parallel.sh so these cells are comparable to the committed grid.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/models/gemma4-rebuild-merged
PY=/workspace/venv_vllm312/bin/python
echo "### rebuild IFEval (full 541) $(date -u +%H:%M)"
$PY -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 --record-as gemma4_ce_rebuild_full \
  --tasks "ifeval=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### rebuild MMLU $(date -u +%H:%M)"
$PY -u analysis/residency/mmlu_gptoss.py \
  --model gemma4_instruct --path $M --arms free,R8,R16 \
  --record-as gemma4_ce_rebuild_full_dual --gpu-mem 0.90
echo "### rebuild LANE-A DONE $(date -u +%H:%M)"
