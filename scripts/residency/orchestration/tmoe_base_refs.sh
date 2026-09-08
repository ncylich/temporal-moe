#!/bin/bash
# Matched base references the rebuild comparison needs: IFEval at the FULL 541 prompts
# (base was only ever run at 200) and MMLU through the same dual producer used for the
# adapted arm. Without these, rebuild's IFEval/MMLU cells have nothing valid to sit against.
set -euo pipefail
cd /workspace/temporal-moe
export TMOE_ROOT=/workspace/temporal-moe
export PATH=/workspace/venv_vllm312/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_XET=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=/dev/shm/gemma4-26b-it
PY=/workspace/venv_vllm312/bin/python
echo "### base IFEval full 541 $(date -u +%H:%M)"
$PY -u analysis/residency/instruct_genbench_vllm.py \
  --model gemma4_instruct --path $B --arms free,R8,R16 --record-as gemma4_instruct_full \
  --tasks "ifeval=0" --gen-cap 2048 --max-model-len 4096 --gpu-mem 0.90
echo "### base MMLU dual $(date -u +%H:%M)"
$PY -u analysis/residency/mmlu_gptoss.py \
  --model gemma4_instruct --path $B --arms free,R8,R16 \
  --record-as gemma4_instruct_full_dual --gpu-mem 0.90
echo "### BASE-REFS DONE $(date -u +%H:%M)"
